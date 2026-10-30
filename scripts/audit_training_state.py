"""Read local run summaries and train/validation data; never evaluate test labels.

Run with PYTHONPATH=src .venv/bin/python scripts/audit_training_state.py.
Outputs an aggregate report under gitignored artifacts/reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from minimol.data.fetch_public import TDC_METRICS
from minimol.jsonutil import write_json


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build_audit(root: Path) -> dict:
    artifacts = root / "artifacts"
    reg = []
    for name in ("reg_5m_malinois", "reg_5m_malinois_v2"):
        path = artifacts / name / "metrics.json"
        if not path.exists():
            continue
        run = read(path)
        reg.append({
            "run": name,
            "source": str(path.relative_to(root)),
            "seed": run["seed"],
            "steps": run["steps"],
            "epochs": len(run["history"]),
            "data_sha256": run["data_sha256"],
            "metrics": run["metrics"],
            "history": run["history"],
        })
    admet = []
    overlap = []
    for path in sorted((artifacts / "tdc_admet_5m").glob("*/metrics.json")):
        name = path.parent.name
        baseline_path = artifacts / "tdc_admet_baselines" / name / "metrics.json"
        manifest_path = root / "data/processed/tdc_admet" / name / "manifest.json"
        if not baseline_path.exists() or not manifest_path.exists():
            continue
        run, baseline, manifest = read(path), read(baseline_path), read(manifest_path)
        gnn, tree = run["endpoints"][name], baseline["endpoints"][name]
        primary = {"roc-auc": "auroc", "pr-auc": "auprc"}.get(TDC_METRICS[name], TDC_METRICS[name])
        hashes_match = run["data_sha256"] == baseline["data_sha256"] == manifest["sha256"]
        comparable = hashes_match and gnn.get(primary) is not None and tree.get(primary) is not None
        winner = None
        if comparable:
            difference = gnn[primary] - tree[primary]
            if primary == "mae":
                difference = -difference
            winner = "tie" if difference == 0 else "gnn" if difference > 0 else "xgboost"
        admet.append({
            "endpoint": name,
            "primary_metric": primary,
            "hashes_match": hashes_match,
            "comparable_on_primary": comparable,
            "winner_on_primary": winner,
            "gnn": gnn,
            "baseline": tree,
            "seed": run["seed"],
            "epochs": len(run["history"]),
            "source": str(path.relative_to(root)),
            "baseline_source": str(baseline_path.relative_to(root)),
            "quarantined_molecules": manifest["n_smiles_quarantined"],
        })
        frame = pd.read_parquet(manifest_path.with_name("data.parquet"),
                                filters=[("split", "in", ["train", "validation"])],
                                columns=["split", "scaffold_id", "inchikey"])
        counts = {}
        for column in ("scaffold_id", "inchikey"):
            grouped = frame.dropna(subset=[column]).groupby(column)["split"].nunique()
            counts[column] = int((grouped > 1).sum())
        if any(counts.values()):
            overlap.append({"endpoint": name, "cross_split_identifiers": counts})
    incomplete = [path.name for path in artifacts.glob("reg_5m_malinois_v2_seed*")
                  if not (path / "metrics.json").exists()]
    return {
        "snapshot_date": "2026-10-30",
        "scope": "Saved validation reports; split overlaps use train/validation identifiers only.",
        "reg": reg,
        "admet": admet,
        "admet_summary": {
            "endpoints": len(admet),
            "gnn_wins": sum(row["winner_on_primary"] == "gnn" for row in admet),
            "xgboost_wins": sum(row["winner_on_primary"] == "xgboost" for row in admet),
            "missing_primary": sum(row["gnn"].get(row["primary_metric"]) is None for row in admet),
            "incomparable_hashes": sum(not row["hashes_match"] for row in admet),
        },
        "train_validation_overlaps": overlap,
        "incomplete_reg_runs": incomplete,
        "limitations": [
            "Single training seed per completed recipe; validation comparisons do not establish significance.",
            "Saved hash equality is checked against the manifest, not rehashed from all data files.",
            "Scaffold and InChIKey overlap counts need molecule-level interpretation and an explicit split policy.",
            "No test inference or training is performed by this audit.",
        ],
    }


def probe_reg_padding(root: Path) -> dict:
    """Compare native length and right N padding using the same real checkpoint."""
    import torch

    from minimol.data.normalize import LabelScaler
    from minimol.data.sequences import one_hot_sequence
    from minimol.models.reg_cnn import MinimolReg

    torch.set_num_threads(2)
    checkpoint = root / "artifacts/reg_5m_malinois_v2/checkpoint_best.pt"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MinimolReg.from_metadata(blob["metadata"]).eval()
    model.load_state_dict(blob["model_state"])
    scaler = LabelScaler().load_state_dict(blob["metadata"]["scaler"])
    width = int(blob["metadata"]["length_max"])
    frame = pd.read_parquet(root / "data/processed/malinois/data.parquet",
                            columns=["sequence"], filters=[("split", "==", "validation")])
    frame = frame.drop_duplicates("sequence")
    frame["length"] = frame.sequence.str.len()
    short = frame.loc[frame.length < width].sort_values("length")
    measurements = []
    for sequence in short.sequence.head(3):
        x = one_hot_sequence(sequence).unsqueeze(0)
        padded = torch.zeros(1, 5, width)
        padded[:, 4, :] = 1
        padded[:, :, :len(sequence)] = x
        mask = torch.arange(width).unsqueeze(0) < len(sequence)
        with torch.no_grad():
            native = scaler.inverse_torch(model(x, torch.ones(1, len(sequence), dtype=torch.bool)))
            fixed = scaler.inverse_torch(model(padded, mask))
        measurements.append({"length_bp": len(sequence),
                             "max_abs_difference_log2fc": float((native - fixed).abs().max())})
    return {"checkpoint": str(checkpoint.relative_to(root)),
            "fixed_width_bp": width, "n_short_validation_sequences": len(short),
            "n_validation_sequences": len(frame), "examples": measurements,
            "note": "Three shortest validation sequences. This probe is not a population-average error estimate."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--probe-padding", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report = build_audit(root)
    if args.probe_padding:
        report["reg_padding_probe"] = probe_reg_padding(root)
    out = args.out or root / "artifacts/reports/project_audit_2026-10-30.json"
    write_json(out, report)
    print(json.dumps(report["admet_summary"], indent=2))
    print(out)


if __name__ == "__main__":
    main()
