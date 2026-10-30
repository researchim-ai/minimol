"""Split audit entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from minimol.data.homology import render_leakage_markdown, scan_leakage
from minimol.data.manifest import read_manifest
from minimol.data.split import assert_genomic_windows_grouped
from minimol.errors import MinimolError, SplitLeakageError


def cross_split_identity(frame: pd.DataFrame) -> dict:
    """Shared identities after standardization. A scaffold match is not a duplicate label."""

    def shared(column: str) -> list[dict]:
        if column not in frame.columns or "split" not in frame.columns:
            return []
        hits = []
        usable = frame.dropna(subset=[column])
        published = usable[usable["split"].astype(str).isin(["train", "validation", "test"])]
        for value, part in published.groupby(column):
            splits = sorted({str(item) for item in part["split"].tolist()})
            if len(splits) > 1:
                hits.append({column: str(value), "splits": splits, "n": int(len(part))})
        return hits

    quarantine = 0
    if "split" in frame.columns:
        quarantine = int((frame["split"].astype(str) == "quarantine").sum())
    return {
        "inchikey_cross_split": shared("inchikey"),
        "scaffold_cross_split": shared("scaffold_id"),
        "n_quarantine_rows": quarantine,
        "official_test_unchanged": quarantine == 0,
        "standardization_policy": (
            "RDKit canonical isomeric SMILES; stereochemistry kept; "
            "tautomers not canonicalized; salts recorded on the standardized record"
        ),
        "benchmark_protocol": "published TDC train_val and test files; train/validation from the scaffold splitter",
        "note": (
            "A shared InChIKey is the same molecule after standardization. "
            "A shared scaffold is a chemical series, not a duplicated measurement. "
            "Rows moved to quarantine are not part of an unchanged official test."
        ),
    }


def split_audit(manifest_path: Path, report_path: Path | None = None) -> dict:
    manifest = read_manifest(manifest_path)
    data_path = Path(manifest_path).parent / manifest["data_file"]
    frame = pd.read_parquet(data_path)
    report = scan_leakage(frame)
    if manifest.get("task") == "cds":
        try:
            assert_genomic_windows_grouped(frame)
            report.notes.append("genomic windows are grouped by assembly and chromosome")
        except SplitLeakageError as exc:
            report.group_cross_split.append({"genomic_windows": str(exc)})
    payload = report.as_dict()
    payload["manifest_sha256"] = manifest.get("sha256")
    destination = report_path or (Path(manifest_path).parent / "leakage_audit.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_leakage_markdown(report))
    (destination.parent / "leakage_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not report.passed:
        raise SplitLeakageError("split audit failed; see leakage_audit.md")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimol.data.split_audit")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    try:
        split_audit(Path(args.manifest), Path(args.report) if args.report else None)
    except (MinimolError, SplitLeakageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("split audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
