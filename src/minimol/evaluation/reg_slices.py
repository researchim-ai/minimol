"""Validation errors broken out by length, chromosome, activity, and label SE.

Train and validation stay in the original log2FC scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from minimol.data.normalize import LabelScaler, pivot_reg
from minimol.evaluation.metrics import mae, pearson, spearman
from minimol.jsonutil import write_json
from minimol.training.loop import move_batch, resolve_device
from minimol.training.train_reg import RegDataset, collate_reg


def length_bin(length: int) -> str:
    if length < 160:
        return "<160"
    if length < 190:
        return "160-189"
    if length < 200:
        return "190-199"
    return "200"


def _cell(y, yhat, mask) -> dict:
    chosen = np.asarray(mask, dtype=bool) & np.isfinite(y) & np.isfinite(yhat)
    return {
        "n": int(chosen.sum()),
        "mae": mae(y, yhat, chosen),
        "pearson": pearson(y, yhat, chosen),
        "spearman": spearman(y, yhat, chosen),
    }


def slice_metrics(rows: list[dict], yhat: np.ndarray, assays: list[str]) -> dict:
    """`yhat` is aligned with `rows` and already in the raw log2FC scale."""

    y = np.stack([row["y"] for row in rows])
    mask = np.stack([row["mask"] for row in rows])
    se = np.stack([row["se"] for row in rows])
    yhat = np.asarray(yhat, dtype=np.float64)
    by_assay = {}
    by_length = {}
    by_chromosome = {}
    by_activity = {}
    by_se = {}
    lengths = np.asarray([len(row["sequence"]) for row in rows])
    chromosomes = [str(row.get("chromosome", "")) for row in rows]
    for index, assay in enumerate(assays):
        by_assay[assay] = _cell(y[:, index], yhat[:, index], mask[:, index])
        valid = mask[:, index]
        if int(valid.sum()) >= 3:
            edges = np.quantile(y[valid, index], [0.33, 0.66])
        else:
            edges = np.array([0.0, 0.0])
        for name, chosen in (
            ("low", valid & (y[:, index] <= edges[0])),
            ("mid", valid & (y[:, index] > edges[0]) & (y[:, index] <= edges[1])),
            ("high", valid & (y[:, index] > edges[1])),
        ):
            by_activity.setdefault(assay, {})[name] = _cell(y[:, index], yhat[:, index], chosen)
        for name, chosen in (
            ("se_le_0.5", valid & (se[:, index] <= 0.5)),
            ("se_0.5_to_1", valid & (se[:, index] > 0.5) & (se[:, index] <= 1.0)),
        ):
            by_se.setdefault(assay, {})[name] = _cell(y[:, index], yhat[:, index], chosen)
    for bin_name in ("<160", "160-189", "190-199", "200"):
        chosen_rows = np.asarray([length_bin(int(length)) == bin_name for length in lengths])
        per_assay = {}
        for index, assay in enumerate(assays):
            per_assay[assay] = _cell(y[:, index], yhat[:, index], mask[:, index] & chosen_rows)
        by_length[bin_name] = {"n_sequences": int(chosen_rows.sum()), "per_assay": per_assay}
    for chromosome in sorted(set(chromosomes)):
        chosen_rows = np.asarray([item == chromosome for item in chromosomes])
        per_assay = {}
        for index, assay in enumerate(assays):
            per_assay[assay] = _cell(y[:, index], yhat[:, index], mask[:, index] & chosen_rows)
        by_chromosome[chromosome] = {"n_sequences": int(chosen_rows.sum()), "per_assay": per_assay}
    return {
        "scale": "raw_log2fc",
        "per_assay": by_assay,
        "by_length": by_length,
        "by_chromosome": by_chromosome,
        "by_activity_tertile": by_activity,
        "by_label_se": by_se,
    }


def score_reg_checkpoint(checkpoint: Path, data_path: Path, out_dir: Path, splits=("validation", "train")) -> dict:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    from minimol.models.reg_cnn import MinimolReg

    model = MinimolReg.from_metadata(meta)
    model.load_state_dict(blob["model_state"])
    model.eval()
    frame = pd.read_parquet(data_path)
    assays = list(meta["assay_names"])
    rows = pivot_reg(frame, assays)
    scaler = LabelScaler().load_state_dict(meta["scaler"])
    device = resolve_device("auto")
    model.to(device)
    reports = {}
    for split in splits:
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        loader = DataLoader(
            RegDataset(subset, scaler, np.ones((len(subset), len(assays)))),
            batch_size=128,
            shuffle=False,
            collate_fn=collate_reg,
        )
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                pred = scaler.inverse_torch(model(batch["x"], batch["pad_mask"]))
                preds.append(pred.float().cpu().numpy())
        yhat = np.concatenate(preds)
        reports[split] = slice_metrics(subset, yhat, assays)
    payload = {
        "task": "reg_slices",
        "padding_policy": model.padding_policy,
        "head_kind": model.head_kind,
        "checkpoint": str(checkpoint),
        "scale": "raw_log2fc",
        "splits": reports,
        "research_use_only": True,
        "clinical_use": False,
        "note": "Train and validation are both inverse-scaled log2FC. The test split was not scored.",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "slice_metrics.json", payload)
    return payload
