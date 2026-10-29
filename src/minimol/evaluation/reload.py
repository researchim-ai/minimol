"""Reload a checkpoint and score the validation split.

Conformal intervals, when reported, are fit on one half of the validation
groups and checked on the other half. The test split is not read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from minimol.data.normalize import LabelScaler, pivot_reg
from minimol.errors import MinimolError
from minimol.evaluation.calibration import split_conformal_intervals
from minimol.evaluation.latency import measure_latency
from minimol.evaluation.metrics import regression_report
from minimol.jsonutil import write_json
from minimol.training.loop import move_batch, resolve_device
from minimol.training.train_cds import CDSDataset, _score, collate_cds, load_cds_rows
from minimol.training.train_reg import RegDataset, collate_reg


def _group_conformal(y, yhat, groups, alpha: float = 0.1) -> dict | None:
    groups = np.asarray(groups)
    unique = sorted(set(groups.tolist()))
    if len(unique) < 4:
        return None
    calibration_groups = set(unique[: len(unique) // 2])
    calibration = np.array([group in calibration_groups for group in groups])
    if int(calibration.sum()) < 2 or int((~calibration).sum()) < 1:
        return None
    intervals = split_conformal_intervals(y[calibration], yhat[calibration], yhat[~calibration], alpha=alpha)
    held = y[~calibration]
    covered = (held >= intervals["lower"]) & (held <= intervals["upper"])
    return {
        "alpha": alpha,
        "quantile": intervals["quantile"],
        "n_calibration": int(calibration.sum()),
        "n_report": int((~calibration).sum()),
        "coverage": float(np.mean(covered)),
        "fit_split": "validation_group_half",
        "note": "Fit on half of the validation groups. The test split was not used.",
    }


def evaluate_reg_checkpoint(checkpoint: Path, data_path: Path, out_dir: Path, repeats: int = 5) -> dict:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    if meta.get("task") != "reg":
        raise MinimolError(f"checkpoint task is {meta.get('task')}, expected reg")
    from minimol.models.reg_cnn import MinimolReg

    model = MinimolReg.from_metadata(meta)
    model.load_state_dict(blob["model_state"])
    model.eval()
    frame = pd.read_parquet(data_path)
    assays = list(meta["assay_names"])
    rows = [row for row in pivot_reg(frame, assays) if row["split"] == "validation"]
    if not rows:
        raise MinimolError("validation split is empty")
    scaler = LabelScaler().load_state_dict(meta["scaler"])
    weights = np.ones((len(rows), len(assays)), dtype=np.float64)
    loader = DataLoader(RegDataset(rows, scaler, weights), batch_size=min(16, len(rows)), shuffle=False, collate_fn=collate_reg)
    device = resolve_device("cpu")
    model.to(device)
    actuals, preds, masks = [], [], []
    first_batch = None
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if first_batch is None:
                first_batch = batch
            pred = scaler.inverse_torch(model(batch["x"], batch["pad_mask"]))
            actuals.append(batch["y_raw"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            masks.append(batch["y_mask"].cpu().numpy())
    y = np.concatenate(actuals)
    yhat = np.concatenate(preds)
    mask = np.concatenate(masks)
    report = regression_report(y, yhat, mask, assays)
    report["per_assay"] = report["per_target"]
    groups = [row["group_id"] for row in rows]
    intervals = {}
    for index, name in enumerate(assays):
        chosen = mask[:, index] > 0.5
        if int(chosen.sum()) < 4:
            continue
        group_slice = [groups[i] for i, flag in enumerate(chosen) if flag]
        intervals[name] = _group_conformal(y[chosen, index], yhat[chosen, index], group_slice)
    latency = None
    if first_batch is not None:
        batch = first_batch

        def _run():
            with torch.no_grad():
                model(batch["x"], batch["pad_mask"])

        latency = measure_latency(_run, warmup=1, repeats=repeats)
        latency["device"] = "cpu"
        latency["batch_size"] = int(batch["x"].shape[0])
    payload = {
        "task": "reg",
        "split": "validation",
        "metrics": report,
        "conformal": intervals,
        "latency": latency,
        "param_count": meta.get("param_count"),
        "research_use_only": True,
        "clinical_use": False,
        "checkpoint": str(checkpoint),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)
    return payload


def evaluate_cds_checkpoint(checkpoint: Path, data_path: Path, out_dir: Path, repeats: int = 3) -> dict:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = blob["metadata"]
    if meta.get("task") != "cds":
        raise MinimolError(f"checkpoint task is {meta.get('task')}, expected cds")
    from minimol.models.cds_cnn import MinimolCDS

    model = MinimolCDS(meta["preset"], **(meta.get("overrides") or {}))
    model.load_state_dict(blob["model_state"])
    model.eval()
    rows, teacher_kind = load_cds_rows(Path(data_path))
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not val_rows:
        raise MinimolError("validation split is empty")
    loader = DataLoader(CDSDataset(val_rows), batch_size=min(4, len(val_rows)), shuffle=False, collate_fn=collate_cds)
    device = resolve_device("cpu")
    model.to(device)
    scores = _score(model, loader, device)
    first = next(iter(loader))
    first = move_batch(first, device)

    def _run():
        with torch.no_grad():
            model(first["x"], first["pad_mask"])

    latency = measure_latency(_run, warmup=1, repeats=repeats)
    latency["device"] = "cpu"
    payload = {
        "task": "cds",
        "split": "validation",
        "metrics": scores,
        "teacher_kind": teacher_kind,
        "latency": latency,
        "receptive_field_bp": meta.get("receptive_field_bp"),
        "research_use_only": True,
        "clinical_use": False,
        "note": "Per-base probabilities are not a GFF annotation.",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)
    return payload
