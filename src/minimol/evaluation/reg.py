"""Regulatory prediction collection."""

from __future__ import annotations

import numpy as np
import torch

from minimol.training.loop import move_batch


def collect_reg_predictions(model, loader, scaler, device) -> dict:
    actuals = []
    preds = []
    masks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = scaler.inverse_torch(model(batch["x"], batch["pad_mask"]))
            actuals.append(batch["y_raw"].detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            masks.append(batch["y_mask"].detach().cpu().numpy())
    return {
        "y": np.concatenate(actuals, axis=0),
        "yhat": np.concatenate(preds, axis=0),
        "mask": np.concatenate(masks, axis=0),
    }


def reg_metric_value(report: dict) -> float | None:
    if report.get("macro_spearman") is not None:
        return float(report["macro_spearman"])
    if report.get("macro_mae") is None:
        return None
    return float(-report["macro_mae"])
