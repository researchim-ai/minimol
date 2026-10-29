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


def reg_metric_value(report: dict) -> dict:
    """Checkpoint score is negative MAE.

    Spearman on a handful of validation rows can prefer an early epoch whose
    predictions are shifted by several units. MAE keeps the weights on the
    label scale. Spearman is still recorded and reported.
    """

    mae = report.get("macro_mae")
    spearman = report.get("macro_spearman")
    return {
        "value": None if mae is None else -float(mae),
        "val_mae": mae,
        "val_spearman": spearman,
    }
