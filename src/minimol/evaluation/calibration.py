"""Calibration on a held-out calibration split. Never fit this on the test split."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from minimol.evaluation.metrics import brier, expected_calibration_error


def conformal_quantile(residuals, alpha: float = 0.1) -> float:
    values = np.sort(np.asarray(residuals, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("calibration residuals are empty")
    rank = int(np.ceil((values.size + 1) * (1 - alpha)))
    rank = min(max(rank, 1), int(values.size))
    return float(values[rank - 1])


def split_conformal_intervals(y_cal, yhat_cal, yhat_test, alpha: float = 0.1) -> dict:
    residual = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
    quantile = conformal_quantile(residual, alpha)
    pred = np.asarray(yhat_test, dtype=np.float64)
    return {
        "lower": pred - quantile,
        "upper": pred + quantile,
        "quantile": quantile,
        "alpha": alpha,
        "n_calibration": int(np.isfinite(residual).sum()),
    }


def fit_binary_temperature(logits, labels, steps: int = 80, lr: float = 0.05) -> float:
    scores = torch.tensor(np.asarray(logits, dtype=np.float32))
    target = torch.tensor(np.asarray(labels, dtype=np.float32))
    log_t = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(1e-2, 100.0)
        loss = F.binary_cross_entropy_with_logits(scores / temperature, target)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_t).detach())


def apply_temperature(logits, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / float(temperature)))


def calibration_summary(probs, labels) -> dict:
    return {
        "ece": expected_calibration_error(probs, labels),
        "brier": brier(probs, labels),
        "n": int(len(np.asarray(labels))),
        "note": "Raw entropy is not a calibrated confidence.",
    }
