"""Negative controls for regulatory models.

A shuffled-label model that matches the real one on validation is treated as
evidence of leakage or a constant predictor, not as a biological result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from minimol.data.normalize import assay_order, pivot_reg
from minimol.errors import MinimolError
from minimol.evaluation.metrics import regression_report
from minimol.jsonutil import write_json
from minimol.training.baselines import (
    fit_gc_baseline,
    fit_ridge,
    kmer_matrix,
    mean_predict,
    median_predict,
    predict_gc,
    predict_ridge,
)


def regulatory_controls(frame: pd.DataFrame, seed: int = 0, k_max: int = 4) -> dict:
    assays = assay_order(frame["assay"].tolist())
    rows = pivot_reg(frame, assays)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if len(train) < 4 or len(val) < 3:
        raise MinimolError("negative controls need at least 4 train and 3 validation sequences")
    seq_train = [row["sequence"] for row in train]
    seq_val = [row["sequence"] for row in val]
    y_train = np.stack([row["y"] for row in train])
    mask_train = np.stack([row["mask"] for row in train])
    y_val = np.stack([row["y"] for row in val])
    mask_val = np.stack([row["mask"] for row in val])
    features_train = kmer_matrix(seq_train, tuple(range(3, k_max + 1)))
    features_val = kmer_matrix(seq_val, tuple(range(3, k_max + 1)))
    fallback = y_train.mean(axis=0)
    real_models = fit_ridge(features_train, y_train, mask_train)
    real_pred = predict_ridge(real_models, features_val, fallback)
    rng = np.random.default_rng(seed)
    shuffled = y_train.copy()
    for index in range(shuffled.shape[1]):
        chosen = np.flatnonzero(mask_train[:, index])
        shuffled[chosen, index] = rng.permutation(shuffled[chosen, index])
    shuffled_models = fit_ridge(features_train, shuffled, mask_train)
    shuffled_pred = predict_ridge(shuffled_models, features_val, fallback)
    gc_pred = predict_gc(fit_gc_baseline(seq_train, y_train, mask_train), seq_val)
    mean_pred = mean_predict(y_train, mask_train, len(val))
    median_pred = median_predict(y_train, mask_train, len(val))
    real = regression_report(y_val, real_pred, mask_val, assays)
    control = regression_report(y_val, shuffled_pred, mask_val, assays)
    gc = regression_report(y_val, gc_pred, mask_val, assays)
    mean = regression_report(y_val, mean_pred, mask_val, assays)
    median = regression_report(y_val, median_pred, mask_val, assays)
    real_s = real["macro_spearman"]
    shuffled_s = control["macro_spearman"]
    suspect = real_s is not None and shuffled_s is not None and shuffled_s >= real_s - 0.05
    return {
        "task": "reg_negative_controls",
        "real_ridge": real,
        "shuffled_label_ridge": control,
        "gc_content": gc,
        "mean": mean,
        "median": median,
        "leakage_suspect": bool(suspect),
        "research_use_only": True,
        "clinical_use": False,
        "note": "Shuffled labels keep the sequences. Similar validation scores imply leakage or no sequence signal.",
    }


def write_regulatory_controls(data_path: Path, out_dir: Path, seed: int = 0) -> dict:
    report = regulatory_controls(pd.read_parquet(data_path), seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", report)
    return report
