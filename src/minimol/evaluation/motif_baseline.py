"""Motif-indicator baseline.

Motifs are strings supplied by the caller. This is not a JASPAR download and
not a neural method. A hit is exact presence of the string in the sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from minimol.data.normalize import assay_order, pivot_reg
from minimol.errors import MinimolError
from minimol.evaluation.metrics import regression_report
from minimol.jsonutil import write_json
from minimol.training.baselines import fit_ridge, mean_predict, predict_ridge


def motif_indicators(sequences: list[str], motifs: list[str]) -> np.ndarray:
    matrix = np.zeros((len(sequences), len(motifs)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        for column, motif in enumerate(motifs):
            matrix[row, column] = 1.0 if motif and motif in sequence else 0.0
    return matrix


def motif_baseline_report(frame: pd.DataFrame, motifs: list[str]) -> dict:
    motifs = [motif.strip().upper() for motif in motifs if motif.strip()]
    if not motifs:
        raise MinimolError("pass at least one motif string")
    assays = assay_order(frame["assay"].tolist())
    rows = pivot_reg(frame, assays)
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if len(train) < 4 or len(val) < 3:
        raise MinimolError("motif baseline needs train and validation sequences")
    y_train = np.stack([row["y"] for row in train])
    mask_train = np.stack([row["mask"] for row in train])
    y_val = np.stack([row["y"] for row in val])
    mask_val = np.stack([row["mask"] for row in val])
    x_train = motif_indicators([row["sequence"] for row in train], motifs)
    x_val = motif_indicators([row["sequence"] for row in val], motifs)
    fallback = y_train.mean(axis=0)
    pred = predict_ridge(fit_ridge(x_train, y_train, mask_train), x_val, fallback)
    mean_pred = mean_predict(y_train, mask_train, len(val))
    motif_metrics = regression_report(y_val, pred, mask_val, assays)
    mean_metrics = regression_report(y_val, mean_pred, mask_val, assays)
    return {
        "task": "motif_baseline",
        "motifs": motifs,
        "source": "caller_supplied_strings",
        "jaspar_download": False,
        "motif": motif_metrics,
        "mean": mean_metrics,
        "research_use_only": True,
        "clinical_use": False,
        "note": "Exact string matches. Not a JASPAR PWM and not a neural model.",
    }


def write_motif_baseline(data_path: Path, out_dir: Path, motifs: list[str]) -> dict:
    report = motif_baseline_report(pd.read_parquet(data_path), motifs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", report)
    return report
