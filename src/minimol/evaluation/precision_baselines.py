"""Precision baselines on matched WT/mutant pairs.

Zero-delta predicts no change. WT-only ignores the mutation. Composition-delta
uses the amino-acid count difference and is still not a docking score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from minimol.errors import MinimolError
from minimol.evaluation.bind_baselines import amino_acid_counts
from minimol.evaluation.metrics import mae
from minimol.jsonutil import write_json


def _fit_predict(train_x, train_y, test_x) -> np.ndarray:
    if train_x.shape[0] < 2:
        return np.full(test_x.shape[0], float(np.mean(train_y)))
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def precision_baseline_report(frame: pd.DataFrame) -> dict:
    rows = []
    for record in frame.itertuples(index=False):
        rows.append(
            {
                "split": str(record.split),
                "y": float(record.measurement),
                "direction": str(record.direction),
                "wt": amino_acid_counts(record.sequence_wt),
                "delta_aa": amino_acid_counts(record.sequence_mut) - amino_acid_counts(record.sequence_wt),
                "protein_id": str(record.protein_id),
            }
        )
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if len(train) < 2 or not val:
        raise MinimolError("precision baselines need train and validation pairs")
    y_train = np.array([row["y"] for row in train])
    y_val = np.array([row["y"] for row in val])
    zero = np.zeros_like(y_val)
    wt_only = _fit_predict(np.stack([row["wt"] for row in train]), y_train, np.stack([row["wt"] for row in val]))
    composition = _fit_predict(
        np.stack([row["delta_aa"] for row in train]),
        y_train,
        np.stack([row["delta_aa"] for row in val]),
    )
    scores = {
        "zero_delta": {"mae": mae(y_val, zero), "n": int(len(y_val))},
        "wt_sequence_only": {"mae": mae(y_val, wt_only), "n": int(len(y_val))},
        "composition_delta": {"mae": mae(y_val, composition), "n": int(len(y_val))},
        "docking_only": {
            "mae": None,
            "reason": "No docking backend is configured. A docking score would not be ground truth anyway.",
        },
    }
    return {
        "task": "precision_baselines",
        "endpoint": str(frame["endpoint"].iloc[0]) if "endpoint" in frame.columns else None,
        "scores": scores,
        "research_use_only": True,
        "clinical_use": False,
        "note": "Synthetic matched pairs. Zero-delta is the no-change control.",
    }


def write_precision_baselines(data_path: Path, out_dir: Path) -> dict:
    report = precision_baseline_report(pd.read_parquet(data_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", report)
    return report
