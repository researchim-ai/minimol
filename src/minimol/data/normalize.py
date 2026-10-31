"""Label scaling and regulatory-table checks.

Normalization statistics are fit on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from minimol.data.sequences import normalize_sequence, sequence_hash

REG_REQUIRED = [
    "sample_id",
    "sequence",
    "source_id",
    "data_revision",
    "assay",
    "split",
    "group_id",
    "chromosome",
    "y_log2fc",
    "y_se",
    "valid_y",
    "sequence_hash",
    "license_id",
]

CANONICAL_ASSAYS = ("K562", "HepG2", "SK-N-SH")
SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}
KNOWN_SPLITS = {"train", "validation", "test", "release_holdout", "calibration", "quarantine"}


def canonicalize_split(value: str) -> str:
    text = SPLIT_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if text not in KNOWN_SPLITS:
        raise ValueError(f"unknown split: {value}")
    return text


def assay_order(present) -> list[str]:
    present = list(dict.fromkeys(present))
    if set(present) <= set(CANONICAL_ASSAYS):
        return [name for name in CANONICAL_ASSAYS if name in present]
    return sorted(present)


class LabelScaler:
    def __init__(self):
        self.mean: list[float] | None = None
        self.std: list[float] | None = None

    def fit(self, y, mask) -> LabelScaler:
        values = np.asarray(y, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if values.ndim == 1:
            values = values[:, None]
            valid = valid[:, None]
        means = []
        stds = []
        for index in range(values.shape[1]):
            column = values[valid[:, index], index]
            column = column[np.isfinite(column)]
            if column.size == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            means.append(float(column.mean()))
            scale = float(column.std())
            stds.append(scale if scale > 1e-6 else 1.0)
        self.mean = means
        self.std = stds
        return self

    def transform(self, y):
        values = np.asarray(y, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        return (values - mean) / std

    def inverse_numpy(self, y):
        values = np.asarray(y, dtype=np.float64)
        return values * np.asarray(self.std) + np.asarray(self.mean)

    def inverse_torch(self, y: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=y.device, dtype=y.dtype)
        std = torch.as_tensor(self.std, device=y.device, dtype=y.dtype)
        return y * std + mean

    def state_dict(self) -> dict:
        return {"mean": list(self.mean or []), "std": list(self.std or [])}

    def load_state_dict(self, state: dict) -> LabelScaler:
        self.mean = [float(value) for value in state["mean"]]
        self.std = [float(value) for value in state["std"]]
        return self


def fit_scaler_train_only(y, mask, splits) -> LabelScaler:
    train = np.asarray(splits) == "train"
    if not np.any(train):
        raise ValueError("training split is empty")
    return LabelScaler().fit(np.asarray(y)[train], np.asarray(mask)[train])


def audit_reg_frame(df: pd.DataFrame) -> dict:
    issues = []
    missing = [column for column in REG_REQUIRED if column not in df.columns]
    if missing:
        issues.append({"kind": "missing_columns", "columns": missing})
        return {"passed": False, "issues": issues, "n": int(len(df))}
    if df["sample_id"].duplicated().any():
        issues.append({"kind": "duplicate_sample_id", "n": int(df["sample_id"].duplicated().sum())})
    for row in df.itertuples(index=False):
        try:
            sequence = normalize_sequence(row.sequence)
        except ValueError as exc:
            issues.append({"kind": "bad_sequence", "sample_id": row.sample_id, "detail": str(exc)})
            continue
        if row.sequence_hash != sequence_hash(sequence):
            issues.append({"kind": "hash_mismatch", "sample_id": row.sample_id})
        try:
            canonicalize_split(row.split)
        except ValueError:
            issues.append({"kind": "bad_split", "sample_id": row.sample_id, "split": row.split})
        if not str(row.assay).strip():
            issues.append({"kind": "empty_assay", "sample_id": row.sample_id})
        if bool(row.valid_y) and not np.isfinite(row.y_log2fc):
            issues.append({"kind": "missing_label", "sample_id": row.sample_id})
    return {"passed": not issues, "issues": issues, "n": int(len(df))}


def pivot_reg(df: pd.DataFrame, assays: list[str] | None = None) -> list[dict]:
    """One row per sequence, with a label vector aligned to `assays`."""

    frame = df.copy()
    frame["split"] = frame["split"].map(canonicalize_split)
    if assays is None:
        assays = assay_order(frame["assay"].tolist())
    rows = []
    for seq_hash, group in frame.groupby("sequence_hash", sort=True):
        sequence = str(group["sequence"].iloc[0])
        y = np.zeros(len(assays), dtype=np.float64)
        mask = np.zeros(len(assays), dtype=bool)
        se = np.ones(len(assays), dtype=np.float64)
        for record in group.itertuples(index=False):
            if record.assay not in assays or not bool(record.valid_y):
                continue
            if not np.isfinite(record.y_log2fc):
                continue
            index = assays.index(record.assay)
            y[index] = float(record.y_log2fc)
            mask[index] = True
            if np.isfinite(record.y_se):
                se[index] = float(record.y_se)
        rows.append(
            {
                "sequence_hash": seq_hash,
                "sequence": normalize_sequence(sequence),
                "split": canonicalize_split(group["split"].iloc[0]),
                "group_id": str(group["group_id"].iloc[0]),
                "locus_group": str(group["locus_group"].iloc[0]) if "locus_group" in group.columns else "",
                "y": y,
                "mask": mask,
                "se": se,
                "chromosome": str(group["chromosome"].iloc[0]) if "chromosome" in group.columns else "",
                "label_quality": str(group["label_quality"].iloc[0]) if "label_quality" in group.columns else "",
            }
        )
    return rows
