"""Split rules. Official test ids stay put; groups and scaffolds are not cut apart."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minimol.data.normalize import canonicalize_split
from minimol.errors import MinimolError, SplitLeakageError


def assert_plinder_release_frozen(release: str | None) -> None:
    text = "" if release is None else str(release).strip()
    if not text or "UNPINNED" in text.upper() or text.startswith("REPLACE"):
        raise MinimolError("PLINDER release is not frozen")


def assert_disjoint_ids(train_ids, test_ids, what: str = "cluster") -> None:
    overlap = set(train_ids) & set(test_ids)
    overlap.discard(None)
    overlap.discard("")
    if overlap:
        sample = ", ".join(sorted(map(str, overlap))[:5])
        raise SplitLeakageError(f"{what} ids overlap train and test: {sample}")


def scaffold_split(scaffold_ids, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> list[str]:
    scaffolds = np.array(sorted({str(item) for item in scaffold_ids}))
    rng = np.random.default_rng(seed)
    order = scaffolds.copy()
    rng.shuffle(order)
    count = int(order.size)
    n_test = int(round(count * fractions[2])) if count >= 3 else 0
    n_val = int(round(count * fractions[1])) if count >= 3 else 0
    if n_test + n_val >= count:
        n_test = 1 if count > 2 else 0
        n_val = 1 if count - n_test > 1 else 0
    test = set(order[:n_test].tolist())
    validation = set(order[n_test : n_test + n_val].tolist())
    assigned = []
    for scaffold in scaffold_ids:
        if scaffold in test:
            assigned.append("test")
        elif scaffold in validation:
            assigned.append("validation")
        else:
            assigned.append("train")
    return assigned


def apply_official_split(df: pd.DataFrame, official_col: str = "official_split") -> tuple[pd.DataFrame, str]:
    frame = df.copy()
    if official_col in frame.columns and frame[official_col].notna().any():
        frame["split"] = frame[official_col].map(canonicalize_split)
        return frame, "official"
    if "scaffold_id" not in frame.columns:
        raise MinimolError("scaffold_id is required when no official split is present")
    frame["split"] = scaffold_split(frame["scaffold_id"].tolist())
    return frame, "scaffold"


def test_ids(df: pd.DataFrame, id_col: str = "sample_id") -> list[str]:
    part = df[df["split"].map(canonicalize_split) == "test"]
    return sorted(part[id_col].astype(str).tolist())


def assert_test_ids_unchanged(before: list[str], after: list[str]) -> None:
    if list(before) != list(after):
        raise SplitLeakageError("official test ids changed")


def make_stricter_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    split_col: str = "split",
    val_fraction: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Quarantine groups that already cross official splits.

    Official test and release_holdout rows stay in those splits. A fraction of
    the remaining clean train groups becomes strict_validation, stored in
    `strict_split` as `validation`.
    """

    frame = df.copy()
    frame["_split"] = frame[split_col].map(canonicalize_split)
    strict = []
    grouped = frame.groupby(group_col, sort=True)
    leaked = set()
    for group_id, group in grouped:
        splits = set(group["_split"])
        if len(splits - {"quarantine"}) > 1:
            leaked.add(group_id)
    train_groups = sorted(
        {
            group_id
            for group_id, group in grouped
            if group_id not in leaked and set(group["_split"]) == {"train"}
        }
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(train_groups)
    n_val = int(round(len(train_groups) * val_fraction))
    held = set(train_groups[:n_val])
    for group_id, split in zip(frame[group_col], frame["_split"], strict=False):
        if group_id in leaked:
            strict.append("quarantine")
        elif split in {"test", "release_holdout", "calibration"}:
            strict.append(split)
        elif group_id in held:
            strict.append("validation")
        else:
            strict.append(split)
    frame["strict_split"] = strict
    return frame.drop(columns=["_split"])


def assert_genomic_windows_grouped(df: pd.DataFrame) -> None:
    """Windows from the same assembly chromosome stay in one split."""

    keys = ["assembly_accession", "chromosome"]
    for column in keys + ["split"]:
        if column not in df.columns:
            raise MinimolError(f"missing column {column}")
    for _, group in df.groupby(keys, sort=False):
        splits = {canonicalize_split(value) for value in group["split"]}
        if len(splits) > 1:
            raise SplitLeakageError("chromosome windows cross splits")


def time_split(dates, test_fraction: float = 0.2) -> dict:
    """Last fraction of dated rows is test. Missing dates stay in train and are flagged."""

    parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    labels = np.array(["train"] * len(parsed), dtype=object)
    missing = [index for index, value in enumerate(parsed) if pd.isna(value)]
    valid_index = [index for index, value in enumerate(parsed) if not pd.isna(value)]
    if valid_index:
        order = sorted(valid_index, key=lambda index: parsed.iloc[index])
        n_test = int(round(len(order) * test_fraction))
        for index in order[-n_test:]:
            labels[index] = "test"
    return {"split": labels.tolist(), "missing_date_index": missing}
