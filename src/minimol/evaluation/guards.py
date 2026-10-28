"""Scaffold overlap, similarity OOD, and endpoint pooling checks."""

from __future__ import annotations

import numpy as np

from minimol.errors import MinimolError


def overlapping_scaffolds(train_scaffolds, test_scaffolds) -> list[str]:
    overlap = set(map(str, train_scaffolds)) & set(map(str, test_scaffolds))
    overlap.discard("")
    overlap.discard("None")
    return sorted(overlap)


def tanimoto(left, right) -> float:
    a = np.asarray(left).astype(bool)
    b = np.asarray(right).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def nearest_tanimoto(query, train_fps) -> float:
    if len(train_fps) == 0:
        return 0.0
    return max(tanimoto(query, other) for other in train_fps)


def ood_from_similarity(similarity: float, threshold: float = 0.4) -> bool:
    return bool(similarity < threshold)


def assert_endpoints_not_pooled(frame) -> None:
    if "endpoint" not in frame.columns or "assay_type" not in frame.columns:
        raise MinimolError("endpoint and assay_type are required")
    mixed = frame.groupby("endpoint")["assay_type"].nunique()
    bad = mixed[mixed > 1]
    if len(bad):
        raise MinimolError("endpoint pools assay types: " + ", ".join(map(str, bad.index)))


def target_prior_flag(full_metric: float, protein_only: float, ligand_only: float, tol: float = 0.02) -> bool:
    """True when a protein-only model matches the full model and beats the ligand."""

    return protein_only >= full_metric - tol and protein_only > ligand_only + tol
