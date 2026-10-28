"""Carbon-X option scoring.

Softmax of conditional log-likelihoods is a pseudo-distribution. It is not a
calibrated probability of pathogenicity, activity, or any other event.
"""

from __future__ import annotations

import numpy as np

from minimol.errors import MinimolError, RemoteCodeNotReviewed

CARBON_X_REPO = "HuggingFaceBio/Carbon-X"


def assert_carbon_x_loadable(revision: str, reviewed_remote_code: bool) -> None:
    if not reviewed_remote_code:
        raise RemoteCodeNotReviewed(
            "Refusing trust_remote_code until the pinned Carbon-X Python is reviewed."
        )
    text = "" if revision is None else str(revision)
    if not text or "UNVERIFIED" in text or "REPLACE" in text:
        raise MinimolError("Carbon-X revision is not a verified commit.")


def scores_to_pseudo_distribution(loglikes) -> dict:
    values = np.asarray(loglikes, dtype=np.float64)
    if values.size == 0:
        raise MinimolError("option list is empty")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    probs = weights / weights.sum()
    return {
        "pseudo_distribution": probs.tolist(),
        "calibrated": False,
        "meaning": "softmax of conditional token log-likelihoods, not an event probability",
    }


def go_nogo(metric_with_teacher: float, metric_without: float, min_delta: float = 0.0) -> dict:
    delta = float(metric_with_teacher) - float(metric_without)
    decision = "GO" if delta > min_delta else "NO-GO"
    return {
        "decision": decision,
        "delta": delta,
        "rule": "Drop the teacher unless the held-out metric improves.",
    }


def prompt_sensitivity(score_rows: list[list[float]]) -> float:
    """Mean variance of option scores across prompt templates. High means unstable."""

    array = np.asarray(score_rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        return 0.0
    return float(array.var(axis=0).mean())
