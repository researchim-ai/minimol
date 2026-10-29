"""Ranking metrics for a fixed candidate list.

Relevance is taken from the task oracle. A perfect order has NDCG 1.
These scores do not measure experimental activity.
"""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevance, k: int | None = None) -> float:
    values = np.asarray(relevance, dtype=np.float64)
    if k is not None:
        values = values[:k]
    if values.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, values.size + 2))
    return float(np.sum(np.maximum(values, 0.0) * discounts))


def ndcg_at_k(relevance, k: int | None = None) -> float | None:
    values = np.asarray(relevance, dtype=np.float64)
    ideal = np.sort(values)[::-1]
    normalizer = dcg_at_k(ideal, k)
    if normalizer <= 0:
        return None
    return dcg_at_k(values, k) / normalizer


def precision_at_k(hits, k: int) -> float | None:
    values = np.asarray(hits, dtype=np.float64)[:k]
    if values.size == 0:
        return None
    return float(np.mean(values > 0))


def oracle_relevance(task: dict) -> list[float]:
    """Non-negative relevance in the task's candidate order. Higher is better."""

    objective = task["objective"]["property"]
    direction = task["objective"]["direction"]
    raw = []
    for candidate in task["candidates"]:
        if not candidate["valid"] or candidate.get("oracle") is None:
            raw.append(None)
            continue
        value = float(candidate["oracle"][objective])
        raw.append(-value if direction == "minimize" else value)
    finite = [value for value in raw if value is not None]
    if not finite:
        return [0.0 for _ in raw]
    floor = min(finite)
    return [0.0 if value is None else value - floor for value in raw]


def ranking_report(task: dict, order: list[int] | None = None, k: int = 3) -> dict:
    relevance = oracle_relevance(task)
    if order is None:
        order = sorted(range(len(relevance)), key=lambda index: relevance[index], reverse=True)
    ordered = [relevance[index] for index in order]
    passes = []
    for index in order:
        candidate = task["candidates"][index]
        ok = candidate["valid"] and _passes(candidate, task.get("constraints") or [])
        passes.append(1.0 if ok else 0.0)
    return {
        "ndcg_at_k": ndcg_at_k(ordered, k),
        "precision_at_k": precision_at_k(passes, k),
        "k": k,
        "n_candidates": len(task["candidates"]),
    }


def _passes(candidate: dict, constraints: list[dict]) -> bool:
    if not candidate.get("valid") or candidate.get("oracle") is None:
        return False
    for constraint in constraints:
        value = candidate["oracle"][constraint["property"]]
        if "min" in constraint and value < constraint["min"]:
            return False
        if "max" in constraint and value > constraint["max"]:
            return False
    return True
