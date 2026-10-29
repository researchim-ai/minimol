"""Compare ranking methods on one fixed candidate list.

Every method sees the same candidates and the same stored oracle. A higher
NDCG is not evidence of experimental activity.
"""

from __future__ import annotations

import numpy as np

from minimol.evaluation.ranking import oracle_relevance, ranking_report


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def fixed_budget_rank_report(tasks: list[dict], k: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    oracle_scores: list[float | None] = []
    random_scores: list[float | None] = []
    reverse_scores: list[float | None] = []
    invalid_rates: list[float] = []
    for task in tasks:
        candidates = task.get("candidates") or []
        count = len(candidates)
        if count == 0:
            continue
        relevance = oracle_relevance(task)
        oracle_order = sorted(range(count), key=lambda index: relevance[index], reverse=True)
        reverse_order = list(reversed(oracle_order))
        random_order = rng.permutation(count).tolist()
        oracle_scores.append(ranking_report(task, oracle_order, k)["ndcg_at_k"])
        reverse_scores.append(ranking_report(task, reverse_order, k)["ndcg_at_k"])
        random_scores.append(ranking_report(task, random_order, k)["ndcg_at_k"])
        invalid_rates.append(float(np.mean([not candidate.get("valid") for candidate in candidates])))
    return {
        "task": "rank_benchmark",
        "k": k,
        "n_tasks": len(oracle_scores),
        "oracle_mean_ndcg": _mean(oracle_scores),
        "random_mean_ndcg": _mean(random_scores),
        "reverse_mean_ndcg": _mean(reverse_scores),
        "mean_invalid_rate": _mean(invalid_rates),
        "candidate_budget": "fixed",
        "research_use_only": True,
        "clinical_use": False,
        "note": "Same candidates and the same stored RDKit oracle for oracle, random, and reverse order.",
    }
