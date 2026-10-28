"""Clustered bootstrap. Resamples groups, not individual rows."""

from __future__ import annotations

import numpy as np


def clustered_bootstrap_ci(values, groups, statistic, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    values = np.asarray(values)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        stats.append(float(statistic(values[index])))
    stats = np.asarray(stats, dtype=np.float64)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(stats.mean()), "low": float(low), "high": float(high), "n_boot": n_boot}
