"""Clustered bootstrap. Resamples groups, not individual rows."""

from __future__ import annotations

import numpy as np


def grouped_macro_mae_ci(y, yhat, mask, groups, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap macro MAE by resampling groups. This is not a final test interval."""

    actual = np.asarray(y, dtype=np.float64)
    pred = np.asarray(yhat, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    labels = np.asarray(groups)
    unique = np.unique(labels)
    index_of = {group: np.flatnonzero(labels == group) for group in unique}
    rng = np.random.default_rng(seed)

    def macro(chosen_index) -> float:
        maes = []
        for column in range(actual.shape[1]):
            chosen = valid[chosen_index, column]
            if int(chosen.sum()) == 0:
                continue
            error = np.abs(actual[chosen_index, column][chosen] - pred[chosen_index, column][chosen])
            maes.append(float(error.mean()))
        return float(np.mean(maes)) if maes else float("nan")

    stats = []
    for _ in range(n_boot):
        chosen_groups = rng.choice(unique, size=unique.size, replace=True)
        chosen_index = np.concatenate([index_of[group] for group in chosen_groups])
        stats.append(macro(chosen_index))
    stats = np.asarray(stats, dtype=np.float64)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(np.nanmean(stats)),
        "low": float(low),
        "high": float(high),
        "n_boot": n_boot,
        "n_groups": int(unique.size),
        "resample": "group",
        "note": "Validation interval after the checkpoint was chosen on this same split. Not an independent test.",
    }


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
