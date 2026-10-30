"""Decisions derived from regression outputs, used as a baseline for learned heads.

No output here is claimed to be a calibrated probability. All questions and
margin filters are fixed before scoring. Filter coverage is always reported.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from minimol.evaluation.metrics import auprc, auroc


def _margin_report(correct, regret, margin, minimum: float) -> dict:
    chosen = margin > minimum if minimum == 0 else margin >= minimum
    return {
        "n": int(chosen.sum()),
        "fraction_of_available": float(chosen.mean()) if chosen.size else None,
        "accuracy": float(correct[chosen].mean()) if chosen.any() else None,
        "mean_regret_log2fc": float(regret[chosen].mean()) if chosen.any() else None,
    }


def regression_decisions(y, prediction, mask, assays, *, train_means, seed: int = 20261030) -> dict:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(truth) & np.isfinite(pred)
    if truth.shape != pred.shape or valid.shape != truth.shape or truth.ndim != 2:
        raise ValueError("labels, predictions, and mask must be aligned matrices")
    if truth.shape[1] != len(assays):
        raise ValueError("assays must match the output columns")
    pairwise, thresholds = {}, {}
    rng = np.random.default_rng(seed)
    for column, assay in enumerate(assays):
        ids = np.flatnonzero(valid[:, column])
        rng.shuffle(ids)
        ids = ids[:len(ids) // 2 * 2].reshape(-1, 2)
        actual_difference = truth[ids[:, 0], column] - truth[ids[:, 1], column]
        pred_difference = pred[ids[:, 0], column] - pred[ids[:, 1], column]
        correct = (actual_difference > 0) == (pred_difference > 0)
        regret = np.where(correct, 0, np.abs(actual_difference))
        pairwise[assay] = {
            "n_disjoint_pairs": int(len(ids)),
            "random_expected_accuracy": 0.5,
            "margins": {str(m): _margin_report(correct, regret, np.abs(actual_difference), m)
                        for m in (0.0, 0.5, 1.0)},
        }
        observed, scores = truth[valid[:, column], column], pred[valid[:, column], column]
        thresholds[assay] = {}
        for threshold in (0.0, 0.5, 1.0):
            labels, guesses = observed > threshold, scores > threshold
            thresholds[assay][str(threshold)] = {
                "n": int(len(labels)),
                "positive_fraction": float(labels.mean()) if labels.size else None,
                "accuracy": float((labels == guesses).mean()) if labels.size else None,
                "balanced_accuracy": float(balanced_accuracy_score(labels, guesses)) if np.unique(labels).size == 2 else None,
                "f1": float(f1_score(labels, guesses, zero_division=0)) if labels.size else None,
                "auroc_using_raw_regression_score": auroc(labels, scores),
                "average_precision_using_raw_regression_score": auprc(labels, scores),
                "probability_calibrated": False,
            }
    all_valid = valid.all(axis=1)
    actual, scores = truth[all_valid], pred[all_valid]
    if truth.shape[1] < 2:
        raise ValueError("assay choice needs at least two assays")
    choice = actual.argmax(axis=1)
    guessed = scores.argmax(axis=1)
    sorted_actual = np.sort(actual, axis=1)
    gap = sorted_actual[:, -1] - sorted_actual[:, -2]
    regret = actual.max(axis=1) - actual[np.arange(len(actual)), guessed]
    constant_assay = int(np.argmax(np.asarray(train_means)))
    choice_report = {
        "n_sequences_all_assays_observed": int(all_valid.sum()),
        "options": list(assays),
        "random_expected_accuracy": 1.0 / len(assays),
        "constant_assay_from_train_mean": assays[constant_assay],
        "margins": {},
    }
    for minimum in (0.0, 0.5, 1.0):
        row = _margin_report(choice == guessed, regret, gap, minimum)
        keep = gap > minimum if minimum == 0 else gap >= minimum
        row["constant_assay_accuracy"] = float((choice[keep] == constant_assay).mean()) if keep.any() else None
        choice_report["margins"][str(minimum)] = row
    return {
        "method": "deterministic_decisions_from_regression_predictions",
        "pair_sampling_seed": seed,
        "pairwise_activity": pairwise,
        "highest_activity_assay": choice_report,
        "threshold_events": thresholds,
        "limitations": [
            "These are regression-derived decisions, not a trained question-conditioned model.",
            "The checkpoint was selected on this validation set; this is exploratory development evaluation.",
            "Pairs are random disjoint validation sequences, not matched biological variants.",
            "Margin filters use measured labels; filtered accuracy is not deployable abstention coverage.",
            "Measured labels have assay noise. No calibrated probabilities or unseen-question transfer is established.",
        ],
    }
