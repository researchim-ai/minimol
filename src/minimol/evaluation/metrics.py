"""Regression, classification, and nucleotide metrics. Empty slices return nulls."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def _pair(y, yhat, mask=None):
    actual = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(yhat, dtype=np.float64).reshape(-1)
    if mask is None:
        valid = np.ones(actual.shape[0], dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool).reshape(-1)
    valid = valid & np.isfinite(actual) & np.isfinite(pred)
    return actual[valid], pred[valid]


def _corr(result):
    coef = getattr(result, "statistic", None)
    if coef is None:
        coef = result[0]
    if coef is None or not np.isfinite(coef):
        return None
    return float(coef)


def spearman(y, yhat, mask=None):
    actual, pred = _pair(y, yhat, mask)
    if actual.size < 3 or np.unique(actual).size < 2 or np.unique(pred).size < 2:
        return None
    return _corr(stats.spearmanr(actual, pred))


def pearson(y, yhat, mask=None):
    actual, pred = _pair(y, yhat, mask)
    if actual.size < 3 or np.unique(actual).size < 2 or np.unique(pred).size < 2:
        return None
    return _corr(stats.pearsonr(actual, pred))


def mae(y, yhat, mask=None):
    actual, pred = _pair(y, yhat, mask)
    if actual.size == 0:
        return None
    return float(np.mean(np.abs(actual - pred)))


def rmse(y, yhat, mask=None):
    actual, pred = _pair(y, yhat, mask)
    if actual.size == 0:
        return None
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def regression_report(y, yhat, mask=None, names=None) -> dict:
    actual = np.asarray(y, dtype=np.float64)
    pred = np.asarray(yhat, dtype=np.float64)
    if actual.ndim == 1:
        actual = actual[:, None]
        pred = pred[:, None]
    if mask is None:
        valid = np.ones_like(actual, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.repeat(valid[:, None], actual.shape[1], axis=1)
    if names is None:
        names = [f"y{index}" for index in range(actual.shape[1])]
    per = {}
    spearmans = []
    maes = []
    for index, name in enumerate(names):
        column_mask = valid[:, index]
        metrics = {
            "spearman": spearman(actual[:, index], pred[:, index], column_mask),
            "pearson": pearson(actual[:, index], pred[:, index], column_mask),
            "mae": mae(actual[:, index], pred[:, index], column_mask),
            "rmse": rmse(actual[:, index], pred[:, index], column_mask),
            "n": int(np.sum(column_mask)),
        }
        per[name] = metrics
        if metrics["spearman"] is not None:
            spearmans.append(metrics["spearman"])
        if metrics["mae"] is not None:
            maes.append(metrics["mae"])
    return {
        "per_target": per,
        "macro_spearman": float(np.mean(spearmans)) if spearmans else None,
        "macro_mae": float(np.mean(maes)) if maes else None,
    }


def brier(probs, labels, mask=None) -> float | None:
    pred, actual = _pair(probs, labels, mask)
    if pred.size == 0:
        return None
    return float(np.mean((pred - actual) ** 2))


def binary_nll(probs, labels, mask=None) -> float | None:
    pred, actual = _pair(probs, labels, mask)
    if pred.size == 0:
        return None
    clipped = np.clip(pred, 1e-6, 1 - 1e-6)
    return float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)))


def expected_calibration_error(probs, labels, n_bins: int = 10, mask=None) -> float | None:
    pred, actual = _pair(probs, labels, mask)
    if pred.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            chosen = (pred >= edges[index]) & (pred <= edges[index + 1])
        else:
            chosen = (pred >= edges[index]) & (pred < edges[index + 1])
        if not np.any(chosen):
            continue
        error += float(chosen.mean()) * abs(float(actual[chosen].mean()) - float(pred[chosen].mean()))
    return float(error)


def auroc(labels, scores, mask=None):
    actual, pred = _pair(labels, scores, mask)
    if actual.size == 0 or np.unique(actual).size < 2:
        return None
    return float(roc_auc_score(actual, pred))


def auprc(labels, scores, mask=None):
    actual, pred = _pair(labels, scores, mask)
    if actual.size == 0 or np.unique(actual).size < 2:
        return None
    return float(average_precision_score(actual, pred))


def prf_counts(y_true, y_pred, mask=None, threshold: float = 0.5):
    actual, pred = _pair(y_true, y_pred, mask)
    if actual.size == 0:
        return 0, 0, 0, 0
    truth = actual > 0.5
    guess = pred >= threshold
    tp = int(np.logical_and(truth, guess).sum())
    fp = int(np.logical_and(~truth, guess).sum())
    fn = int(np.logical_and(truth, ~guess).sum())
    tn = int(np.logical_and(~truth, ~guess).sum())
    return tp, fp, fn, tn


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float | None:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return None
    return float((tp * tn - fp * fn) / math.sqrt(denom))


def base_iou(y_true, y_pred, threshold: float = 0.5) -> float:
    truth = np.asarray(y_true) >= 0.5
    guess = np.asarray(y_pred) >= threshold
    union = np.logical_or(truth, guess).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(truth, guess).sum() / union)


def _segments(binary) -> list[tuple[int, int]]:
    values = np.asarray(binary).astype(bool)
    spans = []
    start = None
    for index, bit in enumerate(values):
        if bit and start is None:
            start = index
        if not bit and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(values)))
    return spans


def _segment_iou(left, right) -> float:
    inter = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = (left[1] - left[0]) + (right[1] - right[0]) - inter
    if union == 0:
        return 0.0
    return inter / union


def mean_segment_iou(y_true, y_pred, threshold: float = 0.5) -> float:
    truth = _segments(np.asarray(y_true) >= 0.5)
    guess = _segments(np.asarray(y_pred) >= threshold)
    if not truth and not guess:
        return 1.0
    if not truth or not guess:
        return 0.0
    return float(np.mean([max(_segment_iou(span, other) for other in guess) for span in truth]))


def change_points(binary, threshold: float = 0.5) -> list[int]:
    values = (np.asarray(binary) >= threshold).astype(int)
    if values.size == 0:
        return []
    return (np.where(values[1:] != values[:-1])[0] + 1).tolist()


def boundary_scores(y_true, y_pred, tolerance: int = 3, threshold: float = 0.5) -> dict:
    truth = change_points(y_true, 0.5)
    pred = change_points(y_pred, threshold)
    if not truth and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not truth or not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    matched = set()
    hits = 0
    for point in pred:
        best = None
        best_distance = None
        for index, target in enumerate(truth):
            if index in matched:
                continue
            distance = abs(point - target)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best = index
                best_distance = distance
        if best is not None:
            matched.add(best)
            hits += 1
    precision = hits / len(pred)
    recall = hits / len(truth)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def prevalence(labels, mask=None) -> float | None:
    actual, _pred = _pair(labels, np.zeros_like(np.asarray(labels, dtype=np.float64)), mask)
    if actual.size == 0:
        return None
    return float(np.mean(actual))


def shuffled_correlation(y, yhat, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    actual = np.asarray(y, dtype=np.float64).copy()
    pred = np.asarray(yhat, dtype=np.float64)
    shuffled = actual.copy()
    rng.shuffle(shuffled)
    return {"real": spearman(actual, pred), "shuffled": spearman(shuffled, pred)}
