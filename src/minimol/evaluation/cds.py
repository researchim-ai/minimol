"""CDS metrics and overlap-add stitching.

Window probabilities are averaged in probability space, not logit space.
"""

from __future__ import annotations

import numpy as np

from minimol.evaluation.metrics import (
    auprc,
    base_iou,
    boundary_scores,
    f1_from_counts,
    mcc_from_counts,
    mean_segment_iou,
    prf_counts,
)


def stitch_probabilities(windows: list[tuple[int, np.ndarray]], length: int) -> np.ndarray:
    if not windows:
        raise ValueError("no windows")
    channels = int(np.asarray(windows[0][1]).shape[0])
    total = np.zeros((channels, length), dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    for start, probs in windows:
        array = np.asarray(probs, dtype=np.float64)
        end = min(length, start + array.shape[-1])
        width = end - start
        if width <= 0:
            continue
        total[:, start:end] += array[:, :width]
        weight[start:end] += 1.0
    return total / np.maximum(weight, 1.0)


def nucleotide_report(y_true, y_prob, mask=None, tolerances=(3, 6, 12)) -> dict:
    truth = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[None, :]
        prob = prob[None, :]
    if mask is None:
        valid = np.ones_like(truth, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.ndim == 1:
            valid = np.repeat(valid[None, :], truth.shape[0], axis=0)
    strands = []
    for index, name in enumerate(("plus", "minus")[: truth.shape[0]]):
        tp, fp, fn, tn = prf_counts(truth[index], prob[index], valid[index])
        strand = {
            "f1": f1_from_counts(tp, fp, fn),
            "mcc": mcc_from_counts(tp, fp, fn, tn),
            "auprc": auprc(truth[index], prob[index], valid[index]),
            "base_iou": base_iou(np.where(valid[index], truth[index], 0), np.where(valid[index], prob[index], 0)),
            "segment_iou": mean_segment_iou(truth[index], prob[index]),
            "boundary_f1": {
                str(tol): boundary_scores(truth[index], prob[index], tolerance=tol)["f1"] for tol in tolerances
            },
        }
        strands.append((name, strand))
    f1s = [item["f1"] for _, item in strands if item["f1"] is not None]
    return {
        "strands": {name: item for name, item in strands},
        "macro_f1": float(np.mean(f1s)) if f1s else None,
    }


def edge_vs_center(correct: np.ndarray, edge: int = 32) -> dict:
    values = np.asarray(correct, dtype=np.float64)
    if values.size == 0:
        return {"edge": None, "center": None}
    edge = min(edge, values.size // 4 if values.size >= 4 else 0)
    if edge == 0:
        return {"edge": float(values.mean()), "center": float(values.mean())}
    edges = np.concatenate([values[:edge], values[-edge:]])
    center = values[edge:-edge] if values.size > 2 * edge else values
    return {"edge": float(np.mean(edges)), "center": float(np.mean(center))}
