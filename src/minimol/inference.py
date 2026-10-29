"""CDS inference with overlap-add for sequences longer than the model window.

Probabilities are averaged in the overlap. This is not a gene-model decoder.
"""

from __future__ import annotations

import numpy as np
import torch

from minimol.data.sequences import one_hot_sequence
from minimol.evaluation.cds import stitch_probabilities


def predict_cds_probabilities(
    model: torch.nn.Module,
    sequence: str,
    window: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    if window < 8:
        raise ValueError("window must cover at least 8 bases")
    overlap = min(overlap, window // 2)
    model.eval()
    if len(sequence) <= window:
        encoded = one_hot_sequence(sequence).unsqueeze(0)
        with torch.no_grad():
            return torch.sigmoid(model(encoded)).squeeze(0).cpu().numpy()
    step = max(window - overlap, 1)
    chunks: list[tuple[int, np.ndarray]] = []
    start = 0
    while start < len(sequence):
        chunk = sequence[start : start + window]
        encoded = one_hot_sequence(chunk).unsqueeze(0)
        with torch.no_grad():
            probs = torch.sigmoid(model(encoded)).squeeze(0).cpu().numpy()
        chunks.append((start, probs))
        if start + len(chunk) >= len(sequence):
            break
        start += step
    stitched = stitch_probabilities(chunks, len(sequence))
    if stitched.shape[-1] != len(sequence):
        raise RuntimeError("stitched CDS output length does not match the sequence")
    return stitched
