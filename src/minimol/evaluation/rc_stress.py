"""Reverse-complement coordinate stress for a CDS model.

p_plus on the reverse complement should match the reversed p_minus of the
original sequence, and the other way around. A large disagreement is a
coordinate bug or a model that has not learned strand symmetry.
"""

from __future__ import annotations

import numpy as np
import torch

from minimol.data.sequences import reverse_complement
from minimol.inference import predict_cds_probabilities


def strand_disagreement(plus, minus, plus_rc, minus_rc) -> float:
    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    plus_rc = np.asarray(plus_rc, dtype=np.float64)
    minus_rc = np.asarray(minus_rc, dtype=np.float64)
    if plus.shape != minus.shape or plus_rc.shape != plus.shape:
        raise ValueError("strand probability vectors must share one length")
    forward = np.mean(np.abs(plus_rc - minus[::-1]))
    backward = np.mean(np.abs(minus_rc - plus[::-1]))
    return float(0.5 * (forward + backward))


def rc_stress(model: torch.nn.Module, sequences: list[str]) -> dict:
    errors = []
    for sequence in sequences:
        probabilities = predict_cds_probabilities(model, sequence, window=len(sequence))
        flipped = predict_cds_probabilities(model, reverse_complement(sequence), window=len(sequence))
        errors.append(strand_disagreement(probabilities[0], probabilities[1], flipped[0], flipped[1]))
    return {
        "task": "cds_rc_stress",
        "n": len(errors),
        "mean_abs_disagreement": float(np.mean(errors)) if errors else None,
        "max_abs_disagreement": float(np.max(errors)) if errors else None,
        "research_use_only": True,
        "clinical_use": False,
        "note": "Near zero means the two strand heads respect reverse-complement coordinates.",
    }
