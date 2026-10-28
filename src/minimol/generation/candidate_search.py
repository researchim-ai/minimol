"""Best-of-N selection on an already scored candidate list."""

from __future__ import annotations

import numpy as np


def select_best(scores, direction: str = "maximize") -> int:
    values = np.asarray(scores, dtype=np.float64)
    if direction == "maximize":
        return int(np.nanargmax(values))
    if direction == "minimize":
        return int(np.nanargmin(values))
    raise ValueError(direction)


def select_random(n: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, n))
