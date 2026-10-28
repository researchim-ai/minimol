"""Latency samples. Numbers come from this process, not from a datasheet."""

from __future__ import annotations

import time

import numpy as np


def measure_latency(fn, warmup: int = 2, repeats: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(repeats),
        "p50_s": float(np.quantile(array, 0.5)),
        "p95_s": float(np.quantile(array, 0.95)),
        "mean_s": float(array.mean()),
    }
