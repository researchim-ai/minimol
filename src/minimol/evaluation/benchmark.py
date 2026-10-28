"""Benchmark reports. Official TDC numbers are refused without official splits."""

from __future__ import annotations

import json
from pathlib import Path

from minimol.errors import MinimolError
from minimol.jsonutil import write_json


def pareto_front(rows: list[dict], maximize=("macro_spearman",), minimize=("param_count", "p50_s")) -> list[dict]:
    def better(left, right) -> bool:
        dominated = True
        strict = False
        for key in maximize:
            if left.get(key) is None or right.get(key) is None:
                return False
            if left[key] < right[key]:
                dominated = False
            if left[key] > right[key]:
                strict = True
        for key in minimize:
            if left.get(key) is None or right.get(key) is None:
                return False
            if left[key] > right[key]:
                dominated = False
            if left[key] < right[key]:
                strict = True
        return dominated and strict

    front = []
    for row in rows:
        if any(better(other, row) for other in rows if other is not row):
            continue
        front.append(row)
    return front


def run_benchmark(suite: str, metrics_paths: list[Path], out: Path, official_tdc: bool = False) -> dict:
    if suite.startswith("tdc") and not official_tdc:
        raise MinimolError(
            "Refusing to emit a TDC leaderboard report without official TDC splits. "
            "Use --suite synthetic-admet for proxy data."
        )
    rows = [json.loads(Path(path).read_text()) for path in metrics_paths]
    payload = {
        "suite": suite,
        "official_tdc": official_tdc,
        "n_runs": len(rows),
        "runs": rows,
        "research_use_only": True,
        "clinical_use": False,
    }
    write_json(out, payload)
    return payload
