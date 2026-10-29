"""Mean and standard deviation across seed reports. Missing values are skipped."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minimol.jsonutil import write_json


def _lookup(payload: dict, dotted: str):
    cursor = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def aggregate_seed_reports(records: list[dict], fields: list[str]) -> dict:
    summary = {}
    for field in fields:
        values = []
        for record in records:
            value = _lookup(record, field)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            summary[field] = {"mean": None, "std": None, "n": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[field] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "n": int(array.size),
        }
    return {
        "n_runs": len(records),
        "fields": summary,
        "research_use_only": True,
        "clinical_use": False,
        "note": "Mean and sample standard deviation across the supplied runs. This is not a clinical interval.",
    }


def write_aggregate(records: list[dict], fields: list[str], out_path: Path) -> dict:
    report = aggregate_seed_reports(records, fields)
    write_json(out_path, report)
    return report
