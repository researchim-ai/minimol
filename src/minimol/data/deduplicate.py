"""Replicate aggregation. Ki/Kd/IC50 relations are not averaged together."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["inchikey", "assay_id", "endpoint", "relation"]
    missing = [column for column in keys + ["measurement"] if column not in df.columns]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    rows = []
    for values, group in df.groupby(keys, dropna=False, sort=True):
        record = group.iloc[0].to_dict()
        measurements = pd.to_numeric(group["measurement"], errors="coerce").to_numpy(dtype=np.float64)
        finite = measurements[np.isfinite(measurements)]
        record["n_replicates"] = int(finite.size)
        record["replicate_values"] = [float(value) for value in finite]
        if finite.size == 0:
            record["measurement"] = np.nan
            record["replicate_std"] = np.nan
        elif str(values[3]) != "=":
            record["measurement"] = float(finite[0])
            record["replicate_std"] = np.nan
            record["aggregation"] = "not_averaged_censored"
        else:
            record["measurement"] = float(finite.mean())
            record["replicate_std"] = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
            record["aggregation"] = "mean" if finite.size > 1 else "single"
        rows.append(record)
    return pd.DataFrame(rows)
