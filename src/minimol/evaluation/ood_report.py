"""Applicability by nearest Morgan fingerprint in the training split.

A low Tanimoto flag is an out-of-domain warning, not a calibrated probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from minimol.data.featurize import morgan_fingerprint
from minimol.errors import MinimolError
from minimol.evaluation.guards import nearest_tanimoto
from minimol.jsonutil import write_json


def applicability_report(frame: pd.DataFrame, threshold: float = 0.4) -> dict:
    if "smiles_canonical" not in frame.columns or "split" not in frame.columns:
        raise MinimolError("smiles_canonical and split are required")
    seen: dict[str, str] = {}
    for record in frame.itertuples(index=False):
        seen.setdefault(str(record.smiles_canonical), str(record.split))
    train = []
    validation = []
    for smiles, split in seen.items():
        fingerprint = morgan_fingerprint(smiles)
        if fingerprint is None:
            continue
        if split == "train":
            train.append(fingerprint)
        elif split == "validation":
            validation.append(fingerprint)
    if not train or not validation:
        raise MinimolError("applicability report needs train and validation molecules")
    nearest = [nearest_tanimoto(query, train) for query in validation]
    flags = [similarity < threshold for similarity in nearest]
    return {
        "task": "applicability",
        "fingerprint": "morgan_radius2_2048",
        "threshold": threshold,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_ood": int(sum(flags)),
        "fraction_ood": float(np.mean(flags)),
        "median_nearest_tanimoto": float(np.median(nearest)),
        "research_use_only": True,
        "clinical_use": False,
        "note": "OOD means the nearest training fingerprint is below the threshold. It is not a probability.",
    }


def write_applicability(data_path: Path, out_dir: Path, threshold: float = 0.4) -> dict:
    report = applicability_report(pd.read_parquet(data_path), threshold=threshold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", report)
    return report
