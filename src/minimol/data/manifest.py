"""Manifests describe a processed table without copying its rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from minimol.tracking import sha256_file


def build_manifest(frame: pd.DataFrame, *, task: str, data_path: Path, extra: dict | None = None) -> dict:
    splits = {}
    if "split" in frame.columns:
        splits = {str(key): int(value) for key, value in frame["split"].value_counts().to_dict().items()}
    licenses = {}
    for column in ("license_id", "upstream_license"):
        if column in frame.columns:
            licenses = {str(key): int(value) for key, value in frame[column].value_counts().to_dict().items()}
            break
    manifest = {
        "task": task,
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "splits": splits,
        "license_ids": licenses,
        "sha256": sha256_file(data_path),
        "data_file": data_path.name,
        "official_tdc": False,
        "benchmark_name": None,
        "research_use_only": True,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())
