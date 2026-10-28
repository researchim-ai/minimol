"""YAML config loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from minimol.errors import MinimolError


def load_yaml(path: str | Path) -> dict:
    file_path = Path(path)
    with file_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise MinimolError(f"config must be a mapping: {file_path}")
    return data


def require(config: dict, *keys: str):
    cursor = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            raise MinimolError("missing config key: " + ".".join(keys))
        cursor = cursor[key]
    return cursor
