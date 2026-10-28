"""Teacher cache. Test and release-holdout rows are refused."""

from __future__ import annotations

import json
from pathlib import Path

from minimol.data.normalize import canonicalize_split
from minimol.errors import SplitLeakageError

FORBIDDEN = {"test", "release_holdout"}


class TeacherCache:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def put(self, sample_id: str, split: str, payload) -> None:
        canonical = canonicalize_split(split)
        if canonical in FORBIDDEN:
            raise SplitLeakageError(f"refusing to cache a teacher label for split {canonical}")
        self.rows[str(sample_id)] = {"split": canonical, "payload": payload}

    def get(self, sample_id: str):
        return self.rows.get(str(sample_id))

    def contains_forbidden_split(self) -> bool:
        return any(row["split"] in FORBIDDEN for row in self.rows.values())

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.rows) + "\n")

    def load(self, path: Path) -> TeacherCache:
        raw = json.loads(Path(path).read_text())
        for sample_id, row in raw.items():
            self.put(sample_id, row["split"], row["payload"])
        return self
