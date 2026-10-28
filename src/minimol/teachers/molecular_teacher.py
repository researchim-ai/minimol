"""Fingerprint teacher fit on the training split only."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from minimol.data.featurize import morgan_fingerprint
from minimol.errors import SplitLeakageError

FORBIDDEN = {"test", "release_holdout"}


class FingerprintTeacher:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self._fitted = False

    def fit(self, smiles, y, splits) -> FingerprintTeacher:
        keep = [split == "train" for split in splits]
        if not any(keep):
            raise ValueError("teacher fit needs training rows")
        features = np.stack([morgan_fingerprint(item) for item, flag in zip(smiles, keep, strict=True) if flag])
        target = np.asarray(y, dtype=np.float64)[np.asarray(keep)]
        self.model.fit(features, target)
        self._fitted = True
        return self

    def predict(self, smiles, splits) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("teacher is not fit")
        for split in splits:
            if split in FORBIDDEN:
                raise SplitLeakageError(f"refusing teacher prediction on {split}")
        features = np.stack([morgan_fingerprint(item) for item in smiles])
        return self.model.predict(features)
