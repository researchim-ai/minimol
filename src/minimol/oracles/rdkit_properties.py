"""Deterministic chemistry oracles. These are not experimental ADMET labels."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

SA_PROXY_DEFINITION = (
    "sa_proxy = 1 + ring_count + 0.25 * stereocenters + 0.05 * rotatable bonds. "
    "This is not the Ertl SAScore."
)
DETERMINISTIC_PROPERTIES = (
    "qed",
    "logp",
    "tpsa",
    "mw",
    "hbd",
    "hba",
    "rotatable",
    "rings",
    "sa_proxy",
)


def sa_proxy(mol) -> float:
    stereo = sum(1 for atom in mol.GetAtoms() if atom.HasProp("_CIPCode"))
    rings = float(Descriptors.RingCount(mol))
    rotatable = float(Descriptors.NumRotatableBonds(mol))
    return 1.0 + rings + 0.25 * stereo + 0.05 * rotatable


def properties_from_mol(mol) -> dict[str, float]:
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": float(Descriptors.NumHDonors(mol)),
        "hba": float(Descriptors.NumHAcceptors(mol)),
        "rotatable": float(Descriptors.NumRotatableBonds(mol)),
        "rings": float(Descriptors.RingCount(mol)),
        "qed": float(QED.qed(mol)),
        "sa_proxy": float(sa_proxy(mol)),
    }


def properties_from_smiles(smiles: str) -> dict[str, float] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return properties_from_mol(mol)


@lru_cache(maxsize=1)
def descriptor_functions() -> tuple:
    probe = Chem.MolFromSmiles("c1ccccc1")
    kept = []
    for name, function in Descriptors.descList:
        if name == "Ipc":
            continue
        try:
            value = float(function(probe))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        kept.append((name, function))
    return tuple(kept)


def descriptor_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    functions = descriptor_functions()
    values = np.full(len(functions), np.nan, dtype=np.float64)
    if mol is None:
        return values
    for index, (_, function) in enumerate(functions):
        try:
            value = float(function(mol))
        except Exception:
            continue
        if np.isfinite(value):
            values[index] = value
    return values


class MedianImputer:
    def fit(self, matrix: np.ndarray) -> MedianImputer:
        values = np.asarray(matrix, dtype=np.float64)
        median = np.nanmedian(values, axis=0)
        self.median_ = np.where(np.isfinite(median), median, 0.0)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.array(matrix, dtype=np.float64, copy=True)
        bad = ~np.isfinite(values)
        rows, cols = np.where(bad)
        values[rows, cols] = self.median_[cols]
        return values
