"""Validity checks. An unparseable SMILES is rejected, not repaired into a prediction."""

from __future__ import annotations

from rdkit import Chem


def is_valid_smiles(smiles: str | None) -> bool:
    if smiles is None or not str(smiles).strip():
        return False
    return Chem.MolFromSmiles(str(smiles)) is not None
