"""BRICS fragments and a few deterministic edits. Enumeration is capped."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.error")


def brics_fragments(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return sorted(BRICS.BRICSDecompose(mol))


def add_methyl(smiles: str, rng: np.random.Generator) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    candidates = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() in {"C", "N", "O"} and atom.GetDegree() < 3 and atom.GetAtomicNum() > 1
    ]
    if not candidates:
        return None
    editable = Chem.RWMol(mol)
    anchor = int(rng.choice(np.array(candidates)))
    added = editable.AddAtom(Chem.Atom(6))
    editable.AddBond(anchor, added, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(editable)
    except Exception:
        return None
    return Chem.MolToSmiles(editable, canonical=True)


def remove_terminal_carbon(smiles: str, rng: np.random.Generator) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 3:
        return None
    terminals = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetDegree() == 1 and not atom.GetIsAromatic()
    ]
    if not terminals:
        return None
    editable = Chem.RWMol(mol)
    editable.RemoveAtom(int(rng.choice(np.array(terminals))))
    try:
        Chem.SanitizeMol(editable)
    except Exception:
        return None
    return Chem.MolToSmiles(editable, canonical=True)


def brics_recombine(smiles_a: str, smiles_b: str, limit: int = 4) -> list[str]:
    mols = [Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)]
    if any(mol is None for mol in mols):
        return []
    found = []
    try:
        products = BRICS.BRICSBuild(mols)
    except Exception:
        return []
    for product in products:
        if len(found) >= limit:
            break
        try:
            Chem.SanitizeMol(product)
            found.append(Chem.MolToSmiles(product, canonical=True))
        except Exception:
            continue
    return found
