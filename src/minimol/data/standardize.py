"""SMILES standardization. Stereochemistry is kept. Salts are recorded, not silently forgotten."""

from __future__ import annotations

import hashlib

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.error")

ALLOWED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
METAL_ATOMIC_NUMBERS = {3, 11, 12, 13, 19, 20, 26, 29, 30, 47, 78, 79, 80}
LARGE_HEAVY_ATOM_COUNT = 150


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def standardize_smiles(smiles: str | None) -> dict:
    result = {
        "smiles_original": None if smiles is None else str(smiles),
        "valid": False,
        "smiles_canonical": None,
        "inchikey": None,
        "scaffold": "",
        "scaffold_id": None,
        "graph_hash": None,
        "salt_stripped": False,
        "removed_fragments": [],
        "mixture": False,
        "unsupported_element": False,
        "metallocomplex": False,
        "too_large": False,
        "isotope": False,
        "explicit_hydrogen": False,
        "formal_charge": None,
        "stereochemistry_status": "absent",
        "tautomer_status": "not_canonicalized",
        "heavy_atoms": 0,
        "error": None,
    }
    if smiles is None or not str(smiles).strip():
        result["error"] = "empty"
        return result
    original = str(smiles).strip()
    result["explicit_hydrogen"] = "[H]" in original
    mol = Chem.MolFromSmiles(original)
    if mol is None:
        result["error"] = "unparseable"
        return result
    try:
        fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    except Exception:
        result["error"] = "unsanitary_fragments"
        return result
    if len(fragments) > 1:
        heavy = [fragment.GetNumHeavyAtoms() for fragment in fragments]
        result["mixture"] = sum(count >= 3 for count in heavy) >= 2
        order = sorted(range(len(fragments)), key=lambda index: heavy[index], reverse=True)
        mol = fragments[order[0]]
        result["removed_fragments"] = [
            Chem.MolToSmiles(fragments[index], canonical=True) for index in order[1:]
        ]
        result["salt_stripped"] = True
    atoms = list(mol.GetAtoms())
    numbers = [atom.GetAtomicNum() for atom in atoms]
    result["unsupported_element"] = any(number not in ALLOWED_ATOMIC_NUMBERS for number in numbers)
    result["metallocomplex"] = any(number in METAL_ATOMIC_NUMBERS for number in numbers)
    result["isotope"] = any(atom.GetIsotope() != 0 for atom in atoms)
    result["explicit_hydrogen"] = result["explicit_hydrogen"] or any(atom.GetSymbol() == "H" for atom in atoms)
    result["too_large"] = mol.GetNumHeavyAtoms() > LARGE_HEAVY_ATOM_COUNT
    result["heavy_atoms"] = int(mol.GetNumHeavyAtoms())
    result["formal_charge"] = int(Chem.GetFormalCharge(mol))
    stereo = any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in atoms) or any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()
    )
    result["stereochemistry_status"] = "retained" if stereo else "absent"
    try:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        result["error"] = "canonicalization_failed"
        return result
    result["smiles_canonical"] = canonical
    result["graph_hash"] = _sha(canonical)
    try:
        result["inchikey"] = Chem.MolToInchiKey(mol)
    except Exception:
        result["inchikey"] = None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or ""
    except Exception:
        scaffold = ""
    result["scaffold"] = scaffold
    if scaffold:
        result["scaffold_id"] = "murcko:" + _sha(scaffold)[:16]
    else:
        identity = result["inchikey"] or result["graph_hash"]
        result["scaffold_id"] = "acyclic:" + identity
    result["valid"] = True
    return result
