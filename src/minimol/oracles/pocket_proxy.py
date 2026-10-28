"""Pocket scores are not binding affinities unless a real docking backend is configured.

The default proxy refuses to emit a number. A decoy exists only for pipeline tests
and is explicitly marked `is_binding_affinity=false`.
"""

from __future__ import annotations

from minimol.oracles.rdkit_properties import properties_from_smiles


def pocket_proxy_score(smiles: str, pocket_sequence: str | None = None, explicit_decoy: bool = False) -> dict:
    del pocket_sequence
    if not explicit_decoy:
        return {
            "score": None,
            "kind": "disabled",
            "is_binding_affinity": False,
            "reason": "No docking backend is configured. This is not a binding affinity.",
        }
    props = properties_from_smiles(smiles)
    if props is None:
        return {
            "score": None,
            "kind": "decoy_invalid",
            "is_binding_affinity": False,
            "reason": "Unparseable SMILES. Not a binding affinity.",
        }
    return {
        "score": -float(props["mw"]),
        "kind": "negative_molecular_weight_decoy",
        "is_binding_affinity": False,
        "reason": "Decoy rank by molecular weight only. Not a binding affinity or docking score.",
    }
