"""Assay catalog. Computed proxies are labeled as proxies."""

from __future__ import annotations

import re

ASSAYS = {
    "K562": {
        "track": "regulatory",
        "units": "log2FC",
        "organism": "Homo sapiens",
        "definition": "MPRA log2 fold-change in the source assay scale",
    },
    "HepG2": {
        "track": "regulatory",
        "units": "log2FC",
        "organism": "Homo sapiens",
        "definition": "MPRA log2 fold-change in the source assay scale",
    },
    "SK-N-SH": {
        "track": "regulatory",
        "units": "log2FC",
        "organism": "Homo sapiens",
        "definition": "MPRA log2 fold-change in the source assay scale",
    },
    "logp_rdkit": {
        "track": "admet",
        "units": "log10_partition",
        "task": "regression",
        "quality": "computed_proxy",
        "definition": "RDKit MolLogP. Not an experimental ADMET endpoint.",
    },
    "tpsa_rdkit": {
        "track": "admet",
        "units": "angstrom2",
        "task": "regression",
        "quality": "computed_proxy",
        "definition": "RDKit TPSA. Not an experimental ADMET endpoint.",
    },
    "qed_rdkit": {
        "track": "admet",
        "units": "unitless_0_1",
        "task": "regression",
        "quality": "computed_proxy",
        "definition": "RDKit QED. Not an experimental drug-likeness measurement.",
    },
    "high_logp_proxy": {
        "track": "admet",
        "units": "indicator",
        "task": "binary",
        "quality": "computed_proxy",
        "definition": "1 if RDKit MolLogP > 2. A computed rule, not toxicity.",
    },
    "tox_proxy_high_logp": {
        "track": "tox",
        "units": "indicator",
        "task": "binary",
        "quality": "computed_proxy",
        "definition": "Same computed logP rule under a tox-track name. Not a clinical toxicity label.",
    },
    "pKi_synthetic": {
        "track": "bind",
        "units": "synthetic_log_units",
        "task": "regression",
        "quality": "synthetic_interaction",
        "definition": "Synthetic interaction of ligand mass and lysine count. Not a Ki.",
    },
}

_MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])")


def parse_mutation(text: str) -> tuple[str, int, str]:
    match = _MUTATION.fullmatch(str(text).strip())
    if match is None:
        raise ValueError(f"unparsed mutation: {text}")
    return match.group(1), int(match.group(2)), match.group(3)


def task_kind(endpoint: str) -> str:
    meta = ASSAYS.get(endpoint)
    if meta is None:
        raise KeyError(endpoint)
    return meta.get("task", "regression")
