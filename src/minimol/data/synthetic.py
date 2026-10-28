"""Synthetic stand-ins. They exercise the pipeline and are not MPRA, RefSeq, or TDC labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from minimol.data.assays import ASSAYS
from minimol.data.codecs import pack_bits, pack_float16
from minimol.data.manifest import build_manifest, write_manifest
from minimol.data.sequences import normalize_sequence, sample_id, sequence_hash
from minimol.data.split import scaffold_split
from minimol.data.standardize import standardize_smiles
from minimol.oracles.brics import add_methyl
from minimol.oracles.rdkit_properties import properties_from_smiles

BUILTIN_SMILES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccncc1",
    "C1CCCCC1",
    "CCN",
    "CCOC",
    "c1ccc(O)cc1",
    "CC(=O)O",
    "CO",
    "CCC",
    "CCCC",
    "c1ccc(N)cc1",
    "CC(C)O",
    "c1ccc2ccccc2c1",
    "C1CCOC1",
    "c1ccccc1O",
    "CC(N)C(=O)O",
    "NCC(=O)O",
    "c1ccc(Cl)cc1",
    "CC(=O)C",
    "C1CCNCC1",
    "c1ccc(F)cc1",
    "CCCCO",
    "c1ccoc1",
    "CCS",
    "c1ccc(C)cc1",
)

REG_ASSAYS = ("K562", "HepG2", "SK-N-SH")
MOTIFS = {"K562": "TATAAA", "HepG2": "CCAAT", "SK-N-SH": "GGGCGG"}


def _write_frame(frame: pd.DataFrame, out_dir: Path, task: str, extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"
    frame.to_parquet(data_path, index=False)
    manifest = build_manifest(frame, task=task, data_path=data_path, extra=extra)
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest


def _activity(sequence: str, assay: str, rng: np.random.Generator) -> float:
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    value = 0.2 * (gc - 0.5)
    if MOTIFS[assay] in sequence:
        value += 2.5
    return float(value + rng.normal(0.0, 0.05))


def write_reg_sample(out_dir: Path, n_groups: int = 40, length: int = 96, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    group_ids = [f"g{index:04d}" for index in range(n_groups)]
    motif_ids = group_ids[: n_groups // 2]
    control_ids = group_ids[n_groups // 2 :]
    assignment = {}
    assignment.update(zip(motif_ids, scaffold_split(motif_ids, seed=seed), strict=False))
    assignment.update(zip(control_ids, scaffold_split(control_ids, seed=seed + 1), strict=False))
    rows = []
    for group_id in group_ids:
        bases = rng.choice(list("ACGT"), size=length)
        sequence = "".join(bases)
        if group_id in motif_ids:
            sequence = sequence[:12] + "TATAAACCAATGGGCGG" + sequence[29:]
            sequence = sequence[:length]
        variant = list(sequence)
        variant[-5] = "A" if variant[-5] != "A" else "C"
        for member_index, text in enumerate((sequence, "".join(variant))):
            text = normalize_sequence(text)
            for assay in REG_ASSAYS:
                rows.append(
                    {
                        "sample_id": sample_id(text, assay, "synthetic-reg-v1", group_id, str(member_index)),
                        "sequence": text,
                        "source_id": "builtin_demo_v1",
                        "data_revision": "synthetic-reg-v1",
                        "assay": assay,
                        "split": assignment[group_id],
                        "group_id": group_id,
                        "chromosome": f"chr{(int(group_id[1:]) % 5) + 1}",
                        "y_log2fc": _activity(text, assay, rng),
                        "y_se": 0.1,
                        "valid_y": True,
                        "sequence_hash": sequence_hash(text),
                        "license_id": "minimol_synthetic",
                        "relation": "=",
                        "units": "log2FC",
                        "label_quality": "synthetic_planted_motif",
                        "target_name": "mpra_log2fc",
                    }
                )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "reg",
        {
            "label_quality": "synthetic_planted_motif",
            "official_mpra": False,
            "note": "Planted motifs, not Malinois measurements.",
        },
    )


def _plant_cds(rng: np.random.Generator, length: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    bases = rng.choice(list("ACGT"), size=length)
    start = int(rng.integers(12, length // 3))
    coding = ("ATG" + "GGT" * 12 + "TAA")[:36]
    bases[start : start + len(coding)] = list(coding)
    plus = np.zeros(length, dtype=np.uint8)
    plus[start : start + len(coding)] = 1
    start2 = int(rng.integers(length // 2, length - 36))
    bases[start2 : start2 + 24] = list("C" * 24)
    minus = np.zeros(length, dtype=np.uint8)
    minus[start2 : start2 + 24] = 1
    if length > 8:
        bases[3] = "N"
        bases[4] = "N"
    noise = rng.normal(0.0, 0.02, size=(2, length))
    teacher = np.clip(np.stack([plus, minus]).astype(np.float64) * 0.8 + 0.1 + noise, 0.02, 0.98)
    mask = (bases != "N").astype(np.uint8)
    return "".join(bases), plus, minus, teacher, mask


def write_cds_sample(out_dir: Path, per_species: int = 8, length: int = 192, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    species = [
        ("tax100", "train"),
        ("tax200", "train"),
        ("tax300", "validation"),
        ("tax400", "test"),
    ]
    rows = []
    for taxid, split in species:
        for index in range(per_species):
            sequence, plus, minus, teacher, mask = _plant_cds(rng, length)
            start = index * length
            rows.append(
                {
                    "sample_id": sample_id(taxid, str(start), sequence_hash(sequence)),
                    "assembly_accession": f"ASM_{taxid}",
                    "species_taxid": taxid,
                    "chromosome": "chr1",
                    "start_0based": start,
                    "end_exclusive": start + length,
                    "genome_build": "synthetic-build",
                    "sequence": sequence,
                    "length": length,
                    "gt_plus": pack_bits(plus),
                    "gt_minus": pack_bits(minus),
                    "teacher_p_plus": pack_float16(teacher[0]),
                    "teacher_p_minus": pack_float16(teacher[1]),
                    "valid_mask": pack_bits(mask),
                    "split": split,
                    "sequence_hash": sequence_hash(sequence),
                    "teacher_revision": "synthetic-local",
                    "teacher_context_bp": length,
                    "teacher_kind": "synthetic_noisy_label",
                    "upstream_license": "minimol_synthetic",
                }
            )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "cds",
        {
            "teacher_kind": "synthetic_noisy_label",
            "official_refseq": False,
            "note": "Noisy copies of planted labels. Not Carbon-A and not RefSeq.",
        },
    )


def _molecule_table(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    molecules = []
    seen = set()
    for smiles in BUILTIN_SMILES:
        standardized = standardize_smiles(smiles)
        if not standardized["valid"] or standardized["unsupported_element"]:
            continue
        candidates = [standardized]
        variant = add_methyl(standardized["smiles_canonical"], rng)
        if variant:
            second = standardize_smiles(variant)
            if second["valid"]:
                candidates.append(second)
        for item in candidates:
            if item["smiles_canonical"] in seen:
                continue
            seen.add(item["smiles_canonical"])
            props = properties_from_smiles(item["smiles_canonical"])
            if props is None:
                continue
            item = dict(item)
            item["props"] = props
            molecules.append(item)
    splits = scaffold_split([item["scaffold_id"] for item in molecules], seed=seed)
    for item, split in zip(molecules, splits, strict=True):
        item["split"] = split
    return molecules


def write_admet_sample(out_dir: Path, seed: int = 5) -> dict:
    molecules = _molecule_table(seed)
    rows = []
    endpoints = (
        ("logp_rdkit", "logp", "regression"),
        ("tpsa_rdkit", "tpsa", "regression"),
        ("qed_rdkit", "qed", "regression"),
        ("high_logp_proxy", None, "binary"),
    )
    for item in molecules:
        props = item["props"]
        for endpoint, key, _kind in endpoints:
            if endpoint == "high_logp_proxy":
                measurement = 1.0 if props["logp"] > 2.0 else 0.0
            else:
                measurement = float(props[key])
            meta = ASSAYS[endpoint]
            rows.append(
                {
                    "sample_id": sample_id(item["smiles_canonical"], endpoint, "synthetic-admet-v1"),
                    "smiles_original": item["smiles_original"],
                    "smiles_canonical": item["smiles_canonical"],
                    "inchikey": item["inchikey"],
                    "scaffold_id": item["scaffold_id"],
                    "scaffold": item["scaffold"],
                    "assay_id": endpoint,
                    "endpoint": endpoint,
                    "target_name": endpoint,
                    "assay_type": meta["track"],
                    "units": meta["units"],
                    "measurement": measurement,
                    "relation": "=",
                    "source": "builtin_demo_v1",
                    "split": item["split"],
                    "label_quality": "computed_proxy",
                    "license_id": "minimol_synthetic",
                    "stereochemistry_status": item["stereochemistry_status"],
                    "organism": "not_applicable",
                    "conditions": "rdkit_descriptor",
                }
            )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "admet",
        {
            "official_tdc": False,
            "benchmark_name": None,
            "label_quality": "computed_proxy",
            "note": "RDKit descriptors. Not TDC and not experimental ADMET.",
        },
    )


def write_tox_sample(out_dir: Path, seed: int = 5) -> dict:
    molecules = _molecule_table(seed)
    rows = []
    for item in molecules:
        measurement = 1.0 if item["props"]["logp"] > 2.0 else 0.0
        meta = ASSAYS["tox_proxy_high_logp"]
        rows.append(
            {
                "sample_id": sample_id(item["smiles_canonical"], "tox_proxy_high_logp", "synthetic-tox-v1"),
                "smiles_original": item["smiles_original"],
                "smiles_canonical": item["smiles_canonical"],
                "inchikey": item["inchikey"],
                "scaffold_id": item["scaffold_id"],
                "scaffold": item["scaffold"],
                "assay_id": "tox_proxy_high_logp",
                "endpoint": "tox_proxy_high_logp",
                "target_name": "tox_proxy_high_logp",
                "assay_type": "tox_proxy",
                "units": meta["units"],
                "measurement": measurement,
                "relation": "=",
                "source": "builtin_demo_v1",
                "split": item["split"],
                "label_quality": "computed_proxy",
                "license_id": "minimol_synthetic",
                "stereochemistry_status": item["stereochemistry_status"],
                "organism": "not_applicable",
                "conditions": "rdkit_logp_gt_2",
            }
        )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "tox",
        {
            "label_quality": "computed_proxy",
            "note": "Computed logP threshold. Not Tox21, ClinTox, or a clinical toxicity label.",
        },
    )


def _proteins() -> list[dict]:
    rows = []
    specs = [
        ("prot_train_a", "train", 4, 1),
        ("prot_train_b", "train", 2, 2),
        ("prot_val", "validation", 3, 1),
        ("prot_test", "test", 5, 0),
    ]
    for name, split, n_k, n_d in specs:
        sequence = ["A"] * 40
        for index in range(n_k):
            sequence[index] = "K"
        for index in range(n_d):
            sequence[30 + index] = "D"
        rows.append({"protein_id": name, "split": split, "sequence": "".join(sequence), "cluster": split})
    return rows


def _interaction(sequence: str, props: dict) -> float:
    return 0.01 * props["mw"] + 0.25 * sequence.count("K") - 0.1 * sequence.count("D")


def write_bind_sample(out_dir: Path, seed: int = 3) -> dict:
    molecules = _molecule_table(seed)
    proteins = _proteins()
    rows = []
    for protein in proteins:
        for item in molecules:
            if item["split"] != protein["split"]:
                continue
            props = item["props"]
            rows.append(
                {
                    "sample_id": sample_id(protein["protein_id"], item["smiles_canonical"], "pKi_synthetic"),
                    "smiles_canonical": item["smiles_canonical"],
                    "inchikey": item["inchikey"],
                    "scaffold_id": item["scaffold_id"],
                    "protein_id": protein["protein_id"],
                    "protein_sequence": protein["sequence"],
                    "cluster_id": protein["cluster"],
                    "endpoint": "pKi_synthetic",
                    "assay_type": "biochemical_ki",
                    "units": "synthetic_log_units",
                    "measurement": _interaction(protein["sequence"], props),
                    "relation": "=",
                    "split": protein["split"],
                    "label_quality": "synthetic_interaction",
                    "license_id": "minimol_synthetic",
                    "source": "builtin_demo_v1",
                }
            )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "bind",
        {"label_quality": "synthetic_interaction", "note": "Not BindingDB and not a Ki."},
    )


def write_precision_sample(out_dir: Path, seed: int = 3) -> dict:
    molecules = _molecule_table(seed)
    proteins = _proteins()
    rows = []
    for protein in proteins:
        ligands = [item for item in molecules if item["split"] == protein["split"]]
        if not ligands:
            continue
        wild = protein["sequence"]
        mutants = {
            "K1A": ("A" + wild[1:], "decrease"),
            "A10K": (wild[:9] + "K" + wild[10:], "increase"),
            "A20L": (wild[:19] + "L" + wild[20:], "no_change"),
        }
        ligand = ligands[0]
        props = ligand["props"]
        base = _interaction(wild, props)
        for mutation, (mutant, _expected) in mutants.items():
            delta = _interaction(mutant, props) - base
            if delta < -0.1:
                direction = "decrease"
            elif delta > 0.1:
                direction = "increase"
            else:
                direction = "no_change"
            rows.append(
                {
                    "sample_id": sample_id(protein["protein_id"], mutation, ligand["smiles_canonical"]),
                    "smiles_canonical": ligand["smiles_canonical"],
                    "inchikey": ligand["inchikey"],
                    "scaffold_id": ligand["scaffold_id"],
                    "protein_id": protein["protein_id"],
                    "cluster_id": protein["cluster"],
                    "sequence_wt": wild,
                    "sequence_mut": mutant,
                    "mutation": mutation,
                    "endpoint": "delta_synthetic_activity",
                    "assay_type": "matched_mutation",
                    "units": "synthetic_log_units",
                    "measurement": delta,
                    "direction": direction,
                    "relation": "=",
                    "split": protein["split"],
                    "label_quality": "synthetic_matched_pair",
                    "license_id": "minimol_synthetic",
                    "source": "builtin_demo_v1",
                }
            )
    frame = pd.DataFrame(rows)
    return _write_frame(
        frame,
        Path(out_dir),
        "precision",
        {
            "label_quality": "synthetic_matched_pair",
            "note": "Matched synthetic pairs. Not an experimental mutation assay.",
        },
    )
