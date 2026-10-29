"""Binding baselines required before a fusion model can claim a gain.

Ligand-only and protein-only models are fit on the training split and scored
on validation. A protein-only model that matches the joint model is a target-prior
risk, not evidence of interaction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from minimol.data.sequences import AA_ALPHABET
from minimol.errors import MinimolError
from minimol.evaluation.metrics import mae, spearman
from minimol.jsonutil import write_json
from minimol.oracles.rdkit_properties import properties_from_smiles


def amino_acid_counts(sequence: str) -> np.ndarray:
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float64)
    for char in sequence:
        if char in AA_ALPHABET:
            counts[AA_ALPHABET.index(char)] += 1.0
    return counts


def _ligand_features(smiles: str) -> np.ndarray | None:
    props = properties_from_smiles(smiles)
    if props is None:
        return None
    return np.array([props["mw"], props["logp"], props["tpsa"], props["qed"]], dtype=np.float64)


def _fit_predict(train_x, train_y, test_x) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def _score(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "spearman": spearman(y_true, y_pred), "n": int(len(y_true))}


def binding_baseline_report(frame: pd.DataFrame) -> dict:
    if frame["endpoint"].nunique() != 1:
        raise MinimolError("score one binding endpoint at a time")
    rows = []
    for record in frame.itertuples(index=False):
        ligand = _ligand_features(record.smiles_canonical)
        if ligand is None:
            continue
        rows.append(
            {
                "split": str(record.split),
                "y": float(record.measurement),
                "ligand": ligand,
                "protein": amino_acid_counts(record.protein_sequence),
                "protein_id": str(record.protein_id),
            }
        )
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    if len(train) < 2 or not val:
        raise MinimolError("binding baselines need a train split and a validation split")
    y_train = np.array([row["y"] for row in train])
    y_val = np.array([row["y"] for row in val])
    ligand_train = np.stack([row["ligand"] for row in train])
    ligand_val = np.stack([row["ligand"] for row in val])
    protein_train = np.stack([row["protein"] for row in train])
    protein_val = np.stack([row["protein"] for row in val])
    joint_train = np.concatenate([ligand_train, protein_train], axis=1)
    joint_val = np.concatenate([ligand_val, protein_val], axis=1)
    scores = {
        "ligand_only": _score(y_val, _fit_predict(ligand_train, y_train, ligand_val)),
        "protein_only": _score(y_val, _fit_predict(protein_train, y_train, protein_val)),
        "ligand_plus_protein": _score(y_val, _fit_predict(joint_train, y_train, joint_val)),
    }
    protein_mae = scores["protein_only"]["mae"]
    joint_mae = scores["ligand_plus_protein"]["mae"]
    ligand_mae = scores["ligand_only"]["mae"]
    prior = (
        protein_mae is not None
        and joint_mae is not None
        and ligand_mae is not None
        and protein_mae <= joint_mae + 0.05
        and protein_mae + 0.05 < ligand_mae
    )
    return {
        "task": "bind_baselines",
        "endpoint": str(frame["endpoint"].iloc[0]),
        "label_quality": str(frame["label_quality"].iloc[0]) if "label_quality" in frame.columns else None,
        "scores": scores,
        "target_prior_risk": bool(prior),
        "unseen_proteins_in_validation": sorted({row["protein_id"] for row in val} - {row["protein_id"] for row in train}),
        "research_use_only": True,
        "clinical_use": False,
        "note": "These are linear baselines. A synthetic interaction label is not a Ki.",
    }


def write_binding_baselines(data_path: Path, out_dir: Path) -> dict:
    report = binding_baseline_report(pd.read_parquet(data_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", report)
    return report
