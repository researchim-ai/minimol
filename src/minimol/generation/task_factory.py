"""Synthetic design tasks with exact RDKit labels.

These tasks train constraint-following and ranking. They do not measure
experimental ADMET, and they are not drawn from ZINC until that source is approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minimol.data.sequences import sample_id
from minimol.data.standardize import standardize_smiles
from minimol.data.synthetic import BUILTIN_SMILES
from minimol.errors import MinimolError
from minimol.oracles.brics import add_methyl, remove_terminal_carbon
from minimol.oracles.rdkit_properties import SA_PROXY_DEFINITION, properties_from_smiles
from minimol.oracles.validity import is_valid_smiles

OBJECTIVES = ("qed", "logp", "sa_proxy")


def _split_for(scaffold_id: str, val_fraction: float = 0.1) -> str:
    bucket = int(hashlib.sha256(scaffold_id.encode()).hexdigest()[:8], 16) % 1000
    if bucket < int(val_fraction * 1000):
        return "validation"
    return "train"


def _candidate(smiles: str, valid: bool, reason: str | None = None) -> dict:
    if not valid:
        return {"smiles": smiles, "valid": False, "oracle": None, "reason": reason or "invalid"}
    props = properties_from_smiles(smiles)
    if props is None:
        return {"smiles": smiles, "valid": False, "oracle": None, "reason": "unparseable"}
    return {"smiles": smiles, "valid": True, "oracle": props, "reason": reason}


def _improves(candidate, reference, objective: str, direction: str) -> bool:
    if not candidate["valid"]:
        return False
    left = candidate["oracle"][objective]
    right = reference[objective]
    return left > right if direction == "maximize" else left < right


def _passes(candidate, constraints) -> bool:
    if not candidate["valid"]:
        return False
    for constraint in constraints:
        value = candidate["oracle"][constraint["property"]]
        if "min" in constraint and value < constraint["min"]:
            return False
        if "max" in constraint and value > constraint["max"]:
            return False
    return True


def verify_task(task: dict, atol: float = 1e-4) -> bool:
    for candidate in task["candidates"]:
        if not candidate["valid"]:
            if is_valid_smiles(candidate["smiles"]):
                return False
            continue
        fresh = properties_from_smiles(candidate["smiles"])
        if fresh is None:
            return False
        for key, value in candidate["oracle"].items():
            if abs(fresh[key] - value) > atol:
                return False
    return True


def make_tasks(limit: int = 20, seed: int = 0, holdout_scaffolds: set[str] | None = None) -> list[dict]:
    holdout = holdout_scaffolds or set()
    rng = np.random.default_rng(seed)
    tasks = []
    for smiles in BUILTIN_SMILES:
        if len(tasks) >= limit:
            break
        standardized = standardize_smiles(smiles)
        if not standardized["valid"] or standardized["scaffold_id"] in holdout:
            continue
        props = properties_from_smiles(standardized["smiles_canonical"])
        if props is None:
            continue
        objective = str(rng.choice(list(OBJECTIVES)))
        direction = "minimize" if objective == "sa_proxy" else "maximize"
        constraints = [
            {"property": "mw", "max": float(props["mw"] + 80.0)},
            {"property": "tpsa", "max": float(props["tpsa"] + 40.0)},
        ]
        split = _split_for(standardized["scaffold_id"])
        edits = []
        methyl = add_methyl(standardized["smiles_canonical"], rng)
        if methyl:
            edits.append(methyl)
        removed = remove_terminal_carbon(standardized["smiles_canonical"], rng)
        if removed:
            edits.append(removed)
        candidates = [_candidate(item, True) for item in edits]
        candidates.append(_candidate("C(", False, "invalid"))
        candidates.append(_candidate("C" * 40, True, "hard_negative_mw"))
        solved = any(
            _passes(candidate, constraints) and _improves(candidate, props, objective, direction)
            for candidate in candidates
        )
        task = {
            "task_id": sample_id(standardized["smiles_canonical"], objective, str(seed), str(len(tasks))),
            "source_smiles": standardized["smiles_canonical"],
            "source_dataset": "builtin_demo_v1",
            "scaffold": standardized["scaffold_id"],
            "objective": {"property": objective, "direction": direction},
            "constraints": constraints,
            "candidates": candidates,
            "reference_oracle": props,
            "solved": solved,
            "evaluation_method": "rdkit-properties",
            "sa_proxy_definition": SA_PROXY_DEFINITION,
            "provenance": {
                "split": split,
                "license_id": "minimol_synthetic",
                "source_record": "builtin_demo_v1",
            },
            "verification": {"all_candidates_recomputed": False, "hard_constraints_enforced": True},
        }
        task["verification"]["all_candidates_recomputed"] = verify_task(task)
        if not task["verification"]["all_candidates_recomputed"]:
            raise MinimolError("task oracle failed recomputation")
        tasks.append(task)
    return tasks


def write_tasks(path: Path, tasks: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(task) for task in tasks) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minimol.generation.task_factory")
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    tasks = make_tasks(limit=args.limit, seed=args.seed)
    write_tasks(Path(args.out), tasks)
    print(f"wrote {len(tasks)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
