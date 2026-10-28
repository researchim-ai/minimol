"""Reward for a verified candidate. Invalid chemistry is rejected, not scored."""

from __future__ import annotations

from minimol.oracles.rdkit_properties import properties_from_smiles


def compute_reward(
    feasible: bool,
    objective: float,
    scale: float,
    ood_penalty: float = 0.0,
    uncertainty: float = 0.0,
    duplicate: float = 0.0,
    oracle_cost: float = 0.0,
    lambdas: dict | None = None,
) -> dict:
    weights = {"ood": 1.0, "uncertainty": 1.0, "duplicate": 1.0, "cost": 1.0}
    if lambdas:
        weights.update(lambdas)
    if not feasible:
        return {"reward": None, "rejected": True, "reason": "infeasible"}
    reward = float(objective) / float(scale)
    reward -= weights["ood"] * float(ood_penalty)
    reward -= weights["uncertainty"] * float(uncertainty)
    reward -= weights["duplicate"] * float(duplicate)
    reward -= weights["cost"] * float(oracle_cost)
    return {"reward": float(reward), "rejected": False, "reason": None}


def red_team_flags(smiles: str, source_smiles: str | None = None) -> list[str]:
    props = properties_from_smiles(smiles)
    if props is None:
        return ["invalid"]
    flags = []
    if props["mw"] > 800:
        flags.append("extreme_mw")
    if props["mw"] < 30:
        flags.append("extreme_small")
    if source_smiles and smiles == source_smiles:
        flags.append("duplicate_of_source")
    return flags


def recommend_next_to_measure(ids, uncertainty, upside, similarity, k: int = 3) -> list[str]:
    """Greedy pick: uncertainty x upside x (1 - similarity to already picked)."""

    remaining = set(range(len(ids)))
    picked = []
    sims = list(similarity)
    while remaining and len(picked) < k:
        best = None
        best_score = None
        for index in remaining:
            diversity = 1.0 - float(sims[index])
            score = float(uncertainty[index]) * float(upside[index]) * diversity
            if best_score is None or score > best_score:
                best = index
                best_score = score
        picked.append(ids[best])
        remaining.remove(best)
        for index in remaining:
            sims[index] = max(sims[index], similarity[best])
    return picked
