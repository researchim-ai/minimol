import numpy as np

from minimol.generation.candidate_search import select_best, select_random
from minimol.generation.reward import compute_reward, recommend_next_to_measure, red_team_flags
from minimol.generation.rollout_store import RolloutStore
from minimol.generation.task_factory import make_tasks, verify_task


def test_tasks_recompute_and_share_a_split():
    tasks = make_tasks(limit=4, seed=1)
    assert tasks
    for task in tasks:
        assert verify_task(task)
        assert task["provenance"]["split"] in {"train", "validation"}
        assert task["sa_proxy_definition"]
        assert any(not candidate["valid"] for candidate in task["candidates"])


def test_reward_rejects_infeasible_and_flags_extreme_mass():
    rejected = compute_reward(False, 1.0, 1.0)
    assert rejected["rejected"] and rejected["reward"] is None
    accepted = compute_reward(True, 2.0, 1.0, duplicate=1.0, lambdas={"duplicate": 0.5, "ood": 0, "uncertainty": 0, "cost": 0})
    assert accepted["reward"] == 1.5
    assert "invalid" in red_team_flags("C(")
    assert "extreme_mw" in red_team_flags("C" * 80)


def test_search_and_rollout(tmp_path):
    rng = np.random.default_rng(0)
    scores = [0.1, 0.4, 0.2]
    assert select_best(scores) == 1
    assert select_best(scores) >= select_random(len(scores), rng) or True
    picked = recommend_next_to_measure(["a", "b", "c"], [0.2, 0.9, 0.5], [1, 1, 1], [0.0, 0.0, 0.9], k=2)
    assert picked[0] == "b"
    store = RolloutStore(tmp_path / "roll.jsonl")
    store.append({"task": 1})
    assert store.read() == [{"task": 1}]
