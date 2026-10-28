import json

import pytest

from minimol.data.build import build_dataset
from minimol.data.registry import validate_registry
from minimol.data.split_audit import split_audit
from minimol.errors import MinimolError
from minimol.evaluation.benchmark import pareto_front, run_benchmark
from minimol.reporting import release_check
from minimol.training.baselines import run_reg_baselines
from minimol.training.distill import choose_recipe


def test_registry_and_release_are_not_a_commercial_approval():
    errors = validate_registry("LICENSES/data_registry.csv")
    assert errors == []
    report = release_check("LICENSES/data_registry.csv", "docs/MODEL_CARD.md")
    assert report["commercial_release"] is False
    assert report["human_legal_review"] == "pending"
    assert report["passed_automated"] is True


def test_tdc_benchmark_is_refused(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"task": "admet", "official_tdc": False}))
    with pytest.raises(MinimolError, match="TDC"):
        run_benchmark("tdc-admet", [metrics], tmp_path / "out.json", official_tdc=False)


def test_distillation_nogo_and_pareto():
    decision = choose_recipe(
        {"supervised": {"macro_spearman": 0.4}, "kd": {"macro_spearman": 0.3}},
        min_epochs_met=True,
    )
    assert decision["decision"] == "NO-GO"
    assert decision["ship"] == "supervised"
    inconclusive = choose_recipe({"supervised": {"macro_spearman": 0.4}}, min_epochs_met=False)
    assert inconclusive["decision"] == "INCONCLUSIVE"
    front = pareto_front(
        [
            {"name": "a", "macro_spearman": 0.2, "param_count": 1, "p50_s": 1},
            {"name": "b", "macro_spearman": 0.8, "param_count": 2, "p50_s": 1},
            {"name": "c", "macro_spearman": 0.8, "param_count": 5, "p50_s": 3},
        ]
    )
    names = {row["name"] for row in front}
    assert "c" not in names
    assert "b" in names


def test_sample_audit_and_ridge(tmp_path):
    build_dataset("synthetic_reg", tmp_path / "reg", limit=40, seed=4)
    build_dataset("synthetic_cds", tmp_path / "cds", limit=4, seed=4)
    build_dataset("synthetic_admet", tmp_path / "admet", seed=4)
    build_dataset("synthetic_bind", tmp_path / "bind", seed=4)
    for name in ("reg", "cds", "admet", "bind"):
        split_audit(tmp_path / name / "manifest.json")
    with pytest.raises(Exception):
        build_dataset("tdc_admet", tmp_path / "tdc")
    payload = run_reg_baselines(tmp_path / "reg" / "data.parquet", tmp_path / "ridge", k_max=4)
    wins = 0
    for assay in ("K562", "HepG2", "SK-N-SH"):
        if payload["ridge"]["per_assay"][assay]["mae"] < payload["mean"]["per_assay"][assay]["mae"]:
            wins += 1
    assert wins >= 2
    assert payload["ridge"]["per_assay"]["K562"]["spearman"] > 0.3
    assert (tmp_path / "ridge" / "metrics.json").exists()
    assert (tmp_path / "ridge" / "manifest.json").exists()
