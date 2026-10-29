import json

from minimol.data.synthetic import write_bind_sample, write_precision_sample, write_reg_sample
from minimol.evaluation.aggregate import aggregate_seed_reports
from minimol.evaluation.bind_baselines import binding_baseline_report
from minimol.evaluation.controls import regulatory_controls
from minimol.evaluation.precision_baselines import precision_baseline_report
from minimol.evaluation.ranking import ndcg_at_k, oracle_relevance, ranking_report
from minimol.generation.task_factory import make_tasks


def test_binding_baselines_do_not_flag_a_protein_shortcut(tmp_path):
    write_bind_sample(tmp_path / "bind", seed=3)
    frame_path = tmp_path / "bind" / "data.parquet"
    import pandas as pd

    report = binding_baseline_report(pd.read_parquet(frame_path))
    assert report["target_prior_risk"] is False
    assert report["scores"]["ligand_only"]["mae"] < report["scores"]["protein_only"]["mae"]
    assert report["unseen_proteins_in_validation"]


def test_precision_composition_beats_zero_delta(tmp_path):
    write_precision_sample(tmp_path / "precision", seed=3)
    import pandas as pd

    report = precision_baseline_report(pd.read_parquet(tmp_path / "precision" / "data.parquet"))
    assert report["scores"]["composition_delta"]["mae"] < report["scores"]["zero_delta"]["mae"]
    assert report["scores"]["docking_only"]["mae"] is None


def test_shuffled_labels_are_weaker_than_the_motif_signal(tmp_path):
    write_reg_sample(tmp_path / "reg", n_groups=40, seed=9)
    import pandas as pd

    report = regulatory_controls(pd.read_parquet(tmp_path / "reg" / "data.parquet"), seed=1)
    real = report["real_ridge"]["macro_spearman"]
    shuffled = report["shuffled_label_ridge"]["macro_spearman"]
    assert real is not None
    assert shuffled is None or real > shuffled + 0.15
    assert report["leakage_suspect"] is False


def test_perfect_ranking_has_unit_ndcg():
    relevance = [3.0, 2.0, 1.0, 0.0]
    assert ndcg_at_k(relevance, k=4) == 1.0
    assert ndcg_at_k(list(reversed(relevance)), k=4) < 1.0
    tasks = make_tasks(limit=2, seed=1)
    report = ranking_report(tasks[0])
    assert report["ndcg_at_k"] == 1.0
    assert max(oracle_relevance(tasks[0])) >= 0.0


def test_seed_aggregate_reports_mean_and_std():
    summary = aggregate_seed_reports(
        [{"metrics": {"macro_spearman": 0.2}}, {"metrics": {"macro_spearman": 0.4}}],
        ["metrics.macro_spearman"],
    )
    assert summary["fields"]["metrics.macro_spearman"]["n"] == 2
    assert abs(summary["fields"]["metrics.macro_spearman"]["mean"] - 0.3) < 1e-8
    assert summary["clinical_use"] is False
    assert json.dumps(summary)
