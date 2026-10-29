import pandas as pd

from minimol.data.synthetic import write_admet_sample, write_reg_sample
from minimol.evaluation.motif_baseline import motif_baseline_report
from minimol.evaluation.ood_report import applicability_report
from minimol.evaluation.rank_benchmark import fixed_budget_rank_report
from minimol.evaluation.rc_stress import rc_stress, strand_disagreement
from minimol.models.cds_cnn import MinimolCDS


def _toy_task():
    return {
        "objective": {"property": "qed", "direction": "maximize"},
        "constraints": [{"property": "mw", "max": 500.0}],
        "candidates": [
            {"valid": True, "oracle": {"qed": 0.1, "mw": 100.0}},
            {"valid": True, "oracle": {"qed": 0.5, "mw": 120.0}},
            {"valid": True, "oracle": {"qed": 0.9, "mw": 140.0}},
            {"valid": False, "oracle": None},
        ],
    }


def test_fixed_budget_oracle_beats_reverse_order():
    report = fixed_budget_rank_report([_toy_task()], k=3, seed=0)
    assert report["oracle_mean_ndcg"] == 1.0
    assert report["reverse_mean_ndcg"] < report["oracle_mean_ndcg"]
    assert report["mean_invalid_rate"] == 0.25
    assert report["candidate_budget"] == "fixed"


def test_strand_swap_of_an_equivariant_map_is_zero():
    plus = [0.1, 0.4, 0.2, 0.8]
    minus = [0.3, 0.0, 0.5, 0.6]
    plus_rc = list(reversed(minus))
    minus_rc = list(reversed(plus))
    assert strand_disagreement(plus, minus, plus_rc, minus_rc) == 0.0
    broken = strand_disagreement(plus, minus, plus, minus)
    assert broken > 0.1


def test_rc_stress_on_a_tiny_model_is_finite():
    model = MinimolCDS("5m", channels=16, n_blocks=2, dropout=0.0).eval()
    report = rc_stress(model, ["ACGTACGTACGT", "ATGCCCAAATTT"])
    assert report["n"] == 2
    assert report["mean_abs_disagreement"] >= 0.0


def test_planted_motifs_beat_the_mean(tmp_path):
    write_reg_sample(tmp_path / "reg", n_groups=40, seed=4)
    frame = pd.read_parquet(tmp_path / "reg" / "data.parquet")
    report = motif_baseline_report(frame, ["TATAAA", "CCAAT", "GGGCGG"])
    assert report["jaspar_download"] is False
    motif_mae = report["motif"]["per_target"]["K562"]["mae"]
    mean_mae = report["mean"]["per_target"]["K562"]["mae"]
    assert motif_mae < mean_mae


def test_applicability_flags_a_distant_fingerprint(tmp_path):
    write_admet_sample(tmp_path / "admet", seed=2)
    frame = pd.read_parquet(tmp_path / "admet" / "data.parquet")
    report = applicability_report(frame, threshold=0.4)
    assert report["n_train"] > 0
    assert report["n_validation"] > 0
    assert 0.0 <= report["fraction_ood"] <= 1.0
    assert report["clinical_use"] is False
