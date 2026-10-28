import numpy as np

from minimol.evaluation.calibration import (
    conformal_quantile,
    fit_binary_temperature,
    split_conformal_intervals,
)
from minimol.evaluation.cds import nucleotide_report, stitch_probabilities
from minimol.evaluation.metrics import boundary_scores, shuffled_correlation, spearman


def test_boundary_f1_matches_a_known_pair():
    truth = np.zeros(60)
    truth[10:] = 1
    pred = np.zeros(60)
    pred[11:] = 1
    pred[50:] = 0
    pred[52:] = 1
    # change points: truth at 10; pred at 11 and 50 and 52
    scores = boundary_scores(truth, pred, tolerance=3)
    assert abs(scores["recall"] - 1.0) < 1e-8
    assert scores["precision"] < 1.0
    assert scores["f1"] > 0


def test_stitch_averages_overlap():
    windows = [(0, np.array([[1.0, 1.0, 0.0]])), (2, np.array([[0.0, 1.0]]))]
    stitched = stitch_probabilities(windows, length=4)
    assert stitched.shape == (1, 4)
    assert abs(stitched[0, 2] - 0.0) < 1e-8


def test_conformal_quantile_rank():
    residuals = np.arange(10, dtype=float)
    assert conformal_quantile(residuals, alpha=0.1) == 9.0
    intervals = split_conformal_intervals([0, 0], [1, -1], [5], alpha=0.5)
    assert intervals["lower"][0] <= 5 <= intervals["upper"][0]


def test_temperature_is_positive():
    temperature = fit_binary_temperature([5, -5, 5, -5], [1, 0, 1, 0], steps=20)
    assert temperature > 0


def test_shuffled_target_is_a_weaker_control():
    rng = np.random.default_rng(0)
    target = rng.normal(size=200)
    pred = target + rng.normal(scale=0.05, size=200)
    report = shuffled_correlation(target, pred, seed=1)
    assert report["real"] > 0.9
    assert report["shuffled"] < report["real"] - 0.5


def test_nucleotide_report_perfect_prediction():
    truth = np.array([[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=float)
    report = nucleotide_report(truth, truth)
    assert report["macro_f1"] == 1.0
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
