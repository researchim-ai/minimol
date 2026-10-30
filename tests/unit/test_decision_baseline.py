import numpy as np

from minimol.evaluation.decision_baseline import regression_decisions


def test_perfect_regression_gives_correct_decisions_with_explicit_margin_coverage():
    y = np.array([[2, 0, -1], [-1, 2, 0], [0, -1, 2], [1, 0, -1], [-2, 1, 0], [0, -2, 1]], dtype=float)
    r = regression_decisions(y, y.copy(), np.ones_like(y, dtype=bool), ['a', 'b', 'c'], train_means=[1, 0, 0])
    assert r['highest_activity_assay']['margins']['0.0']['accuracy'] == 1
    assert r['highest_activity_assay']['margins']['0.0']['mean_regret_log2fc'] == 0
    for assay in ['a', 'b', 'c']:
        assert r['threshold_events'][assay]['0.5']['accuracy'] == 1
        assert r['threshold_events'][assay]['0.5']['probability_calibrated'] is False


def test_missing_measurements_and_ties_are_not_counted_as_confident_choices():
    y = np.array([[1, 1, 0], [1.1, 1, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    mask = np.ones_like(y, dtype=bool)
    mask[-1, 1] = False
    r = regression_decisions(y, y.copy(), mask, ['a', 'b', 'c'], train_means=[1, 0, 0])
    choice = r['highest_activity_assay']
    assert choice['n_sequences_all_assays_observed'] == 3
    assert choice['margins']['0.0']['n'] == 2
    assert choice['margins']['0.5']['n'] == 1
    assert np.isclose(choice['margins']['0.5']['fraction_of_available'], 1 / 3)
    assert choice['margins']['0.5']['accuracy'] == 1
