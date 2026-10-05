import numpy as np

from gcap.analysis.bootstrap import cluster_bootstrap_ci, crossed_bootstrap_ci


def test_cluster_bootstrap_ci_contains_true_mean_on_repeated_sampling():
    rng = np.random.default_rng(0)
    coverage = 0
    trials = 200
    true_mean = 0.1
    for i in range(trials):
        sample = rng.normal(true_mean, 1.0, size=50)
        _point, lower, upper = cluster_bootstrap_ci(sample, np.mean, n_boot=300, confidence=0.95, seed=i)
        if lower <= true_mean <= upper:
            coverage += 1
    # Loose coverage check (not a strict calibration test): should be
    # comfortably above 80% for a nominal 95% percentile-bootstrap interval.
    assert coverage / trials > 0.80


def test_cluster_bootstrap_ci_point_matches_statistic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    point, lower, upper = cluster_bootstrap_ci(values, np.mean, n_boot=500, seed=1)
    assert point == 3.0
    assert lower <= point <= upper


def test_cluster_bootstrap_ci_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        cluster_bootstrap_ci([], np.mean)


def test_crossed_bootstrap_ci_matches_matrix_mean():
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    point, lower, upper = crossed_bootstrap_ci(matrix, np.mean, n_boot=500, seed=2)
    assert point == 5.0
    assert lower <= point <= upper


def test_crossed_bootstrap_ci_narrower_with_more_draws():
    rng = np.random.default_rng(3)
    small_matrix = rng.normal(0.2, 1.0, size=(10, 3))
    large_matrix = rng.normal(0.2, 1.0, size=(10, 30))
    _p1, l1, u1 = crossed_bootstrap_ci(small_matrix, np.mean, n_boot=500, seed=4)
    _p2, l2, u2 = crossed_bootstrap_ci(large_matrix, np.mean, n_boot=500, seed=4)
    assert (u2 - l2) < (u1 - l1)
