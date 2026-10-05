from gcap.analysis.separation import detect_separation, sign_flip_permutation_test


def test_zero_events_in_one_arm_is_separation():
    assert detect_separation(events_a=0, n_a=20, events_b=5, n_b=20) is True


def test_all_events_in_one_arm_is_separation():
    assert detect_separation(events_a=20, n_a=20, events_b=5, n_b=20) is True


def test_no_zero_cell_and_finite_log_odds_is_not_separation():
    assert detect_separation(events_a=8, n_a=20, events_b=5, n_b=20, fitted_log_odds=0.9) is False


def test_extreme_log_odds_is_separation():
    assert detect_separation(events_a=8, n_a=20, events_b=5, n_b=20, fitted_log_odds=15.0) is True


def test_singular_boundary_flag_is_separation():
    assert detect_separation(events_a=8, n_a=20, events_b=5, n_b=20, singular_boundary=True) is True


def test_sign_flip_exact_all_zero_diffs_gives_pvalue_one():
    p = sign_flip_permutation_test([0.0, 0.0, 0.0, 0.0])
    assert p == 1.0


def test_sign_flip_exact_all_same_sign_gives_small_pvalue():
    # 6 clusters all with diff = +1: only the all-plus sign pattern (and its
    # mirror, all-minus, which gives the same |mean|) reach the observed
    # statistic, out of 2**6=64 patterns.
    diffs = [1.0] * 6
    p = sign_flip_permutation_test(diffs)
    assert p == 2 / 64


def test_sign_flip_matches_exact_for_small_n():
    diffs = [1.0, 1.0, 1.0, -1.0, 1.0]
    p_exact = sign_flip_permutation_test(diffs, n_perm=100000, seed=1)
    # Recompute exact value by hand via brute force for cross-check.
    import itertools

    import numpy as np

    arr = np.array(diffs)
    observed = abs(arr.mean())
    count = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        total += 1
        stat = abs((arr * np.array(signs)).mean())
        if stat >= observed - 1e-15:
            count += 1
    assert p_exact == count / total


def test_sign_flip_monte_carlo_fallback_for_large_n():
    diffs = [0.5] * 25  # exceeds EXACT_ENUMERATION_LIMIT (20)
    p = sign_flip_permutation_test(diffs, n_perm=2000, seed=1)
    assert 0.0 <= p <= 1.0
    assert p < 0.01  # very strong, consistent one-directional signal
