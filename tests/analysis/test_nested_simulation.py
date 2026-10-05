from gcap.analysis.nested_simulation import (
    derive_kmin,
    draw_success_probability,
    stability_power_for_k,
)


def test_draw_success_probability_increases_with_k():
    low_k = draw_success_probability(k=5, n_cases=20, g_lower_bound=0.5, n_mc=20000, seed=1)
    high_k = draw_success_probability(k=18, n_cases=20, g_lower_bound=0.5, n_mc=20000, seed=1)
    assert high_k > low_k


def test_draw_success_probability_increases_with_g():
    low_g = draw_success_probability(k=10, n_cases=20, g_lower_bound=0.2, n_mc=20000, seed=1)
    high_g = draw_success_probability(k=10, n_cases=20, g_lower_bound=0.8, n_mc=20000, seed=1)
    assert high_g > low_g


def test_stability_power_monotonic_in_draw_success_probability():
    _p1, power1 = stability_power_for_k(k=5, n_cases=20, g_lower_bound=0.3, n_mc=20000, seed=1)
    _p2, power2 = stability_power_for_k(k=15, n_cases=20, g_lower_bound=0.3, n_mc=20000, seed=1)
    assert power2 >= power1


def test_derive_kmin_returns_smallest_qualifying_candidate():
    # g_lower_bound=0.4 puts the qualifying k in the interior of the
    # candidate range (verified numerically: k=10 is the first candidate at
    # which both the stability-power and joint-power floors clear 0.80),
    # so this test actually exercises "smallest qualifying candidate",
    # rather than trivially picking the first candidate tried.
    k, results = derive_kmin(
        candidates=range(5, 21),
        n_cases=20,
        g_lower_bound=0.4,
        central_test_power=0.9,
        n_mc=20000,
        seed=1,
    )
    assert k is not None
    assert 5 < k < 20  # strictly interior: neither the first nor last candidate
    # Every candidate below the selected k must have failed at least one
    # of the two floors (otherwise it would have been selected first).
    by_k = {r.k: r for r in results}
    for candidate in range(5, k):
        r = by_k[candidate]
        assert not (r.stability_power >= 0.80 and r.joint_power_product >= 0.80)
    assert by_k[k].stability_power >= 0.80
    assert by_k[k].joint_power_product >= 0.80


def test_derive_kmin_returns_none_when_g_lower_bound_too_weak():
    k, results = derive_kmin(
        candidates=range(5, 21),
        n_cases=20,
        g_lower_bound=0.05,  # far too weak a detection signal to ever qualify
        central_test_power=0.9,
        n_mc=20000,
        seed=1,
    )
    assert k is None
    assert len(results) == 16
