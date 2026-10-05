import pytest

from gcap.analysis.conformance import conformance_gap_one_sided_lower_bound, conformance_gap_point


def test_conformance_gap_point_matches_hand_computation():
    d1 = [0.1, 0.2, 0.3]
    d2 = [0.4, 0.5, 0.6]
    g = conformance_gap_point(d1, d2)
    assert g == pytest.approx(0.3)


def test_conformance_gap_zero_when_identical():
    d1 = [0.2, 0.3, 0.4, 0.5]
    g = conformance_gap_point(d1, d1)
    assert g == 0.0


def test_lower_bound_below_point_estimate_when_positive_gap():
    d1 = [0.1] * 40
    d2 = [0.4] * 40
    # add small per-cluster jitter so the bootstrap distribution is non-degenerate
    import random

    rng = random.Random(0)
    d1 = [x + rng.uniform(-0.02, 0.02) for x in d1]
    d2 = [x + rng.uniform(-0.02, 0.02) for x in d2]
    point = conformance_gap_point(d1, d2)
    lower = conformance_gap_one_sided_lower_bound(d1, d2, confidence=0.95, n_boot=2000, seed=1)
    assert lower < point
    assert lower > 0  # strong, consistent gap should still show a positive LCB


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        conformance_gap_point([0.1, 0.2], [0.1])
