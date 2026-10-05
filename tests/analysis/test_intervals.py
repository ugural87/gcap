import pytest

from gcap.analysis.intervals import clopper_pearson, clopper_pearson_lower


def test_29_of_30_matches_spec_reference_value():
    # Section 9.5: "the lower endpoint of the two-sided 95% Clopper-Pearson
    # interval is approximately 0.828 for 29/30 and 0.780 for 28/30."
    lower, upper = clopper_pearson(29, 30, confidence=0.95)
    assert lower == pytest.approx(0.828, abs=0.001)
    assert upper <= 1.0


def test_28_of_30_matches_spec_reference_value():
    lower, _upper = clopper_pearson(28, 30, confidence=0.95)
    assert lower == pytest.approx(0.780, abs=0.001)


def test_zero_successes_lower_bound_is_zero():
    lower, upper = clopper_pearson(0, 10, confidence=0.95)
    assert lower == 0.0
    assert upper > 0.0


def test_all_successes_upper_bound_is_one():
    lower, upper = clopper_pearson(10, 10, confidence=0.95)
    assert upper == 1.0
    assert lower > 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        clopper_pearson(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 10)


def test_one_sided_lower_bound_leq_two_sided_lower_bound():
    # A one-sided 95% bound is less conservative than a two-sided 95% bound's
    # lower endpoint (which is really a one-sided 97.5% bound).
    two_sided_lower, _ = clopper_pearson(29, 30, confidence=0.95)
    one_sided = clopper_pearson_lower(29, 30, confidence=0.95)
    assert one_sided >= two_sided_lower
