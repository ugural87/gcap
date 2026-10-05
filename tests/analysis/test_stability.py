import pytest

from gcap.analysis.stability import stability_rule


def test_29_of_30_passes():
    passes, lower = stability_rule(29, of=30)
    assert passes is True
    assert lower == pytest.approx(0.828, abs=0.001)


def test_28_of_30_fails():
    passes, lower = stability_rule(28, of=30)
    assert passes is False
    assert lower == pytest.approx(0.780, abs=0.001)


def test_30_of_30_passes():
    passes, _lower = stability_rule(30, of=30)
    assert passes is True


def test_count_below_floor_fails_even_if_ci_would_pass_hypothetically():
    # 29 is the floor; 25 of 30 should fail regardless of its CI value.
    passes, _lower = stability_rule(25, of=30)
    assert passes is False
