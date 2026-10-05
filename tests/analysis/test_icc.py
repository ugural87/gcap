from __future__ import annotations

import pytest

from gcap.analysis.icc import one_way_icc


def test_perfect_within_group_agreement_gives_icc_one():
    groups = {"a": [1, 1, 1], "b": [5, 5, 5], "c": [9, 9, 9]}
    assert one_way_icc(groups) == pytest.approx(1.0)


def test_hand_computed_unbalanced_two_group_example():
    # group A: [1, 3] (mean 2), group B: [5, 7] (mean 6); grand mean 4.
    # SSB=16 (df=1, MSB=16); SSW=4 (df=2, MSW=2); n0=2 (balanced n_i=[2,2]).
    # ICC = (16-2) / (16 + (2-1)*2) = 14/18.
    groups = {"a": [1, 3], "b": [5, 7]}
    assert one_way_icc(groups) == pytest.approx(14 / 18)


def test_single_group_returns_none():
    assert one_way_icc({"a": [1, 2, 3]}) is None


def test_no_group_with_more_than_one_observation_returns_none():
    assert one_way_icc({"a": [1], "b": [2], "c": [3]}) is None


def test_empty_input_returns_none():
    assert one_way_icc({}) is None


def test_can_return_negative_icc_without_clipping():
    # Both groups share the same mean (no between-group variance) but have
    # nonzero within-group spread -> the ANOVA estimator goes negative.
    groups = {"a": [0.0, 4.0], "b": [0.0, 4.0]}
    icc = one_way_icc(groups)
    assert icc is not None
    assert icc < 0
