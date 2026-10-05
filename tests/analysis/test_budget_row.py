from __future__ import annotations

import pytest

from gcap.analysis.budget_row import (
    design_effect,
    effective_n_clusters,
    select_c_d1_budget_row,
)
from gcap.config.registry import ClusterFloors, ReplicateCounts

_FAST = {"n_sims": 60, "n_perm": 100}


def test_design_effect_is_one_at_a_single_replicate_regardless_of_icc():
    assert design_effect(1, 0.0) == 1.0
    assert design_effect(1, 0.9) == 1.0


def test_design_effect_matches_donner_klar_formula():
    assert design_effect(5, 0.5) == pytest.approx(1 + 4 * 0.5)
    assert design_effect(3, 0.2) == pytest.approx(1 + 2 * 0.2)


def test_design_effect_rejects_non_positive_replicate_count():
    with pytest.raises(ValueError):
        design_effect(0, 0.1)


def test_effective_n_clusters_scales_linearly_with_m_at_zero_icc():
    # DEFF(m, 0) == 1 always -> effective_n_clusters == n_clusters * m exactly.
    assert effective_n_clusters(36, 3, 0.0) == pytest.approx(108.0)
    assert effective_n_clusters(72, 5, 0.0) == pytest.approx(360.0)


def test_effective_n_clusters_shows_diminishing_returns_with_positive_icc():
    eff_3 = effective_n_clusters(72, 3, 0.3)
    eff_5 = effective_n_clusters(72, 5, 0.3)
    # Both m=3 and m=5 values sit strictly below their icc=0 (proportional)
    # counterparts, and the ratio eff_5/eff_3 is less than 5/3 -- the
    # defining diminishing-returns property of a positive design effect.
    assert eff_5 > eff_3
    assert eff_5 / eff_3 < 5 / 3


def test_select_prefers_the_fewest_replicates_that_clear_a_lenient_target():
    selection = select_c_d1_budget_row(icc=0.0, p10=0.35, p01=0.0, target_power=0.5, **_FAST)
    assert selection.selected_row == "3"
    assert selection.meets_target is True
    assert len(selection.candidates) == 3
    assert [c.replicates for c in selection.candidates] == [3, 4, 5]


def test_select_raises_the_row_when_more_replicates_are_needed():
    # A near-zero effect (psi tiny) at the default n_clusters is very hard
    # to power even at m=5 -> falls back to the ceiling row rather than
    # silently understating what's achievable.
    selection = select_c_d1_budget_row(
        icc=0.9, n_clusters=4, p10=0.02, p01=0.0, target_power=0.999, **_FAST
    )
    assert selection.selected_row == "5"
    assert selection.meets_target is False


def test_select_with_icc_none_falls_back_to_zero_and_documents_the_source():
    selection = select_c_d1_budget_row(icc=None, **_FAST)
    assert selection.icc_source == "fallback_zero"
    assert selection.icc_used == 0.0


def test_select_clamps_a_negative_pilot_icc_to_zero_but_keeps_pilot_source():
    selection = select_c_d1_budget_row(icc=-0.15, **_FAST)
    assert selection.icc_source == "pilot"
    assert selection.icc_used == 0.0


def test_select_defaults_n_clusters_to_the_c_d1_cluster_floor_sum():
    floors = ClusterFloors()
    selection = select_c_d1_budget_row(icc=0.0, target_power=0.5, p10=0.35, p01=0.0, **_FAST)
    assert selection.n_clusters == floors.c_d1_single + floors.c_d1_episodes


def test_select_rejects_a_replicate_table_outside_3_4_5():
    with pytest.raises(ValueError):
        select_c_d1_budget_row(icc=0.0, replicate_counts=ReplicateCounts(c_d1_floor=2), **_FAST)
    with pytest.raises(ValueError):
        select_c_d1_budget_row(icc=0.0, replicate_counts=ReplicateCounts(c_d1_ceiling=6), **_FAST)
