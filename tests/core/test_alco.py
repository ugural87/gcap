from decimal import Decimal

from gcap.config.registry import AlcoTransitionThresholds
from gcap.core.alco import AlcoObservations, determine_package

THRESH = AlcoTransitionThresholds()


def test_all_clear_from_normal_stays_normal():
    obs = AlcoObservations(
        stress_liquidity_buffer_pct=Decimal("0.95"),
        funding_spread_percentile=Decimal("0.50"),
        maturity_gap_headroom_pct=Decimal("0.30"),
        collateral_usage_pct=Decimal("0.50"),
    )
    result = determine_package("NORMAL", obs, THRESH)
    assert result.package == "NORMAL"
    assert result.triggering_observations == ()


def test_single_breach_moves_to_caution():
    obs = AlcoObservations(
        stress_liquidity_buffer_pct=Decimal("0.80"),  # below 0.85 floor
        funding_spread_percentile=Decimal("0.50"),
        maturity_gap_headroom_pct=Decimal("0.30"),
        collateral_usage_pct=Decimal("0.50"),
    )
    result = determine_package("NORMAL", obs, THRESH)
    assert result.package == "CAUTION"
    assert "stress_liquidity_buffer_below_floor" in result.triggering_observations


def test_multiple_breaches_move_to_stress():
    obs = AlcoObservations(
        stress_liquidity_buffer_pct=Decimal("0.70"),
        funding_spread_percentile=Decimal("0.95"),
        maturity_gap_headroom_pct=Decimal("0.05"),
        collateral_usage_pct=Decimal("0.95"),
    )
    result = determine_package("NORMAL", obs, THRESH)
    assert result.package == "STRESS"
    assert len(result.triggering_observations) >= 2


def test_deescalation_passes_through_recovery():
    clean = AlcoObservations(
        stress_liquidity_buffer_pct=Decimal("0.95"),
        funding_spread_percentile=Decimal("0.50"),
        maturity_gap_headroom_pct=Decimal("0.30"),
        collateral_usage_pct=Decimal("0.50"),
    )
    step1 = determine_package("STRESS", clean, THRESH)
    assert step1.package == "RECOVERY"
    step2 = determine_package(step1.package, clean, THRESH)
    assert step2.package == "NORMAL"


def test_transition_is_a_pure_function_of_observations_and_previous():
    obs = AlcoObservations(
        stress_liquidity_buffer_pct=Decimal("0.80"),
        funding_spread_percentile=Decimal("0.50"),
        maturity_gap_headroom_pct=Decimal("0.30"),
        collateral_usage_pct=Decimal("0.50"),
    )
    r1 = determine_package("NORMAL", obs, THRESH)
    r2 = determine_package("NORMAL", obs, THRESH)
    assert r1 == r2
