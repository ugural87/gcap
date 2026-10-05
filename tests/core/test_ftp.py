from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gcap.core.ftp import (
    FTPAddonVector,
    FTPInputs,
    FTPOutput,
    assert_no_action_field,
    compute_ftp,
    tenor_stratum,
)
from gcap.core.numerics import horizon_from_annual
from gcap.errors import FTPValidationError

small_rate = st.decimals(min_value="0.00", max_value="0.02", places=6, allow_nan=False, allow_infinity=False)
base_rate = st.decimals(min_value="-0.02", max_value="0.15", places=6, allow_nan=False, allow_infinity=False)
tau = st.decimals(min_value="1", max_value="3650", places=0, allow_nan=False, allow_infinity=False)
trade_alpha = st.decimals(min_value="-0.5", max_value="0.5", places=6, allow_nan=False, allow_infinity=False)


def _inputs(base_ann, lp, tp, cp, rp, cc, tau_days, trade_alpha_h, day_count=Decimal("365")):
    return FTPInputs(
        tau_days=tau_days,
        day_count_basis=day_count,
        base_funding_curve_rate_ann=base_ann,
        addons=FTPAddonVector(lp=lp, tp=tp, cp=cp, rp=rp, cc=cc),
        trade_alpha_horizon=trade_alpha_h,
    )


def test_output_schema_has_no_action_field():
    # Direct schema assertion (invariant 2), independent of the import-time guard.
    for name in FTPOutput.model_fields:
        lowered = name.lower()
        assert not any(t in lowered for t in ("action", "approve", "reject", "defer", "decision", "recommend"))
    # And the guard function itself must not raise.
    assert_no_action_field()


def test_zero_addons_equals_pure_base_curve_conversion():
    inputs = _inputs(Decimal("0.04"), *([Decimal("0")] * 5), Decimal("90"), Decimal("0.02"))
    out = compute_ftp(inputs)
    expected_horizon = horizon_from_annual(Decimal("0.04"), Decimal("90"), Decimal("365"))
    assert out.ftp_horizon == expected_horizon
    assert out.all_in_annualised_rate == Decimal("0.04")


@given(
    base_ann=base_rate,
    lp=small_rate,
    extra_lp=small_rate,
    tau_days=tau,
    trade_alpha_h=trade_alpha,
)
def test_ftp_non_decreasing_in_addon(base_ann, lp, extra_lp, tau_days, trade_alpha_h):
    low = _inputs(base_ann, lp, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), tau_days, trade_alpha_h)
    high = _inputs(
        base_ann, lp + extra_lp, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), tau_days, trade_alpha_h
    )
    out_low = compute_ftp(low)
    out_high = compute_ftp(high)
    assert out_high.ftp_horizon >= out_low.ftp_horizon


def test_double_counted_term_spread_rejected():
    with pytest.raises(FTPValidationError):
        FTPInputs(
            tau_days=Decimal("90"),
            day_count_basis=Decimal("365"),
            base_funding_curve_rate_ann=Decimal("0.04"),
            addons=FTPAddonVector(tp=Decimal("0.001")),
            trade_alpha_horizon=Decimal("0.01"),
            tp_double_counts_term_spread=True,
        )


def test_trade_alpha_le_minus_one_rejected():
    with pytest.raises(FTPValidationError):
        FTPInputs(
            tau_days=Decimal("90"),
            day_count_basis=Decimal("365"),
            base_funding_curve_rate_ann=Decimal("0.04"),
            addons=FTPAddonVector(),
            trade_alpha_horizon=Decimal("-1"),
        )
    with pytest.raises(FTPValidationError):
        FTPInputs(
            tau_days=Decimal("90"),
            day_count_basis=Decimal("365"),
            base_funding_curve_rate_ann=Decimal("0.04"),
            addons=FTPAddonVector(),
            trade_alpha_horizon=Decimal("-1.2"),
        )


def test_band_conversion_matches_closed_form():
    inputs = _inputs(Decimal("0.04"), Decimal("0.001"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("180"), Decimal("0.03"))
    out = compute_ftp(inputs)
    beta_ann = Decimal("0.0050")
    beta_horizon = horizon_from_annual(beta_ann, Decimal("180"), Decimal("365"))
    assert out.in_band(beta_ann) == (abs(out.economic_alpha_horizon) <= beta_horizon)
    assert out.distance_from_band_edge(beta_ann) == beta_horizon - abs(out.economic_alpha_horizon)


def test_economic_alpha_is_trade_alpha_minus_ftp():
    inputs = _inputs(Decimal("0.03"), Decimal("0.002"), Decimal("0.001"), Decimal("0.0015"), Decimal("0.0005"), Decimal("0.0005"), Decimal("270"), Decimal("0.025"))
    out = compute_ftp(inputs)
    assert out.economic_alpha_horizon == out.trade_alpha_horizon - out.ftp_horizon


@pytest.mark.parametrize(
    "tau_days,day_count,expected",
    [
        (Decimal("20"), Decimal("365"), "<1m"),
        (Decimal("180"), Decimal("365"), "1-12m"),
        (Decimal("400"), Decimal("365"), ">12m"),
        (Decimal("365"), Decimal("365"), "1-12m"),
    ],
)
def test_tenor_stratum(tau_days, day_count, expected):
    assert tenor_stratum(tau_days, day_count) == expected
