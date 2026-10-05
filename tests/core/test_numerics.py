from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gcap.core.numerics import annual_from_horizon, horizon_from_annual, to_decimal
from gcap.errors import CoreValidationError

rates = st.decimals(min_value="-0.30", max_value="0.50", places=6, allow_nan=False, allow_infinity=False)
taus = st.decimals(min_value="1", max_value="3650", places=0, allow_nan=False, allow_infinity=False)
bases = st.sampled_from([Decimal("360"), Decimal("365")])


def test_to_decimal_rejects_float():
    with pytest.raises(CoreValidationError):
        to_decimal(0.1)  # type: ignore[arg-type]


def test_to_decimal_accepts_str_and_decimal():
    assert to_decimal("1.5") == Decimal("1.5")
    assert to_decimal(Decimal("1.5")) == Decimal("1.5")


@given(rate_ann=rates, tau=taus, basis=bases)
def test_annualisation_round_trip(rate_ann, tau, basis):
    horizon = horizon_from_annual(rate_ann, tau, basis)
    back = annual_from_horizon(horizon, tau, basis)
    assert abs(back - rate_ann) < Decimal("1e-15")


def test_horizon_from_annual_zero_rate_is_zero():
    assert horizon_from_annual(Decimal("0"), Decimal("30"), Decimal("365")) == Decimal("0")


def test_annual_from_horizon_rejects_sub_total_loss():
    with pytest.raises(CoreValidationError):
        annual_from_horizon(Decimal("-1"), Decimal("30"), Decimal("365"))
    with pytest.raises(CoreValidationError):
        annual_from_horizon(Decimal("-1.5"), Decimal("30"), Decimal("365"))
