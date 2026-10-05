from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st

from gcap.core.portfolio import MarginalContributionInputs, marginal_contribution

decimals = st.decimals(min_value="-100", max_value="100", places=4, allow_nan=False, allow_infinity=False)


def test_zero_risk_and_concentration_delta_equals_expected_return():
    inputs = MarginalContributionInputs(
        delta_expected_return=Decimal("0.05"),
        delta_risk=Decimal("0"),
        delta_concentration=Decimal("0"),
        lam=Decimal("0.5"),
        gamma=Decimal("0.25"),
    )
    out = marginal_contribution(inputs)
    assert out.delta_u == Decimal("0.05")


@given(dr=decimals, drisk=decimals, dconc=decimals, lam=st.decimals(min_value="0", max_value="5", places=2), gamma=st.decimals(min_value="0", max_value="5", places=2))
def test_delta_u_matches_closed_form(dr, drisk, dconc, lam, gamma):
    inputs = MarginalContributionInputs(delta_expected_return=dr, delta_risk=drisk, delta_concentration=dconc, lam=lam, gamma=gamma)
    out = marginal_contribution(inputs)
    assert out.delta_u == dr - lam * drisk - gamma * dconc
