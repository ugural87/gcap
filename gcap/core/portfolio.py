"""Portfolio contribution (Section 4.5 / proposal Section 5.3).

Delta U_i = E[Delta R_p,i] - lambda*Delta Risk_p,i - gamma*Delta Concentration_p,i

lambda and gamma come from the parameter registry; they are never set by an
agent.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

ENGINE_VERSION = "portfolio-1.0.0"


class MarginalContributionInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta_expected_return: Decimal
    delta_risk: Decimal
    delta_concentration: Decimal
    lam: Decimal
    gamma: Decimal


class MarginalContributionOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta_u: Decimal
    delta_expected_return: Decimal
    risk_penalty: Decimal
    concentration_penalty: Decimal
    engine_version: str = ENGINE_VERSION


def marginal_contribution(inputs: MarginalContributionInputs) -> MarginalContributionOutput:
    risk_penalty = inputs.lam * inputs.delta_risk
    concentration_penalty = inputs.gamma * inputs.delta_concentration
    delta_u = inputs.delta_expected_return - risk_penalty - concentration_penalty
    return MarginalContributionOutput(
        delta_u=delta_u,
        delta_expected_return=inputs.delta_expected_return,
        risk_penalty=risk_penalty,
        concentration_penalty=concentration_penalty,
    )
