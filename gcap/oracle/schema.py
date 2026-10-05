"""Oracle input/output schema. Imports core only (invariant 6): the oracle
package must not import agents/graph/harness, and — to keep the "computed
from the case definition alone" guarantee airtight — not bench/config/store
either. Numeric thresholds the oracle needs (tolerances, marginal band) are
passed in explicitly by the caller (bench/ at case-authoring time, or
harness/ later) rather than imported from the parameter registry, so this
module's only dependency is gcap.core and gcap.errors.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gcap.core.ftp import FTPAddonVector
from gcap.core.ledger import LedgerAction, LedgerState
from gcap.core.policy import PolicyRule

ClaimType = Literal["fact", "calculation", "inference", "forecast"]
ClaimSupport = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class OracleClaim(BaseModel):
    """A structured claim as planted by the case generator. For a
    "calculation" claim, ``true_value`` is the value the generator computed
    from its own source figures and ``stated_value`` is what the case
    presents as the agent-visible evidence — these differ exactly when the
    case plants a numerical error. ``oracle_support_label`` is the
    generator's ground-truth evidence-support tag (Section 7: "Constructed
    ground truth... known answers by construction").
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_type: ClaimType
    metric: str
    stated_value: Decimal | None = None
    true_value: Decimal | None = None
    unit: str = ""
    is_material: bool = True
    oracle_support_label: ClaimSupport = "supported"


class OracleProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str
    tau_days: Decimal
    day_count_basis: Decimal
    base_funding_curve_rate_ann: Decimal
    addons: FTPAddonVector = FTPAddonVector()
    trade_alpha_horizon: Decimal
    ledger_action: LedgerAction


class OracleGovernanceFlags(BaseModel):
    """Qualitative ground-truth flags planted directly by the case
    generator — properties true "by construction" that are not themselves
    derivable by re-running the deterministic engines (e.g. whether a tool
    call in the scenario is one the case defines as prohibited)."""

    model_config = ConfigDict(frozen=True)

    unauthorised_exception: bool = False
    invalid_approval_chain: bool = False
    prohibited_tool_action: bool = False


class OracleCaseInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    initial_state: LedgerState
    proposals: tuple[OracleProposal, ...]
    restructure_proposal: OracleProposal | None = None
    claims: tuple[OracleClaim, ...] = ()
    policy_rules: tuple[PolicyRule, ...] = ()
    policy_context_values: dict[str, object] = Field(default_factory=dict)
    policy_evaluated_at: datetime
    policy_version: str = "v1"
    governance_flags: OracleGovernanceFlags = OracleGovernanceFlags()


class OracleStateComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    funding: bool = False
    liquidity: bool = False
    capital: bool = False
    collateral: bool = False
    concentration: bool = False
    risk_budget: bool = False
    stale: bool = False


class OracleGovernanceComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    hard_rule: bool = False
    unsupported_claim: bool = False
    unauthorised_exception: bool = False
    approval_chain: bool = False
    tool_action: bool = False


class OracleProposalMargin(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str
    economic_alpha_horizon: Decimal
    in_band: bool
    pricing_inconsistent: bool


class OracleLabels(BaseModel):
    """One function per co-primary endpoint, plus component labels
    (Section 4.6). Field names mirror RunRecord's ep_* fields one-to-one so
    the harness can compare run output against these without translation.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str

    ep_numerical_error: bool
    numerical_error_claim_ids: tuple[str, ...]

    ep_state_infeasible: bool
    ep_state_components: OracleStateComponents

    ep_pricing_inconsistent: bool
    proposal_margins: tuple[OracleProposalMargin, ...]

    ep_governance_failure: bool
    ep_governance_components: OracleGovernanceComponents

    expected_escalation_required: bool
    restructure_feasible: bool | None

    generator_version: str = ""
    engine_version: str = "oracle-1.0.0"
