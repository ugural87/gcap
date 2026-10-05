"""Agent-visible case view: the vocabulary of "what an agent (and the
deterministic tools called on its behalf) is allowed to see."

These schemas deliberately carry LESS than ``bench.schema.BenchCase`` /
``oracle.schema.OracleClaim``: no ``true_value``, no
``oracle_support_label``, no ``governance_flags`` (the case generator's
ground-truth planting), no ``restructure_proposal`` unless the scenario
actually surfaces one to the desk. A case source builds one of these
*from* richer ground truth (harness/cases.py does this for BenchCase; a
live production integration would build one directly from real evidence,
never having ground truth to strip in the first place) — agents/ itself
never imports anything that could construct one from privileged data,
which is exactly what the import-linter contract ("agents cannot see
oracle... or bench") makes structurally impossible, not just documented.

Kept in agents/ (not core/, not store/) because this is specifically the
agent-facing contract — core/ has no notion of "what an LLM is shown" and
store/ persists what happened, not what was shown.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gcap.core.ledger import LedgerState

ClaimType = Literal["fact", "calculation", "inference", "forecast"]
AssetClass = Literal["equity", "fixed_income"]


class AgentVisibleClaim(BaseModel):
    """Strictly a subset of oracle.schema.OracleClaim's fields — no
    true_value, no oracle_support_label."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_type: ClaimType
    metric: str
    stated_value: Decimal | None = None
    unit: str = ""
    is_material: bool = True


class AgentVisibleInstrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_class: AssetClass
    fields: dict[str, str] = {}  # flattened EquityFundamentals/FixedIncomeAnalytics, display-only


class ResourceFootprint(BaseModel):
    """The trade's own mechanical resource consumption — capital,
    liquidity, funding, collateral, risk-budget usage, and which state
    version the proposal was drafted against. This is NOT ground truth in
    the oracle sense (it is not "the correct answer" to a hidden question);
    it is the proposal's own metadata, exactly as visible in a real trade
    ticket. A deliberately stale ``state_version_seen`` (the case
    generator's family-8 stale-state scenarios) is part of the scenario
    itself, not a leaked answer — the system is meant to be able to detect
    staleness precisely because this field is visible."""

    model_config = ConfigDict(frozen=True)

    state_version_seen: int = 0
    capital_usage: Decimal = Decimal("0")
    liquidity_usage: Decimal = Decimal("0")
    funding_usage: Decimal = Decimal("0")
    collateral_usage: Decimal = Decimal("0")
    risk_budget_usage: Decimal = Decimal("0")
    concentration_key: str | None = None
    concentration_usage: Decimal = Decimal("0")
    notional_delta: Decimal = Decimal("0")


class AgentVisibleProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str
    instrument: AgentVisibleInstrument
    claims: tuple[AgentVisibleClaim, ...] = ()
    # Tenor/rate context an agent legitimately sees to structure a trade;
    # never the case generator's true_value fields.
    tau_days: Decimal
    day_count_basis: Decimal
    base_funding_curve_rate_ann: Decimal
    resource_footprint: ResourceFootprint = ResourceFootprint()


class AgentVisibleCase(BaseModel):
    """What the graph actually operates on. Built by harness/cases.py from
    either a BenchCase (ground truth stripped) or, in a future production
    integration, directly from real evidence with no stripping needed."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    family: str
    materiality: str
    kind: Literal["single", "episode"]
    proposals: tuple[AgentVisibleProposal, ...]
    initial_state: LedgerState
    policy_version: str
