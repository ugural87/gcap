"""Shared graph-test fixtures, deliberately not in a test_*.py file (see
tests/store/factories.py for the precedent: importing fixtures across
test_*.py modules is fragile, so anything shared lives here instead).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from gcap.agents.view import (
    AgentVisibleCase,
    AgentVisibleClaim,
    AgentVisibleInstrument,
    AgentVisibleProposal,
    ResourceFootprint,
)
from gcap.core.ledger import LedgerState
from gcap.core.policy import PolicyRule


def make_proposal(
    *,
    proposal_id: str = "p1",
    claim_id: str = "c1",
    is_material: bool = True,
    stated_value: Decimal = Decimal("0.15"),
    tau_days: Decimal = Decimal("180"),
    base_funding_curve_rate_ann: Decimal = Decimal("0.01"),
    resource_footprint: ResourceFootprint | None = None,
    asset_class: str = "equity",
) -> AgentVisibleProposal:
    claim = AgentVisibleClaim(
        claim_id=claim_id,
        claim_type="calculation",
        metric="portfolio_return_pct",
        stated_value=stated_value,
        is_material=is_material,
    )
    # Display-only "fields" (Section 4's equity-is-primary-input, fixed-
    # income-pluggable-via-the-same-architecture requirement): the graph
    # itself never branches on asset_class, only the flattened label shown
    # to the agents differs.
    fields = {"ticker": "SYN0001"} if asset_class == "equity" else {"isin": "SYNFI000001"}
    return AgentVisibleProposal(
        proposal_id=proposal_id,
        instrument=AgentVisibleInstrument(asset_class=asset_class, fields=fields),  # type: ignore[arg-type]
        claims=(claim,),
        tau_days=tau_days,
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=base_funding_curve_rate_ann,
        resource_footprint=resource_footprint or ResourceFootprint(),
    )


def make_case(
    *, proposals: tuple[AgentVisibleProposal, ...], case_id: str = "case-1"
) -> AgentVisibleCase:
    return AgentVisibleCase(
        case_id=case_id,
        family="fam-1",
        materiality="material",
        kind="single" if len(proposals) == 1 else "episode",
        proposals=proposals,
        initial_state=make_ledger_state(),
        policy_version="policy-v1",
    )


def make_ledger_state(
    *,
    version: int = 0,
    capital_available: Decimal = Decimal("1000"),
    liquidity_available: Decimal = Decimal("1000"),
    funding_capacity: Decimal = Decimal("1000"),
    collateral_available: Decimal = Decimal("1000"),
    risk_budget_available: Decimal = Decimal("1000"),
) -> LedgerState:
    return LedgerState(
        version=version,
        capital_available=capital_available,
        liquidity_available=liquidity_available,
        funding_capacity=funding_capacity,
        collateral_available=collateral_available,
        risk_budget_available=risk_budget_available,
    )


def make_footprint(**overrides: object) -> ResourceFootprint:
    return ResourceFootprint(**overrides)  # type: ignore[arg-type]


def make_policy_rule(
    *,
    rule_id: str = "R-unsupported-claim",
    severity: str = "hard",
    predicate_id: str = "hard.unsupported_material_claim",
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        owner="test-owner",
        scope="test-scope",
        severity=severity,  # type: ignore[arg-type]
        effective_from=datetime(2020, 1, 1, tzinfo=UTC),
        effective_to=None,
        version="policy-v1",
        required_evidence=(),
        predicate_id=predicate_id,
    )
