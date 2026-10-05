"""Event-sourced state ledger (Section 4.2 / proposal Section 5.3).

State S = (C, L, F, P, X, B, K, V) is a frozen Pydantic model with an
integer version. The current state is the fold of committed events over the
initial state (invariant 3): St+1 = T(St, at), and apply() is called only
for executed or committed actions. The caller (graph/harness, built in M2)
is responsible for never calling ``apply`` on a REJECT/DEFER/blocked
decision; this module makes that easy to honour by keeping ``apply`` a pure
function of (state, action) with no notion of "which action kinds commit".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gcap.errors import LedgerInfeasibleError, StaleStateError

FeasibilityComponent = Literal[
    "funding", "liquidity", "capital", "collateral", "concentration", "risk_budget"
]


class LedgerState(BaseModel):
    """S = (C, L, F, P, X, B, K, V).

    C -> capital_available, L -> liquidity_available, F -> funding_capacity
    (+ maturity profile approximated here by funding_capacity alone; a
    richer maturity ladder is a store/ concern), P -> positions, X ->
    concentration_used (exposures by concentration key), B ->
    risk_budget_available, K -> collateral_available, V -> policy_version.
    """

    model_config = ConfigDict(frozen=True)

    version: int = 0
    capital_available: Decimal
    liquidity_available: Decimal
    funding_capacity: Decimal
    collateral_available: Decimal
    risk_budget_available: Decimal
    concentration_used: dict[str, Decimal] = Field(default_factory=dict)
    concentration_limits: dict[str, Decimal] = Field(default_factory=dict)
    positions: dict[str, Decimal] = Field(default_factory=dict)
    policy_version: str = "v1"
    alco_package: Literal["NORMAL", "CAUTION", "STRESS", "RECOVERY"] = "NORMAL"


class LedgerAction(BaseModel):
    """A proposed resource consumption. ``state_version_seen`` is the
    version the proposer believed current at package-build time; a mismatch
    at apply time is staleness (Section 4.2: "the package's recorded state
    version differs from the ledger's current version at gate time")."""

    model_config = ConfigDict(frozen=True)

    action_id: str
    instrument_id: str
    state_version_seen: int
    capital_usage: Decimal = Decimal("0")
    liquidity_usage: Decimal = Decimal("0")
    funding_usage: Decimal = Decimal("0")
    collateral_usage: Decimal = Decimal("0")
    risk_budget_usage: Decimal = Decimal("0")
    concentration_key: str | None = None
    concentration_usage: Decimal = Decimal("0")
    notional_delta: Decimal = Decimal("0")


def is_stale(state: LedgerState, action: LedgerAction) -> bool:
    return action.state_version_seen != state.version


def check_feasibility(state: LedgerState, action: LedgerAction) -> list[FeasibilityComponent]:
    """Returns the list of violated components; an empty list means
    feasible. Does NOT check staleness — that is a separate, orthogonal gate
    (Section 4.2: "No separate freshness rule is needed" refers to staleness
    being just a version-mismatch check, not folded into feasibility)."""
    violations: list[FeasibilityComponent] = []
    if action.funding_usage > state.funding_capacity:
        violations.append("funding")
    if action.liquidity_usage > state.liquidity_available:
        violations.append("liquidity")
    if action.capital_usage > state.capital_available:
        violations.append("capital")
    if action.collateral_usage > state.collateral_available:
        violations.append("collateral")
    if action.risk_budget_usage > state.risk_budget_available:
        violations.append("risk_budget")
    if action.concentration_key is not None:
        used = state.concentration_used.get(action.concentration_key, Decimal(0))
        limit = state.concentration_limits.get(action.concentration_key)
        if limit is not None and (used + action.concentration_usage) > limit:
            violations.append("concentration")
    return violations


def apply(state: LedgerState, action: LedgerAction) -> LedgerState:
    """Raises on an infeasible or stale action; otherwise returns a new
    state with version incremented by exactly 1. Called only for executed
    or committed actions (invariant 3) — REJECT, DEFER and a blocked
    execution must never reach this function."""
    if is_stale(state, action):
        raise StaleStateError(
            f"Action {action.action_id!r} was built against state version "
            f"{action.state_version_seen}, but the ledger is at version {state.version}."
        )
    violations = check_feasibility(state, action)
    if violations:
        raise LedgerInfeasibleError(
            f"Action {action.action_id!r} is infeasible: violated components {violations}."
        )

    new_concentration = dict(state.concentration_used)
    if action.concentration_key is not None:
        new_concentration[action.concentration_key] = (
            new_concentration.get(action.concentration_key, Decimal(0)) + action.concentration_usage
        )

    new_positions = dict(state.positions)
    if action.notional_delta != 0:
        new_positions[action.instrument_id] = (
            new_positions.get(action.instrument_id, Decimal(0)) + action.notional_delta
        )

    return state.model_copy(
        update={
            "version": state.version + 1,
            "capital_available": state.capital_available - action.capital_usage,
            "liquidity_available": state.liquidity_available - action.liquidity_usage,
            "funding_capacity": state.funding_capacity - action.funding_usage,
            "collateral_available": state.collateral_available - action.collateral_usage,
            "risk_budget_available": state.risk_budget_available - action.risk_budget_usage,
            "concentration_used": new_concentration,
            "positions": new_positions,
        }
    )


class LedgerEvent(BaseModel):
    """One committed event. The event log is append-only (Section 9.2 rule
    5); state at any decision point is reconstructable by folding events up
    to that point."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    action: LedgerAction
    resulting_version: int
    committed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def fold(initial: LedgerState, events: Sequence[LedgerEvent]) -> LedgerState:
    """The current state is the fold of committed events over the initial
    state. Folding must equal sequential incremental application."""
    state = initial
    for event in sorted(events, key=lambda e: e.resulting_version):
        state = apply(state, event.action)
        if state.version != event.resulting_version:
            raise LedgerInfeasibleError(
                f"Event {event.event_id!r} recorded resulting_version="
                f"{event.resulting_version} but folding produced version={state.version}; "
                "the event log is inconsistent with the action sequence."
            )
    return state


def fold_up_to(initial: LedgerState, events: Sequence[LedgerEvent], version: int) -> LedgerState:
    """Reconstructs state at a specific historical decision point."""
    subset = [e for e in events if e.resulting_version <= version]
    return fold(initial, subset)
