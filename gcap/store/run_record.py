"""RunRecord: the sole contract between execution and analysis (Section 6.2).

Defined here in store/ (not agents/ or harness/, neither of which exists
yet) so that analysis/ can import it without ever importing agents/graph/
harness (the forbidden-import contract enforced by import-linter). Once
agents/ is built (M2), gcap.agents.treasury re-exports TreasuryAction from
this module rather than duplicating it, since store sits above agents in
the dependency order (Section 2 diagram) and RunRecord is the natural home
for a field this central to it.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ConfigName = Literal["A", "B", "C", "D1", "D2", "E"]
Split = Literal["dev", "pilot", "confirm", "audit", "roadmap"]
TenorStratum = Literal["<1m", "1-12m", ">12m"]
FailureKind = Literal["none", "runtime", "schema", "missing_evidence"]


class TreasuryAction(StrEnum):
    """Section 5.3."""

    APPROVE = "APPROVE"
    APPROVE_WITH_LIMIT = "APPROVE_WITH_LIMIT"
    APPROVE_WITH_HEDGE = "APPROVE_WITH_HEDGE"
    RESTRUCTURE = "RESTRUCTURE"
    DEFER = "DEFER"
    REJECT = "REJECT"

    @property
    def is_approval_family(self) -> bool:
        """APPROVE, APPROVE_WITH_LIMIT and APPROVE_WITH_HEDGE are the
        approval family, full stop — an approval-with-a-condition is still
        a decision to approve. Section 9.6's deterministic recomputation
        for the latter two (graph.pipeline._revised_addons) does NOT change
        membership here; it changes whether that approval's PRICING is
        judged consistent (RunRecord.ep_pricing_inconsistent) and, in
        config E, whether CONTROL_CLEARANCE can pass — both computed by the
        caller from the revised margin when one exists, never by mutating
        this property. See config.registry.RevisedMarginRecomputation for
        the modelling choice and its to-decide reduction fractions."""
        return self in (
            TreasuryAction.APPROVE,
            TreasuryAction.APPROVE_WITH_LIMIT,
            TreasuryAction.APPROVE_WITH_HEDGE,
        )


class StateComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    funding: bool = False
    liquidity: bool = False
    capital: bool = False
    collateral: bool = False
    concentration: bool = False
    risk_budget: bool = False
    stale: bool = False


class GovernanceComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    hard_rule: bool = False
    unsupported_claim: bool = False
    unauthorised_exception: bool = False
    approval_chain: bool = False
    tool_action: bool = False


class RunRecord(BaseModel):
    """One row per stochastic inference run. Field groups mirror Section
    6.2 exactly: identity/pairing, versions, pricing/decision, endpoints
    (with components), operations."""

    model_config = ConfigDict(frozen=True)

    # ---- identity and pairing ----
    run_id: UUID
    config: ConfigName
    split: Split
    block: str
    case_id: str
    episode_id: str | None = None
    proposal_position: int | None = None
    ordering_id: str | None = None
    replicate: int
    draw_id: str | None = None
    family: str
    materiality: str
    tenor_stratum: TenorStratum | None = None

    # ---- versions ----
    model_snapshot: str
    # P2-10: one node can make several physical calls (research_node makes
    # one per claim), so every call's own prompt hash is kept — not just
    # whichever call happened to run last.
    prompt_hashes: dict[str, list[str]] = Field(default_factory=dict)
    # P2-10: per-node count of calls that were answered from the shared
    # cache rather than a fresh physical model call. Kept separate from
    # tokens_in/tokens_out/latency_ms (which already exclude cache hits'
    # cost) so a cache hit is visible in the usage record without being
    # double-counted as spend.
    cache_hits: dict[str, int] = Field(default_factory=dict)
    policy_version: str
    ftp_params: dict[str, Decimal] = Field(default_factory=dict)
    state_version_in: int
    provider_seed: int | None = None
    manifest_hash: str

    # ---- pricing and decision ----
    shown_margin: Decimal | None = None
    oracle_margin: Decimal | None = None
    aggregate_addon: Decimal | None = None
    in_band: dict[str, bool] = Field(default_factory=dict)
    treasury_action: TreasuryAction | None = None
    revised_margin: Decimal | None = None
    approval_family: bool | None = None

    # ---- endpoints ----
    ep_numerical_error: bool
    ep_state_infeasible: bool
    ep_state_components: StateComponents
    ep_pricing_inconsistent: bool
    ep_governance_failure: bool
    ep_governance_components: GovernanceComponents
    false_block: bool | None = None
    escalation_correct: bool | None = None
    # P2-03 (v2.0.0 code review): the raw, ungated diagnostic counterpart to
    # each ep_* field above -- see harness/grading.py's module docstring
    # for the full rationale. Defaulted (not required) so every existing
    # RunRecord construction site (including in already-written tests) that
    # predates this field keeps working unchanged; harness/record.py is the
    # one production writer and always supplies a real value.
    detected_numerical_error: bool = False
    detected_state_infeasible: bool = False
    detected_pricing_inconsistent: bool = False
    # Raw hard-rule governance detection, consistently across every arm.
    # The full five-component oracle composite remains available in
    # ``ep_governance_components``; it is deliberately not overloaded into
    # this scalar because E's policy engine can observe hard rules only.
    detected_governance_failure: bool = False

    # ---- operations ----
    tokens_in: dict[str, int] = Field(default_factory=dict)
    tokens_out: dict[str, int] = Field(default_factory=dict)
    latency_ms: dict[str, int] = Field(default_factory=dict)
    retries: int = 0
    rework: bool = False
    failure: FailureKind = "none"
    # P1-8 (code review): the exception's own str(), so a failed run is
    # diagnosable from the store alone instead of only "failure=runtime"
    # with no further detail. None whenever failure == "none"; harness.
    # runner._failure_run_write is the only writer of a non-None value.
    failure_reason: str | None = None
