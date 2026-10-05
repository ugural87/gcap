"""SQLAlchemy 2.0 ORM models for the Section 6.2 seven-entity data model:
runs, decision_packages, evidence_claims, tool_results, policy_state,
approvals, execution_events.

Relationship to ``store.run_record.RunRecord``: RunRecord is "the sole
contract between execution and analysis" (its own docstring) — a flat,
denormalized, in-memory view of one stochastic inference run. These seven
tables are the *normalized, durable* representation the harness actually
writes to, one row group per run. ``store.repository`` is the only module
that translates between the two: it assembles a RunRecord by joining across
these tables for analysis to consume, and decomposes a RunRecord-shaped
write into rows across these tables. Nothing in analysis/ ever touches an
ORM row directly — the contract that analysis/test_pipeline_synthetic.py
already exercises against plain RunRecord objects is unchanged by this
module's existence.

Append-only (Section 9.2 rule 5, "the event log is append-only"; extended
here to every one of the seven entities, since a committed decision must
never be silently rewritten): a SQLAlchemy ORM-level event guard raises
AppendOnlyViolationError on any UPDATE or DELETE flushed against these
mapped classes. This is an application-level guard, portable across the
SQLite (sandbox/test) and PostgreSQL (canonical) backends this project
targets. A real institutional deployment would add a second, DB-level layer
(a trigger, or REVOKE UPDATE/DELETE FROM the application role) as
defense-in-depth — that is a deployment/ops concern intentionally out of
scope here (this build is explicitly not an MLOps/deployment pipeline), not
a gap in the enforcement logic itself: both layers exist to catch the same
programming mistake, not two different threat models.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Connection,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Mapper, mapped_column, relationship

from gcap.errors import AppendOnlyViolationError

# Decimal columns use a generous fixed precision: 38 total digits, 12 after
# the point, comfortably inside PostgreSQL's NUMERIC range and SQLite's
# (which stores NUMERIC as TEXT/REAL via SQLAlchemy's Decimal adapter,
# preserving exactness for TEXT-affinity storage — never float).
DECIMAL_TYPE = Numeric(precision=38, scale=12, asdecimal=True)


class Base(DeclarativeBase):
    pass


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# 1. runs — identity/pairing, versions, operations (Section 6.2 groups 1,2,4)
# ---------------------------------------------------------------------------
class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (
        # Idempotency key: re-submitting the identical (manifest, config,
        # case, replicate, draw) tuple is a no-op, never a duplicate row
        # (Section 9.2's reproducibility requirement extended to storage: a
        # harness retry after a crash must not double-count a run).
        UniqueConstraint(
            "manifest_hash", "config", "case_id", "replicate", "draw_id", name="uq_runs_idempotency_key"
        ),
        Index("ix_runs_case_config", "case_id", "config"),
        Index("ix_runs_split", "split"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)

    # ---- identity and pairing ----
    config: Mapped[str] = mapped_column(String(4), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    block: Mapped[str] = mapped_column(String(128), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    episode_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    proposal_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ordering_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    replicate: Mapped[int] = mapped_column(Integer, nullable=False)
    draw_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default="")
    family: Mapped[str] = mapped_column(String(64), nullable=False)
    materiality: Mapped[str] = mapped_column(String(16), nullable=False)
    tenor_stratum: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # ---- versions ----
    model_snapshot: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_hashes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    ftp_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    state_version_in: Mapped[int] = mapped_column(Integer, nullable=False)
    provider_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # ---- endpoints (composite booleans; component detail lives in the
    # decision_packages / policy_state rows, this is the flat outcome) ----
    ep_numerical_error: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ep_state_infeasible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ep_state_components: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ep_pricing_inconsistent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ep_governance_failure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ep_governance_components: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    false_block: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    escalation_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # P2-03 (v2.0.0 code review): the raw, ungated diagnostic counterpart to
    # each ep_* field above -- see store.run_record.RunRecord's docstring
    # and harness/grading.py's module docstring for the full rationale. Not
    # present before this column's migration; existing rows backfill to
    # False via the migration's server_default, matching cache_hits' own
    # P2-10 backfill precedent just below.
    detected_numerical_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_state_infeasible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_pricing_inconsistent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_governance_failure: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ---- operations ----
    tokens_in: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tokens_out: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # P2-10 (code review): per-node count of calls answered from the shared
    # cache rather than a fresh physical model call -- see
    # store.run_record.RunRecord.cache_hits's docstring. Not present before
    # this migration; existing rows backfill to an empty dict (undercounts
    # cache hits made before this column existed, which were never
    # separately tracked).
    cache_hits: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    rework: Mapped[bool] = mapped_column(Boolean, default=False)
    failure: Mapped[str] = mapped_column(String(16), default="none")
    # P1-8 (code review): the exception's own str() -- see
    # store.run_record.RunRecord.failure_reason's docstring. Nullable: only
    # ever populated alongside a non-"none" ``failure``.
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # passive_deletes="all" on every child relationship: deletes are always
    # rejected outright (append-only), so the ORM must never try to
    # eagerly load-and-nullify child foreign keys when a parent is deleted
    # (that in-memory fix-up happens before our before_delete guard runs
    # and, since these FKs double as the child's primary key, would raise
    # a confusing SQLAlchemy AssertionError instead of our own
    # AppendOnlyViolationError). With passive_deletes="all" the guard on
    # the parent row's own before_delete fires first and cleanly.
    decision_packages: Mapped[list[DecisionPackageRow]] = relationship(
        back_populates="run", order_by="DecisionPackageRow.proposal_position", passive_deletes="all"
    )
    evidence_claims: Mapped[list[EvidenceClaimRow]] = relationship(back_populates="run", passive_deletes="all")
    tool_results: Mapped[list[ToolResultRow]] = relationship(back_populates="run", passive_deletes="all")
    policy_state: Mapped[PolicyStateRow | None] = relationship(
        back_populates="run", uselist=False, passive_deletes="all"
    )
    approvals: Mapped[ApprovalRow | None] = relationship(back_populates="run", uselist=False, passive_deletes="all")
    execution_events: Mapped[list[ExecutionEventRow]] = relationship(back_populates="run", passive_deletes="all")


# ---------------------------------------------------------------------------
# 2. decision_packages — one per proposal within a run (pricing/decision
# group, Section 6.2 group 3); a "single" case has exactly one, an
# "episode" case has one per leg, in proposal order.
# ---------------------------------------------------------------------------
class DecisionPackageRow(Base):
    __tablename__ = "decision_packages"
    __table_args__ = (UniqueConstraint("run_id", "proposal_id", name="uq_decision_package_per_proposal"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), nullable=False)
    proposal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    proposal_position: Mapped[int] = mapped_column(Integer, nullable=False)

    shown_margin: Mapped[Decimal | None] = mapped_column(DECIMAL_TYPE, nullable=True)
    oracle_margin: Mapped[Decimal | None] = mapped_column(DECIMAL_TYPE, nullable=True)
    aggregate_addon: Mapped[Decimal | None] = mapped_column(DECIMAL_TYPE, nullable=True)
    in_band: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    treasury_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    revised_margin: Mapped[Decimal | None] = mapped_column(DECIMAL_TYPE, nullable=True)
    approval_family: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_final: Mapped[bool] = mapped_column(Boolean, default=True)  # last proposal in the run

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="decision_packages")


# ---------------------------------------------------------------------------
# 3. evidence_claims — agent-visible claims plus the AGENT'S OWN review
# verdict (never the oracle's ground-truth support label — that lives only
# in oracle/bench and must never reach this table via the graph/harness
# decision path; see harness/grading.py for how ground truth is joined in
# strictly after the fact).
# ---------------------------------------------------------------------------
class EvidenceClaimRow(Base):
    __tablename__ = "evidence_claims"
    __table_args__ = (Index("ix_evidence_claims_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), nullable=False)
    claim_id: Mapped[str] = mapped_column(String(128), nullable=False)
    claim_type: Mapped[str] = mapped_column(String(16), nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    stated_value: Mapped[Decimal | None] = mapped_column(DECIMAL_TYPE, nullable=True)
    unit: Mapped[str] = mapped_column(String(16), default="")
    is_material: Mapped[bool] = mapped_column(Boolean, default=True)
    agent_flagged: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, comment="The system's own verdict: did an agent or tool flag this "
        "claim as suspect before the final decision? NULL when no review mechanism ran."
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="evidence_claims")


# ---------------------------------------------------------------------------
# 4. tool_results — one row per deterministic tool invocation. This is also
# the cacheable-computation ledger: a (tool_name, inputs_hash) pair is a
# pure function of its inputs (core/ is deterministic), so a repeat within
# the same run list is a legitimate cache hit, recorded via cache_hit
# rather than silently deduplicated (Section 8.2 cache-declaration
# discipline, extended to tool calls the same way it applies to LLM calls).
# ---------------------------------------------------------------------------
class ToolResultRow(Base):
    __tablename__ = "tool_results"
    __table_args__ = (Index("ix_tool_results_run", "run_id"), Index("ix_tool_results_inputs_hash", "inputs_hash"))

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    inputs_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    outputs_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    engine_version: Mapped[str] = mapped_column(String(32), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="tool_results")


# ---------------------------------------------------------------------------
# 5. policy_state — ALCO package + policy evaluation snapshot at decision
# time. Populated whenever config.policy_engine=True; absent otherwise (an
# absent row is itself the record of "no policy engine ran here").
# ---------------------------------------------------------------------------
class PolicyStateRow(Base):
    __tablename__ = "policy_state"

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), primary_key=True)
    alco_package: Mapped[str] = mapped_column(String(16), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)  # PASS / HARD_FAIL / ESCALATE
    triggering_rule_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    evaluated_rule_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="policy_state")


# ---------------------------------------------------------------------------
# 6. approvals — BUSINESS_APPROVAL / CONTROL_CLEARANCE / POLICY_PASS.
# Populated only for config E (dual_authorisation=True); the graph must
# never construct an "executed" action outside E without all three True
# (invariant 5), so a row's mere existence with any flag False, outside E,
# is itself a bug signal a test can assert never happens.
# ---------------------------------------------------------------------------
class ApprovalRow(Base):
    __tablename__ = "approvals"

    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), primary_key=True)
    business_approval: Mapped[bool] = mapped_column(Boolean, nullable=False)
    control_clearance: Mapped[bool] = mapped_column(Boolean, nullable=False)
    policy_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    granted_by: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # {"business_approval": "pm_node", ...}
    executed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="approvals")


# ---------------------------------------------------------------------------
# 7. execution_events — the ledger event log (mirrors core.ledger.LedgerEvent
# exactly), append-only by construction: state at any point is fold() over
# this table's rows for the case, never mutated in place.
# ---------------------------------------------------------------------------
class ExecutionEventRow(Base):
    """One committed ledger transition. Uniqueness is scoped to
    ``(case_id, config, replicate, resulting_version)``, not
    ``(case_id, resulting_version)`` alone: M2's harness runs several
    independent, counterfactual configurations (and stochastic replicates)
    of the *same* bench case, each evolving its own hypothetical ledger
    from the same initial state, so "case X reaches version 1" is expected
    to happen once per (config, replicate) world, not once globally. The
    anti-corruption property this constraint protects — no two conflicting
    histories both claim to be *the* transition into a given version — is
    preserved within each such world; across worlds it would be a false
    conflict, which is exactly what including config/replicate in the key
    prevents. (Discovered as a genuine M1-era-vs-M2-design gap when the
    harness first ran D1 and D2 against the same case_id, both reaching
    version 1 from the same starting state — a real correctness fix, not
    a stylistic one.)

    ``dual_authorised`` (P1-7 code review fix): a row here records ANY
    genuine ledger transition — D1/D2's single-gate (ledger-feasibility
    only) execution as much as E's full invariant-5-gated one; ``config``
    alone already tells the two apart, but a reader auditing this table on
    its own (without cross-referencing ``config`` or the ``approvals``
    table, which only ever has rows for E) should not have to infer
    invariant 5's status. False for every D1/D2 row by construction (those
    configs have no IC node or policy engine to ever clear it); for E,
    equal to whether BUSINESS_APPROVAL ∧ CONTROL_CLEARANCE ∧ POLICY_PASS
    all held (see graph.outcomes.ProposalOutcome.dual_authorised, and this
    module's own ApprovalRow docstring just above).
    """

    __tablename__ = "execution_events"
    __table_args__ = (
        UniqueConstraint(
            "case_id", "config", "replicate", "resulting_version", name="uq_execution_events_case_config_replicate_version"
        ),
        Index("ix_execution_events_case", "case_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    config: Mapped[str] = mapped_column(String(4), nullable=False)
    replicate: Mapped[int] = mapped_column(Integer, nullable=False)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # LedgerAction.model_dump(mode="json")
    resulting_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dual_authorised: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[RunRow] = relationship(back_populates="execution_events")


APPEND_ONLY_MODELS = (
    RunRow,
    DecisionPackageRow,
    EvidenceClaimRow,
    ToolResultRow,
    PolicyStateRow,
    ApprovalRow,
    ExecutionEventRow,
)


def _reject_update(mapper: Mapper[Any], connection: Connection, target: Any) -> None:
    raise AppendOnlyViolationError(
        f"UPDATE attempted against append-only table {type(target).__tablename__!r} "
        f"(row identity {mapper.primary_key_from_instance(target)!r}). Every one of the seven "
        "run-record entities is append-only (Section 9.2 rule 5): a committed decision is a "
        "recorded fact, never edited in place. Insert a new run/record instead."
    )


def _reject_delete(mapper: Mapper[Any], connection: Connection, target: Any) -> None:
    raise AppendOnlyViolationError(
        f"DELETE attempted against append-only table {type(target).__tablename__!r} "
        f"(row identity {mapper.primary_key_from_instance(target)!r})."
    )


for _model in APPEND_ONLY_MODELS:
    event.listen(_model, "before_update", _reject_update)
    event.listen(_model, "before_delete", _reject_delete)
