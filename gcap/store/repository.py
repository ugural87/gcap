"""Bridges the normalized seven-table schema (store.models) and the flat
RunRecord pydantic contract (store.run_record) that agents/graph/harness on
one side, and analysis on the other, actually work with.

Write side: the harness assembles a ``RunWrite`` bundle (everything the
graph produced for one stochastic inference run, including the
proposal-level and tool-call-level detail RunRecord itself does not carry)
and calls ``save_run``, which decomposes it across the seven tables inside
one transaction — either all seven writes land, or none do.

Read side: ``load_run_record``/``load_run_records`` reconstruct the flat
RunRecord view analysis/ already knows how to consume, so M1's
analysis/pipeline.py needed zero changes for M2's DB-backed storage to
exist.

Idempotency (Section 9.2's reproducibility discipline, extended to
storage): ``save_run`` is safe to call twice with the same natural key
(manifest_hash, config, case_id, replicate, draw_id) — the second call is a
no-op that returns the first call's run_id, never a duplicate row. This
matters because a harness worker can crash after committing a run's DB
transaction but before acknowledging success upstream; the retry that
follows must not double-count the run in the analysis layer's cluster
counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gcap.store.models import (
    ApprovalRow,
    DecisionPackageRow,
    EvidenceClaimRow,
    ExecutionEventRow,
    PolicyStateRow,
    RunRow,
    ToolResultRow,
)
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents, TreasuryAction


class DecisionPackageWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str
    proposal_position: int
    shown_margin: Decimal | None = None
    oracle_margin: Decimal | None = None
    aggregate_addon: Decimal | None = None
    in_band: dict[str, bool] = {}
    treasury_action: TreasuryAction | None = None
    revised_margin: Decimal | None = None
    approval_family: bool | None = None
    is_final: bool = True


class EvidenceClaimWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_type: str
    metric: str
    stated_value: Decimal | None = None
    unit: str = ""
    is_material: bool = True
    agent_flagged: bool | None = None


class ToolResultWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    inputs_hash: str
    outputs_summary: dict[str, Any] = {}
    cache_hit: bool = False
    latency_ms: int = 0
    engine_version: str = ""


class PolicyStateWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    alco_package: str
    verdict: str
    triggering_rule_ids: tuple[str, ...] = ()
    evaluated_rule_ids: tuple[str, ...] = ()
    policy_version: str


class ApprovalWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_approval: bool
    control_clearance: bool
    policy_pass: bool
    granted_by: dict[str, str] = {}
    executed: bool


class ExecutionEventWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    action: dict[str, Any]
    resulting_version: int
    # P1-7 (code review): see models.ExecutionEventRow's docstring — False
    # for every D1/D2 event by construction, equal to invariant 5's full
    # conjunction for a config E event.
    dual_authorised: bool = False


@dataclass(frozen=True)
class RunWrite:
    """Everything one stochastic inference run produced, ready to persist.
    ``run_record`` supplies every column of the flat ``runs`` row (identity,
    versions, endpoints, operations); the remaining fields supply the
    proposal/claim/tool/policy/approval/event detail RunRecord itself does
    not carry."""

    run_record: RunRecord
    decision_packages: tuple[DecisionPackageWrite, ...] = ()
    evidence_claims: tuple[EvidenceClaimWrite, ...] = ()
    tool_results: tuple[ToolResultWrite, ...] = ()
    policy_state: PolicyStateWrite | None = None
    approval: ApprovalWrite | None = None
    execution_events: tuple[ExecutionEventWrite, ...] = ()


def save_run(session: Session, write: RunWrite) -> tuple[str, bool]:
    """Returns (run_id, was_inserted). was_inserted=False means an
    identical natural key already existed and nothing new was written."""
    rr = write.run_record
    run_id = str(rr.run_id)

    row = RunRow(
        run_id=run_id,
        config=rr.config,
        split=rr.split,
        block=rr.block,
        case_id=rr.case_id,
        episode_id=rr.episode_id,
        proposal_position=rr.proposal_position,
        ordering_id=rr.ordering_id,
        replicate=rr.replicate,
        draw_id=rr.draw_id or "",
        family=rr.family,
        materiality=rr.materiality,
        tenor_stratum=rr.tenor_stratum,
        model_snapshot=rr.model_snapshot,
        prompt_hashes={k: list(v) for k, v in rr.prompt_hashes.items()},
        policy_version=rr.policy_version,
        ftp_params={k: str(v) for k, v in rr.ftp_params.items()},
        state_version_in=rr.state_version_in,
        provider_seed=rr.provider_seed,
        manifest_hash=rr.manifest_hash,
        ep_numerical_error=rr.ep_numerical_error,
        ep_state_infeasible=rr.ep_state_infeasible,
        ep_state_components=rr.ep_state_components.model_dump(),
        ep_pricing_inconsistent=rr.ep_pricing_inconsistent,
        ep_governance_failure=rr.ep_governance_failure,
        ep_governance_components=rr.ep_governance_components.model_dump(),
        detected_numerical_error=rr.detected_numerical_error,
        detected_state_infeasible=rr.detected_state_infeasible,
        detected_pricing_inconsistent=rr.detected_pricing_inconsistent,
        detected_governance_failure=rr.detected_governance_failure,
        false_block=rr.false_block,
        escalation_correct=rr.escalation_correct,
        tokens_in=dict(rr.tokens_in),
        tokens_out=dict(rr.tokens_out),
        latency_ms=dict(rr.latency_ms),
        cache_hits=dict(rr.cache_hits),
        retries=rr.retries,
        rework=rr.rework,
        failure=rr.failure,
        failure_reason=rr.failure_reason,
    )

    for dp in write.decision_packages:
        row.decision_packages.append(
            DecisionPackageRow(
                run_id=run_id,
                proposal_id=dp.proposal_id,
                proposal_position=dp.proposal_position,
                shown_margin=dp.shown_margin,
                oracle_margin=dp.oracle_margin,
                aggregate_addon=dp.aggregate_addon,
                in_band=dict(dp.in_band),
                treasury_action=dp.treasury_action.value if dp.treasury_action is not None else None,
                revised_margin=dp.revised_margin,
                approval_family=dp.approval_family,
                is_final=dp.is_final,
            )
        )
    for ec in write.evidence_claims:
        row.evidence_claims.append(
            EvidenceClaimRow(
                run_id=run_id,
                claim_id=ec.claim_id,
                claim_type=ec.claim_type,
                metric=ec.metric,
                stated_value=ec.stated_value,
                unit=ec.unit,
                is_material=ec.is_material,
                agent_flagged=ec.agent_flagged,
            )
        )
    for tr in write.tool_results:
        row.tool_results.append(
            ToolResultRow(
                run_id=run_id,
                tool_name=tr.tool_name,
                inputs_hash=tr.inputs_hash,
                outputs_summary=dict(tr.outputs_summary),
                cache_hit=tr.cache_hit,
                latency_ms=tr.latency_ms,
                engine_version=tr.engine_version,
            )
        )
    if write.policy_state is not None:
        ps = write.policy_state
        row.policy_state = PolicyStateRow(
            run_id=run_id,
            alco_package=ps.alco_package,
            verdict=ps.verdict,
            triggering_rule_ids=list(ps.triggering_rule_ids),
            evaluated_rule_ids=list(ps.evaluated_rule_ids),
            policy_version=ps.policy_version,
        )
    if write.approval is not None:
        ap = write.approval
        row.approvals = ApprovalRow(
            run_id=run_id,
            business_approval=ap.business_approval,
            control_clearance=ap.control_clearance,
            policy_pass=ap.policy_pass,
            granted_by=dict(ap.granted_by),
            executed=ap.executed,
        )
    for ev in write.execution_events:
        row.execution_events.append(
            ExecutionEventRow(
                run_id=run_id,
                case_id=rr.case_id,
                config=rr.config,
                replicate=rr.replicate,
                event_id=ev.event_id,
                action=ev.action,
                resulting_version=ev.resulting_version,
                dual_authorised=ev.dual_authorised,
            )
        )

    savepoint = session.begin_nested()
    try:
        session.add(row)
        session.flush()
        savepoint.commit()
        return run_id, True
    except IntegrityError:
        savepoint.rollback()
        existing = session.execute(
            select(RunRow.run_id).where(
                RunRow.manifest_hash == rr.manifest_hash,
                RunRow.config == rr.config,
                RunRow.case_id == rr.case_id,
                RunRow.replicate == rr.replicate,
                RunRow.draw_id == (rr.draw_id or ""),
            )
        ).scalar_one()
        return existing, False


def _row_to_run_record(row: RunRow) -> RunRecord:
    final_dp = next((dp for dp in row.decision_packages if dp.is_final), None)
    return RunRecord(
        run_id=UUID(row.run_id),
        config=row.config,  # type: ignore[arg-type]
        split=row.split,  # type: ignore[arg-type]
        block=row.block,
        case_id=row.case_id,
        episode_id=row.episode_id,
        proposal_position=row.proposal_position,
        ordering_id=row.ordering_id,
        replicate=row.replicate,
        draw_id=row.draw_id or None,
        family=row.family,
        materiality=row.materiality,
        tenor_stratum=row.tenor_stratum,  # type: ignore[arg-type]
        model_snapshot=row.model_snapshot,
        prompt_hashes={k: list(v) for k, v in row.prompt_hashes.items()},
        policy_version=row.policy_version,
        ftp_params={k: Decimal(v) for k, v in row.ftp_params.items()},
        state_version_in=row.state_version_in,
        provider_seed=row.provider_seed,
        manifest_hash=row.manifest_hash,
        shown_margin=final_dp.shown_margin if final_dp else None,
        oracle_margin=final_dp.oracle_margin if final_dp else None,
        aggregate_addon=final_dp.aggregate_addon if final_dp else None,
        in_band=dict(final_dp.in_band) if final_dp else {},
        treasury_action=TreasuryAction(final_dp.treasury_action) if final_dp and final_dp.treasury_action else None,
        revised_margin=final_dp.revised_margin if final_dp else None,
        approval_family=final_dp.approval_family if final_dp else None,
        ep_numerical_error=row.ep_numerical_error,
        ep_state_infeasible=row.ep_state_infeasible,
        ep_state_components=StateComponents(**row.ep_state_components),
        ep_pricing_inconsistent=row.ep_pricing_inconsistent,
        ep_governance_failure=row.ep_governance_failure,
        ep_governance_components=GovernanceComponents(**row.ep_governance_components),
        detected_numerical_error=row.detected_numerical_error,
        detected_state_infeasible=row.detected_state_infeasible,
        detected_pricing_inconsistent=row.detected_pricing_inconsistent,
        detected_governance_failure=row.detected_governance_failure,
        false_block=row.false_block,
        escalation_correct=row.escalation_correct,
        tokens_in=dict(row.tokens_in),
        tokens_out=dict(row.tokens_out),
        latency_ms=dict(row.latency_ms),
        cache_hits=dict(row.cache_hits),
        retries=row.retries,
        rework=row.rework,
        failure=row.failure,  # type: ignore[arg-type]
        failure_reason=row.failure_reason,
    )


def load_run_record(session: Session, run_id: str) -> RunRecord:
    row = session.get(RunRow, run_id)
    if row is None:
        raise KeyError(f"No run persisted with run_id={run_id!r}.")
    return _row_to_run_record(row)


def load_run_records(
    session: Session,
    *,
    manifest_hash: str | None = None,
    split: str | None = None,
    config: str | None = None,
) -> list[RunRecord]:
    """Bulk loader feeding analysis.pipeline.run_all_primary_contrasts
    exactly the ``list[RunRecord]`` shape it already expects."""
    stmt = select(RunRow)
    if manifest_hash is not None:
        stmt = stmt.where(RunRow.manifest_hash == manifest_hash)
    if split is not None:
        stmt = stmt.where(RunRow.split == split)
    if config is not None:
        stmt = stmt.where(RunRow.config == config)
    rows = session.execute(stmt).scalars().all()
    return [_row_to_run_record(row) for row in rows]
