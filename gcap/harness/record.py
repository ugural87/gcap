"""Rolls up one graph.CaseOutcome (richer, graph-native) into one
store.repository.RunWrite (flat RunRecord + the six detail-table writes) —
the harness's own translation layer, matching outcomes.py's own docstring
("harness/store (M2d) is what condenses this down into a RunRecord + the
six other store entities").

One RunRecord row = one whole-case run (Section 6.2's "one row per
stochastic inference run"): a "single" case has one proposal and an
episode has several, sequentially state-linked. ``DecisionPackageWrite``
carries one row per proposal in the case; the final proposal's row is
marked ``is_final=True`` and is what the flat RunRecord's own pricing/
decision columns (shown_margin, treasury_action, approval_family,
revised_margin, ...) are read back from — matching
CaseOutcome.final_proposal, which exists for exactly this purpose.
``block`` is the one RunRecord column with no source in a CaseOutcome at
all: it is a run-list scheduling label assigned when the batch of runs is
planned, not derivable from one run's own inputs, so it arrives as this
function's own ``block`` parameter rather than being read off ``outcome``.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.agents.view import AgentVisibleCase, AgentVisibleClaim
from gcap.config.flags import ConfigFlags, ConfigName
from gcap.graph.context import GraphContext
from gcap.graph.outcomes import CaseOutcome, ProposalOutcome
from gcap.harness.grading import GradedEndpoints
from gcap.oracle.schema import OracleLabels
from gcap.store.repository import (
    ApprovalWrite,
    DecisionPackageWrite,
    EvidenceClaimWrite,
    ExecutionEventWrite,
    PolicyStateWrite,
    RunWrite,
    ToolResultWrite,
)
from gcap.store.run_record import FailureKind, RunRecord, Split, TenorStratum

_STATE_COMPONENT_NAMES = ("funding", "liquidity", "capital", "collateral", "concentration", "risk_budget")


def _node_call_summaries(
    outcome: CaseOutcome,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, int], dict[str, int], dict[str, int], int, bool]:
    """One dict entry per distinct node_name.

    P2-10: ``prompt_hashes`` now accumulates every call's hash (not just the
    last one) — a multi-claim node such as research_node makes one call per
    claim, and every claim's own prompt hash needs to survive into the run
    record, not just whichever claim happened to be processed last.
    Token/latency/cache_hits dicts still accumulate across every call to a
    node (e.g. every claim's own Research call), but token/latency figures
    now come from ``NodeCall``'s own computed properties, which already
    exclude cache-hit events from the cost totals (see agents/nodes.py's
    ``NodeCall``/``CallEvent`` — a cache hit costs nothing physically and is
    tracked separately in ``cache_hits`` instead of being double-counted as
    a fresh physical call)."""
    prompt_hashes: dict[str, list[str]] = {}
    tokens_in: dict[str, int] = {}
    tokens_out: dict[str, int] = {}
    latency_ms: dict[str, int] = {}
    cache_hits: dict[str, int] = {}
    retries = 0
    rework = False
    for proposal in outcome.proposals:
        for call in proposal.node_calls:
            prompt_hashes.setdefault(call.node_name, []).append(call.prompt_hash)
            tokens_in[call.node_name] = tokens_in.get(call.node_name, 0) + call.tokens_in
            tokens_out[call.node_name] = tokens_out.get(call.node_name, 0) + call.tokens_out
            latency_ms[call.node_name] = latency_ms.get(call.node_name, 0) + call.latency_ms
            cache_hits[call.node_name] = cache_hits.get(call.node_name, 0) + call.cache_hit_count
            if call.reworked:
                retries += 1
                rework = True
    return prompt_hashes, tokens_in, tokens_out, latency_ms, cache_hits, retries, rework


def _decision_package(proposal: ProposalOutcome, *, is_final: bool) -> DecisionPackageWrite:
    ftp = proposal.ftp_output
    return DecisionPackageWrite(
        proposal_id=proposal.proposal_id,
        proposal_position=proposal.proposal_position,
        shown_margin=ftp.economic_alpha_horizon if ftp is not None else None,
        aggregate_addon=ftp.addons.aggregate if ftp is not None else None,
        treasury_action=proposal.final_action,
        revised_margin=proposal.revised_margin,
        approval_family=proposal.approval_family,
        is_final=is_final,
    )


def _evidence_claims(
    proposal: ProposalOutcome, agent_proposal_claims: tuple[AgentVisibleClaim, ...]
) -> tuple[EvidenceClaimWrite, ...]:
    reviews_by_id = {r.claim_id: r for r in proposal.claim_reviews}
    writes = []
    for claim in agent_proposal_claims:
        review = reviews_by_id.get(claim.claim_id)
        writes.append(
            EvidenceClaimWrite(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                metric=claim.metric,
                stated_value=claim.stated_value,
                unit=claim.unit,
                is_material=claim.is_material,
                agent_flagged=review.flagged if review is not None else None,
            )
        )
    return tuple(writes)


def _tool_results(proposal: ProposalOutcome) -> tuple[ToolResultWrite, ...]:
    return tuple(
        ToolResultWrite(
            tool_name=tr.tool_name,
            inputs_hash=tr.inputs_hash,
            outputs_summary=dict(tr.outputs_summary),
            cache_hit=False,  # per-tool-call cache hits are tracked by ToolCallCache's own counters, not per row
            engine_version=tr.engine_version,
        )
        for tr in proposal.tool_results
    )


def _false_block(final: ProposalOutcome, oracle_labels: OracleLabels) -> bool | None:
    if final.approval_family is None:
        return None
    any_true_endpoint = (
        oracle_labels.ep_numerical_error
        or oracle_labels.ep_state_infeasible
        or oracle_labels.ep_pricing_inconsistent
        or oracle_labels.ep_governance_failure
    )
    return (final.approval_family is False) and not any_true_endpoint


def _escalation_correct(final: ProposalOutcome, oracle_labels: OracleLabels) -> bool | None:
    if final.final_action is None:
        return None
    system_escalated = final.final_action.value == "DEFER"
    return system_escalated == oracle_labels.expected_escalation_required


def build_run_write(
    *,
    case: AgentVisibleCase,
    outcome: CaseOutcome,
    graded: GradedEndpoints,
    oracle_labels: OracleLabels,
    config: ConfigFlags,
    config_name: ConfigName,
    ctx: GraphContext,
    split: Split,
    block: str,
    replicate: int,
    draw_id: str | None,
    manifest_hash: str,
    provider_seed: int | None,
    tenor_stratum: TenorStratum | None = None,
    failure: FailureKind = "none",
) -> RunWrite:
    final = outcome.final_proposal
    prompt_hashes, tokens_in, tokens_out, latency_ms, cache_hits, retries, rework = _node_call_summaries(outcome)

    oracle_margin = next(
        (m.economic_alpha_horizon for m in oracle_labels.proposal_margins if m.proposal_id == final.proposal_id), None
    )
    in_band: dict[str, bool] = {}
    ftp_params: dict[str, Decimal] = {}
    if final.ftp_output is not None:
        in_band["primary"] = final.ftp_output.in_band(ctx.marginal_band_ann)
        addons = final.ftp_output.addons
        ftp_params = {
            "lp": addons.lp,
            "tp": addons.tp,
            "cp": addons.cp,
            "rp": addons.rp,
            "cc": addons.cc,
            "aggregate": addons.aggregate,
        }

    run_record = RunRecord(
        run_id=uuid.uuid4(),
        config=config_name,
        split=split,
        block=block,
        case_id=case.case_id,
        episode_id=case.case_id if case.kind == "episode" else None,
        proposal_position=final.proposal_position,
        replicate=replicate,
        draw_id=draw_id,
        family=case.family,
        materiality=case.materiality,
        tenor_stratum=tenor_stratum,
        model_snapshot=getattr(ctx.adapter, "model_snapshot", "unknown-model-snapshot"),
        prompt_hashes=prompt_hashes,
        cache_hits=cache_hits,
        policy_version=case.policy_version,
        ftp_params=ftp_params,
        state_version_in=case.initial_state.version,
        provider_seed=provider_seed,
        manifest_hash=manifest_hash,
        shown_margin=final.ftp_output.economic_alpha_horizon if final.ftp_output is not None else None,
        oracle_margin=oracle_margin,
        aggregate_addon=final.ftp_output.addons.aggregate if final.ftp_output is not None else None,
        in_band=in_band,
        treasury_action=final.final_action,
        revised_margin=final.revised_margin,
        approval_family=final.approval_family,
        ep_numerical_error=graded.ep_numerical_error,
        ep_state_infeasible=graded.ep_state_infeasible,
        ep_state_components=graded.ep_state_components,
        ep_pricing_inconsistent=graded.ep_pricing_inconsistent,
        ep_governance_failure=graded.ep_governance_failure,
        ep_governance_components=graded.ep_governance_components,
        # P2-03 (v2.0.0 code review): persist the raw, ungated diagnostic
        # counterpart alongside each ep_* field -- see harness/grading.py's
        # module docstring.
        detected_numerical_error=graded.detected_numerical_error,
        detected_state_infeasible=graded.detected_state_infeasible,
        detected_pricing_inconsistent=graded.detected_pricing_inconsistent,
        detected_governance_failure=graded.detected_governance_failure,
        false_block=_false_block(final, oracle_labels),
        escalation_correct=_escalation_correct(final, oracle_labels),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        retries=retries,
        rework=rework,
        failure=failure,
    )

    decision_packages = tuple(
        _decision_package(p, is_final=(p.proposal_position == final.proposal_position)) for p in outcome.proposals
    )
    evidence_claims: tuple[EvidenceClaimWrite, ...] = ()
    tool_results: tuple[ToolResultWrite, ...] = ()
    policy_state: PolicyStateWrite | None = None
    approval: ApprovalWrite | None = None
    execution_events: tuple[ExecutionEventWrite, ...] = ()

    # AgentVisibleProposal carries no proposal_position of its own; the
    # graph assigns positions purely by the case's proposal ORDER (see
    # graph/pipeline.py's run_case), so pairing back up here must use the
    # same positional correspondence, never a lookup by an id the view
    # schema doesn't have.
    case_claims_by_position = {i: p.claims for i, p in enumerate(case.proposals)}
    for proposal_outcome in outcome.proposals:
        evidence_claims += _evidence_claims(
            proposal_outcome, case_claims_by_position.get(proposal_outcome.proposal_position, ())
        )
        tool_results += _tool_results(proposal_outcome)

    if final.policy_result is not None:
        policy_state = PolicyStateWrite(
            alco_package=ctx.alco_package,
            verdict=final.policy_result.verdict,
            triggering_rule_ids=final.policy_result.triggering_rule_ids,
            evaluated_rule_ids=final.policy_result.evaluated_rule_ids,
            policy_version=final.policy_result.policy_version,
        )
    if final.business_approval is not None:
        approval = ApprovalWrite(
            business_approval=bool(final.business_approval),
            control_clearance=bool(final.control_clearance),
            policy_pass=bool(final.policy_pass),
            executed=bool(final.executed),
        )
    for p in outcome.proposals:
        if p.executed and p.state_out is not None:
            execution_events += (
                ExecutionEventWrite(
                    event_id=f"{case.case_id}:{p.proposal_id}:{replicate}",
                    action={"proposal_id": p.proposal_id, "final_action": p.final_action.value if p.final_action else None},
                    resulting_version=p.state_out.version,
                    # P1-7: carries invariant 5's status through to the
                    # store so this table is self-describing without a
                    # join back to `config` or the (E-only) approvals
                    # table — see store.models.ExecutionEventRow's
                    # docstring.
                    dual_authorised=p.dual_authorised,
                ),
            )

    return RunWrite(
        run_record=run_record,
        decision_packages=decision_packages,
        evidence_claims=evidence_claims,
        tool_results=tool_results,
        policy_state=policy_state,
        approval=approval,
        execution_events=execution_events,
    )
