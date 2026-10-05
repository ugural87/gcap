from __future__ import annotations

from decimal import Decimal

from gcap.agents.llm import FakeLLMAdapter
from gcap.agents.nodes import CallEvent, NodeCall
from gcap.config.flags import CONFIG_TABLE
from gcap.core.ftp import FTPAddonVector, FTPInputs, compute_ftp
from gcap.graph.context import GraphContext, ProposalHints
from gcap.graph.outcomes import CaseOutcome, ProposalOutcome
from gcap.graph.pipeline import run_proposal
from gcap.harness.grading import grade_case_outcome
from gcap.harness.record import _node_call_summaries
from gcap.oracle.schema import (
    OracleGovernanceComponents,
    OracleLabels,
    OracleProposalMargin,
    OracleStateComponents,
)
from gcap.store.run_record import TreasuryAction
from tests.graph.factories import make_case, make_ledger_state, make_proposal


def _labels_for(final_proposal_id: str, economic_alpha: Decimal) -> OracleLabels:
    return OracleLabels(
        case_id="episode-1",
        ep_numerical_error=False,
        numerical_error_claim_ids=(),
        ep_state_infeasible=False,
        ep_state_components=OracleStateComponents(),
        ep_pricing_inconsistent=False,
        proposal_margins=(
            OracleProposalMargin(
                proposal_id=final_proposal_id, economic_alpha_horizon=economic_alpha, in_band=True, pricing_inconsistent=False
            ),
        ),
        ep_governance_failure=False,
        ep_governance_components=OracleGovernanceComponents(),
        expected_escalation_required=False,
        restructure_feasible=None,
    )


def test_build_run_write_rolls_up_a_d1_case_outcome():
    from gcap.harness.record import build_run_write

    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.01"))
    case = make_case(proposals=(proposal,), case_id="episode-1")
    ctx = GraphContext(
        adapter=FakeLLMAdapter(),
        hints_by_proposal={"p1": ProposalHints(trade={"trade_alpha_horizon": "0.05"}, treasury={"suggested_action": "APPROVE"})},
    )
    config = CONFIG_TABLE["D1"]
    from gcap.graph.outcomes import CaseOutcome

    outcome_proposal = run_proposal(
        proposal=proposal, state=make_ledger_state(), config=config, config_name="D1", ctx=ctx,
        case_id="episode-1", replicate=0, proposal_position=0,
    )
    outcome = CaseOutcome(case_id="episode-1", proposals=(outcome_proposal,), final_state=outcome_proposal.state_out)

    expected_ftp = compute_ftp(
        FTPInputs(
            tau_days=proposal.tau_days, day_count_basis=proposal.day_count_basis,
            base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann, addons=FTPAddonVector(),
            trade_alpha_horizon=Decimal("0.05"),
        )
    )
    labels = _labels_for("p1", expected_ftp.economic_alpha_horizon)
    graded = grade_case_outcome(outcome, config, labels)

    write = build_run_write(
        case=case, outcome=outcome, graded=graded, oracle_labels=labels, config=config, config_name="D1",
        ctx=ctx, split="dev", block="block-1", replicate=0, draw_id=None, manifest_hash="deadbeef",
        provider_seed=123, tenor_stratum="1-12m",
    )

    rr = write.run_record
    assert rr.config == "D1"
    assert rr.case_id == "episode-1"
    assert rr.split == "dev"
    assert rr.block == "block-1"
    assert rr.tenor_stratum == "1-12m"
    assert rr.manifest_hash == "deadbeef"
    assert rr.provider_seed == 123
    assert rr.treasury_action == TreasuryAction.APPROVE
    assert rr.approval_family is True
    assert rr.shown_margin == expected_ftp.economic_alpha_horizon
    assert rr.oracle_margin == expected_ftp.economic_alpha_horizon  # same math -> should agree
    assert rr.failure == "none"
    assert "research" in rr.tokens_in
    assert "trade" in rr.tokens_in
    assert len(write.decision_packages) == 1
    assert write.decision_packages[0].is_final is True
    assert len(write.evidence_claims) == 1
    assert write.evidence_claims[0].claim_id == "c1"
    assert any(t.tool_name == "ftp" for t in write.tool_results)
    assert any(t.tool_name == "ledger_feasibility" for t in write.tool_results)
    assert len(write.execution_events) == 1  # D1 executed (feasible + approval-family)
    assert rr.revised_margin is None  # bare APPROVE never triggers Section 9.6's recomputation
    assert write.decision_packages[0].revised_margin is None


def test_build_run_write_rolls_up_a_revised_margin_from_approve_with_hedge():
    """Closes the gap flagged in the M2 status report: Section 9.6's
    APPROVE_WITH_HEDGE recomputation (graph.pipeline._revised_addons) must
    actually reach RunRecord.revised_margin and
    DecisionPackageWrite.revised_margin through build_run_write, not just
    exist on the graph's own ProposalOutcome."""
    from gcap.harness.record import build_run_write

    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    case = make_case(proposals=(proposal,), case_id="episode-hedge")
    ctx = GraphContext(
        adapter=FakeLLMAdapter(),
        hints_by_proposal={
            "p1": ProposalHints(
                trade={"trade_alpha_horizon": "0.00365", "tau_days": "365"},
                treasury={"suggested_action": "APPROVE_WITH_HEDGE"},
            )
        },
    )
    config = CONFIG_TABLE["D2"]
    from gcap.graph.outcomes import CaseOutcome

    outcome_proposal = run_proposal(
        proposal=proposal, state=make_ledger_state(), config=config, config_name="D2", ctx=ctx,
        case_id="episode-hedge", replicate=0, proposal_position=0,
    )
    assert outcome_proposal.revised_margin is not None and outcome_proposal.revised_margin > 0
    assert outcome_proposal.ftp_output is not None and outcome_proposal.ftp_output.economic_alpha_horizon < 0
    outcome = CaseOutcome(case_id="episode-hedge", proposals=(outcome_proposal,), final_state=outcome_proposal.state_out)

    labels = _labels_for("p1", outcome_proposal.revised_margin)
    graded = grade_case_outcome(outcome, config, labels)
    write = build_run_write(
        case=case, outcome=outcome, graded=graded, oracle_labels=labels, config=config, config_name="D2",
        ctx=ctx, split="dev", block="block-1", replicate=0, draw_id=None, manifest_hash="deadbeef", provider_seed=None,
    )

    rr = write.run_record
    assert rr.revised_margin == outcome_proposal.revised_margin
    assert rr.shown_margin == outcome_proposal.ftp_output.economic_alpha_horizon  # unrevised figure, kept separately
    assert rr.ep_pricing_inconsistent is False  # graded on the revised margin, not the shown one
    assert write.decision_packages[0].revised_margin == outcome_proposal.revised_margin


def test_build_run_write_marks_false_block_when_oracle_says_case_is_clean():
    from gcap.harness.record import build_run_write

    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.01"))
    case = make_case(proposals=(proposal,), case_id="episode-2")
    ctx = GraphContext(
        adapter=FakeLLMAdapter(),
        hints_by_proposal={"p1": ProposalHints(pm={"recommend": False})},
    )
    config = CONFIG_TABLE["B"]
    from gcap.graph.outcomes import CaseOutcome

    outcome_proposal = run_proposal(
        proposal=proposal, state=make_ledger_state(), config=config, config_name="B", ctx=ctx,
        case_id="episode-2", replicate=0, proposal_position=0,
    )
    outcome = CaseOutcome(case_id="episode-2", proposals=(outcome_proposal,), final_state=None)
    labels = OracleLabels(
        case_id="episode-2",
        ep_numerical_error=False,
        numerical_error_claim_ids=(),
        ep_state_infeasible=False,
        ep_state_components=OracleStateComponents(),
        ep_pricing_inconsistent=False,
        proposal_margins=(),
        ep_governance_failure=False,
        ep_governance_components=OracleGovernanceComponents(),
        expected_escalation_required=False,
        restructure_feasible=None,
    )
    graded = grade_case_outcome(outcome, config, labels)
    write = build_run_write(
        case=case, outcome=outcome, graded=graded, oracle_labels=labels, config=config, config_name="B",
        ctx=ctx, split="dev", block="block-1", replicate=0, draw_id=None, manifest_hash="deadbeef", provider_seed=None,
    )
    assert write.run_record.approval_family is False  # PM rejected
    assert write.run_record.false_block is True  # oracle says nothing was actually wrong


# ---------------------------------------------------------------------------
# P2-10 (code review): usage/cost accounting. _node_call_summaries is the
# rollup point where these three sub-bugs previously lived; exercised
# directly against hand-built NodeCall/CallEvent fixtures rather than a full
# graph run, since the bug is in the rollup arithmetic, not the graph.
# ---------------------------------------------------------------------------
def _outcome_with(*calls: NodeCall) -> CaseOutcome:
    proposal = ProposalOutcome(proposal_id="p1", proposal_position=0, node_calls=tuple(calls))
    return CaseOutcome(case_id="case-1", proposals=(proposal,), final_state=None)


def test_node_call_summaries_preserves_every_claims_own_prompt_hash():
    """research_node makes one call per claim; before this fix,
    prompt_hashes[node_name] = call.prompt_hash overwrote earlier claims'
    hashes with whichever claim's call happened to run last."""
    call_for_c1 = NodeCall(
        node_name="research", prompt_hash="hash-c1", reworked=False,
        events=(CallEvent(cache_hit=False, tokens_in=10, tokens_out=5, latency_ms=100),),
    )
    call_for_c2 = NodeCall(
        node_name="research", prompt_hash="hash-c2", reworked=False,
        events=(CallEvent(cache_hit=False, tokens_in=11, tokens_out=6, latency_ms=110),),
    )
    prompt_hashes, tokens_in, _tokens_out, _latency_ms, _cache_hits, _retries, _rework = _node_call_summaries(
        _outcome_with(call_for_c1, call_for_c2)
    )
    assert prompt_hashes["research"] == ["hash-c1", "hash-c2"]
    assert tokens_in["research"] == 21  # still accumulates across both claims' calls


def test_node_call_summaries_counts_a_reworked_attempts_real_cost():
    """A rework attempt that ultimately fails still cost real tokens; that
    cost must survive into the usage record even though only the final,
    successful attempt's output is used. Previously NodeCall stored only
    the last attempt's flat token/latency figures, silently dropping every
    earlier attempt's spend."""
    reworked_call = NodeCall(
        node_name="trade", prompt_hash="hash-1", reworked=True,
        events=(
            CallEvent(cache_hit=False, tokens_in=10, tokens_out=5, latency_ms=100),  # failed first attempt
            CallEvent(cache_hit=False, tokens_in=12, tokens_out=6, latency_ms=110),  # succeeded on rework
        ),
    )
    _prompt_hashes, tokens_in, tokens_out, latency_ms, _cache_hits, retries, rework = _node_call_summaries(
        _outcome_with(reworked_call)
    )
    assert tokens_in["trade"] == 22
    assert tokens_out["trade"] == 11
    assert latency_ms["trade"] == 210
    assert retries == 1
    assert rework is True


def test_node_call_summaries_excludes_cache_hits_from_cost_but_still_counts_them():
    """A cache hit replays an earlier, already-paid-for response: it must
    not add to tokens_in/tokens_out/latency_ms (that would double-count the
    same physical call's cost across D1/D2's two replicates), but it must
    still show up somewhere so a cache hit isn't invisible to the usage
    record."""
    cache_hit_call = NodeCall(
        node_name="research", prompt_hash="hash-1", reworked=False,
        events=(CallEvent(cache_hit=True, tokens_in=999, tokens_out=999, latency_ms=999),),
    )
    _prompt_hashes, tokens_in, tokens_out, latency_ms, cache_hits, _retries, _rework = _node_call_summaries(
        _outcome_with(cache_hit_call)
    )
    assert tokens_in.get("research", 0) == 0
    assert tokens_out.get("research", 0) == 0
    assert latency_ms.get("research", 0) == 0
    assert cache_hits["research"] == 1
