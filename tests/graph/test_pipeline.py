"""Exercises run_proposal/run_case across all six ConfigFlags arms, plus
the D1/D2 SharedCallCache wiring the graph now routes Research/Trade calls
through. Uses FakeLLMAdapter exclusively (per the standing no-real-API-key
constraint) and ProposalHints to deterministically steer node outputs."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gcap.agents.cache import SharedCallCache
from gcap.agents.llm import FakeLLMAdapter
from gcap.config.flags import CONFIG_TABLE
from gcap.core.alco import PACKAGES
from gcap.core.ftp import FTPAddonVector, FTPInputs, compute_ftp
from gcap.errors import UndeclaredCacheHitError
from gcap.graph.context import GraphContext, ProposalHints
from gcap.graph.pipeline import _midpoint_addons, run_case, run_proposal
from tests.graph.factories import (
    make_case,
    make_footprint,
    make_ledger_state,
    make_policy_rule,
    make_proposal,
)


def _ctx(**overrides: object) -> GraphContext:
    base: dict[str, object] = {"adapter": FakeLLMAdapter()}
    base.update(overrides)
    return GraphContext(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config A: single generalist LLM, no decomposition, no tools/ledger/policy.
# ---------------------------------------------------------------------------
def test_config_a_uses_single_agent_node_only():
    proposal = make_proposal()
    ctx = _ctx(
        hints_by_proposal={"p1": ProposalHints(single_agent={"suggested_action": "APPROVE"})}
    )
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["A"],
        config_name="A",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.single_agent is not None
    assert outcome.trade is None
    assert outcome.pm is None
    assert outcome.final_action is not None and outcome.final_action.value == "APPROVE"
    assert outcome.approval_family is True
    # No mechanism exists in A: every endpoint stays harness-graded (False).
    assert outcome.ep_numerical_error is False
    assert outcome.ep_state_infeasible is False
    assert outcome.ep_pricing_inconsistent is False
    assert outcome.ep_governance_failure is False
    assert outcome.executed is None


# ---------------------------------------------------------------------------
# Config B: Research/Trade/PM, no deterministic tools.
# ---------------------------------------------------------------------------
def test_config_b_final_action_follows_pm_recommendation():
    proposal = make_proposal()
    ctx = _ctx(hints_by_proposal={"p1": ProposalHints(pm={"recommend": False})})
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["B"],
        config_name="B",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.pm is not None and outcome.pm.recommend is False
    assert outcome.final_action is not None and outcome.final_action.value == "REJECT"
    assert outcome.approval_family is False
    assert outcome.tool_results == ()  # no deterministic tools in B
    assert outcome.ep_numerical_error is False  # not applicable in B; harness-graded


# ---------------------------------------------------------------------------
# Config C: + deterministic reconciliation tool -> in-graph ep_numerical_error.
# ---------------------------------------------------------------------------
def test_config_c_ep_numerical_error_detected_when_claim_flagged_and_approved():
    proposal = make_proposal(stated_value=Decimal("0.15"))
    ctx = _ctx(
        hints_by_proposal={
            "p1": ProposalHints(
                claim_hints={
                    "c1": {"authoritative_value": "0.50"}
                },  # far off stated 0.15 -> flagged
                pm={"recommend": True},
            )
        }
    )
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["C"],
        config_name="C",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert len(outcome.tool_results) == 1
    assert outcome.tool_results[0].tool_name == "reconciliation"
    assert outcome.tool_results[0].outputs_summary["flagged"] is True
    assert outcome.approval_family is True
    assert outcome.ep_numerical_error is True
    assert outcome.detected_numerical_error is True  # P2-03: same raw signal, approved this time


def test_config_c_ep_numerical_error_false_when_flagged_but_rejected():
    """Rule is (flagged) AND (approval_family) — a flagged claim that gets
    rejected anyway is not a numerical-error endpoint: the system caught it."""
    proposal = make_proposal(stated_value=Decimal("0.15"))
    ctx = _ctx(
        hints_by_proposal={
            "p1": ProposalHints(
                claim_hints={"c1": {"authoritative_value": "0.50"}},
                pm={"recommend": False},
            )
        }
    )
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["C"],
        config_name="C",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.tool_results[0].outputs_summary["flagged"] is True
    assert outcome.approval_family is False
    assert outcome.ep_numerical_error is False
    # P2-03 (v2.0.0 code review): the raw, ungated diagnostic still shows
    # the tool caught the problem -- the system's own mechanism worked,
    # even though the final (approval_family-gated) endpoint is False
    # because the decision was REJECT, not an approval.
    assert outcome.detected_numerical_error is True


def test_config_b_detected_numerical_error_stays_false_no_tool_at_all():
    """B has no reconciliation tool (config.deterministic_tools is False),
    so detected_numerical_error must stay at its inapplicable-default False
    -- never accidentally True just because a claim's stated/authoritative
    values would have diverged had the tool run."""
    proposal = make_proposal(stated_value=Decimal("0.15"))
    ctx = _ctx(
        hints_by_proposal={
            "p1": ProposalHints(claim_hints={"c1": {"authoritative_value": "0.50"}}, pm={"recommend": True})
        }
    )
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["B"],
        config_name="B",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.tool_results == ()
    assert outcome.detected_numerical_error is False


# ---------------------------------------------------------------------------
# D1/D2: + Treasury, real FTP + ledger feasibility, in-graph
# ep_state_infeasible / ep_pricing_inconsistent.
# ---------------------------------------------------------------------------
def test_config_d1_state_infeasible_blocks_execution_and_sets_endpoint():
    footprint = make_footprint(
        capital_usage=Decimal("5000")
    )  # exceeds ledger capital_available=1000
    proposal = make_proposal(resource_footprint=footprint)
    ctx = _ctx(hints_by_proposal={"p1": ProposalHints(treasury={"suggested_action": "APPROVE"})})
    state = make_ledger_state(capital_available=Decimal("1000"))
    outcome = run_proposal(
        proposal=proposal,
        state=state,
        config=CONFIG_TABLE["D1"],
        config_name="D1",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.ledger_stale is False
    assert "capital" in outcome.ledger_violations
    assert outcome.approval_family is True
    assert outcome.ep_state_infeasible is True
    assert outcome.detected_state_infeasible is True  # P2-03: same raw signal, approval-family case
    assert outcome.executed is False
    # apply() must never be called speculatively: state is unchanged.
    assert outcome.state_out == state


def test_config_d1_detected_state_infeasible_true_even_when_rejected():
    """P2-03 regression: the ledger-feasibility check runs regardless of
    Treasury's own action, so an infeasible proposal that Treasury itself
    REJECTs must still show detected_state_infeasible=True (the mechanism
    genuinely caught the problem) alongside ep_state_infeasible=False (not
    an endpoint: nothing was let through) -- proving the diagnostic is
    genuinely ungated by approval_family, not just an alias for ep_*."""
    footprint = make_footprint(capital_usage=Decimal("5000"))  # exceeds capital_available=1000
    proposal = make_proposal(resource_footprint=footprint)
    ctx = _ctx(hints_by_proposal={"p1": ProposalHints(treasury={"suggested_action": "REJECT"})})
    state = make_ledger_state(capital_available=Decimal("1000"))
    outcome = run_proposal(
        proposal=proposal,
        state=state,
        config=CONFIG_TABLE["D1"],
        config_name="D1",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.approval_family is False
    assert outcome.ep_state_infeasible is False
    assert outcome.detected_state_infeasible is True


def test_config_d1_pricing_inconsistent_detected_with_zero_addons():
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.05"))
    ctx = _ctx(
        hints_by_proposal={
            "p1": ProposalHints(
                trade={
                    "trade_alpha_horizon": "0.02"
                },  # below the zero-addon FTP horizon at 5% funding
                treasury={"suggested_action": "APPROVE"},
            )
        }
    )
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D1"],
        config_name="D1",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.ftp_output is not None
    assert outcome.ftp_output.economic_alpha_horizon <= 0
    assert outcome.approval_family is True
    assert outcome.ep_pricing_inconsistent is True
    assert outcome.detected_pricing_inconsistent is True  # P2-03: same raw signal, approval-family case
    assert outcome.executed is True  # D1 has no policy engine; ledger-feasible approval executes
    # P1-7: real execution, but D1 has no IC node/policy engine at all, so
    # it structurally cannot be a *dually authorised* one (invariant 5).
    assert outcome.dual_authorised is False
    assert outcome.revised_margin is None  # bare APPROVE never triggers Section 9.6's recomputation


# ---------------------------------------------------------------------------
# Section 9.6: APPROVE_WITH_LIMIT/HEDGE trigger a deterministic recomputation
# before the pricing question is settled — the fix for the gap flagged in
# the M2 status report (TreasuryAction.is_approval_family's own docstring
# used to point at this as unimplemented).
# ---------------------------------------------------------------------------
def _revision_ctx(*, action: str, trade_alpha_horizon: str) -> GraphContext:
    return _ctx(
        hints_by_proposal={
            "p1": ProposalHints(
                # tau_days must match make_proposal's own tau_days=365 below:
                # trade_node's fake output otherwise defaults tau_days to
                # 180 regardless of the proposal's own tau_days, which would
                # break the tau_days == day_count_basis identity the
                # hand-computed margins in these tests rely on.
                trade={"trade_alpha_horizon": trade_alpha_horizon, "tau_days": "365"},
                treasury={"suggested_action": action},
            )
        }
    )


def test_approve_with_hedge_revised_margin_flips_pricing_consistent():
    """base_funding_curve_rate_ann=0, tau_days=day_count_basis=365 keeps the
    horizon/annual conversion an identity, so the arithmetic below is exact:
    NORMAL's midpoint addons sum to 0.00375; trade_alpha=0.00365 sits below
    that (margin -0.0001) but above the RP-halved revised rate 0.00350
    (revised margin +0.00015) — the hedge's own risk-premium offset is what
    turns a negative shown margin into a positive one."""
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    ctx = _revision_ctx(action="APPROVE_WITH_HEDGE", trade_alpha_horizon="0.00365")
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D2"],
        config_name="D2",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.ftp_output is not None
    assert outcome.ftp_output.economic_alpha_horizon < 0  # the shown/unrevised margin
    assert outcome.revised_margin is not None
    assert outcome.revised_margin > 0
    assert outcome.approval_family is True  # still an approval, hedge or not
    assert outcome.ep_pricing_inconsistent is False  # graded on the revised margin
    assert (
        outcome.executed is True
    )  # D2 has no policy engine; ledger-feasible + revised-consistent executes
    assert outcome.dual_authorised is False  # P1-7: D2 can never clear invariant 5


def test_approve_with_hedge_still_pricing_inconsistent_when_revision_insufficient():
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    ctx = _revision_ctx(action="APPROVE_WITH_HEDGE", trade_alpha_horizon="0.00345")
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D2"],
        config_name="D2",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.revised_margin is not None
    assert outcome.revised_margin < 0  # the hedge helps, but not enough
    assert outcome.ep_pricing_inconsistent is True
    # D2 has no policy engine/control-clearance gate at all (only E does) —
    # a ledger-feasible approval executes regardless of pricing, so this
    # endpoint is purely diagnostic here; test_config_e_control_clearance_
    # uses_revised_margin_for_approve_with_hedge below covers the one
    # config where the revised margin actually gates execution.
    assert outcome.executed is True
    assert outcome.dual_authorised is False  # P1-7


def test_approve_with_limit_revised_margin_flips_pricing_consistent():
    """Same NORMAL-package arithmetic, but APPROVE_WITH_LIMIT offsets CP+CC
    instead of RP (0.00175 halved to 0.000875, a bigger drop than the hedge
    case's RP-only offset), so a lower trade_alpha still flips positive."""
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    ctx = _revision_ctx(action="APPROVE_WITH_LIMIT", trade_alpha_horizon="0.003")
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D2"],
        config_name="D2",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.ftp_output is not None
    assert outcome.ftp_output.economic_alpha_horizon < 0
    assert outcome.revised_margin is not None
    assert outcome.revised_margin > 0
    assert outcome.ep_pricing_inconsistent is False
    assert outcome.executed is True
    assert outcome.dual_authorised is False  # P1-7: D2 again, not E


def test_config_e_control_clearance_uses_revised_margin_for_approve_with_hedge():
    """Invariant 5's CONTROL_CLEARANCE must see the same repriced deal that
    would actually execute, not the pre-revision figure Treasury originally
    saw — otherwise a well-behaved hedge that fixes the pricing would still
    be blocked on stale economics."""
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    # Built directly rather than via _e_ctx: that helper hard-codes
    # treasury={"suggested_action": "APPROVE"} and can't be overridden
    # through its **hint_overrides (a duplicate-keyword TypeError), since
    # every other E test wants plain APPROVE.
    hints = ProposalHints(
        claim_hints={"c1": {"tool_flagged": False}},
        treasury={"suggested_action": "APPROVE_WITH_HEDGE"},
        ic={"business_approval": True},
        trade={"trade_alpha_horizon": "0.00365", "tau_days": "365"},
    )
    ctx = _ctx(policy_rules=(make_policy_rule(),), hints_by_proposal={"p1": hints})
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.ftp_output is not None
    assert outcome.ftp_output.economic_alpha_horizon < 0
    assert outcome.revised_margin is not None and outcome.revised_margin > 0
    assert outcome.control_clearance is True
    assert outcome.executed is True
    assert outcome.dual_authorised is True  # P1-7: E, and it actually executed


def test_config_e_control_clearance_still_blocks_when_revision_insufficient():
    """The E-config mirror of the D2 'insufficient hedge' case above: here
    CONTROL_CLEARANCE (and therefore invariant 5's execution conjunction)
    DOES gate on the revised margin, unlike D1/D2's unconditional
    ledger-feasible execution."""
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0"), tau_days=Decimal("365"))
    hints = ProposalHints(
        claim_hints={"c1": {"tool_flagged": False}},
        treasury={"suggested_action": "APPROVE_WITH_HEDGE"},
        ic={"business_approval": True},
        trade={"trade_alpha_horizon": "0.00345", "tau_days": "365"},
    )
    ctx = _ctx(policy_rules=(make_policy_rule(),), hints_by_proposal={"p1": hints})
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.revised_margin is not None and outcome.revised_margin < 0
    assert outcome.control_clearance is False
    assert outcome.executed is False
    assert outcome.dual_authorised is False  # P1-7: E, but the gate failed


def test_d1_masks_a_pricing_problem_that_d2_catches():
    """The pricing-placebo contrast this research design exists to measure:
    with the SAME trade economics, D1's forced-zero addons under-detect
    relative to D2's real addon vector."""
    base_rate = Decimal("0.01")
    tau_days = Decimal("180")
    day_count = Decimal("365")
    d1_ftp = compute_ftp(
        FTPInputs(
            tau_days=tau_days,
            day_count_basis=day_count,
            base_funding_curve_rate_ann=base_rate,
            addons=FTPAddonVector(),
            trade_alpha_horizon=Decimal("0"),
        )
    )
    d2_ftp = compute_ftp(
        FTPInputs(
            tau_days=tau_days,
            day_count_basis=day_count,
            base_funding_curve_rate_ann=base_rate,
            addons=_midpoint_addons(PACKAGES["NORMAL"]),
            trade_alpha_horizon=Decimal("0"),
        )
    )
    d1_horizon = d1_ftp.ftp_horizon
    d2_horizon = d2_ftp.ftp_horizon
    assert d2_horizon > d1_horizon  # addons strictly increase the true cost
    trade_alpha = (d1_horizon + d2_horizon) / 2  # sits between the two thresholds

    def _run(config_name: str) -> bool:
        proposal = make_proposal(base_funding_curve_rate_ann=base_rate, tau_days=tau_days)
        ctx = _ctx(
            hints_by_proposal={
                "p1": ProposalHints(
                    trade={"trade_alpha_horizon": str(trade_alpha)},
                    treasury={"suggested_action": "APPROVE"},
                )
            }
        )
        outcome = run_proposal(
            proposal=proposal,
            state=make_ledger_state(),
            config=CONFIG_TABLE[config_name],  # type: ignore[index]
            config_name=config_name,  # type: ignore[arg-type]
            ctx=ctx,
            case_id="case-1",
            replicate=0,
            proposal_position=0,
        )
        assert outcome.approval_family is True
        return outcome.ep_pricing_inconsistent

    assert _run("D1") is False  # masked: D1's own math cannot see the addon cost
    assert _run("D2") is True  # caught: D2 computes the full vector itself


# ---------------------------------------------------------------------------
# Config E: + IC, policy engine, dual authorisation (invariant 5).
# ---------------------------------------------------------------------------
def _e_ctx(*, unsupported_claim_flag: bool, **hint_overrides: object) -> GraphContext:
    hints = ProposalHints(
        claim_hints={"c1": {"tool_flagged": unsupported_claim_flag}},
        treasury={"suggested_action": "APPROVE"},
        ic={"business_approval": True},
        **hint_overrides,  # type: ignore[arg-type]
    )
    return _ctx(
        policy_rules=(make_policy_rule(),),
        hints_by_proposal={"p1": hints},
    )


def test_config_e_governance_hard_fail_blocks_execution():
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.01"))
    ctx = _e_ctx(unsupported_claim_flag=True, trade={"trade_alpha_horizon": "0.05"})
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.policy_result is not None
    assert outcome.policy_result.verdict == "HARD_FAIL"
    assert outcome.business_approval is True
    assert outcome.control_clearance is True
    assert outcome.policy_pass is False
    # Invariant 5: execution requires the full conjunction.
    assert outcome.executed is False
    assert outcome.dual_authorised is False  # P1-7
    assert outcome.ep_governance_failure is True
    assert outcome.detected_governance_failure is True  # P2-03: same raw signal, approval-family case
    assert outcome.state_out == make_ledger_state()  # apply() never called


def test_config_e_detected_governance_failure_true_even_when_rejected():
    """P2-03 regression, E's variant: the policy engine evaluates
    regardless of Treasury's own action, so a HARD_FAIL case that Treasury
    itself REJECTs must still show detected_governance_failure=True (the
    engine genuinely caught it) alongside ep_governance_failure=False (not
    an endpoint: nothing was let through).

    Built directly rather than via _e_ctx (as test_config_e_control_
    clearance_uses_revised_margin_for_approve_with_hedge does above): that
    helper hard-codes treasury={"suggested_action": "APPROVE"} and can't be
    overridden through its **hint_overrides (a duplicate-keyword TypeError).
    """
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.01"))
    hints = ProposalHints(
        claim_hints={"c1": {"tool_flagged": True}},
        treasury={"suggested_action": "REJECT"},
        ic={"business_approval": True},
        trade={"trade_alpha_horizon": "0.05"},
    )
    ctx = _ctx(policy_rules=(make_policy_rule(),), hints_by_proposal={"p1": hints})
    outcome = run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.policy_result is not None
    assert outcome.policy_result.verdict == "HARD_FAIL"
    assert outcome.approval_family is False
    assert outcome.ep_governance_failure is False
    assert outcome.detected_governance_failure is True


def test_config_e_clean_case_executes_and_increments_ledger_version():
    proposal = make_proposal(base_funding_curve_rate_ann=Decimal("0.01"))
    ctx = _e_ctx(unsupported_claim_flag=False, trade={"trade_alpha_horizon": "0.05"})
    state = make_ledger_state()
    outcome = run_proposal(
        proposal=proposal,
        state=state,
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.policy_result is not None and outcome.policy_result.verdict == "PASS"
    assert outcome.business_approval is True
    assert outcome.control_clearance is True
    assert outcome.policy_pass is True
    assert outcome.executed is True
    assert outcome.dual_authorised is True  # P1-7
    assert outcome.ep_governance_failure is False
    assert outcome.state_out is not None
    assert outcome.state_out.version == state.version + 1


def test_config_e_clean_case_executes_identically_for_a_fixed_income_proposal():
    """Equity is the primary input, but fixed income must be pluggable
    through the exact same architecture (Section 4/graph-agnosticism
    requirement) -- the only difference from the equity test directly
    above is asset_class; every other input, hint, and assertion is
    identical, proving no node branches on it."""
    proposal = make_proposal(
        base_funding_curve_rate_ann=Decimal("0.01"), asset_class="fixed_income"
    )
    assert proposal.instrument.asset_class == "fixed_income"
    ctx = _e_ctx(unsupported_claim_flag=False, trade={"trade_alpha_horizon": "0.05"})
    state = make_ledger_state()
    outcome = run_proposal(
        proposal=proposal,
        state=state,
        config=CONFIG_TABLE["E"],
        config_name="E",
        ctx=ctx,
        case_id="case-1",
        replicate=0,
        proposal_position=0,
    )
    assert outcome.policy_result is not None and outcome.policy_result.verdict == "PASS"
    assert outcome.business_approval is True
    assert outcome.control_clearance is True
    assert outcome.policy_pass is True
    assert outcome.executed is True
    assert outcome.dual_authorised is True  # P1-7
    assert outcome.ep_governance_failure is False
    assert outcome.state_out is not None
    assert outcome.state_out.version == state.version + 1


# ---------------------------------------------------------------------------
# P1-7 (code review): ``dual_authorised`` is the field invariant 5 and
# README's "execution ... only in config E" claim actually govern -- unlike
# ``executed`` (a real, consequential ledger transition D1/D2 can and do
# produce, see the tests above), no config other than E has the IC
# node/policy engine that could ever set it True. Locked in directly here,
# across every arm, rather than only inferred from the scattered per-config
# assertions above.
# ---------------------------------------------------------------------------
def test_only_config_e_can_ever_produce_a_dually_authorised_execution():
    """Runs every config in the maximally execution-favourable scenario
    each one supports (full approval, ledger-feasible, no governance
    problems) and checks ``dual_authorised``: True only for E, False (never
    None -- it is a plain bool, not tri-state like ``executed``) for every
    other arm, even though A/B/C/D1/D2 differ wildly in whether ``executed``
    itself ends up True, False or None."""
    footprint = make_footprint(capital_usage=Decimal("10"))  # trivially feasible
    state = make_ledger_state(capital_available=Decimal("1000"))

    def _run(config_name: str, **hint_kwargs: object) -> bool:
        proposal = make_proposal(resource_footprint=footprint)
        ctx = _ctx(hints_by_proposal={"p1": ProposalHints(**hint_kwargs)})
        outcome = run_proposal(
            proposal=proposal,
            state=state,
            config=CONFIG_TABLE[config_name],  # type: ignore[index]
            config_name=config_name,  # type: ignore[arg-type]
            ctx=ctx,
            case_id="case-dual-auth",
            replicate=0,
            proposal_position=0,
        )
        return outcome.dual_authorised

    assert _run("A", single_agent={"suggested_action": "APPROVE"}) is False
    assert _run("B", pm={"recommend": True}) is False
    assert _run("C", pm={"recommend": True}) is False
    assert _run("D1", treasury={"suggested_action": "APPROVE"}) is False
    assert _run("D2", treasury={"suggested_action": "APPROVE"}) is False

    e_outcome_dual_authorised = _run(
        "E",
        claim_hints={"c1": {"tool_flagged": False}},
        treasury={"suggested_action": "APPROVE"},
        ic={"business_approval": True},
    )
    assert e_outcome_dual_authorised is True


# ---------------------------------------------------------------------------
# SharedCallCache wiring: D1/D2 Research+Trade calls are shared; any other
# cross-config hit on those nodes must raise.
# ---------------------------------------------------------------------------
def test_shared_cache_hits_across_declared_d1_d2_pair():
    adapter = FakeLLMAdapter()
    cache = SharedCallCache()
    proposal = make_proposal()
    hints = {"p1": ProposalHints(treasury={"suggested_action": "APPROVE"})}

    ctx_d1 = _ctx(shared_cache=cache, hints_by_proposal=hints, adapter=adapter)
    run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D1"],
        config_name="D1",
        ctx=ctx_d1,
        case_id="case-shared",
        replicate=0,
        proposal_position=0,
    )
    misses_after_d1 = cache.misses
    assert misses_after_d1 == 2  # research (1 claim) + trade

    ctx_d2 = _ctx(shared_cache=cache, hints_by_proposal=hints, adapter=adapter)
    run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["D2"],
        config_name="D2",
        ctx=ctx_d2,
        case_id="case-shared",
        replicate=0,
        proposal_position=0,
    )
    assert cache.hits == 2  # research + trade reused verbatim from D1
    assert cache.misses == misses_after_d1  # no new upstream calls for D2


def test_shared_cache_raises_for_non_d1_d2_cross_config_reuse():
    adapter = FakeLLMAdapter()
    cache = SharedCallCache()
    proposal = make_proposal()
    hints = {"p1": ProposalHints(pm={"recommend": True})}

    ctx_b = _ctx(shared_cache=cache, hints_by_proposal=hints, adapter=adapter)
    run_proposal(
        proposal=proposal,
        state=make_ledger_state(),
        config=CONFIG_TABLE["B"],
        config_name="B",
        ctx=ctx_b,
        case_id="case-undeclared",
        replicate=0,
        proposal_position=0,
    )

    ctx_d1 = _ctx(
        shared_cache=cache,
        hints_by_proposal={"p1": ProposalHints(treasury={"suggested_action": "APPROVE"})},
        adapter=adapter,
    )
    with pytest.raises(UndeclaredCacheHitError):
        run_proposal(
            proposal=proposal,
            state=make_ledger_state(),
            config=CONFIG_TABLE["D1"],
            config_name="D1",
            ctx=ctx_d1,
            case_id="case-undeclared",
            replicate=0,
            proposal_position=0,
        )


# ---------------------------------------------------------------------------
# run_case: state threading across an episode's proposals.
# ---------------------------------------------------------------------------
def test_run_case_carries_ledger_state_across_proposals():
    footprint = make_footprint(capital_usage=Decimal("100"))
    p1 = make_proposal(proposal_id="p1", resource_footprint=footprint)
    p2 = make_proposal(
        proposal_id="p2", resource_footprint=footprint.model_copy(update={"state_version_seen": 1})
    )
    case = make_case(proposals=(p1, p2), case_id="episode-1")
    ctx = _ctx(
        hints_by_proposal={
            "p1": ProposalHints(treasury={"suggested_action": "APPROVE"}),
            "p2": ProposalHints(treasury={"suggested_action": "APPROVE"}),
        }
    )
    outcome = run_case(case=case, config=CONFIG_TABLE["D1"], config_name="D1", ctx=ctx, replicate=0)
    assert len(outcome.proposals) == 2
    first, second = outcome.proposals
    assert first.executed is True
    assert first.state_out is not None and first.state_out.version == 1
    # p2's footprint declares state_version_seen=1, matching the state p1 left behind.
    assert second.executed is True
    assert outcome.final_state is not None and outcome.final_state.version == 2
    assert outcome.final_proposal is second
