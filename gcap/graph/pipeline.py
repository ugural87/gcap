"""The one graph codebase for all six configurations (Section 3: "All six
configurations run through one graph codebase; a configuration is a frozen
dataclass of flags, and nothing else may differ between arms"). Every
branch below is keyed off ``ConfigFlags`` fields — never off a
config-name string comparison — so a new arm added to CONFIG_TABLE gets
correct behaviour automatically from its flags alone.

Endpoint detection rule (this module's central design decision, since it
governs invariant 6 — "oracle independent of system under test" — at the
graph boundary): an endpoint is detected **in-graph**, from the system's
own deterministic tool/engine output, whenever the active config actually
has the relevant mechanism; RunRecord's ep_* fields are left at their
default (False) otherwise, and it is the HARNESS's job (never graph's —
graph never imports oracle, per the import-linter contract) to grade the
remainder post-hoc against oracle ground truth for a config that lacks the
mechanism:

  ep_numerical_error     in-graph iff config.deterministic_tools  (C, D1, D2, E)
  ep_state_infeasible    in-graph iff config.ledger               (D1, D2, E)
  ep_pricing_inconsistent in-graph iff config.treasury_node        (D1, D2, E)
  ep_governance_failure  in-graph iff config.policy_engine         (E)

P0-02 (v2.0.0 code review): E's in-graph ep_governance_failure below is,
and can only ever be, ``policy_result.verdict == "HARD_FAIL" and
approval_family`` -- ONE of the oracle's five governance components
(oracle.labels.label_governance's hard_rule/unsupported_claim/
unauthorised_exception/approval_chain/tool_action OR). The other four are
oracle-authored ground-truth flags (``BenchCase.governance_flags``,
``OracleClaim.oracle_support_label``) that invariant 6 and the
import-linter contract forbid this module from ever importing or seeing,
so E structurally cannot detect them in-graph -- there is no amount of
policy-rule authoring inside this module that changes that, since the
information itself is on the wrong side of the oracle/system boundary.
harness.grading.grade_case_outcome's oracle-graded branch is narrowed to
the SAME single hard_rule component for exactly this reason -- see its
module docstring for the full rationale and what would be needed to widen
this honestly (new in-graph detection mechanisms fed by graph-visible
signals, not oracle flags, plus matching bench-generator wiring, which do
not exist yet for unauthorised_exception/approval_chain/tool_action).

Each in-graph rule follows the same shape: "the system's own check flagged
a problem AND the final decision proceeded (approval-family) anyway" — the
system had the means to catch it and didn't. This is why D1 (FTP addons
forced to zero) is expected to show a near-zero in-graph
ep_pricing_inconsistent rate even for cases whose TRUE full-addon economics
are negative: D1's own math genuinely cannot see the masked cost. That
gap between D1 and D2's in-graph pricing detection is not a bug — it is
the pricing-placebo contrast the research design exists to measure.

P1-7 (code review): ``executed`` (ProposalOutcome.executed) is a genuine,
consequential ledger transition in D1, D2 AND E alike -- core.ledger.apply()
really runs, state_out really differs from state, and a later proposal in
the same episode really sees the result (capital/liquidity/etc. actually
used up). D1/D2's execution gate is single-gate: ``approval_family and
ledger_ok``, with NO dual-authorisation mechanism at all (no IC node, no
policy engine in those configs) -- so it is not "fake"/"simulated" in the
sense of having no effect, but it also never clears invariant 5's bar,
because D1/D2 structurally cannot: there is no IC vote or policy result to
even evaluate. Invariant 5 (execution = BUSINESS_APPROVAL AND
CONTROL_CLEARANCE AND POLICY_PASS, via core.validator
.validate_integrated_package) is what ``dual_authorised`` tracks, and THAT
field -- not ``executed`` -- is the one that can only be True in the E
branch: D1/D2 always pass ``dual_authorised=False`` alongside whatever
``executed`` value ledger-feasibility produced. This is the fix for a
real prior documentation defect (this module's own docstring, and
README's invariant-5 bullet, used to describe ``executed`` itself as
E-only, which the code never actually enforced -- see
graph/outcomes.py's field-level docstrings for the full rationale).

core.ledger.apply() is never called speculatively in either branch — an
infeasible/stale action is always caught before apply() would raise, and
recorded as a non-executed outcome (invariant 9: failures are recorded
outcomes, never silent).

P2-03 (v2.0.0 code review): alongside every ep_* field above, this module
now also populates the matching ``detected_*`` field on ProposalOutcome —
the SAME underlying tool/ledger/pricing/policy verdict, but read BEFORE the
``and approval_family`` gate rather than after it. ep_* answers "did the
system's own mechanism catch this AND let it through anyway" (the research
question); detected_* answers only "did the system's own mechanism notice
the problem at all", regardless of what the final decision did with it.
Before this existed, harness.grading's oracle-vs-in-graph ``detected_*``
diagnostic had nothing ungated to read for an in-graph-graded config and
fell back to ep_* itself — silently comparing two different things
depending on which half of the D2/E (or B/C) pair produced the number. See
graph/outcomes.py's ProposalOutcome docstring and harness/grading.py's
module docstring for the full before/after.
"""

from __future__ import annotations

from decimal import Decimal

from gcap.agents.nodes import (
    ic_node,
    pm_node,
    research_node,
    single_agent_node,
    trade_node,
    treasury_node,
)
from gcap.agents.tools import ftp_tool, ledger_feasibility_tool, reconciliation_tool
from gcap.agents.treasury import TreasuryAction
from gcap.agents.view import AgentVisibleCase, AgentVisibleProposal
from gcap.config.flags import ConfigFlags, ConfigName
from gcap.core.alco import PACKAGES, AlcoPackage
from gcap.core.ftp import FTPAddonVector, FTPInputs
from gcap.core.ledger import LedgerAction, LedgerState, apply
from gcap.core.policy import PolicyContext, evaluate
from gcap.core.validator import validate_integrated_package
from gcap.graph.context import GraphContext
from gcap.graph.outcomes import CaseOutcome, ProposalOutcome


def _midpoint_addons(package: AlcoPackage) -> FTPAddonVector:
    def mid(name: str) -> Decimal:
        lo, hi = package.ftp_addon_ranges[name]
        return (lo + hi) / 2

    return FTPAddonVector(lp=mid("lp"), tp=mid("tp"), cp=mid("cp"), rp=mid("rp"), cc=mid("cc"))


def _revised_addons(action: TreasuryAction, addons: FTPAddonVector, ctx: GraphContext) -> FTPAddonVector | None:
    """Section 9.6's deterministic recomputation, required for
    APPROVE_WITH_LIMIT/HEDGE before the pricing-consistency question is
    settled for that proposal — never for a bare APPROVE (nothing about
    the deal changed, so there is nothing to recompute) and never for
    RESTRUCTURE/DEFER/REJECT (not approval-family to begin with). Returns
    None when the action does not trigger a revision at all.

    Modelling choice (documented in full on
    config.registry.RevisedMarginRecomputation, the to-decide parameter
    this reads from ``ctx``): APPROVE_WITH_HEDGE offsets the residual-risk
    addon (RP) a hedge is meant to reduce; APPROVE_WITH_LIMIT offsets the
    capital-consumption and concentration addons (CP+CC) a size/exposure
    limit is meant to reduce. Neither ever touches LP or TP, which reflect
    funding-liquidity and tenor properties the limit/hedge does not alter.
    """
    if action == TreasuryAction.APPROVE_WITH_HEDGE:
        return FTPAddonVector(
            lp=addons.lp,
            tp=addons.tp,
            cp=addons.cp,
            rp=addons.rp * (Decimal("1") - ctx.hedge_rp_reduction_pct),
            cc=addons.cc,
        )
    if action == TreasuryAction.APPROVE_WITH_LIMIT:
        factor = Decimal("1") - ctx.limit_addon_reduction_pct
        return FTPAddonVector(lp=addons.lp, tp=addons.tp, cp=addons.cp * factor, rp=addons.rp, cc=addons.cc * factor)
    return None


def run_proposal(
    *,
    proposal: AgentVisibleProposal,
    state: LedgerState,
    config: ConfigFlags,
    config_name: ConfigName,
    ctx: GraphContext,
    case_id: str,
    replicate: int,
    proposal_position: int,
) -> ProposalOutcome:
    hints = ctx.hints_for(proposal.proposal_id)
    node_calls = []
    tool_results = []

    # ---- Config A: one generalist LLM, no decomposition, no tools, no
    # ledger, no policy. Every endpoint is harness-graded. ----
    if not config.multi_agent:
        decision, call = single_agent_node(
            adapter=ctx.adapter, proposal=proposal, token_budget=ctx.token_budget, metadata=hints.single_agent
        )
        node_calls.append(call)
        return ProposalOutcome(
            proposal_id=proposal.proposal_id,
            proposal_position=proposal_position,
            single_agent=decision,
            final_action=decision.treasury_action,
            approval_family=decision.treasury_action.is_approval_family,
            node_calls=tuple(node_calls),
            state_out=state,
        )

    # ---- B/C/D1/D2/E: Research -> Trade -> PM, with a deterministic
    # reconciliation tool from C onward. ----
    metadata_by_claim = {cid: dict(h) for cid, h in hints.claim_hints.items()}
    ep_numerical_error = False
    if config.deterministic_tools:
        for claim in proposal.claims:
            claim_hint = hints.claim_hints.get(claim.claim_id, {})
            authoritative_value = claim_hint.get("authoritative_value")
            if authoritative_value is None or claim.stated_value is None:
                continue
            tolerance = ctx.numerical_tolerances.get(claim.metric, ctx.default_tolerance)
            tool_result = reconciliation_tool(
                claim_id=claim.claim_id,
                metric=claim.metric,
                stated_value=claim.stated_value,
                authoritative_value=Decimal(str(authoritative_value)),
                tolerance=tolerance,
            )
            tool_results.append(tool_result)
            metadata_by_claim.setdefault(claim.claim_id, {})["tool_flagged"] = tool_result.outputs_summary["flagged"]

    claim_reviews, calls = research_node(
        adapter=ctx.adapter,
        proposal=proposal,
        token_budget=ctx.token_budget,
        metadata_by_claim=metadata_by_claim,
        shared_cache=ctx.shared_cache,
        config_name=config_name,
        case_id=case_id,
        replicate=replicate,
        manifest_hash=ctx.manifest_hash,
    )
    node_calls.extend(calls)

    any_material_claim_flagged_by_tool = any(
        tr.tool_name == "reconciliation" and tr.outputs_summary.get("flagged") for tr in tool_results
    )

    trade, call = trade_node(
        adapter=ctx.adapter,
        proposal=proposal,
        token_budget=ctx.token_budget,
        metadata=hints.trade,
        shared_cache=ctx.shared_cache,
        config_name=config_name,
        case_id=case_id,
        replicate=replicate,
        claim_reviews=claim_reviews,
        manifest_hash=ctx.manifest_hash,
        tenor_bounds=ctx.tenor_restructuring_bounds,
    )
    node_calls.append(call)

    pm, call = pm_node(
        adapter=ctx.adapter,
        proposal=proposal,
        token_budget=ctx.token_budget,
        metadata=hints.pm,
        claim_reviews=claim_reviews,
        trade=trade,
    )
    node_calls.append(call)

    unsupported_claim_present = any(
        review.flagged for review, claim in zip(claim_reviews, proposal.claims, strict=True) if claim.is_material
    )

    # ---- B/C: no Treasury node. PM's recommendation is the final decision. ----
    if not config.treasury_node:
        final_action = TreasuryAction.APPROVE if pm.recommend else TreasuryAction.REJECT
        approval_family = final_action.is_approval_family
        if config.deterministic_tools:
            ep_numerical_error = any_material_claim_flagged_by_tool and approval_family
        # P2-03: raw, ungated detection. any_material_claim_flagged_by_tool
        # is itself already False whenever this config has no
        # reconciliation tool at all (tool_results stays empty above when
        # not config.deterministic_tools), so no extra guard is needed here
        # to preserve ep_numerical_error's own "left False when not
        # applicable" convention.
        detected_numerical_error = any_material_claim_flagged_by_tool
        return ProposalOutcome(
            proposal_id=proposal.proposal_id,
            proposal_position=proposal_position,
            claim_reviews=tuple(claim_reviews),
            tool_results=tuple(tool_results),
            trade=trade,
            pm=pm,
            final_action=final_action,
            approval_family=approval_family,
            ep_numerical_error=ep_numerical_error,
            detected_numerical_error=detected_numerical_error,
            node_calls=tuple(node_calls),
            state_out=state,
        )

    # ---- D1/D2/E: Treasury proposes an action; the graph computes FTP and
    # ledger feasibility for real, itself, never trusting Treasury's own
    # arithmetic (invariant 1: LLMs never compute). ----
    treasury_decision, call = treasury_node(
        adapter=ctx.adapter,
        proposal=proposal,
        token_budget=ctx.token_budget,
        metadata=hints.treasury,
        trade=trade,
        # P1-02 (v2.0.0 code review): PM's recommendation now reaches
        # Treasury's own prompt (agents.prompts.treasury_prompt), not just
        # this outcome's `pm` field -- see that function's docstring.
        pm=pm,
    )
    node_calls.append(call)

    addons = FTPAddonVector() if config.ftp_addons == "zero" else _midpoint_addons(PACKAGES[ctx.alco_package])
    ftp_inputs = FTPInputs(
        tau_days=trade.tau_days,
        day_count_basis=ctx.day_count_basis,
        base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann,
        addons=addons,
        trade_alpha_horizon=trade.trade_alpha_horizon,
    )
    ftp_tool_result, ftp_output = ftp_tool(ftp_inputs)
    tool_results.append(ftp_tool_result)

    # Section 9.6: APPROVE_WITH_LIMIT/HEDGE must trigger a deterministic
    # recomputation before the pricing question is settled for this
    # proposal. revised_margin is None (no recomputation ran) for a bare
    # APPROVE; pricing_margin is what every downstream pricing check below
    # actually consults — the revised figure when one exists, the original
    # ftp_output figure otherwise — since the revised deal, not the
    # pre-revision one, is what would actually go on to execute.
    revised_margin: Decimal | None = None
    pricing_ftp_output = ftp_output
    revised_addons = _revised_addons(treasury_decision.treasury_action, addons, ctx)
    if revised_addons is not None:
        revised_ftp_tool_result, revised_ftp_output = ftp_tool(
            FTPInputs(
                tau_days=ftp_inputs.tau_days,
                day_count_basis=ftp_inputs.day_count_basis,
                base_funding_curve_rate_ann=ftp_inputs.base_funding_curve_rate_ann,
                addons=revised_addons,
                trade_alpha_horizon=ftp_inputs.trade_alpha_horizon,
            )
        )
        tool_results.append(revised_ftp_tool_result)
        revised_margin = revised_ftp_output.economic_alpha_horizon
        pricing_ftp_output = revised_ftp_output
    pricing_margin = revised_margin if revised_margin is not None else ftp_output.economic_alpha_horizon

    footprint = proposal.resource_footprint
    action = LedgerAction(
        action_id=proposal.proposal_id,
        instrument_id=proposal.proposal_id,
        state_version_seen=footprint.state_version_seen,
        capital_usage=footprint.capital_usage,
        liquidity_usage=footprint.liquidity_usage,
        funding_usage=footprint.funding_usage,
        collateral_usage=footprint.collateral_usage,
        risk_budget_usage=footprint.risk_budget_usage,
        concentration_key=footprint.concentration_key,
        concentration_usage=footprint.concentration_usage,
        notional_delta=footprint.notional_delta,
    )
    ledger_tool_result, ledger_stale, ledger_violations = ledger_feasibility_tool(state, action)
    tool_results.append(ledger_tool_result)
    ledger_ok = (not ledger_stale) and (not ledger_violations)

    approval_family = treasury_decision.treasury_action.is_approval_family
    if config.deterministic_tools:
        ep_numerical_error = any_material_claim_flagged_by_tool and approval_family
    ep_state_infeasible = (not ledger_ok) and approval_family
    ep_pricing_inconsistent = (pricing_margin <= 0) and approval_family

    # P2-03: raw, ungated detection -- the same three underlying checks as
    # above, read before the "and approval_family" gate. ledger and pricing
    # checks always run in D1/D2/E (unlike the reconciliation tool, which is
    # config-gated), so these two are unconditionally meaningful here;
    # detected_numerical_error follows any_material_claim_flagged_by_tool's
    # own already-False-when-inapplicable convention exactly as in the B/C
    # branch above.
    detected_numerical_error = any_material_claim_flagged_by_tool
    detected_state_infeasible = not ledger_ok
    detected_pricing_inconsistent = pricing_margin <= 0

    # ---- D1/D2: no dual-authorisation gate. Treasury alone decides;
    # execution follows automatically whenever the deterministic checks
    # this config actually has (ledger only — no policy engine here)
    # allow it. Never call apply() on an infeasible/stale action: the
    # attempt is recorded as a non-executed outcome (invariant 9), not a
    # crash. ----
    if not config.policy_engine:
        executed = approval_family and ledger_ok
        state_out = apply(state, action) if executed else state
        return ProposalOutcome(
            proposal_id=proposal.proposal_id,
            proposal_position=proposal_position,
            claim_reviews=tuple(claim_reviews),
            tool_results=tuple(tool_results),
            trade=trade,
            pm=pm,
            treasury=treasury_decision,
            ftp_output=ftp_output,
            ledger_stale=ledger_stale,
            ledger_violations=ledger_violations,
            executed=executed,
            # P1-7: D1/D2 have no IC node/policy engine at all, so there is
            # structurally no way for this branch to ever clear invariant
            # 5's conjunction -- always False here, never derived from
            # ``executed`` (see this module's docstring).
            dual_authorised=False,
            final_action=treasury_decision.treasury_action,
            approval_family=approval_family,
            revised_margin=revised_margin,
            ep_numerical_error=ep_numerical_error,
            ep_state_infeasible=ep_state_infeasible,
            ep_pricing_inconsistent=ep_pricing_inconsistent,
            detected_numerical_error=detected_numerical_error,
            detected_state_infeasible=detected_state_infeasible,
            detected_pricing_inconsistent=detected_pricing_inconsistent,
            node_calls=tuple(node_calls),
            state_out=state_out,
        )

    # ---- E: + Investment Committee, policy engine, dual authorisation.
    # Invariant 5: execution = BUSINESS_APPROVAL AND CONTROL_CLEARANCE AND
    # POLICY_PASS. Nothing here is executed on anything less. ----
    ic_vote, call = ic_node(
        adapter=ctx.adapter,
        proposal=proposal,
        token_budget=ctx.token_budget,
        metadata=hints.ic,
        treasury=treasury_decision,
        pricing_margin=pricing_margin,
        pricing_in_band=pricing_ftp_output.in_band(ctx.marginal_band_ann) if pricing_ftp_output is not None else None,
        ledger_ok=ledger_ok,
        ledger_violations=tuple(str(v) for v in ledger_violations),
    )
    node_calls.append(call)

    policy_context = dict(ctx.policy_context_values)
    policy_context["unsupported_claim_present"] = unsupported_claim_present
    policy_context["state_is_stale"] = ledger_stale
    policy_result = evaluate(
        ctx.policy_rules, PolicyContext(values=policy_context), ctx.policy_evaluated_at, ctx.policy_version
    )

    business_approval = ic_vote.business_approval
    control_clearance = ledger_ok and (pricing_margin > 0)
    policy_pass = policy_result.verdict == "PASS"

    # The Integrated Validator (core.validator) is THE gate invariant 5
    # refers to: one named, independently testable function, not five
    # scattered booleans. business_approval/control_clearance/policy_pass
    # above are kept as their own ProposalOutcome fields (the graph's
    # existing per-component detail other code and tests already read),
    # but ``executed`` itself is derived from the validator's verdict, so
    # there is exactly one place that decides whether capital moves.
    integrated_validation = validate_integrated_package(
        ledger_stale=ledger_stale,
        ledger_violations=ledger_violations,
        pricing_ftp_output=pricing_ftp_output,
        pricing_margin=pricing_margin,
        business_approval=business_approval,
        policy_result=policy_result,
    )
    executed = approval_family and integrated_validation.approved

    ep_governance_failure = (policy_result.verdict == "HARD_FAIL") and approval_family
    # P2-03: raw, ungated detection -- E-only, mirroring ep_governance_failure
    # itself being E-only (the policy engine only ever runs here).
    detected_governance_failure = policy_result.verdict == "HARD_FAIL"

    state_out = apply(state, action) if executed else state

    return ProposalOutcome(
        proposal_id=proposal.proposal_id,
        proposal_position=proposal_position,
        claim_reviews=tuple(claim_reviews),
        tool_results=tuple(tool_results),
        trade=trade,
        pm=pm,
        treasury=treasury_decision,
        ic=ic_vote,
        ftp_output=ftp_output,
        ledger_stale=ledger_stale,
        ledger_violations=ledger_violations,
        policy_result=policy_result,
        business_approval=business_approval,
        control_clearance=control_clearance,
        policy_pass=policy_pass,
        executed=executed,
        # P1-7: in E, "executed" IS invariant 5's conjunction (see
        # ``executed = approval_family and integrated_validation.approved``
        # just above) -- so dual_authorised tracks it exactly here, the one
        # branch where the two concepts coincide.
        dual_authorised=executed,
        integrated_validation_reasons=integrated_validation.reasons,
        final_action=treasury_decision.treasury_action,
        approval_family=approval_family,
        revised_margin=revised_margin,
        ep_numerical_error=ep_numerical_error,
        ep_state_infeasible=ep_state_infeasible,
        ep_pricing_inconsistent=ep_pricing_inconsistent,
        ep_governance_failure=ep_governance_failure,
        detected_numerical_error=detected_numerical_error,
        detected_state_infeasible=detected_state_infeasible,
        detected_pricing_inconsistent=detected_pricing_inconsistent,
        detected_governance_failure=detected_governance_failure,
        node_calls=tuple(node_calls),
        state_out=state_out,
    )


def run_case(
    *,
    case: AgentVisibleCase,
    config: ConfigFlags,
    config_name: ConfigName,
    ctx: GraphContext,
    replicate: int,
) -> CaseOutcome:
    """Processes every proposal in the case in order, carrying ledger state
    forward across proposals exactly as an episode's sequential-exhaustion
    scenario requires (Section 8.3). A single-proposal case is simply an
    episode of length 1."""
    state = case.initial_state
    outcomes: list[ProposalOutcome] = []
    for position, proposal in enumerate(case.proposals):
        outcome = run_proposal(
            proposal=proposal,
            state=state,
            config=config,
            config_name=config_name,
            ctx=ctx,
            case_id=case.case_id,
            replicate=replicate,
            proposal_position=position,
        )
        outcomes.append(outcome)
        if outcome.state_out is not None:
            state = outcome.state_out
    return CaseOutcome(case_id=case.case_id, proposals=tuple(outcomes), final_state=state)
