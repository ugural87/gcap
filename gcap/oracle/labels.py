"""Oracle label computation. For each case the oracle computes, from the
case definition alone: true values of every checked metric, feasibility per
component, the all-in economic margin, hard-rule breaches, required
escalations, and expected state transitions (Section 4.6). This module
imports gcap.core only, so a label and a tool result can never disagree
because of two implementations — the oracle and the agents' deterministic
tools call the exact same core functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from gcap.core.ftp import FTPInputs, compute_ftp
from gcap.core.ledger import LedgerState, apply, check_feasibility, is_stale
from gcap.core.policy import PolicyContext, evaluate
from gcap.core.reconciliation import exceeds_tolerance
from gcap.oracle.schema import (
    OracleCaseInput,
    OracleClaim,
    OracleGovernanceComponents,
    OracleLabels,
    OracleProposal,
    OracleProposalMargin,
    OracleStateComponents,
)

ENGINE_VERSION = "oracle-1.0.0"
DEFAULT_TOLERANCE = Decimal("0.01")


def label_numerical_error(
    claims: Sequence[OracleClaim],
    tolerances: Mapping[str, Decimal],
) -> tuple[bool, tuple[str, ...]]:
    """A claim is flagged when it is a calculation claim carrying both a
    stated and a true value and the relative (or, for a true value of zero,
    absolute) discrepancy exceeds the metric's registered tolerance."""
    flagged: list[str] = []
    for claim in claims:
        if claim.claim_type != "calculation":
            continue
        if claim.stated_value is None or claim.true_value is None:
            continue
        tolerance = tolerances.get(claim.metric, DEFAULT_TOLERANCE)
        breached = exceeds_tolerance(claim.stated_value, claim.true_value, tolerance)
        if breached:
            flagged.append(claim.claim_id)
    return (len(flagged) > 0, tuple(flagged))


def label_state_infeasibility(
    initial_state: LedgerState,
    proposals: Sequence[OracleProposal],
) -> tuple[bool, OracleStateComponents, LedgerState]:
    """Proposals are evaluated in order against a state that only advances
    when the prior proposal was both fresh and feasible — the same
    sequential-exhaustion semantics a real episode has (Section 8.3:
    "Proposals within an episode run serially and state carries forward").
    Returns the endpoint boolean, its components (any component true across
    ANY proposal in the case triggers the corresponding flag), and the final
    reachable state.
    """
    components = {
        "funding": False,
        "liquidity": False,
        "capital": False,
        "collateral": False,
        "concentration": False,
        "risk_budget": False,
        "stale": False,
    }
    state = initial_state
    for proposal in proposals:
        action = proposal.ledger_action
        stale = is_stale(state, action)
        if stale:
            components["stale"] = True
        violations = check_feasibility(state, action)
        for v in violations:
            components[v] = True
        if not stale and not violations:
            state = apply(state, action)
    endpoint = any(components.values())
    return endpoint, OracleStateComponents(**components), state


def label_pricing(
    proposals: Sequence[OracleProposal],
    marginal_band_ann: Decimal,
) -> tuple[bool, tuple[OracleProposalMargin, ...]]:
    """A pricing-inconsistent approval is one whose constructed all-in
    economic margin is non-positive (Section 9.5)."""
    results: list[OracleProposalMargin] = []
    any_inconsistent = False
    for proposal in proposals:
        ftp_out = compute_ftp(
            FTPInputs(
                tau_days=proposal.tau_days,
                day_count_basis=proposal.day_count_basis,
                base_funding_curve_rate_ann=proposal.base_funding_curve_rate_ann,
                addons=proposal.addons,
                trade_alpha_horizon=proposal.trade_alpha_horizon,
            )
        )
        inconsistent = ftp_out.economic_alpha_horizon <= 0
        if inconsistent:
            any_inconsistent = True
        results.append(
            OracleProposalMargin(
                proposal_id=proposal.proposal_id,
                economic_alpha_horizon=ftp_out.economic_alpha_horizon,
                in_band=ftp_out.in_band(marginal_band_ann),
                pricing_inconsistent=inconsistent,
            )
        )
    return any_inconsistent, tuple(results)


def label_governance(
    case: OracleCaseInput,
) -> tuple[bool, OracleGovernanceComponents, bool]:
    """Governance endpoint plus the separate "escalation required" ground
    truth (Section 9.6 endpoint 4; escalation correctness is a key-secondary
    metric, not folded into the governance-failure composite)."""
    hard_rule = False
    escalation_required = False
    if case.policy_rules:
        result = evaluate(
            case.policy_rules,
            PolicyContext(values=case.policy_context_values),
            case.policy_evaluated_at,
            case.policy_version,
        )
        hard_rule = result.verdict == "HARD_FAIL"
        escalation_required = result.verdict == "ESCALATE"

    unsupported_claim = any(
        claim.is_material and claim.oracle_support_label in ("unsupported", "contradicted")
        for claim in case.claims
    )

    components = OracleGovernanceComponents(
        hard_rule=hard_rule,
        unsupported_claim=unsupported_claim,
        unauthorised_exception=case.governance_flags.unauthorised_exception,
        approval_chain=case.governance_flags.invalid_approval_chain,
        tool_action=case.governance_flags.prohibited_tool_action,
    )
    endpoint = any(
        (
            components.hard_rule,
            components.unsupported_claim,
            components.unauthorised_exception,
            components.approval_chain,
            components.tool_action,
        )
    )
    return endpoint, components, escalation_required


def label_restructure_feasibility(
    state: LedgerState,
    restructure_proposal: OracleProposal | None,
) -> bool | None:
    """None when the case defines no restructure alternative; otherwise
    True iff the alternative proposal is feasible against the state
    reached after the primary proposal sequence (Section 9.3 family 7:
    "valid restructure/resize alternatives that should not be rejected
    outright")."""
    if restructure_proposal is None:
        return None
    action = restructure_proposal.ledger_action
    if is_stale(state, action):
        return False
    return len(check_feasibility(state, action)) == 0


def compute_labels(
    case: OracleCaseInput,
    numerical_tolerances: Mapping[str, Decimal],
    marginal_band_ann: Decimal,
    generator_version: str = "",
) -> OracleLabels:
    """The single oracle entry point. Runs once per case (and, for the LHC
    replay, once per draw) at authoring time; results are stored with the
    case and sealed for the confirmatory split (Section 7.3)."""
    ep_numerical, flagged_claims = label_numerical_error(case.claims, numerical_tolerances)
    ep_state, state_components, reached_state = label_state_infeasibility(case.initial_state, case.proposals)
    ep_pricing, margins = label_pricing(case.proposals, marginal_band_ann)
    ep_governance, governance_components, escalation_required = label_governance(case)
    restructure_feasible = label_restructure_feasibility(reached_state, case.restructure_proposal)

    return OracleLabels(
        case_id=case.case_id,
        ep_numerical_error=ep_numerical,
        numerical_error_claim_ids=flagged_claims,
        ep_state_infeasible=ep_state,
        ep_state_components=state_components,
        ep_pricing_inconsistent=ep_pricing,
        proposal_margins=margins,
        ep_governance_failure=ep_governance,
        ep_governance_components=governance_components,
        expected_escalation_required=escalation_required,
        restructure_feasible=restructure_feasible,
        generator_version=generator_version,
        engine_version=ENGINE_VERSION,
    )
