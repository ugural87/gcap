from __future__ import annotations

from gcap.bench.generator import generate_slice
from gcap.harness.cases import authoritative_claim_hints, to_agent_visible_case


def test_to_agent_visible_case_preserves_identity_and_structure():
    cases = generate_slice(seed=3, split="dev", n_per_family=1)
    for bench_case in cases:
        agent_case = to_agent_visible_case(bench_case)
        assert agent_case.case_id == bench_case.case_id
        assert agent_case.family == bench_case.family
        assert agent_case.materiality == bench_case.materiality
        assert agent_case.kind == bench_case.kind
        assert agent_case.policy_version == bench_case.policy_version
        assert agent_case.initial_state == bench_case.initial_state
        assert len(agent_case.proposals) == len(bench_case.proposals)


def test_to_agent_visible_case_strips_ground_truth_fields():
    """AgentVisibleClaim/AgentVisibleProposal structurally have no
    true_value/oracle_support_label/trade_alpha_horizon/addons fields at
    all (the import-linter contract + these schemas make it impossible to
    construct one carrying them) — this test instead checks that the
    values that ARE carried over are the legitimately-visible ones, not
    silently substituted ground truth."""
    cases = generate_slice(seed=3, split="dev", n_per_family=1)
    numerical_error_case = next(c for c in cases if c.family == "numerical_reconciliation_error")
    agent_case = to_agent_visible_case(numerical_error_case)

    for bench_proposal, agent_proposal in zip(numerical_error_case.proposals, agent_case.proposals, strict=True):
        assert not hasattr(agent_proposal, "trade_alpha_horizon")
        assert not hasattr(agent_proposal, "addons")
        op = bench_proposal.oracle_proposal
        assert agent_proposal.tau_days == op.tau_days
        assert agent_proposal.day_count_basis == op.day_count_basis
        assert agent_proposal.base_funding_curve_rate_ann == op.base_funding_curve_rate_ann

        for bench_claim, agent_claim in zip(bench_proposal.claims, agent_proposal.claims, strict=True):
            assert not hasattr(agent_claim, "true_value")
            assert not hasattr(agent_claim, "oracle_support_label")
            assert agent_claim.claim_id == bench_claim.claim_id
            assert agent_claim.stated_value == bench_claim.stated_value  # stated, never true_value


def test_to_agent_visible_case_copies_resource_footprint_from_ledger_action():
    cases = generate_slice(seed=3, split="dev", n_per_family=1)
    bench_case = cases[0]
    agent_case = to_agent_visible_case(bench_case)
    for bench_proposal, agent_proposal in zip(bench_case.proposals, agent_case.proposals, strict=True):
        action = bench_proposal.oracle_proposal.ledger_action
        fp = agent_proposal.resource_footprint
        assert fp.state_version_seen == action.state_version_seen
        assert fp.capital_usage == action.capital_usage
        assert fp.liquidity_usage == action.liquidity_usage
        assert fp.notional_delta == action.notional_delta


def test_authoritative_claim_hints_carries_true_value_for_every_planted_calculation_claim():
    cases = generate_slice(seed=2, split="dev", n_per_family=1)
    numerical_error_case = next(c for c in cases if c.family == "numerical_reconciliation_error")
    hints = authoritative_claim_hints(numerical_error_case)

    for proposal in numerical_error_case.proposals:
        claims_with_true_value = [c for c in proposal.claims if c.true_value is not None]
        if not claims_with_true_value:
            assert proposal.proposal_id not in hints
            continue
        proposal_hints = hints[proposal.proposal_id]
        for claim in claims_with_true_value:
            assert proposal_hints.claim_hints[claim.claim_id]["authoritative_value"] == claim.true_value


def test_authoritative_claim_hints_omits_claims_without_a_true_value():
    """A claim_type the oracle itself never numerically checks (fact/
    inference/forecast) typically carries no true_value; such a claim must
    be left out of the hint entirely rather than defaulted to some
    sentinel — the graph already treats a missing authoritative_value as
    'not measured', which is the correct behaviour to preserve."""
    cases = generate_slice(seed=2, split="dev", n_per_family=2)
    for case in cases:
        hints = authoritative_claim_hints(case)
        for proposal in case.proposals:
            proposal_hints = hints.get(proposal.proposal_id)
            claim_ids_without_true_value = {c.claim_id for c in proposal.claims if c.true_value is None}
            if proposal_hints is None:
                continue
            assert claim_ids_without_true_value.isdisjoint(proposal_hints.claim_hints.keys())
