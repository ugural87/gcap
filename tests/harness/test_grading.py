from __future__ import annotations

from gcap.config.flags import CONFIG_TABLE
from gcap.graph.outcomes import CaseOutcome, ProposalOutcome
from gcap.harness.grading import grade_case_outcome
from gcap.oracle.schema import (
    OracleGovernanceComponents,
    OracleLabels,
    OracleStateComponents,
)


def _labels(**overrides: object) -> OracleLabels:
    base: dict[str, object] = dict(
        case_id="case-1",
        ep_numerical_error=True,
        numerical_error_claim_ids=("c1",),
        ep_state_infeasible=True,
        ep_state_components=OracleStateComponents(capital=True),
        ep_pricing_inconsistent=True,
        proposal_margins=(),
        ep_governance_failure=True,
        ep_governance_components=OracleGovernanceComponents(hard_rule=True),
        expected_escalation_required=False,
        restructure_feasible=None,
    )
    base.update(overrides)
    return OracleLabels(**base)  # type: ignore[arg-type]


def _outcome(**proposal_overrides: object) -> CaseOutcome:
    proposal_overrides.setdefault("approval_family", True)
    proposal = ProposalOutcome(proposal_id="p1", proposal_position=0, **proposal_overrides)  # type: ignore[arg-type]
    return CaseOutcome(case_id="case-1", proposals=(proposal,), final_state=None)


def test_config_a_defers_every_endpoint_to_oracle():
    """A has no deterministic mechanism at all: every endpoint must come
    from the oracle (P0-2: ANDed with this run's own approval_family — the
    same "problem present AND the system approved anyway" shape the
    in-graph half already has), regardless of what (nothing) the graph
    itself set for the in-graph ep_* fields."""
    outcome = _outcome()  # approval_family=True (see _outcome default); all in-graph ep_* default False
    labels = _labels()
    graded = grade_case_outcome(outcome, CONFIG_TABLE["A"], labels)
    assert graded.ep_numerical_error is True
    assert graded.ep_state_infeasible is True
    assert graded.ep_pricing_inconsistent is True
    assert graded.ep_governance_failure is True
    # Raw detection is available separately even though the endpoint itself
    # requires approval_family too.
    assert graded.detected_numerical_error is True


def test_config_a_oracle_problem_without_approval_is_not_an_endpoint_hit():
    """P0-2's core fix: an oracle-flagged problem the system did NOT
    approve (e.g. it was correctly rejected) must not count as an endpoint
    hit — mirroring exactly how the in-graph half already requires
    approval_family, not just tool/engine detection, before counting."""
    outcome = _outcome(approval_family=False)
    labels = _labels()  # oracle says every endpoint's problem is present
    graded = grade_case_outcome(outcome, CONFIG_TABLE["A"], labels)
    assert graded.ep_numerical_error is False
    assert graded.ep_state_infeasible is False
    assert graded.ep_pricing_inconsistent is False
    assert graded.ep_governance_failure is False
    # But the raw detection diagnostic still reports the problem existed.
    assert graded.detected_numerical_error is True
    assert graded.detected_state_infeasible is True
    assert graded.detected_pricing_inconsistent is True
    assert graded.detected_governance_failure is True


def test_config_c_uses_in_graph_numerical_but_defers_the_rest():
    outcome = _outcome(ep_numerical_error=True)
    labels = _labels(ep_numerical_error=False)  # oracle disagrees; must NOT be consulted for this endpoint
    graded = grade_case_outcome(outcome, CONFIG_TABLE["C"], labels)
    assert graded.ep_numerical_error is True  # in-graph value wins (C has deterministic_tools)
    assert graded.ep_state_infeasible is True  # C has no ledger -> oracle-graded, and approval_family=True
    assert graded.ep_pricing_inconsistent is True  # C has no treasury_node -> oracle-graded
    assert graded.ep_governance_failure is True  # C has no policy_engine -> oracle-graded


def test_config_d1_uses_in_graph_state_and_pricing_but_defers_governance():
    outcome = _outcome(ep_numerical_error=True, ep_state_infeasible=False, ep_pricing_inconsistent=False)
    labels = _labels(ep_state_infeasible=True, ep_pricing_inconsistent=True, ep_governance_failure=True)
    graded = grade_case_outcome(outcome, CONFIG_TABLE["D1"], labels)
    assert graded.ep_numerical_error is True  # in-graph (D1 has deterministic_tools)
    assert graded.ep_state_infeasible is False  # in-graph (D1 has ledger) — disagrees with oracle, wins anyway
    assert graded.ep_pricing_inconsistent is False  # in-graph (D1 has treasury_node)
    assert graded.ep_governance_failure is True  # D1 has no policy_engine -> oracle-graded, approval_family=True


def test_oracle_graded_and_in_graph_graded_branches_agree_on_the_same_fact_pattern():
    """P0-2's comparability test: the same underlying fact pattern — a
    problem the system's mechanism (in-graph) or the oracle (post-hoc)
    calls present, on a proposal that was approved anyway — must grade
    identically whichever half answers it. This is exactly the property
    that was broken before the fix: an oracle-graded arm's endpoint used to
    depend only on the case, never on approval_family, so it could disagree
    with an in-graph-graded arm's endpoint for what is otherwise the same
    "problem detected, approved anyway" situation."""
    approved_with_problem = _outcome(ep_numerical_error=True)  # in-graph: detected AND approval_family=True
    via_in_graph = grade_case_outcome(approved_with_problem, CONFIG_TABLE["C"], _labels(ep_numerical_error=False))
    via_oracle = grade_case_outcome(
        _outcome(ep_numerical_error=False), CONFIG_TABLE["A"], _labels(ep_numerical_error=True)
    )
    assert via_in_graph.ep_numerical_error is via_oracle.ep_numerical_error is True

    rejected_with_problem = _outcome(approval_family=False)
    via_oracle_rejected = grade_case_outcome(rejected_with_problem, CONFIG_TABLE["A"], _labels(ep_numerical_error=True))
    assert via_oracle_rejected.ep_numerical_error is False


def test_config_e_uses_in_graph_for_all_four_endpoints():
    outcome = _outcome(
        ep_numerical_error=False, ep_state_infeasible=False, ep_pricing_inconsistent=False, ep_governance_failure=False
    )
    labels = _labels()  # oracle says everything is broken; E must ignore it entirely
    graded = grade_case_outcome(outcome, CONFIG_TABLE["E"], labels)
    assert graded.ep_numerical_error is False
    assert graded.ep_state_infeasible is False
    assert graded.ep_pricing_inconsistent is False
    assert graded.ep_governance_failure is False


def test_episode_level_or_across_proposals():
    """A case-level endpoint fires if ANY proposal in the episode showed
    the in-graph problem — mirrors oracle.labels' own case-level OR."""
    p1 = ProposalOutcome(proposal_id="p1", proposal_position=0, ep_state_infeasible=False)
    p2 = ProposalOutcome(proposal_id="p2", proposal_position=1, ep_state_infeasible=True)
    outcome = CaseOutcome(case_id="case-1", proposals=(p1, p2), final_state=None)
    graded = grade_case_outcome(outcome, CONFIG_TABLE["D1"], _labels(ep_state_infeasible=False))
    assert graded.ep_state_infeasible is True


def test_d2_e_parity_for_each_non_hard_governance_component():
    """P0-02 regression: the review's own reproduction case. A case whose
    ONLY governance problem is a non-hard-rule component (unsupported_claim,
    unauthorised_exception, approval_chain, or tool_action) must NOT count
    as ep_governance_failure for D2 (oracle-graded, no policy_engine) --
    because E (in-graph-graded) has no way to ever detect any of those four
    components either, and the two arms feeding the same co-primary contrast
    (analysis.endpoints.PRIMARY_CONTRASTS' ("governance_failure_hard_rule_only",
    "D2", "E") -- P1-05 renamed this report-facing identifier; the
    underlying RunRecord column tested below is still ep_governance_failure)
    must be asking the same question for the contrast to be honest.

    Before the P0-02 fix, D2's oracle_governance used the full 5-component
    OR (oracle_labels.ep_governance_failure) here and would have graded
    every one of these cases as a governance failure while E structurally
    never could -- a false arm asymmetry manufactured by which half graded
    the case, not a real behavioural difference between D2 and E."""
    for component in ("unsupported_claim", "unauthorised_exception", "approval_chain", "tool_action"):
        labels = _labels(
            ep_governance_failure=True,  # the oracle's raw, ungated composite is still True...
            ep_governance_components=OracleGovernanceComponents(hard_rule=False, **{component: True}),
        )
        # D2: oracle-graded (no policy_engine) -- must ignore the non-hard component.
        d2_graded = grade_case_outcome(_outcome(), CONFIG_TABLE["D2"], labels)
        assert d2_graded.ep_governance_failure is False, component
        # The scalar raw diagnostic is hard-rule-only in every arm; the
        # full oracle composite remains in ep_governance_components.
        assert d2_graded.detected_governance_failure is False, component

        # E: in-graph-graded, and has no in-graph mechanism to ever set
        # ep_governance_failure=True from a non-hard-rule signal either --
        # this ProposalOutcome represents exactly that (E's own policy
        # engine found nothing, since the signal is oracle-only ground
        # truth it structurally cannot see).
        e_graded = grade_case_outcome(_outcome(ep_governance_failure=False), CONFIG_TABLE["E"], labels)
        assert e_graded.ep_governance_failure is False, component

        # P2-02 (v2.1.0 third code review round): the RAW diagnostic is
        # where the asymmetry this docstring warns about actually shows up
        # -- D2's detected_governance_failure reports the oracle's full
        # five-component composite (True here, asserted above), while E's
        # reports its own in-graph engine's raw pre-gate verdict, which for
        # this non-hard-rule-only case is False (ProposalOutcome.detected_
        # governance_failure defaults to False and this test never sets it
        # otherwise -- E's engine genuinely found nothing to report, since
        # the signal is oracle-only ground truth it structurally cannot
        # see). Same column name, same case, deliberately different
        # numbers -- see store.run_record.RunRecord.detected_governance_
        # failure's own field comment and harness.grading's module
        # docstring for why this is documented, not a bug.
        assert e_graded.detected_governance_failure is False, component

        assert d2_graded.ep_governance_failure is e_graded.ep_governance_failure, component


def test_d2_e_parity_when_hard_rule_is_the_governance_problem():
    """The one component both arms CAN genuinely detect: hard_rule alone
    must still count as ep_governance_failure for D2 (oracle-graded),
    matching what E's own in-graph policy engine would produce for the same
    fact pattern -- this is the comparable event the D2-E contrast is meant
    to measure, and P0-02's narrowing must not have broken it."""
    labels = _labels(
        ep_governance_failure=True,
        ep_governance_components=OracleGovernanceComponents(hard_rule=True),
    )
    d2_graded = grade_case_outcome(_outcome(), CONFIG_TABLE["D2"], labels)
    assert d2_graded.ep_governance_failure is True

    # E's own policy engine caught the hard rule in-graph.
    e_graded = grade_case_outcome(_outcome(ep_governance_failure=True), CONFIG_TABLE["E"], labels)
    assert e_graded.ep_governance_failure is True


def test_p2_03_in_graph_detected_reads_the_raw_proposaloutcome_field_not_ep():
    """P2-03 regression: before this fix, an in-graph-graded config's
    detected_* fell back to reading ep_* itself (there was nothing else to
    read), so it was silently identical to the endpoint it was supposed to
    be an ungated diagnostic for. Now it reads ProposalOutcome's own new
    detected_* field -- proven here by constructing a proposal where the
    two disagree (as graph/pipeline.py's own C/D1/D2/E branches now can:
    the mechanism caught the problem on a proposal that was NOT let through,
    e.g. REJECTed) and confirming grading surfaces that disagreement rather
    than collapsing it."""
    proposal = ProposalOutcome(
        proposal_id="p1",
        proposal_position=0,
        approval_family=False,  # not an endpoint hit...
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        detected_numerical_error=True,  # ...but the mechanism genuinely caught every one of these
        detected_state_infeasible=True,
        detected_pricing_inconsistent=True,
        detected_governance_failure=True,
    )
    outcome = CaseOutcome(case_id="case-1", proposals=(proposal,), final_state=None)
    labels = _labels(
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
    )  # oracle disagrees too -- must not be consulted for a config that grades all four in-graph
    graded = grade_case_outcome(outcome, CONFIG_TABLE["E"], labels)
    assert graded.ep_numerical_error is False
    assert graded.ep_state_infeasible is False
    assert graded.ep_pricing_inconsistent is False
    assert graded.ep_governance_failure is False
    assert graded.detected_numerical_error is True
    assert graded.detected_state_infeasible is True
    assert graded.detected_pricing_inconsistent is True
    assert graded.detected_governance_failure is True


def test_components_always_come_from_oracle_labels():
    outcome = _outcome()
    labels = _labels(
        ep_state_components=OracleStateComponents(liquidity=True),
        ep_governance_components=OracleGovernanceComponents(unsupported_claim=True),
    )
    graded = grade_case_outcome(outcome, CONFIG_TABLE["E"], labels)
    assert graded.ep_state_components.liquidity is True
    assert graded.ep_governance_components.unsupported_claim is True
