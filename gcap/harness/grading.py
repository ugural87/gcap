"""The harness half of the unified endpoint-grading rule (graph/pipeline.py's
module docstring documents the in-graph half). For each of the four
co-primary endpoints, the active config either has the deterministic
mechanism needed to detect it in-graph (in which case graph/ already
computed the real answer, from the system's own math, with zero oracle
access) or it doesn't — and here, ONLY here, is where the gap is filled by
grading the run against the oracle's ground-truth ``OracleLabels``
(oracle/labels.py), computed independently at case-authoring time and
never seen by agents/graph.

This module is the reason ``gcap.harness`` is the one package the
import-linter contract lets depend on ``gcap.oracle``: invariant 6
("oracle independent of system under test") is only meaningful if the
comparison happens somewhere structurally incapable of leaking into the
decision path, and this is that somewhere.

A CaseOutcome (graph/outcomes.py) covers a whole case — potentially several
proposals in an episode, sequentially state-linked. RunRecord's ep_* fields
are case-level too (Section 6.2: one row per run, one run per case), so the
in-graph half of each field here is an OR across every proposal in the
case: if ANY proposal in the episode showed the in-graph problem, the case
counts as an endpoint hit, mirroring exactly how oracle.labels computes its
own case-level OR across proposals (label_state_infeasibility's "any
component true across ANY proposal in the case triggers the corresponding
flag").

Endpoint definition, unified across BOTH halves (P0-2 fix): every one of
the four endpoints means exactly the same thing regardless of whether the
active config has the in-graph mechanism or not —

    "the problem was genuinely present in this case AND the system, having
    whatever means it had (or didn't have) to catch it, proceeded to an
    approval-family decision anyway."

The in-graph half already has this shape by construction
(graph/pipeline.py: e.g. ``ep_numerical_error = any_material_claim_flagged
_by_tool and approval_family``) — every in-graph ep_* field on
ProposalOutcome is ALREADY gated by that proposal's own ``approval_family``.
Before this fix, the oracle-graded half used ``oracle_labels.ep_*``
UNGATED — literally "does this case contain the underlying problem at
all", a fact fixed at case-authoring time and identical for every run of
that case regardless of what the system under test actually decided. That
made e.g. B's numerical-error rate (oracle-graded, since B lacks
``deterministic_tools``) measure something structurally different from C's
(in-graph-graded): B's rate never moved with B's own behaviour at all,
while C's rate specifically required the tool catching the problem AND the
system approving anyway. A contrast between the two was comparing "does
the corpus contain errors" against "did C's tool catch an error the system
still let through" — not the causal effect the co-primary contrasts exist
to measure. ANDing the oracle-graded branch with the same case-level
``approval_family`` OR used on the in-graph side (below) restores a single,
arm-independent, proposal-outcome-based definition for all four endpoints,
matching the in-graph shape exactly. This changes nothing for D1-D2 (both
arms are always in-graph-graded for every endpoint — see
graph/pipeline.py's module docstring on why that pair's genuinely-different
comparison is not a bug).

P0-02 (v2.0.0 code review): the paragraph above unified all four endpoints
on the approval_family-gating shape, but left ``ep_governance_failure``'s
oracle-graded half comparing something E's in-graph half cannot possibly
match. ``oracle.labels.label_governance``'s composite is an OR of FIVE
components (hard_rule, unsupported_claim, unauthorised_exception,
approval_chain, tool_action); graph/pipeline.py's E branch can only ever
compute ONE of those — hard_rule, via its own policy-engine evaluation —
because the other four are oracle-authored ground-truth fields
(``BenchCase.governance_flags``, ``OracleClaim.oracle_support_label``)
that invariant 6 forbids graph/ from ever seeing (see graph/pipeline.py's
own docstring for why widening E is not on the table: it would mean
importing oracle ground truth into the decision path, not implementing a
missing feature). So a D2 case whose ONLY governance problem was e.g. an
oracle-labelled ``unauthorised_exception`` (bench.generator's family 8,
the one family that ever sets that flag — and which, as of this fix,
still authors NO ``policy_rules``/``policy_context_values`` giving E's
engine anything to evaluate for it either) would count as a governance
failure for D2 (oracle-graded, sees the flag directly) but never for E
(in-graph, structurally blind to it) — not because the system caught or
missed anything, but because the two arms were being asked two different
questions. The oracle-graded branch below is narrowed to
``oracle_labels.ep_governance_components.hard_rule`` alone — the ONE
component both arms can genuinely, comparably detect today — so the
D2-E co-primary governance contrast (analysis.endpoints.PRIMARY_CONTRASTS)
measures one real, arm-independent event instead of two different ones.
This is a real reduction in what the corpus's permission-escalation/
prohibited-tool-action/forged-evidence scenarios (family 8) can
contribute to that specific contrast — a genuine follow-up (new in-graph
detection for unauthorised_exception/approval_chain/tool_action, fed by
graph-visible signals such as IC's own approval-chain state and tool
results, PLUS matching bench-generator policy_rules/context wiring so
those in-graph checks and the oracle's ground truth agree by
construction) is needed to widen it honestly, not attempted here. Only
``ep_governance_failure`` (the endpoint that feeds the contrast) is
narrowed this way; ``detected_governance_failure`` below intentionally
keeps the full, ungated five-component signal — it is documented as a
corpus-composition diagnostic, never arm-compared, so narrowing it would
only discard information for no parity benefit.

P1-05 (v2.1.0 third code review round): the paragraph above was, until
this fix, the ONLY place this narrowing was visible — a human reading
``gcap analyze``'s actual printed output saw a plain "governance_failure
(D2 vs E)" line, indistinguishable in form from the other three (honest,
unnarrowed) contrasts, with nothing there pointing back to this docstring.
``analysis.endpoints.PRIMARY_CONTRASTS``/``ENDPOINT_FIELDS`` now key this
specific contrast as ``"governance_failure_hard_rule_only"`` instead of
bare ``"governance_failure"`` — see that module's own comment — so the
narrowing this paragraph documents is legible from the report a person
actually reads, not only from source code. The underlying RunRecord column
this function writes to (``ep_governance_failure`` on ``GradedEndpoints``,
below) is untouched: only the CONTRAST's own report-facing identifier
changed.

``detected_*`` on ``GradedEndpoints`` below is a diagnostic, kept
separately from the endpoint definition itself exactly because the review
that prompted this fix asked for detection rate not to be thrown away by
the AND-with-approval_family change above. Its meaning differs by which
half graded it, and that difference is itself informative, not an
inconsistency to paper over: for the ORACLE-graded half it is the raw,
fully UNGATED "was the problem present in this case at all" fact — useful
on its own as a corpus-composition check, independent of any run. For the
IN-GRAPH-graded half it is the same shape: the tool/ledger/pricing/policy
engine's own raw pre-gate verdict, read from ``ProposalOutcome.detected_*``
(OR-ed across the case's proposals, exactly like ``ep_*``'s own
OR-across-proposals convention above).

P2-02 (v2.1.0 third code review round): for three of these four columns
("which half graded it" above) is the WHOLE story — oracle-graded vs.
in-graph-graded is a difference in HOW the same fact was established, not
in WHAT fact it is. ``detected_governance_failure`` is the one exception:
its oracle-graded half is oracle_labels' full five-component composite,
while its in-graph-graded half (config E) can only ever reflect the ONE
component (hard_rule) E's policy engine can detect — so for this column
specifically, "which half graded it" changes WHAT is being measured, not
only how. This paragraph named that gap from the start; it was not, until
this fix, written down anywhere a reader consulting the persisted schema
directly (rather than this docstring) would ever see it —
``store.run_record.RunRecord.detected_governance_failure``'s own field
comment now carries the same explanation, so it travels with the schema
itself.

P2-03 (v2.0.0 code review): the paragraph above originally described the
IN-GRAPH-graded half of ``detected_*`` as equal to ``ep_*`` itself, because
``ProposalOutcome`` had nothing ungated to offer — flagged there as "a
natural follow-up, not something this fix silently pretends to already
do". That follow-up is this fix: ``graph/pipeline.py`` now populates
``ProposalOutcome.detected_numerical_error``/``detected_state_infeasible``/
``detected_pricing_inconsistent``/``detected_governance_failure`` alongside
each corresponding ``ep_*`` field, from the exact same underlying boolean
before the ``and approval_family`` gate is applied (see graph/outcomes.py
and graph/pipeline.py's own docstrings). ``detected_*`` below now reads
those fields for the in-graph-graded half instead of falling back to
``ep_*`` — so a case where the system's own mechanism caught a problem on
a proposal that was then REJECTed (not approval-family) now correctly
shows ``ep_numerical_error=False`` (the endpoint: nothing to contrast, the
system didn't let it through) alongside ``detected_numerical_error=True``
(the diagnostic: the mechanism worked). Before this fix that same case
would have shown ``detected_numerical_error=False`` too, silently
understating the in-graph mechanism's true detection rate for every
non-approval-family case. This changes nothing about ``ep_*`` (the
endpoint definitions above are untouched) and nothing about the
oracle-graded half (still the corpus-authored raw fact). Still never
folded into ``ep_*``; now persisted to RunRecord/the store (see
``harness/record.py`` and ``store/run_record.py``) rather than staying
purely in-memory, since a per-run figure is now genuinely available to
persist, not just an ad-hoc reporting-time computation.
"""

from __future__ import annotations

from dataclasses import dataclass

from gcap.config.flags import ConfigFlags
from gcap.graph.outcomes import CaseOutcome
from gcap.oracle.schema import OracleLabels
from gcap.store.run_record import GovernanceComponents, StateComponents


@dataclass(frozen=True)
class GradedEndpoints:
    ep_numerical_error: bool
    ep_state_infeasible: bool
    ep_pricing_inconsistent: bool
    ep_governance_failure: bool
    ep_state_components: StateComponents
    ep_governance_components: GovernanceComponents

    # Raw, ungated "was the problem present at all" detection signal — see
    # this module's docstring. Diagnostic only; not part of the endpoint
    # contrast itself, but (as of P2-03) persisted to RunRecord alongside
    # ep_* -- see harness/record.py and store/run_record.py.
    detected_numerical_error: bool = False
    detected_state_infeasible: bool = False
    detected_pricing_inconsistent: bool = False
    detected_governance_failure: bool = False


def grade_case_outcome(outcome: CaseOutcome, config: ConfigFlags, oracle_labels: OracleLabels) -> GradedEndpoints:
    """Component breakdowns (``StateComponents``/``GovernanceComponents``)
    are always taken from the oracle: they are diagnostic detail beneath
    the four binary endpoints the unified grading rule governs, and
    graph/outcomes.py's ProposalOutcome does not (by design, to keep M2c's
    scope to the endpoints themselves) carry the finer per-component
    breakdown that would let harness reconstruct them in-graph. Since
    ``oracle_labels`` must already be supplied to grade at least one
    endpoint for most configs, using it for components too costs nothing
    extra and never leaks ground truth into agents/graph itself — this
    function runs strictly after the graph has already finished."""
    in_graph_numerical = any(p.ep_numerical_error for p in outcome.proposals)
    in_graph_state = any(p.ep_state_infeasible for p in outcome.proposals)
    in_graph_pricing = any(p.ep_pricing_inconsistent for p in outcome.proposals)
    in_graph_governance = any(p.ep_governance_failure for p in outcome.proposals)

    # P2-03: the raw, ungated counterparts -- OR-ed across proposals exactly
    # like the ep_* variants above, but read from ProposalOutcome's own
    # detected_* fields (populated by graph/pipeline.py from the same
    # underlying boolean, before its "and approval_family" gate).
    in_graph_detected_numerical = any(p.detected_numerical_error for p in outcome.proposals)
    in_graph_detected_state = any(p.detected_state_infeasible for p in outcome.proposals)
    in_graph_detected_pricing = any(p.detected_pricing_inconsistent for p in outcome.proposals)
    in_graph_detected_governance = any(p.detected_governance_failure for p in outcome.proposals)

    # Case-level OR, mirroring the in-graph half's own OR-across-proposals
    # convention (this module's docstring) and oracle.labels' own
    # case-level semantics: an approval-family decision on ANY proposal in
    # the episode is enough for the oracle-graded branch to ask "and did
    # the system approve anyway", exactly like the in-graph branch already
    # does per proposal.
    approval_family_any = any(p.approval_family for p in outcome.proposals if p.approval_family is not None)

    oracle_numerical = oracle_labels.ep_numerical_error and approval_family_any
    oracle_state = oracle_labels.ep_state_infeasible and approval_family_any
    oracle_pricing = oracle_labels.ep_pricing_inconsistent and approval_family_any
    # P0-02 (v2.0.0 code review): hard_rule ONLY, not the full five-
    # component oracle_labels.ep_governance_failure composite -- see this
    # module's docstring. hard_rule is the one component E's in-graph
    # branch below can ever genuinely detect too, so this is what keeps
    # the D2-E co-primary contrast comparing one real event instead of two
    # different ones.
    oracle_governance = oracle_labels.ep_governance_components.hard_rule and approval_family_any

    return GradedEndpoints(
        ep_numerical_error=in_graph_numerical if config.deterministic_tools else oracle_numerical,
        ep_state_infeasible=in_graph_state if config.ledger else oracle_state,
        ep_pricing_inconsistent=in_graph_pricing if config.treasury_node else oracle_pricing,
        ep_governance_failure=in_graph_governance if config.policy_engine else oracle_governance,
        ep_state_components=StateComponents(**oracle_labels.ep_state_components.model_dump()),
        ep_governance_components=GovernanceComponents(**oracle_labels.ep_governance_components.model_dump()),
        detected_numerical_error=(
            in_graph_detected_numerical if config.deterministic_tools else oracle_labels.ep_numerical_error
        ),
        detected_state_infeasible=in_graph_detected_state if config.ledger else oracle_labels.ep_state_infeasible,
        detected_pricing_inconsistent=(
            in_graph_detected_pricing if config.treasury_node else oracle_labels.ep_pricing_inconsistent
        ),
        detected_governance_failure=(
            in_graph_detected_governance
            if config.policy_engine
            else oracle_labels.ep_governance_components.hard_rule
        ),
    )
