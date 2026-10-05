"""ProposalOutcome / CaseOutcome: the graph's own return contract, richer
than RunRecord (it carries the full node-call trail, tool results, and
per-mechanism detail) — harness/store (M2d) is what condenses this down
into a RunRecord + the six other store entities. Keeping the graph's
native return type separate from RunRecord means graph/ never needs to
import store/ just to shape its own output.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gcap.agents.nodes import NodeCall
from gcap.agents.schemas import (
    ClaimReview,
    ICVote,
    PMRecommendation,
    TradeProposal,
    TreasuryDecision,
)
from gcap.agents.tools import ToolCallResult
from gcap.agents.treasury import TreasuryAction
from gcap.core.ftp import FTPOutput
from gcap.core.ledger import FeasibilityComponent, LedgerState
from gcap.core.policy import PolicyEvaluationResult


@dataclass(frozen=True)
class ProposalOutcome:
    proposal_id: str
    proposal_position: int

    claim_reviews: tuple[ClaimReview, ...] = ()
    tool_results: tuple[ToolCallResult, ...] = ()

    trade: TradeProposal | None = None
    pm: PMRecommendation | None = None
    treasury: TreasuryDecision | None = None
    ic: ICVote | None = None
    single_agent: TreasuryDecision | None = None

    ftp_output: FTPOutput | None = None
    ledger_stale: bool | None = None
    ledger_violations: tuple[FeasibilityComponent, ...] = ()

    policy_result: PolicyEvaluationResult | None = None

    # Invariant 5's conjunction, populated only for config E.
    business_approval: bool | None = None
    control_clearance: bool | None = None
    policy_pass: bool | None = None
    # ``executed``: did core.ledger.apply() actually run for this proposal
    # and produce state_out? None = "no ledger in this config; not
    # applicable" (A/B/C); a real bool in D1/D2/E, since all three have a
    # working ledger and a genuine, consequential state transition — a
    # later proposal in the same episode really does see the reduced
    # capital/liquidity/etc. an earlier D1/D2 proposal's execution used.
    # This is NOT gated by invariant 5 in D1/D2: those configs lack the
    # dual-authorisation mechanism (no IC vote, no policy engine) invariant
    # 5 is about, so their execution gate is single-gate ledger-feasibility
    # only (pipeline.py's ``not config.policy_engine`` branch). See
    # ``dual_authorised`` below for the field that is actually scoped to
    # invariant 5 and README's "execution ... only in config E" claim —
    # P1-7's fix (previously `executed` alone was read as making that
    # claim, which was never true of it).
    executed: bool | None = None
    # True iff this proposal's execution additionally cleared invariant
    # 5's full conjunction (BUSINESS_APPROVAL ∧ CONTROL_CLEARANCE ∧
    # POLICY_PASS via core.validator.validate_integrated_package) — the
    # one thing only config E can ever produce, since only E runs the IC
    # node and policy engine at all. Always False for A/B/C/D1/D2
    # (structurally: those configs have no mechanism that could set it),
    # even when ``executed`` is True there. A RunRecord/event-log row can
    # therefore be read on its own as "was this execution dually
    # authorised" without cross-referencing ``config`` — see
    # harness/record.py's ExecutionEventWrite, which now carries this
    # flag through to the store.
    dual_authorised: bool = False
    # core.validator.validate_integrated_package's reason codes (config E
    # only; empty for every other config, and empty in E whenever the gate
    # approved). See core/validator.py's module docstring.
    integrated_validation_reasons: tuple[str, ...] = ()

    final_action: TreasuryAction | None = None
    approval_family: bool | None = None
    # Section 9.6's deterministic recomputation for APPROVE_WITH_LIMIT/HEDGE
    # (see pipeline.py's ``_revised_addons``): the re-priced economic margin
    # after modelling the limit/hedge's addon offset. None for APPROVE (no
    # revision to compute) and for any config/action without a Treasury
    # node or FTP output at all.
    revised_margin: Decimal | None = None

    # In-graph endpoint detection — populated whenever THIS config has the
    # relevant mechanism (see pipeline.py's module docstring for the full
    # per-endpoint rule); left False (never None) when not applicable, so
    # the harness's post-hoc oracle-grading step for a config lacking the
    # mechanism can freely overwrite it without a tri-state dance.
    ep_numerical_error: bool = False
    ep_state_infeasible: bool = False
    ep_pricing_inconsistent: bool = False
    ep_governance_failure: bool = False

    # P2-03 (v2.0.0 code review): the SAME in-graph mechanism's raw,
    # UNGATED pre-approval_family verdict -- "did the tool/ledger/pricing/
    # policy engine find the problem at all", never ANDed with
    # approval_family the way ep_* above always is. Before these fields
    # existed, harness.grading's oracle-vs-in-graph ``detected_*``
    # diagnostic (GradedEndpoints) could only ever be genuinely ungated for
    # an ORACLE-graded config; for an IN-GRAPH-graded config it had nothing
    # to read but ep_* itself (already approval_family-gated), so the two
    # halves of the same diagnostic measured two different things
    # depending purely on which config produced them -- this module's own
    # docstring's "harness/store... condenses this down" is exactly what
    # let that asymmetry hide, since ProposalOutcome simply never carried
    # the raw pre-gate signal for harness/record.py to read. Populated by
    # pipeline.py alongside each corresponding ep_* field, from the exact
    # same underlying boolean before the ``and approval_family`` is
    # applied; left False (never None) wherever this config has no
    # relevant mechanism at all, mirroring ep_*'s own convention.
    detected_numerical_error: bool = False
    detected_state_infeasible: bool = False
    detected_pricing_inconsistent: bool = False
    detected_governance_failure: bool = False

    state_out: LedgerState | None = None
    node_calls: tuple[NodeCall, ...] = ()


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    proposals: tuple[ProposalOutcome, ...]
    final_state: LedgerState | None

    @property
    def final_proposal(self) -> ProposalOutcome:
        return self.proposals[-1]
