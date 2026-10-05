"""The Integrated Validator (proposal Section 4 / Section 6.3): an
independent, immutable final control gate for configuration E. It never
mutates or repairs a package — read-only, adjudication-only, per invariant
9's "failures are recorded outcomes, never silently rewritten."

This module was extracted deliberately: graph/pipeline.py's E-branch used
to compute ``executed`` as an inline conjunction of booleans with no single
named, independently testable unit standing for "the Integrated Validator"
the proposal describes as a discrete governance component. Folding the
same logic into one pure function here means (a) it can be unit-tested on
its own, against synthetic packages, without spinning up the whole graph,
and (b) a future reviewer auditing "what does the final gate actually
check" has one function to read, with one reason code per check, instead
of five scattered booleans.

Every reason code here maps to a check this codebase can actually make
today, using fields the graph already computes — nothing here invents a
concept (evidence-id tracking, an ESCALATE treasury action, ...) that does
not otherwise exist in this schema; see the module docstring's companion
note in graph/pipeline.py for why MISSING_RESEARCH_EVIDENCE and
MISSING_REQUIRED_ESCALATION (present in an earlier, independent
implementation of this same proposal) are not reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gcap.core.ftp import FTPOutput
from gcap.core.ledger import FeasibilityComponent
from gcap.core.policy import PolicyEvaluationResult

# The FTP engine (gcap.core.ftp) is trusted and single-implementation
# (invariant 1: LLMs never compute; invariant 2 guards its output schema).
# This reconciliation is defense-in-depth against a corrupted or
# hand-assembled FTPOutput reaching the final gate, not a claim that the
# engine itself is ever wrong -- the same spirit as assert_no_action_field
# running at import time rather than trusting the engine never to regress.
_RECONCILIATION_TOLERANCE = Decimal("0.000001")

IntegratedValidationReason = str


@dataclass(frozen=True)
class IntegratedValidationDecision:
    """``approved`` is True iff ``reasons`` is empty. A caller may still
    choose not to execute even when ``approved`` is True (e.g. the
    proposal's own action was not in the approval family) -- this decision
    covers only the governance gate, never the treasury action itself."""

    approved: bool
    reasons: tuple[IntegratedValidationReason, ...]


def validate_integrated_package(
    *,
    ledger_stale: bool,
    ledger_violations: tuple[FeasibilityComponent, ...],
    pricing_ftp_output: FTPOutput,
    pricing_margin: Decimal,
    business_approval: bool,
    policy_result: PolicyEvaluationResult,
) -> IntegratedValidationDecision:
    """``pricing_ftp_output`` must be whichever FTPOutput is authoritative
    for THIS decision -- the original one for a bare APPROVE, or the
    Section-9.6 revised one when APPROVE_WITH_LIMIT/HEDGE triggered a
    recomputation (see graph/pipeline.py's ``_revised_addons``). Passing
    the pre-revision output for a revised deal would validate economics
    nobody is actually about to execute.
    """
    reasons: list[IntegratedValidationReason] = []

    if ledger_stale:
        reasons.append("STALE_STATE")
    if ledger_violations:
        reasons.append("INFEASIBLE")

    recomputed_economic_alpha = pricing_ftp_output.trade_alpha_horizon - pricing_ftp_output.ftp_horizon
    if abs(recomputed_economic_alpha - pricing_ftp_output.economic_alpha_horizon) > _RECONCILIATION_TOLERANCE:
        reasons.append("FTP_RECONCILIATION_ERROR")
    if pricing_margin <= 0:
        reasons.append("NON_POSITIVE_ECONOMIC_ALPHA")

    if policy_result.verdict == "HARD_FAIL":
        reasons.append("HARD_POLICY_FAIL")
    elif policy_result.verdict == "ESCALATE":
        reasons.append("POLICY_ESCALATED")

    if not business_approval:
        reasons.append("MISSING_BUSINESS_APPROVAL")

    return IntegratedValidationDecision(approved=not reasons, reasons=tuple(reasons))
