"""Pure numeric-discrepancy comparison (extracted from oracle/labels.py's
inline logic so it can be shared, without either side importing the
other): a stated value is flagged when its discrepancy from an
authoritative value exceeds a registered tolerance, relative to the
authoritative value's magnitude, or absolute when that magnitude is zero.

This module exists specifically so ``gcap.agents.tools``'s deterministic
reconciliation tool (config C+, invariant-1-safe: it never emits an
action label, only a flagged/not-flagged verdict a downstream node must
still act on) and ``gcap.oracle.labels.label_numerical_error`` compute
discrepancy the exact same way — one implementation, two call sites, so a
label and a tool result can never disagree because of a subtle rounding
difference between two hand-written copies of the same comparison.
"""

from __future__ import annotations

from decimal import Decimal


def relative_discrepancy(stated: Decimal, authoritative: Decimal) -> Decimal:
    """Absolute discrepancy when the authoritative value is zero (a
    relative discrepancy is undefined there); otherwise the discrepancy as
    a fraction of the authoritative value's magnitude."""
    discrepancy = abs(stated - authoritative)
    if authoritative == 0:
        return discrepancy
    return discrepancy / abs(authoritative)


def exceeds_tolerance(stated: Decimal, authoritative: Decimal, tolerance: Decimal) -> bool:
    return relative_discrepancy(stated, authoritative) > tolerance
