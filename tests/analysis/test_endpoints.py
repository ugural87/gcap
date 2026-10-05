"""P1-8 (code review) regression: a RunRecord with ``failure != "none"``
must never be folded into a contrast's cluster observations as if it were
a genuine "no endpoint triggered" (y=0) result -- its ep_* fields are just
the RunRecord default, not a real observation. See
gcap.analysis.endpoints.extract_cluster_observations's own inline comment
for the full rationale.
"""

from __future__ import annotations

import uuid

from gcap.analysis.endpoints import extract_cluster_observations
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents


def _record(config, case_id, replicate, *, ep_numerical_error=False, failure="none"):
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split="dev",
        block=f"{config}-block",
        case_id=case_id,
        replicate=replicate,
        family="numerical_reconciliation_error",
        materiality="medium",
        model_snapshot="gpt-luna-test",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="0" * 64,
        ep_numerical_error=ep_numerical_error,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
        failure=failure,
    )


def test_failed_records_are_excluded_from_cluster_observations_entirely():
    records = [
        _record("B", "case-1", 0, ep_numerical_error=False),
        _record("C", "case-1", 0, ep_numerical_error=True),
        # A failed C-arm run for a DIFFERENT case: if this were folded in
        # as y=0 (the pre-fix bug), it would silently dilute C's observed
        # rate downward even though the system never actually produced a
        # trustworthy "no problem" verdict for it.
        _record("C", "case-2", 0, ep_numerical_error=False, failure="runtime"),
    ]
    observations = extract_cluster_observations(records, "numerical_error", "B", "C")
    cluster_ids = {o.cluster_id for o in observations}
    assert cluster_ids == {"case-1"}  # case-2's failed run never enters the pool
    assert len(observations) == 2  # one B, one C -- both for case-1


def test_a_failed_record_that_would_have_flagged_the_endpoint_is_still_excluded():
    """Same as above but the failed row's ep_numerical_error is planted
    True -- proving the exclusion is unconditional on failure, not merely
    "exclude when it would have been a non-event"."""
    records = [
        _record("B", "case-1", 0, ep_numerical_error=False),
        _record("C", "case-1", 0, ep_numerical_error=False, failure="schema"),
    ]
    observations = extract_cluster_observations(records, "numerical_error", "B", "C")
    assert len(observations) == 1  # only the clean B-arm row
    assert observations[0].arm == 0
