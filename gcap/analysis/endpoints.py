"""Extracts ClusterObservation rows from RunRecords for a primary contrast
and endpoint (Section 9.6 / 6). analysis/ reads RunRecord only — this
module is the boundary between "rows in the store" and "rows a statistical
model consumes".
"""

from __future__ import annotations

from collections.abc import Sequence

from gcap.analysis.mixed_model import ClusterObservation
from gcap.store.run_record import ConfigName, RunRecord

ENDPOINT_FIELDS: dict[str, str] = {
    "numerical_error": "ep_numerical_error",
    "state_infeasible": "ep_state_infeasible",
    "pricing_inconsistent": "ep_pricing_inconsistent",
    # P1-05 (v2.1.0 third code review round): labelled "..._hard_rule_only"
    # here, not the bare "governance_failure" the other three endpoints'
    # names would suggest by analogy, and not merely in a source comment --
    # this is the REPORT-FACING identifier (analysis.report.ContrastSummary
    # .endpoint, printed verbatim by ConfirmatoryReport.to_text() as
    # `gcap analyze`'s actual output) that names the underlying RunRecord
    # column below. gcap.harness.grading's module docstring explains WHY
    # this endpoint's oracle-graded half is narrowed to
    # oracle_labels.ep_governance_components.hard_rule alone rather than
    # the full five-component governance-failure composite every other
    # config's D2/E-comparable definition would otherwise use -- but that
    # explanation lived only in source code a reader of the PRINTED report
    # would never see. A report reading "governance_failure (D2 vs E):
    # ARD=..., p=..." looks exactly as complete as the other three
    # endpoints' lines, with nothing telling a reader that this one
    # specific contrast quietly measures a narrower thing (hard_rule
    # governance breaches only -- not unauthorised_exception/
    # approval_chain/tool_action/unsupported_claim) than its name would
    # suggest. The RunRecord column itself stays `ep_governance_failure`
    # (renaming a persisted schema column is a real breaking migration,
    # out of scope for a report-layer honesty fix, and every OTHER
    # consumer of that column -- e.g. `detected_governance_failure`'s own
    # full five-component semantics, gcap.harness.grading's own docstring
    # -- is unaffected by this key); only the CONTRAST'S OWN identifier,
    # the one a human actually reads in `gcap analyze` output, changes.
    "governance_failure_hard_rule_only": "ep_governance_failure",
}

PRIMARY_CONTRASTS: tuple[tuple[str, ConfigName, ConfigName], ...] = (
    ("numerical_error", "B", "C"),
    ("state_infeasible", "C", "D1"),
    ("pricing_inconsistent", "D1", "D2"),
    # See ENDPOINT_FIELDS's own comment on this key's name (P1-05).
    ("governance_failure_hard_rule_only", "D2", "E"),
)


def cluster_id_of(record: RunRecord) -> str:
    """The independent experimental cluster is a complete single-decision
    scenario or a complete sequential episode (Section 9.6)."""
    return record.episode_id if record.episode_id is not None else record.case_id


def extract_cluster_observations(
    records: Sequence[RunRecord],
    endpoint: str,
    arm_a: ConfigName,
    arm_b: ConfigName,
) -> list[ClusterObservation]:
    """Rolls proposal-level records up to the replicate level: "for an
    episode, a replicate's endpoint is 1 if any constituent proposal
    triggers it" (Section 6, restated in 9.6 for the separation fallback,
    applied uniformly here as the general rollup rule)."""
    field = ENDPOINT_FIELDS[endpoint]
    groups: dict[tuple[int, str, int, str | None], list[bool]] = {}
    for record in records:
        if record.config not in (arm_a, arm_b):
            continue
        # P1-8 (code review): a failed run's ep_* fields are all left at
        # their RunRecord default (False, harness.runner._failure_run_write)
        # -- not "no endpoint triggered", but "we don't know". Folding it in
        # as y=0 would silently bias every contrast's rate downward exactly
        # when the system is least trustworthy (API errors, exhausted
        # rework cycles). ``analysis.report.ConfirmatoryReport`` surfaces
        # how many records this excludes so a failure-heavy run is visible,
        # not just quietly underweighted.
        if record.failure != "none":
            continue
        arm = 0 if record.config == arm_a else 1
        key = (arm, cluster_id_of(record), record.replicate, record.ordering_id)
        groups.setdefault(key, []).append(bool(getattr(record, field)))

    observations: list[ClusterObservation] = []
    for (arm, cluster_id, replicate, ordering_id), values in groups.items():
        y = 1 if any(values) else 0
        observations.append(
            ClusterObservation(
                cluster_id=cluster_id,
                replicate=str(replicate),
                ordering=ordering_id,
                arm=arm,  # type: ignore[arg-type]
                y=y,
            )
        )
    return observations
