"""Extracts the paired per-cluster D1/D2 nonapproval-rate inputs
analysis.conformance's g estimator needs, straight from RunRecords.

Section 9.5, verbatim: "Define the pilot conformance gap as
g = Pr(nonapproval | D2, m<=0) - Pr(nonapproval | D1, m<=0)." The "m<=0"
condition restricts to runs whose SHOWN (pre-decision) economic margin is
non-positive — RunRecord.shown_margin, never revised_margin: g measures
whether the system's OWN engine output already told it the deal was
underpriced, not whether a later recomputation happened to fix it up.
"nonapproval" is simply "not approval_family": the system declined,
deferred, or restructured a case it should not have approved as-is.

Ties analysis/ to store/ only (RunRecord + endpoints.cluster_id_of, both
already inside analysis's allowed dependency set) — this module never
touches harness/agents/graph, matching every other module in this package.
"""

from __future__ import annotations

from collections.abc import Sequence

from gcap.analysis.endpoints import cluster_id_of
from gcap.store.run_record import RunRecord


def extract_conformance_gap_inputs(records: Sequence[RunRecord]) -> tuple[list[float], list[float]]:
    """Rolls up every D1/D2 run with shown_margin <= 0 to its cluster's
    nonapproval rate (fraction of that cluster's m<=0 runs, in that arm,
    that are NOT approval_family), restricted to clusters with at least
    one qualifying m<=0 run in BOTH arms. Returns (d1_rates, d2_rates),
    paired by cluster_id in a stable (sorted) order — the direct input to
    analysis.conformance.conformance_gap_point /
    conformance_gap_one_sided_lower_bound.
    """
    by_cluster: dict[str, dict[str, list[bool]]] = {}
    for r in records:
        if r.config not in ("D1", "D2"):
            continue
        if r.shown_margin is None or r.shown_margin > 0:
            continue
        if r.approval_family is None:
            continue
        arms = by_cluster.setdefault(cluster_id_of(r), {"D1": [], "D2": []})
        arms[r.config].append(not r.approval_family)

    d1_rates: list[float] = []
    d2_rates: list[float] = []
    for cluster_id in sorted(by_cluster):
        d1_runs = by_cluster[cluster_id]["D1"]
        d2_runs = by_cluster[cluster_id]["D2"]
        if not d1_runs or not d2_runs:
            continue
        d1_rates.append(sum(d1_runs) / len(d1_runs))
        d2_rates.append(sum(d2_runs) / len(d2_runs))
    return d1_rates, d2_rates
