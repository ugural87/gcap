"""End-to-end analysis pipeline over RunRecords: the four co-primary
contrasts, Holm-corrected across them (Section 9.6). This is the module
Milestone 1's acceptance test targets: "Full pipeline on synthetic
RunRecords with a known planted effect recovers it."

Guards invariant 8 at the analysis boundary too (Section 7.3: "the analysis
code must refuse to read confirmatory endpoints before the run log is
closed"). Confirmatory data always requires a real
``manifest_store``/``run_log_store`` pair.  The historical
``run_log_closed`` argument remains in the API for source compatibility but
is never accepted as closure authority: closure is read from
   ``RunLogStore.is_closed()``'s genuinely persisted, timestamped state
   (never from whatever a caller happened to pass on the command line),
   every confirm-split record's ``manifest_hash`` must be the SAME single
   hash (a confirmatory analysis mixing rows stamped under two different
   manifests is exactly the kind of silent apples-to-oranges Section 9.2
   rule 2 exists to prevent), and that hash must validate against
   ``FreezeManifestStore.assert_run_is_confirmatory_valid``.
   ``harness.cli``'s ``analyze`` subcommand always supplies both — this is
   the gap the review's P1-5 finding actually pointed at ("the CLI takes
   this from a bare flag; it never checks a closed run-log record or
   FreezeManifestStore.assert_run_is_confirmatory_valid").

P0-02 (v2.1.0 third code review round): the instant ANY ``split="confirm"``
row is present, the ENTIRE input record set must be confirm-split and
single-manifest — not merely the confirm subset. Before this fix,
``_assert_confirmatory_readable`` only ever inspected ``confirm_records``
(the ``split="confirm"`` subset) for manifest_hash uniqueness and closure;
``run_all_primary_contrasts`` then fed the FULL, unfiltered ``records`` list
straight into ``extract_cluster_observations`` for every one of the four
contrasts. ``extract_cluster_observations`` filters only by config arm and
``failure``, never by ``split``/``manifest_hash`` (``analysis.endpoints``'s
own module docstring: "analysis/ reads RunRecord only"), so a store
containing dev/pilot/audit rows under the SAME config names as a genuine
confirm-split run (B/C/D1/D2/E are shared across every split) would have
those rehearsal rows silently pooled into the confirmatory statistical
estimate — while ``is_confirmatory_analysis = any(r.split == "confirm" for
r in records)`` still called the resulting report confirmatory off the
strength of a single genuine row. ``gcap analyze`` with no ``--split``/
``--manifest-hash`` filter (both optional) against a store holding a mix of
splits was the direct path to this. Refusing outright (never silently
filtering ``records`` down to ``confirm_records`` and proceeding) matches
this codebase's invariant-9 posture: a data-integrity problem this
fundamental is surfaced loudly, not quietly worked around.

A confirmatory analysis additionally never returns a PARTIAL result: once
readability clears, EVERY one of the four co-primary contrasts
(``analysis.endpoints.PRIMARY_CONTRASTS``) must have paired data, or the
whole call raises rather than silently reporting three-out-of-four as if
that were the complete confirmatory result (Section 9.6 rule: the four
contrasts are Holm-corrected JOINTLY — a report missing one of them is not
a smaller true result, it is not yet a confirmatory result at all). This
requirement applies only when the record set actually contains a
``split="confirm"`` row; a dev/pilot/audit/roadmap analysis may legitimately
report however many of the four contrasts its (smaller, exploratory) record
set happens to support.
"""

from __future__ import annotations

from dataclasses import dataclass

from gcap.analysis.endpoints import PRIMARY_CONTRASTS, extract_cluster_observations
from gcap.analysis.mixed_model import ContrastResult, fit_primary_contrast, paired_cluster_rates
from gcap.analysis.multiplicity import holm_correction
from gcap.config.hashing import hash_run_list
from gcap.config.manifest import FreezeManifestStore, RunLogStore
from gcap.errors import ConfirmatoryUnsealError
from gcap.store.run_record import RunRecord


@dataclass(frozen=True)
class PrimaryContrastOutcome:
    endpoint: str
    arm_a: str
    arm_b: str
    result: ContrastResult
    holm_reject: bool
    holm_adjusted_p: float


def _assert_confirmatory_readable(
    records: list[RunRecord],
    run_log_closed: bool,
    *,
    manifest_store: FreezeManifestStore | None,
    run_log_store: RunLogStore | None,
) -> None:
    confirm_records = [r for r in records if r.split == "confirm"]
    if not confirm_records:
        return

    # P0-02 (v2.1.0 third code review round): once this record set contains
    # any confirm-split row, it must contain ONLY confirm-split rows -- see
    # this module's docstring for exactly what silently proceeding used to
    # let through (dev/pilot/audit rows pooled into a "confirmatory" result
    # via a shared config arm).
    if len(confirm_records) != len(records):
        other_splits = sorted({r.split for r in records if r.split != "confirm"})
        raise ConfirmatoryUnsealError(
            "Refusing to read confirmatory endpoints: this record set mixes split='confirm' rows with "
            f"other split(s) {other_splits!r} ({len(records) - len(confirm_records)} non-confirm row(s) "
            f"of {len(records)} total). A confirmatory analysis must be scoped to split='confirm' "
            "records only (Section 9.2 rule 2) -- pass --split confirm (and --manifest-hash) to `gcap "
            "analyze`, or otherwise filter the record set before calling this pipeline, so dev/pilot/"
            "audit/roadmap rehearsal data can never silently contaminate a confirmatory statistical "
            "estimate."
        )

    if manifest_store is None or run_log_store is None:
        raise ConfirmatoryUnsealError(
            "Refusing to read confirmatory endpoints: both a committed FreezeManifestStore and "
            "a persisted RunLogStore are required. The run_log_closed boolean is not closure "
            "authority and cannot unlock confirmatory analysis."
        )

    # A store was supplied: it is authoritative, the bare bool is not even
    # consulted (see this module's docstring).
    manifest_hashes = {r.manifest_hash for r in confirm_records}
    if len(manifest_hashes) > 1:
        raise ConfirmatoryUnsealError(
            "Refusing to read confirmatory endpoints: these records were stamped with "
            f"{len(manifest_hashes)} different manifest_hash values ({sorted(manifest_hashes)!r}); "
            "a confirmatory analysis must be scoped to exactly one committed manifest "
            "(Section 9.2 rule 2)."
        )
    (manifest_hash,) = manifest_hashes

    manifest_store.assert_run_is_confirmatory_valid(manifest_hash)

    if not run_log_store.is_closed(manifest_hash):
        raise ConfirmatoryUnsealError(
            f"Refusing to read confirmatory endpoints: no RunLogStore closure record exists for "
            f"manifest_hash {manifest_hash!r} (invariant 8 / Section 7.3). A bare --run-log-closed "
            "flag is not sufficient once a RunLogStore is configured; close the run log for this "
            "manifest first."
        )

    # Recheck completeness at every read.  Besides detecting accidental
    # store damage, this refuses old/direct closure markers that predate the
    # atomic close API and therefore carry no proof that the frozen plan was
    # complete when they were written.
    manifest = manifest_store.active()
    actual_hash = hash_run_list(
        [(r.config, r.case_id, r.replicate, r.draw_id) for r in confirm_records]
    )
    if actual_hash != manifest.run_list_hash:
        raise ConfirmatoryUnsealError(
            "Refusing to read confirmatory endpoints: the persisted corpus does not match the "
            f"active manifest's frozen run list ({actual_hash!r} != {manifest.run_list_hash!r})."
        )


def run_all_primary_contrasts(
    records: list[RunRecord],
    run_log_closed: bool,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = 0,
    *,
    manifest_store: FreezeManifestStore | None = None,
    run_log_store: RunLogStore | None = None,
    require_complete: bool = True,
) -> dict[str, PrimaryContrastOutcome]:
    """Runs all four co-primary contrasts (B-C, C-D1, D1-D2, D2-E) and
    applies Holm correction across them jointly. See this module's
    docstring for ``manifest_store``/``run_log_store``.

    ``require_complete`` defaults to True: for a record set containing any
    ``split="confirm"`` row, a contrast with no pairable data raises rather
    than being silently omitted (this module's docstring). The one honest
    override is a caller that is deliberately exercising only PART of this
    pipeline against confirm-split records — e.g. a test proving the
    invariant-8 readability gate alone, not full four-contrast completeness
    (which has its own dedicated tests) — and says so explicitly by passing
    ``require_complete=False``. ``harness.cli``'s real ``gcap analyze``
    command never overrides this default.
    """
    _assert_confirmatory_readable(records, run_log_closed, manifest_store=manifest_store, run_log_store=run_log_store)
    is_confirmatory_analysis = any(r.split == "confirm" for r in records)

    raw_results: dict[str, ContrastResult] = {}
    for endpoint, arm_a, arm_b in PRIMARY_CONTRASTS:
        observations = extract_cluster_observations(records, endpoint, arm_a, arm_b)
        if not observations:
            continue
        # Skip a contrast with no cluster observed in BOTH arms rather than
        # letting fit_primary_contrast raise: a given record set (e.g. a
        # dev slice, or a mid-run snapshot) may simply not yet contain
        # paired data for every one of the four contrasts. For a
        # confirmatory analysis this is caught below instead -- ALL four
        # must be present, never silently reported as a partial result.
        if not paired_cluster_rates(observations):
            continue
        raw_results[endpoint] = fit_primary_contrast(observations, n_boot=n_boot, seed=seed)

    if is_confirmatory_analysis and require_complete:
        missing = [e for e, _a, _b in PRIMARY_CONTRASTS if e not in raw_results]
        if missing:
            raise ConfirmatoryUnsealError(
                "Refusing to report a confirmatory result: contrast(s) "
                f"{missing!r} have no paired cluster data yet. The four co-primary contrasts are "
                "Holm-corrected jointly (Section 9.6) — a confirmatory report is not meaningful "
                "until all four are complete."
            )

    endpoints_in_order = list(raw_results)
    p_values = [raw_results[e].p_value for e in endpoints_in_order]
    reject_flags, adjusted_p = holm_correction(p_values, alpha=alpha)

    outcomes: dict[str, PrimaryContrastOutcome] = {}
    for i, endpoint in enumerate(endpoints_in_order):
        arm_a, arm_b = next((a, b) for e, a, b in PRIMARY_CONTRASTS if e == endpoint)
        outcomes[endpoint] = PrimaryContrastOutcome(
            endpoint=endpoint,
            arm_a=arm_a,
            arm_b=arm_b,
            result=raw_results[endpoint],
            holm_reject=reject_flags[i],
            holm_adjusted_p=adjusted_p[i],
        )
    return outcomes
