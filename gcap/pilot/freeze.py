"""M3 close-out (Section 9.1): assembles a complete ``FreezeManifest`` from
this build's M3 pieces -- the pilot report's eligibility numbers (the g
lower bound and frozen kmin), the LHC candidate pool's 30 selected draws
and 20 replay case ids, and the C-D1 Table 5 budget-row selection -- ready
for a human to review and commit via
``gcap.config.manifest.FreezeManifestStore.commit``.

Deliberately stops short of committing: Section 9.1 calls the freeze
itself a governance act ("the freeze is a file, not a meeting"), and
``harness.cli``'s own module docstring already treats committing a
manifest as "a separate, human governance act... deliberately not
exposed" by the sandbox tooling. This module follows the same rule:
nothing here calls ``FreezeManifestStore.commit``, ever. It only ever
returns a ``FreezeManifest`` value for a human to inspect first.

Pure glue, by design: every number this module assembles was computed by
an existing module (``pilot.report``, ``analysis.lhc``, ``pilot.lhc_pool``,
``analysis.budget_row``, ``config.hashing``) and is passed in already
computed -- ``assemble_freeze_manifest`` does not itself run a pilot,
sample an LHC pool, or hash a directory. That keeps each of those
computations independently testable (as they already are, in their own
modules' test files) and this module's own tests focused on one thing:
does the assembly map the pieces onto ``FreezeManifest``'s schema
correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from gcap.analysis.budget_row import BudgetRowSelection
from gcap.analysis.lhc import DrawEvaluation
from gcap.analysis.nested_simulation import KminCandidateResult
from gcap.config.flags import ConfigName
from gcap.config.manifest import BudgetRow, FreezeManifest, LhcDraw
from gcap.config.registry import EligibilityThresholds, ParameterRegistry
from gcap.pilot.report import PilotReport


def _lhc_draw(evaluation: DrawEvaluation) -> LhcDraw:
    return LhcDraw(
        draw_id=evaluation.draw_id,
        lp=evaluation.params["lp"],
        tp=evaluation.params["tp"],
        cp=evaluation.params["cp"],
        rp=evaluation.params["rp"],
        cc=evaluation.params["cc"],
        eligible=evaluation.eligible,
        median_addon_bps=evaluation.median_addon_bps,
        sign_flip_count=evaluation.sign_flip_count,
    )


def kmin_monte_carlo_error(
    kmin_candidates: Sequence[KminCandidateResult], kmin: int | None, n_mc: int
) -> Decimal:
    """The binomial Monte Carlo standard error of the frozen kmin's own
    ``draw_success_probability`` estimate: sqrt(p*(1-p)/n_mc). A documented
    stand-in (see ``pilot.report``'s own module docstring on this build's
    rehearsal-only status) for a real pilot's calibration-uncertainty
    figure -- not a placeholder zero, but not a substitute for re-deriving
    kmin against real pilot data either. ``kmin=None`` (nothing frozen)
    reports ``Decimal("0")`` rather than an error on a value that was
    never selected.
    """
    if kmin is None:
        return Decimal("0")
    result = next(c for c in kmin_candidates if c.k == kmin)
    p = result.draw_success_probability
    return Decimal(str(round((p * (1 - p) / n_mc) ** 0.5, 8)))


def assemble_freeze_manifest(
    *,
    manifest_version: str,
    committed_by: str,
    committed_at: datetime | None = None,
    model_snapshots: dict[str, str],
    node_prompt_hashes: dict[str, str],
    config_flags_table_hash: str,
    registry: ParameterRegistry,
    pilot_report: PilotReport,
    nested_simulation_seed: int,
    nested_simulation_parameter_distributions_hash: str,
    nested_simulation_n_mc: int,
    lhc_candidate_pool_seed: int,
    lhc_selected_draws: Sequence[DrawEvaluation],
    lhc_replay_case_ids: Sequence[str],
    budget_selection: BudgetRowSelection,
    run_list_hash: str,
    realised_max_in_flight_calls: int,
    run_list: Sequence[tuple[ConfigName, str, int, str | None]] = (),
    analysis_code_hash: str,
    confirmatory_label_store_hash: str,
    benchmark_case_files_hash: str,
) -> FreezeManifest:
    """Maps this build's M3 outputs onto ``FreezeManifest``'s schema,
    field group by field group (Section 9.1's own bullet order):

    * eligibility / nested simulation: taken from ``pilot_report`` (g lower
      bound, frozen kmin) plus the Monte Carlo error computed above; the
      candidate range and false-positive-rate inputs the pilot report used
      are not re-derived here, only their *result*.
    * replicate counts: ``registry.fixed.replicates`` with ``c_d1_floor``
      AND ``c_d1_ceiling`` both pinned to the Table 5 row
      ``budget_selection`` selected -- freezing turns a 3/4/5 *range* into
      one specific decision to actually run at (Section 9.2 rule 4's
      conservative-revision spirit means a LATER manifest version may only
      raise this pinned value, never lower it below what this one froze;
      that comparison already runs in
      ``FreezeManifest.validate_conservative_revision``).
    * LHC candidate pool: the 30 selected draws (mapped to ``LhcDraw``) and
      the 20 replay case ids, exactly as ``pilot.lhc_pool`` and
      ``analysis.lhc.build_eligible_pool`` produced them.
    * everything else (parameter registry version/hash, run list hash,
      concurrency, the four content hashes) is either read straight off
      ``registry`` or passed through from the caller's own
      ``config.hashing`` calls.

    ``run_list`` (P1-01, v2.1.0 third review round) is the optional actual
    ``(config, case_id, replicate, draw_id)`` list backing ``run_list_hash``,
    passed straight through to ``FreezeManifest.run_list`` unchanged -- see
    that field's own docstring for why a hash alone cannot support the
    per-invocation subset checking a real multi-``gcap run --config X``
    confirmatory corpus needs. Defaults to ``()`` so a caller that only
    computed ``run_list_hash`` (the pre-existing contract) keeps assembling
    an identical manifest to before this field existed.
    """
    eligibility = EligibilityThresholds(
        median_addon_floor_bps=registry.fixed.eligibility.median_addon_floor_bps,
        kmin=pilot_report.kmin,
        g_lower_bound=(
            Decimal(str(round(pilot_report.g_lower_bound, 8)))
            if pilot_report.g_lower_bound is not None
            else None
        ),
    )
    selected_m = int(budget_selection.selected_row)
    replicate_counts = registry.fixed.replicates.model_copy(
        update={"c_d1_floor": selected_m, "c_d1_ceiling": selected_m}
    )
    budget_row: BudgetRow = budget_selection.selected_row

    # committed_at is left to FreezeManifest's own default_factory (real
    # wall-clock "now") unless a caller supplies one explicitly -- tests
    # that need a byte-identical manifest_hash across two assemblies pass
    # a fixed value; ordinary callers never need to.
    committed_at_kwargs = {} if committed_at is None else {"committed_at": committed_at}

    return FreezeManifest(
        manifest_version=manifest_version,
        committed_by=committed_by,
        **committed_at_kwargs,
        model_snapshots=dict(model_snapshots),
        node_prompt_hashes=dict(node_prompt_hashes),
        config_flags_table_hash=config_flags_table_hash,
        replicate_counts=replicate_counts,
        eligibility=eligibility,
        nested_simulation_seed=nested_simulation_seed,
        nested_simulation_parameter_distributions_hash=nested_simulation_parameter_distributions_hash,
        nested_simulation_monte_carlo_error=kmin_monte_carlo_error(
            pilot_report.kmin_candidates, pilot_report.kmin, nested_simulation_n_mc
        ),
        lhc_candidate_pool_seed=lhc_candidate_pool_seed,
        lhc_selected_draws=tuple(_lhc_draw(d) for d in lhc_selected_draws),
        lhc_replay_case_ids=tuple(lhc_replay_case_ids),
        parameter_registry_version=registry.registry_version,
        parameter_registry_hash=registry.content_hash(),
        run_list_hash=run_list_hash,
        run_list=tuple(run_list),
        budget_row=budget_row,
        concurrency=registry.fixed.concurrency,
        realised_max_in_flight_calls=realised_max_in_flight_calls,
        analysis_code_hash=analysis_code_hash,
        confirmatory_label_store_hash=confirmatory_label_store_hash,
        benchmark_case_files_hash=benchmark_case_files_hash,
    )
