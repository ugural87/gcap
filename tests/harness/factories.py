from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from gcap.config.flags import ConfigName
from gcap.config.hashing import hash_run_list
from gcap.config.manifest import FreezeManifest, LhcDraw
from gcap.config.registry import (
    ConcurrencyConfig,
    EligibilityThresholds,
    ParameterRegistry,
    ReplicateCounts,
    default_registry,
)
from gcap.harness.manifest_binding import (
    hash_analysis_code,
    hash_config_table_default,
    hash_node_prompts,
)

# Fixed, not datetime.now(): FreezeManifest.committed_at defaults to
# "now" and manifest_hash() hashes the whole manifest, so two make_manifest()
# calls a few microseconds apart would otherwise hash differently even
# with every other field identical — exactly the kind of non-determinism
# the manifest_hash is supposed to rule out (invariant 7).
_FIXED_COMMITTED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _draw(i: int) -> LhcDraw:
    return LhcDraw(
        draw_id=f"draw-{i}",
        lp=Decimal("0.001"),
        tp=Decimal("0.0005"),
        cp=Decimal("0.001"),
        rp=Decimal("0.0002"),
        cc=Decimal("0.0003"),
        eligible=True,
        median_addon_bps=Decimal("30"),
        sign_flip_count=14,
    )


def make_manifest(
    version: str = "test-1.0.0",
    *,
    confirmatory_label_store_hash: str = "1" * 64,
    benchmark_case_files_hash: str = "2" * 64,
    registry: ParameterRegistry | None = None,
    run_list: tuple[tuple[ConfigName, str, int, str | None], ...] = (),
) -> FreezeManifest:
    # P1-01 (v2.0.0 code review): the "1"*64/"2"*64 defaults are placeholders
    # -- fine for tests that never actually call SealedLabelStore.get()/
    # SealedCaseStore.get() on a confirm-split case against a real sealed
    # store rooted here. A caller that DOES (i.e. runs `gcap run` against a
    # real --sealed-store built by `gcap generate-cases --split confirm`)
    # must pass gcap.config.hashing.hash_sealed_store(...) of that store's
    # actual "labels"/"cases" subdirectories, or get() now refuses with a
    # hash-mismatch LabelDecryptionRefusedError.
    #
    # P0-01 (v2.1.0 third code review round): node_prompt_hashes/
    # config_flags_table_hash/analysis_code_hash are now REAL, computed the
    # same way gcap.harness.manifest_binding's runtime-side checks compute
    # them (rather than the old "a"*64/"c"*64/"0"*64 placeholders), and
    # parameter_registry_hash/version are derived from a ``registry``
    # parameter -- defaulting to a CALIBRATED registry, since every
    # existing split="confirm" test in tests/harness/test_runner.py already
    # passes ``registry=default_registry().mark_all_calibrated()`` to
    # Harness (confirm/audit both require calibration; see
    # ParameterRegistry.assert_ready_for_split). A caller whose Harness
    # uses a DIFFERENT registry (a different calibration, or one with
    # overridden ``fixed`` values) must pass that same registry object
    # here, or Harness.run_case's new runtime-vs-manifest binding barrier
    # (gcap.harness.manifest_binding.validate_runtime_against_manifest)
    # correctly refuses the mismatch -- exactly the drift-detection this
    # fixture exists to make easy to get right in tests, not to work
    # around.
    registry = registry if registry is not None else default_registry().mark_all_calibrated()
    return FreezeManifest(
        manifest_version=version,
        committed_at=_FIXED_COMMITTED_AT,
        committed_by="test-harness",
        model_snapshots={"primary": "fake-luna-sandbox-1.0", "lightweight": "fake-luna-sandbox-1.0-mini"},
        node_prompt_hashes=hash_node_prompts(),
        config_flags_table_hash=hash_config_table_default(),
        replicate_counts=ReplicateCounts(),
        eligibility=EligibilityThresholds(kmin=11, g_lower_bound=Decimal("0.05")),
        nested_simulation_seed=42,
        nested_simulation_parameter_distributions_hash="d" * 64,
        nested_simulation_monte_carlo_error=Decimal("0.004"),
        lhc_candidate_pool_seed=7,
        lhc_selected_draws=tuple(_draw(i) for i in range(3)),
        lhc_replay_case_ids=("case-0", "case-1"),
        parameter_registry_version=registry.registry_version,
        parameter_registry_hash=registry.content_hash(),
        run_list_hash=hash_run_list(run_list) if run_list else "f" * 64,
        run_list=run_list,
        budget_row="3",
        concurrency=ConcurrencyConfig(),
        realised_max_in_flight_calls=64,
        analysis_code_hash=hash_analysis_code(),
        confirmatory_label_store_hash=confirmatory_label_store_hash,
        benchmark_case_files_hash=benchmark_case_files_hash,
    )
