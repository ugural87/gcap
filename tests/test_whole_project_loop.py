"""The capstone end-to-end test: every M1-M3 piece chained in the order a
real deployment would actually use them, once. This is deliberately the
one test in the suite that crosses every package boundary in sequence --
pilot statistics -> LHC candidate pool -> Table 5 budget row -> freeze
manifest assembly -> a genuinely committed FreezeManifestStore -> a
confirm-split harness run gated on that committed manifest -> analysis
reading the sealed split back out, only once the run log is explicitly
closed. Every other test file in this suite covers one of these pieces in
isolation (tests/pilot/, tests/analysis/, tests/harness/test_integration.py,
tests/harness/test_cli.py); this file is the one place that proves they
still fit together as a whole.

The freeze manifest IS committed here, by the test itself, standing in for
the human governance act Section 9.1 requires ("the freeze is a file, not
a meeting") -- gcap.pilot.freeze.assemble_freeze_manifest itself still
never calls FreezeManifestStore.commit; this test calls it explicitly,
exactly as a real reviewer would after inspecting the assembled manifest.

The pilot/LHC/budget-row inputs below are deliberately synthetic (hand-
built RunRecords and a permissive LHC oracle) rather than a full 32-case
pilot rehearsal through FakeLLMAdapter -- that machinery is already
proven correct and is exercised end-to-end in tests/pilot/test_end_to_end.py
and tests/pilot/test_freeze.py; repeating a 480-run rehearsal here would
only slow this test down without covering anything new. What this test
adds that no other single test does is the confirm-split harness run
under a REAL committed manifest, and the analysis layer reading that
sealed split back out.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import gcap.analysis as _analysis_pkg
from gcap.agents.llm import FakeLLMAdapter
from gcap.analysis.budget_row import select_c_d1_budget_row
from gcap.analysis.lhc import build_eligible_pool
from gcap.analysis.report import build_confirmatory_report
from gcap.bench.generator import generate_slice
from gcap.bench.labelling import label_case
from gcap.config.hashing import hash_bytes_map, hash_directory, hash_run_list
from gcap.config.manifest import FreezeManifestStore, RunLogStore
from gcap.config.registry import ConcurrencyConfig, default_registry
from gcap.errors import ConfirmatoryUnsealError
from gcap.harness.manifest_binding import hash_config_table_default, hash_node_prompts
from gcap.harness.runner import Harness, RunSpec
from gcap.pilot.freeze import assemble_freeze_manifest
from gcap.pilot.report import build_pilot_report
from gcap.store.db import build_engine, get_session_factory, init_schema
from gcap.store.repository import load_run_records
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents

_GENEROUS_CONCURRENCY = ConcurrencyConfig(
    min_requests_per_minute_gate=10_000,
    min_input_tokens_per_minute_gate=10_000_000,
    min_output_tokens_per_minute_gate=10_000_000,
)
_FAST = {"n_sims": 60, "n_perm": 100}


def _pilot_rr(*, config, case_id, shown_margin, approval_family):
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split="pilot",
        block="pilot",
        case_id=case_id,
        replicate=0,
        family="alpha_eliminated_by_funding_cost",
        materiality="high",
        model_snapshot="fake-luna-sandbox-1.0",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="0" * 64,
        shown_margin=shown_margin,
        approval_family=approval_family,
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
    )


def _synthetic_pilot_report():
    records = []
    for i in range(40):
        cid = f"case-{i}"
        records.append(
            _pilot_rr(config="D1", case_id=cid, shown_margin=Decimal("-0.01"), approval_family=True)
        )
        records.append(
            _pilot_rr(
                config="D2", case_id=cid, shown_margin=Decimal("-0.01"), approval_family=False
            )
        )
    return build_pilot_report(
        records, g_n_boot=1000, central_test_p10=0.25, central_test_n_sims=100, kmin_n_mc=20_000
    )


def _synthetic_lhc_draws():
    def permissive_oracle_eval_fn(_params):
        return Decimal("40"), 15

    selected, _all = build_eligible_pool(
        seed=1,
        n_dims=2,
        bounds=[(0.0, 0.01), (0.0, 0.01)],
        param_names=["lp", "cp"],
        oracle_eval_fn=permissive_oracle_eval_fn,
        addon_floor_bps=Decimal("25"),
        kmin=10,
        n_target=5,
        initial_pool_size=20,
    )
    padded = []
    for d in selected:
        params = {
            "lp": d.params["lp"],
            "tp": Decimal("0"),
            "cp": d.params["cp"],
            "rp": Decimal("0"),
            "cc": Decimal("0"),
        }
        padded.append(
            type(d)(
                draw_id=d.draw_id,
                params=params,
                median_addon_bps=d.median_addon_bps,
                sign_flip_count=d.sign_flip_count,
                eligible=d.eligible,
            )
        )
    return padded


def test_whole_project_loop_from_freeze_assembly_to_confirmatory_report(tmp_path):
    # P0-01 (v2.0.0 code review): this test exercises a genuine
    # split="confirm" run through Harness.run_case, which now enforces
    # ParameterRegistry.assert_ready_for_split -- calibrated here (values
    # unchanged) so this rehearsal honestly represents what a real,
    # post-pilot confirmatory run requires rather than silently coasting
    # on the uncalibrated sandbox default.
    # P0-01 (v2.1.0 third code review round): registry.fixed.concurrency is
    # overridden to the generous, non-rate-limiting config this test has
    # always needed (many FakeLLMAdapter calls run synchronously here) --
    # done on the REGISTRY itself, rather than passed only to Harness as a
    # separate _GENEROUS_CONCURRENCY value, so that assemble_freeze_manifest
    # (which always freezes ``registry.fixed.concurrency`` verbatim, never a
    # caller-supplied concurrency) and the Harness constructed below agree
    # on the SAME concurrency -- otherwise the new runtime-vs-manifest
    # binding barrier (validate_runtime_against_manifest) correctly refuses
    # the mismatch this test used to construct unnoticed.
    registry = default_registry().mark_all_calibrated()
    registry = registry.model_copy(
        update={"fixed": registry.fixed.model_copy(update={"concurrency": _GENEROUS_CONCURRENCY})}
    )

    # --- 1. M3 statistics: pilot report, LHC pool, Table 5 budget row ---
    pilot_report = _synthetic_pilot_report()
    lhc_draws = _synthetic_lhc_draws()
    replay_case_ids = tuple(
        f"confirm-alpha_eliminated_by_funding_cost-{i:04d}" for i in range(1, 21)
    )
    budget_selection = select_c_d1_budget_row(icc=0.0, p10=0.35, p01=0.0, target_power=0.5, **_FAST)

    # --- 2. Assemble the freeze manifest (pure glue, never commits itself) ---
    confirm_cases = generate_slice(seed=42, split="confirm", n_per_family=1)[:1]
    run_specs_identity = [("B", c.case_id, r, None) for c in confirm_cases for r in range(1)]
    run_specs_identity += [("C", c.case_id, r, None) for c in confirm_cases for r in range(1)]

    analysis_code_hash = hash_directory(next(iter(_analysis_pkg.__path__)))
    manifest = assemble_freeze_manifest(
        manifest_version="whole-loop-test-1",
        committed_by="test-reviewer",
        model_snapshots={"primary": "fake-luna-sandbox-1.0"},
        # P0-01 (v2.1.0 third code review round): real hashes, computed the
        # same way gcap.harness.manifest_binding's runtime-side checks
        # compute them, rather than the old arbitrary placeholders
        # ("deadbeef" / hash of a literal config_table string) -- this test
        # now runs a genuine split="confirm" case through the new runtime-
        # vs-manifest binding barrier, so a placeholder here would make this
        # capstone test fail the very check it should be proving works.
        node_prompt_hashes=hash_node_prompts(),
        config_flags_table_hash=hash_config_table_default(),
        registry=registry,
        pilot_report=pilot_report,
        nested_simulation_seed=0,
        nested_simulation_parameter_distributions_hash=hash_bytes_map(
            {"g_lower_bound": b"synthetic"}
        ),
        nested_simulation_n_mc=20_000,
        lhc_candidate_pool_seed=1,
        lhc_selected_draws=lhc_draws,
        lhc_replay_case_ids=replay_case_ids,
        budget_selection=budget_selection,
            run_list_hash=hash_run_list(run_specs_identity),
            run_list=run_specs_identity,
        realised_max_in_flight_calls=registry.fixed.concurrency.model_call_semaphore,
        analysis_code_hash=analysis_code_hash,
        # No sealed confirmatory label store exists in this rehearsal (see
        # gcap.bench.sealing's own module docstring on the real-deployment
        # key-release split); an empty directory's hash documents that
        # honestly rather than reusing a zeroed placeholder.
        confirmatory_label_store_hash=hash_directory(tmp_path / "no-sealed-store-yet"),
        benchmark_case_files_hash=hash_bytes_map(
            {c.case_id: c.model_dump_json().encode("utf-8") for c in confirm_cases}
        ),
    )

    # --- 3. Commit it (the human governance act, performed here by the
    #        test standing in for a reviewer -- never by library code) ---
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    assert manifest_store.is_committed()

    # --- 4. Run a confirm-split slice through the harness under that
    #        committed manifest (invariant 7: every run traceable to it) ---
    engine = build_engine(f"sqlite:///{tmp_path / 'whole_loop.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    # P1-01 (v2.1.0 third code review round): run_log_store is now
    # mandatory for split="confirm" -- constructed here (rather than only
    # later, for the read-side check below) so it can be wired into the
    # Harness itself; the same instance is reused for the read-side
    # closure check further down.
    run_log_store = RunLogStore(tmp_path / "run_log")

    return_pct_tolerance = registry.to_decide.numerical_error_tolerance.return_pct
    harness = Harness(
        adapter=FakeLLMAdapter(),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        registry=registry,
        concurrency=registry.fixed.concurrency,
        token_budget=registry.fixed.token_budget,
        numerical_tolerances={
            "revenue_growth_yoy": return_pct_tolerance,
            "portfolio_return_pct": return_pct_tolerance,
        },
        default_tolerance=registry.to_decide.numerical_error_tolerance.ratio_generic,
        day_count_basis=registry.to_decide.day_count_basis.days,
        marginal_band_ann=registry.fixed.marginal_band.primary_ann,
        hedge_rp_reduction_pct=registry.to_decide.revised_margin_recomputation.hedge_rp_reduction_pct,
        limit_addon_reduction_pct=registry.to_decide.revised_margin_recomputation.limit_addon_reduction_pct,
    )

    labels_by_case = {
        c.case_id: label_case(c, {}, registry.fixed.marginal_band.primary_ann)
        for c in confirm_cases
    }
    specs = [
        RunSpec(
            bench_case=case,
            config_name=config_name,
            replicate=0,
            split="confirm",
            block="whole-loop",
            oracle_labels=labels_by_case[case.case_id],
        )
        for case in confirm_cases
        for config_name in ("B", "C")
    ]
    results = harness.run_batch(specs)
    assert len(results) == len(specs)
    assert all(inserted for _run_id, inserted in results)

    # --- 5. Read the sealed split back out and analyze it ---
    with session_factory() as session:
        confirm_records = load_run_records(
            session, manifest_hash=manifest.manifest_hash(), split="confirm"
        )
    assert len(confirm_records) == len(specs)
    assert {r.failure for r in confirm_records} == {"none"}

    # Invariant 8, enforced end to end: the same sealed records are refused
    # before the run log is marked closed, and accepted once it is -- via a
    # genuine RunLogStore closure record (P1-5), not merely a CLI flag.
    try:
        build_confirmatory_report(
            confirm_records, run_log_closed=True, manifest_store=manifest_store, run_log_store=run_log_store,
            require_complete=False,
        )
        raise AssertionError("expected ConfirmatoryUnsealError before the run log is closed")
    except ConfirmatoryUnsealError:
        pass
    run_log_store.close(manifest.manifest_hash(), closed_by="test-reviewer")

    # This capstone test deliberately runs only B and C (one contrast) --
    # see the module docstring on why repeating a full six-arm rehearsal
    # here would only slow this test down. require_complete=False is the
    # explicit, honest opt-out analysis.pipeline's module docstring
    # describes for exactly this situation; full four-contrast-complete
    # confirmatory behaviour has its own dedicated tests in
    # tests/analysis/test_pipeline_synthetic.py and
    # tests/analysis/test_confirmatory_report.py.
    report = build_confirmatory_report(
        confirm_records,
        run_log_closed=False,  # superseded by run_log_store below, not consulted
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        require_complete=False,
        n_boot=200,
        seed=0,
    )
    assert report.n_records == len(confirm_records)
    assert any(c.endpoint == "numerical_error" for c in report.contrasts)
    text = report.to_text()
    assert "numerical_error" in text
