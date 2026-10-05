from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from gcap.analysis.budget_row import select_c_d1_budget_row
from gcap.analysis.lhc import build_eligible_pool
from gcap.config.manifest import FreezeManifestStore
from gcap.config.registry import default_registry
from gcap.pilot.freeze import assemble_freeze_manifest, kmin_monte_carlo_error
from gcap.pilot.report import build_pilot_report
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents

_FAST_KMIN_N_MC = 20_000
_FAST_CENTRAL_TEST_SIMS = 100
_FAST = {"n_sims": 60, "n_perm": 100}


def _rr(*, config, case_id, shown_margin, approval_family):
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


def _frozen_pilot_report():
    """A pilot report with a genuinely frozen kmin (mirrors
    tests/pilot/test_report.py's own "raise central_test_p10" mirror case)
    -- assemble_freeze_manifest needs a report that actually froze
    something to exercise its eligibility/nested-simulation mapping fully.
    """
    records = []
    for i in range(40):
        cid = f"case-{i}"
        records.append(
            _rr(config="D1", case_id=cid, shown_margin=Decimal("-0.01"), approval_family=True)
        )
        records.append(
            _rr(config="D2", case_id=cid, shown_margin=Decimal("-0.01"), approval_family=False)
        )
    return build_pilot_report(
        records,
        g_n_boot=1000,
        central_test_p10=0.25,
        central_test_n_sims=_FAST_CENTRAL_TEST_SIMS,
        kmin_n_mc=_FAST_KMIN_N_MC,
    )


def _permissive_lhc_draws():
    def oracle_eval_fn(params):
        return Decimal("40"), 15

    selected, _all = build_eligible_pool(
        seed=1,
        n_dims=2,
        bounds=[(0.0, 0.01), (0.0, 0.01)],
        param_names=["lp", "cp"],
        oracle_eval_fn=oracle_eval_fn,
        addon_floor_bps=Decimal("25"),
        kmin=10,
        n_target=5,
        initial_pool_size=20,
    )
    # LhcDraw requires all five components; fill the ones the 2-D toy
    # search above didn't sample with zero so _lhc_draw's mapping has
    # every key it needs.
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
            d.__class__(
                draw_id=d.draw_id,
                params=params,
                median_addon_bps=d.median_addon_bps,
                sign_flip_count=d.sign_flip_count,
                eligible=d.eligible,
            )
        )
    return padded


def _lenient_budget_selection():
    return select_c_d1_budget_row(icc=0.0, p10=0.35, p01=0.0, target_power=0.5, **_FAST)


def _assemble(**overrides):
    registry = default_registry()
    kwargs = dict(
        manifest_version="freeze-test-1",
        committed_by="test-harness",
        committed_at=datetime(2026, 9, 22, tzinfo=UTC),
        model_snapshots={"primary": "fake-luna-sandbox-1.0"},
        node_prompt_hashes={"research": "deadbeef"},
        config_flags_table_hash="a" * 64,
        registry=registry,
        pilot_report=_frozen_pilot_report(),
        nested_simulation_seed=0,
        nested_simulation_parameter_distributions_hash="b" * 64,
        nested_simulation_n_mc=_FAST_KMIN_N_MC,
        lhc_candidate_pool_seed=1,
        lhc_selected_draws=_permissive_lhc_draws(),
        lhc_replay_case_ids=tuple(
            f"confirm-alpha_eliminated_by_funding_cost-{i:04d}" for i in range(1, 21)
        ),
        budget_selection=_lenient_budget_selection(),
        run_list_hash="c" * 64,
        realised_max_in_flight_calls=8,
        analysis_code_hash="d" * 64,
        confirmatory_label_store_hash="e" * 64,
        benchmark_case_files_hash="f" * 64,
    )
    kwargs.update(overrides)
    return assemble_freeze_manifest(**kwargs)


def test_assembled_manifest_carries_through_every_scalar_field():
    manifest = _assemble()
    assert manifest.manifest_version == "freeze-test-1"
    assert manifest.committed_by == "test-harness"
    assert manifest.model_snapshots == {"primary": "fake-luna-sandbox-1.0"}
    assert manifest.config_flags_table_hash == "a" * 64
    assert manifest.run_list_hash == "c" * 64
    assert manifest.analysis_code_hash == "d" * 64
    assert manifest.confirmatory_label_store_hash == "e" * 64
    assert manifest.benchmark_case_files_hash == "f" * 64
    assert manifest.realised_max_in_flight_calls == 8


def test_replicate_counts_are_pinned_to_the_selected_budget_row():
    selection = _lenient_budget_selection()
    manifest = _assemble()
    m = int(selection.selected_row)
    assert manifest.replicate_counts.c_d1_floor == m
    assert manifest.replicate_counts.c_d1_ceiling == m
    assert manifest.budget_row == selection.selected_row


def test_eligibility_reflects_the_pilot_reports_g_and_kmin():
    report = _frozen_pilot_report()
    manifest = _assemble(pilot_report=report)
    assert manifest.eligibility.kmin == report.kmin
    assert report.g_lower_bound is not None
    assert manifest.eligibility.g_lower_bound == Decimal(str(round(report.g_lower_bound, 8)))
    # median_addon_floor_bps comes from the registry, not the pilot report.
    assert (
        manifest.eligibility.median_addon_floor_bps
        == default_registry().fixed.eligibility.median_addon_floor_bps
    )


def test_lhc_pool_fields_carry_the_selected_draws_and_replay_case_ids():
    manifest = _assemble()
    draws = _permissive_lhc_draws()
    assert len(manifest.lhc_selected_draws) == len(draws)
    assert {d.draw_id for d in manifest.lhc_selected_draws} == {d.draw_id for d in draws}
    assert len(manifest.lhc_replay_case_ids) == 20
    assert manifest.lhc_candidate_pool_seed == 1


def test_kmin_monte_carlo_error_is_zero_when_nothing_is_frozen():
    unfrozen_records = [
        _rr(config="D1", case_id="only-one", shown_margin=Decimal("0.05"), approval_family=True),
        _rr(config="D2", case_id="only-one", shown_margin=Decimal("0.05"), approval_family=True),
    ]
    report = build_pilot_report(
        unfrozen_records, kmin_n_mc=_FAST_KMIN_N_MC, central_test_n_sims=_FAST_CENTRAL_TEST_SIMS
    )
    assert report.kmin is None
    assert kmin_monte_carlo_error(
        report.kmin_candidates, report.kmin, n_mc=_FAST_KMIN_N_MC
    ) == Decimal("0")


def test_kmin_monte_carlo_error_is_positive_when_kmin_is_frozen():
    report = _frozen_pilot_report()
    assert report.kmin is not None
    error = kmin_monte_carlo_error(report.kmin_candidates, report.kmin, n_mc=_FAST_KMIN_N_MC)
    assert error > Decimal("0")


def test_assembled_manifest_hash_is_deterministic_given_identical_inputs():
    a = _assemble()
    b = _assemble()
    assert a.manifest_hash() == b.manifest_hash()


def test_assembled_manifest_is_valid_and_committable(tmp_path):
    manifest = _assemble()
    store = FreezeManifestStore(tmp_path / "manifests")
    store.commit(manifest)
    assert store.is_committed()
    assert store.active().manifest_hash() == manifest.manifest_hash()
