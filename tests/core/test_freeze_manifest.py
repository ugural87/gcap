from decimal import Decimal

import pytest

from gcap.config.manifest import FreezeManifest, FreezeManifestStore, LhcDraw
from gcap.config.registry import ConcurrencyConfig, EligibilityThresholds, ReplicateCounts
from gcap.errors import ManifestImmutableError, ManifestMissingError


def _draw(i: int, eligible: bool = True) -> LhcDraw:
    return LhcDraw(
        draw_id=f"draw-{i}",
        lp=Decimal("0.001"),
        tp=Decimal("0.0005"),
        cp=Decimal("0.001"),
        rp=Decimal("0.0002"),
        cc=Decimal("0.0003"),
        eligible=eligible,
        median_addon_bps=Decimal("30"),
        sign_flip_count=14,
    )


def _manifest(version: str = "1.0.0") -> FreezeManifest:
    return FreezeManifest(
        manifest_version=version,
        committed_by="ugur",
        model_snapshots={"primary": "gpt-luna-2026-09-01", "lightweight": "gpt-luna-mini-2026-09-01"},
        node_prompt_hashes={"research": "a" * 64, "treasury": "b" * 64},
        config_flags_table_hash="c" * 64,
        replicate_counts=ReplicateCounts(),
        eligibility=EligibilityThresholds(kmin=11, g_lower_bound=Decimal("0.05")),
        nested_simulation_seed=42,
        nested_simulation_parameter_distributions_hash="d" * 64,
        nested_simulation_monte_carlo_error=Decimal("0.004"),
        lhc_candidate_pool_seed=7,
        lhc_selected_draws=tuple(_draw(i) for i in range(30)),
        lhc_replay_case_ids=tuple(f"case-{i}" for i in range(20)),
        parameter_registry_version="1.0.0-dev",
        parameter_registry_hash="e" * 64,
        run_list_hash="f" * 64,
        budget_row="3",
        concurrency=ConcurrencyConfig(),
        realised_max_in_flight_calls=64,
        analysis_code_hash="0" * 64,
        confirmatory_label_store_hash="1" * 64,
        benchmark_case_files_hash="2" * 64,
    )


def test_manifest_not_committed_by_default(tmp_path):
    store = FreezeManifestStore(tmp_path)
    assert store.is_committed() is False
    with pytest.raises(ManifestMissingError):
        store.active()


def test_commit_then_active_roundtrips(tmp_path):
    store = FreezeManifestStore(tmp_path)
    m = _manifest()
    store.commit(m)
    assert store.is_committed() is True
    active = store.active()
    assert active.manifest_hash() == m.manifest_hash()


def test_recommitting_identical_content_is_idempotent(tmp_path):
    store = FreezeManifestStore(tmp_path)
    m = _manifest()
    store.commit(m)
    store.commit(m)  # same content, same version -> no error


def test_recommitting_same_version_different_content_raises(tmp_path):
    store = FreezeManifestStore(tmp_path)
    m = _manifest()
    store.commit(m)
    mutated = m.model_copy(update={"committed_by": "someone-else"})
    with pytest.raises(ManifestImmutableError):
        store.commit(mutated)


def test_new_version_is_a_separate_immutable_record(tmp_path):
    store = FreezeManifestStore(tmp_path)
    m1 = _manifest("1.0.0")
    m2 = _manifest("1.1.0")
    store.commit(m1)
    store.commit(m2)
    assert store.active().manifest_version == "1.1.0"
    assert store.get("1.0.0").manifest_hash() == m1.manifest_hash()


def test_run_record_hash_must_match_active_manifest(tmp_path):
    store = FreezeManifestStore(tmp_path)
    m = _manifest()
    store.commit(m)
    store.assert_run_is_confirmatory_valid(m.manifest_hash())
    with pytest.raises(ManifestMissingError):
        store.assert_run_is_confirmatory_valid("wrong-hash")


def test_kmin_cannot_be_lowered_on_revision():
    prev = _manifest("1.0.0")
    nxt = prev.model_copy(
        update={
            "manifest_version": "1.1.0",
            "eligibility": EligibilityThresholds(kmin=9, g_lower_bound=Decimal("0.05")),
        }
    )
    problems = nxt.validate_conservative_revision(prev)
    assert any("kmin" in p for p in problems)


def test_replicate_floor_cannot_be_lowered_on_revision():
    prev = _manifest("1.0.0")
    lowered = ReplicateCounts(c_d1_floor=2)
    nxt = prev.model_copy(update={"manifest_version": "1.1.0", "replicate_counts": lowered})
    problems = nxt.validate_conservative_revision(prev)
    assert any("C-D1 replicate floor lowered" in p for p in problems)
