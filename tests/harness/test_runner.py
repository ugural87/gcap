from __future__ import annotations

from decimal import Decimal

import pytest

from gcap.agents.cache_store import SqliteCacheStore
from gcap.agents.llm import FakeLLMAdapter
from gcap.bench.generator import generate_slice
from gcap.bench.labelling import label_case
from gcap.config.manifest import FreezeManifestStore, RunLogStore
from gcap.config.registry import default_registry
from gcap.errors import (
    ManifestMissingError,
    RunLogClosedError,
    RunLogStoreRequiredError,
    RuntimeManifestMismatchError,
    UncalibratedParameterError,
)
from gcap.graph.context import ProposalHints
from gcap.harness.runner import Harness, RunSpec, assert_confirmatory_run_permitted
from gcap.store.db import build_engine, get_session_factory, init_schema
from gcap.store.repository import load_run_record, load_run_records
from tests.harness.factories import make_manifest

TOLERANCES: dict[str, object] = {}
BAND_ANN = __import__("decimal").Decimal("0.0050")


@pytest.fixture()
def session_factory(tmp_path):
    # File-backed, not in-memory: run_batch spreads work across threads,
    # and SQLAlchemy gives each thread its own (empty) database for
    # sqlite:///:memory:, so a shared, on-disk file is required here.
    engine = build_engine(f"sqlite:///{tmp_path / 'harness-test.db'}")
    init_schema(engine)
    factory = get_session_factory(engine)
    yield factory
    engine.dispose()


def _one_case():
    cases = generate_slice(seed=5, split="dev", n_per_family=1)
    case = next(c for c in cases if c.family == "hard_mandate_breach")
    labels = label_case(case, TOLERANCES, BAND_ANN)
    return case, labels


def test_run_case_persists_a_run_record(session_factory):
    case, labels = _one_case()
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    run_id, inserted = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="dev", block="blk-1", oracle_labels=labels
    )
    assert inserted is True

    with session_factory() as session:
        record = load_run_record(session, run_id)
    assert record.case_id == case.case_id
    assert record.config == "B"
    assert record.split == "dev"
    assert record.manifest_hash == make_manifest().manifest_hash()
    assert record.failure == "none"


def test_run_case_is_idempotent(session_factory):
    case, labels = _one_case()
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    run_id_1, inserted_1 = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="dev", block="blk-1", oracle_labels=labels
    )
    run_id_2, inserted_2 = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="dev", block="blk-1", oracle_labels=labels
    )
    assert inserted_1 is True
    assert inserted_2 is False
    assert run_id_1 == run_id_2


def test_run_case_different_replicate_is_a_new_run(session_factory):
    case, labels = _one_case()
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    run_id_1, _ = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="dev", block="blk-1", oracle_labels=labels
    )
    run_id_2, inserted_2 = harness.run_case(
        bench_case=case, config_name="B", replicate=1, split="dev", block="blk-1", oracle_labels=labels
    )
    assert inserted_2 is True
    assert run_id_1 != run_id_2


def test_run_case_provider_seed_is_deterministic_per_manifest_case_replicate(session_factory):
    case, labels = _one_case()
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    run_id, _ = harness.run_case(
        bench_case=case, config_name="C", replicate=2, split="dev", block="blk-1", oracle_labels=labels
    )
    with session_factory() as session:
        record = load_run_record(session, run_id)

    from gcap.harness.seeding import derive_seed

    expected = derive_seed(make_manifest().manifest_hash(), case.case_id, 2)
    assert record.provider_seed == expected


def test_confirmatory_run_refused_without_committed_manifest(session_factory, tmp_path):
    """Isolates the manifest guard from the P0-01 calibration guard: a
    calibrated registry is passed so ManifestMissingError (not
    UncalibratedParameterError) is what this test actually exercises."""
    case, labels = _one_case()
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
    )
    with pytest.raises(ManifestMissingError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_refused_without_any_manifest_store(session_factory):
    case, labels = _one_case()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory,
        registry=default_registry().mark_all_calibrated(),
    )
    with pytest.raises(ManifestMissingError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_permitted_once_manifest_is_committed(session_factory, tmp_path):
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    # P1-01 (v2.1.0 third code review round): run_log_store is now
    # mandatory for split="confirm" -- an open (never-closed) one, exactly
    # like test_confirmatory_run_permitted_when_run_log_store_given_but_
    # not_closed below already uses.
    run_log_store = RunLogStore(tmp_path / "run_log")
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
        run_log_store=run_log_store,
    )
    run_id, inserted = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
    )
    assert inserted is True


def test_confirmatory_run_refused_when_no_run_log_store_is_configured(session_factory, tmp_path):
    """P1-01 (v2.1.0 third code review round): before this fix,
    ``Harness.run_log_store=None`` (the default) silently meant "the
    closure check is skipped" -- so a confirm run against a manifest_hash
    whose run log genuinely WAS already closed could still be run
    successfully just by omitting --run-log-store. Closure is now a
    mandatory barrier, exactly like manifest_store already was."""
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
    )  # run_log_store left at its default: None
    with pytest.raises(RunLogStoreRequiredError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_refused_when_the_run_log_closes_while_graph_work_is_in_flight(
    session_factory, tmp_path, monkeypatch
):
    """P1-01 (v2.1.0 third code review round) TOCTOU regression: this
    reproduces the review's own repro steps -- (1) a confirm run starts,
    passing the early (before-graph-work) closure check because the log is
    still open; (2) the run log is closed WHILE this case's graph/LLM work
    is in flight; (3) the run finishes. Before this fix, step 3 still wrote
    the RunRecord successfully, since nothing re-checked closure between
    the early check and the DB write. It must now be refused instead, and
    nothing must be persisted."""
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log")
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
        run_log_store=run_log_store,
    )

    import gcap.harness.runner as runner_module

    real_run_case_graph = runner_module.run_case_graph

    def _close_run_log_mid_flight(*args, **kwargs):
        # Both checks above (mandatory-store, not-yet-closed) already
        # passed -- the log was genuinely open at that point. This
        # simulates it being closed by a concurrent `gcap close-run-log`
        # while THIS case's graph/LLM work is still running.
        run_log_store.close(manifest.manifest_hash(), closed_by="race-simulator")
        return real_run_case_graph(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_case_graph", _close_run_log_mid_flight)

    with pytest.raises(RunLogClosedError, match="in flight"):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )

    with session_factory() as session:
        assert load_run_records(session, manifest_hash=manifest.manifest_hash()) == []


def test_confirmatory_run_refused_when_registry_has_drifted_from_the_committed_manifest(session_factory, tmp_path):
    """P0-01 (v2.1.0 third code review round): assert_confirmatory_run_
    permitted alone (the test above) only proves the manifest OBJECT
    matches what was committed -- it says nothing about whether the
    registry actually in use still matches what that manifest's own
    parameter_registry_hash recorded. A calibrated registry whose VALUES
    have since drifted (same registry_version, different content_hash())
    must now be refused by Harness.run_case itself, not just by a unit
    test of validate_runtime_against_manifest in isolation."""
    case, labels = _one_case()
    manifest = make_manifest()  # frozen against default_registry().mark_all_calibrated()
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    drifted_registry = default_registry().mark_all_calibrated()
    drifted_registry = drifted_registry.model_copy(
        update={
            "fixed": drifted_registry.fixed.model_copy(
                update={
                    "marginal_band": drifted_registry.fixed.marginal_band.model_copy(
                        update={"primary_bps": Decimal("99")}
                    )
                }
            )
        }
    )
    assert drifted_registry.registry_version == manifest.parameter_registry_version
    assert drifted_registry.content_hash() != manifest.parameter_registry_hash
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=drifted_registry,
    )
    with pytest.raises(RuntimeManifestMismatchError, match="content_hash"):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_refused_when_model_snapshot_has_drifted_from_the_committed_manifest(
    session_factory, tmp_path
):
    """Same barrier, different field: an adapter whose real model_snapshot
    no longer matches manifest.model_snapshots['primary'] (e.g. a Luna
    build bump the freeze never accounted for) must also be refused."""
    case, labels = _one_case()
    manifest = make_manifest()  # model_snapshots={"primary": "fake-luna-sandbox-1.0", ...}
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    harness = Harness(
        adapter=FakeLLMAdapter(model_snapshot="fake-luna-sandbox-2.0-drifted"),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
    )
    with pytest.raises(RuntimeManifestMismatchError, match="model_snapshot"):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_batch_refused_when_the_planned_run_list_diverges_from_the_manifest(
    session_factory, tmp_path
):
    """The batch-level half of P0-01: Harness.run_batch must refuse a
    confirm-split batch whose actual (config, case_id, replicate, draw_id)
    units do not hash-match manifest.run_list_hash -- checked BEFORE any
    unit in the batch is dispatched, so a diverged plan never partially
    executes."""
    case, labels = _one_case()
    manifest = make_manifest()  # run_list_hash="f" * 64 -- matches nothing real
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
    )
    spec = RunSpec(
        bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
    )
    with pytest.raises(RuntimeManifestMismatchError, match="run_list_hash"):
        harness.run_batch([spec])


def test_confirmatory_run_batch_permitted_when_the_planned_run_list_matches_the_manifest(
    session_factory, tmp_path
):
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log")
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
        registry=default_registry().mark_all_calibrated(),
        run_log_store=run_log_store,
    )
    spec = RunSpec(
        bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
    )
    results = harness.run_batch([spec])
    assert len(results) == 1
    assert results[0][1] is True


def test_direct_confirmatory_run_case_rejects_unit_outside_frozen_plan(session_factory, tmp_path):
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", "some-other-case", 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    harness = Harness(
        adapter=FakeLLMAdapter(),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        run_log_store=RunLogStore(tmp_path / "run-log"),
        registry=default_registry().mark_all_calibrated(),
    )
    with pytest.raises(RuntimeManifestMismatchError, match="not part"):
        harness.run_case(
            bench_case=case,
            config_name="B",
            replicate=0,
            split="confirm",
            block="blk",
            oracle_labels=labels,
        )


def test_confirmatory_run_refused_when_registry_still_has_uncalibrated_placeholders(session_factory, tmp_path):
    """P0-01 (v2.0.0 code review): ParameterRegistry.assert_ready_for_split
    existed since M1 but nothing on the real run path ever called it, so a
    confirm-split run against the fully uncalibrated default registry
    silently succeeded. Harness's own default registry (default_registry(),
    never calibrated) must now refuse BEFORE the manifest check even runs."""
    case, labels = _one_case()
    manifest = make_manifest()
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, manifest_store=manifest_store,
    )  # registry left at its default: default_registry(), uncalibrated
    with pytest.raises(UncalibratedParameterError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_audit_split_run_also_requires_a_calibrated_registry(session_factory):
    """assert_ready_for_split treats "audit" the same as "confirm" -- and
    unlike confirm, audit has no manifest_store guard at all, so the
    calibration guard is the ONLY protection audit-split runs have."""
    case, labels = _one_case()
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    with pytest.raises(UncalibratedParameterError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="audit", block="blk-1", oracle_labels=labels
        )

    calibrated_harness = Harness(
        adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory,
        registry=default_registry().mark_all_calibrated(),
    )
    run_id, inserted = calibrated_harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="audit", block="blk-2", oracle_labels=labels
    )
    assert inserted is True


def test_confirmatory_run_refused_once_the_run_log_is_closed(session_factory, tmp_path):
    """P1-03 (v2.0.0 code review): closing the run log must be a real
    WRITE barrier, not just something `gcap analyze`/build_confirmatory_
    report checks on read. Before this fix, Harness had no run_log_store
    field at all, so a caller could close the log and then still run more
    confirmatory cases against the same manifest_hash, silently
    invalidating the "this corpus is complete" declaration closure is
    supposed to make."""
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log")
    run_log_store.close(manifest.manifest_hash(), closed_by="test-reviewer")

    harness = Harness(
        adapter=FakeLLMAdapter(),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        registry=default_registry().mark_all_calibrated(),
    )
    with pytest.raises(RunLogClosedError):
        harness.run_case(
            bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
        )


def test_confirmatory_run_permitted_when_run_log_store_given_but_not_closed(session_factory, tmp_path):
    """The RunLogStore-aware check above must not become a blanket refusal:
    an open (never-closed) run log for this manifest_hash must still let a
    confirmatory run through exactly as before."""
    case, labels = _one_case()
    manifest = make_manifest(run_list=(("B", case.case_id, 0, None),))
    manifest_store = FreezeManifestStore(tmp_path / "manifests")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log")  # never closed

    harness = Harness(
        adapter=FakeLLMAdapter(),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        registry=default_registry().mark_all_calibrated(),
    )
    run_id, inserted = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="confirm", block="blk-1", oracle_labels=labels
    )
    assert inserted is True


def test_assert_confirmatory_run_permitted_rejects_mismatched_manifest(tmp_path):
    committed = make_manifest("committed-1.0.0")
    other = make_manifest("other-1.0.0")
    store = FreezeManifestStore(tmp_path / "manifests")
    store.commit(committed)
    with pytest.raises(ManifestMissingError):
        assert_confirmatory_run_permitted(other, store)


def test_run_case_records_schema_failure_without_crashing(session_factory):
    """A rework-exhausted SchemaConformanceError from a node must be
    recorded as failure='schema', never propagate and crash the batch
    (invariant 9)."""

    class AlwaysBrokenAdapter:
        model_snapshot = "broken-test"

        def complete(self, request):
            from gcap.agents.llm import LLMResponse

            return LLMResponse(
                raw_text="{}", parsed={"garbage": True}, tokens_in=1, tokens_out=1, latency_ms=1,
                model_snapshot="broken-test",
            )

    case, labels = _one_case()
    harness = Harness(adapter=AlwaysBrokenAdapter(), manifest=make_manifest(), session_factory=session_factory)
    run_id, inserted = harness.run_case(
        bench_case=case, config_name="B", replicate=0, split="dev", block="blk-1", oracle_labels=labels
    )
    assert inserted is True
    with session_factory() as session:
        record = load_run_record(session, run_id)
    assert record.failure == "schema"
    assert record.ep_numerical_error is False


def test_run_batch_runs_many_units_concurrently_and_persists_all(session_factory):
    cases = generate_slice(seed=6, split="dev", n_per_family=1)[:4]
    specs = [
        RunSpec(
            bench_case=case,
            config_name="B",
            replicate=0,
            split="dev",
            block="blk-batch",
            oracle_labels=label_case(case, TOLERANCES, BAND_ANN),
        )
        for case in cases
    ]
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)
    results = harness.run_batch(specs, max_workers=4)
    assert len(results) == 4
    assert all(inserted for _run_id, inserted in results)

    with session_factory() as session:
        records = [load_run_record(session, run_id) for run_id, _ in results]
    assert {r.case_id for r in records} == {c.case_id for c in cases}


def test_d1_d2_cache_sharing_across_run_case_calls(session_factory):
    """The SharedCallCache lives on the Harness instance, so two run_case
    calls for the D1/D2 pair on the same case/replicate reuse Research and
    Trade calls — exercised end-to-end here, not just at the cache-unit
    level (tests/agents/test_cache.py) or the graph level (tests/graph)."""
    case, labels = _one_case()
    proposal_id = case.proposals[0].proposal_id
    hints = {proposal_id: ProposalHints(treasury={"suggested_action": "APPROVE"})}
    harness = Harness(adapter=FakeLLMAdapter(), manifest=make_manifest(), session_factory=session_factory)

    harness.run_case(
        bench_case=case, config_name="D1", replicate=0, split="dev", block="blk-1", oracle_labels=labels,
        hints_by_proposal=hints,
    )
    misses_after_d1 = harness._shared_cache.misses
    harness.run_case(
        bench_case=case, config_name="D2", replicate=0, split="dev", block="blk-1", oracle_labels=labels,
        hints_by_proposal=hints,
    )
    assert harness._shared_cache.hits > 0
    assert harness._shared_cache.misses == misses_after_d1


def test_d1_d2_cache_sharing_persists_across_two_separate_harness_instances(session_factory, tmp_path):
    """agents.cache_store.SqliteCacheStore's defining capability: two
    separate Harness instances (standing in for two separate `gcap run`
    process invocations, exactly as harness.cli's `--cache-db` flag wires
    up) pointed at the same file share a D1/D2 Research/Trade call across
    that process boundary -- the in-memory-only default cannot do this at
    all, since it dies with the Harness instance that owns it."""
    case, labels = _one_case()
    proposal_id = case.proposals[0].proposal_id
    hints = {proposal_id: ProposalHints(treasury={"suggested_action": "APPROVE"})}
    db_path = tmp_path / "shared-call-cache.sqlite3"

    store1 = SqliteCacheStore(db_path)
    adapter1 = FakeLLMAdapter()
    harness1 = Harness(
        adapter=adapter1, manifest=make_manifest(), session_factory=session_factory, cache_store=store1
    )
    harness1.run_case(
        bench_case=case, config_name="D1", replicate=0, split="dev", block="blk-1", oracle_labels=labels,
        hints_by_proposal=hints,
    )
    store1.close()

    # A fresh Harness, fresh adapter, fresh SharedCallCache -- only the
    # SQLite file at db_path connects it to the first invocation.
    store2 = SqliteCacheStore(db_path)
    adapter2 = FakeLLMAdapter()
    harness2 = Harness(
        adapter=adapter2, manifest=make_manifest(), session_factory=session_factory, cache_store=store2
    )
    harness2.run_case(
        bench_case=case, config_name="D2", replicate=0, split="dev", block="blk-1", oracle_labels=labels,
        hints_by_proposal=hints,
    )
    store2.close()

    assert harness2._shared_cache.hits > 0
    # The second process's own adapter never had to answer the shared
    # (research/trade) calls -- only whatever D2-only calls aren't
    # shareable (treasury, pm, ic) actually reached it.
    assert adapter2.call_count < adapter1.call_count
