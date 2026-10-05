"""End-to-end M3 rehearsal: bench (pilot corpus) -> harness (execution,
FakeLLMAdapter) -> store (real SQLite DB) -> pilot.metrics/report reading
the persisted RunRecords back — the M3 counterpart to M2's own
tests/harness/test_integration.py.

No ProposalHints are supplied here (matching harness.cli's own `gcap run`,
which never wires per-case hints either): every node call gets
FakeLLMAdapter's context-free defaults, so this rehearsal's D1/D2 runs
never actually show a non-positive shown margin and g/kmin correctly come
back as "not derivable from this data" rather than a fabricated number —
see pilot/report.py's own module docstring for why that is the honest
outcome here, and tests/pilot/test_report.py for the (synthetic-RunRecord)
tests proving the g/kmin machinery itself works correctly when the
underlying data DOES show a signal.
"""

from __future__ import annotations

from gcap.agents.llm import FakeLLMAdapter
from gcap.config.registry import ConcurrencyConfig
from gcap.harness.runner import Harness
from gcap.pilot.corpus import build_pilot_corpus
from gcap.pilot.report import build_pilot_report
from gcap.pilot.run import PILOT_ARMS, run_pilot
from gcap.store.db import build_engine, get_session_factory, init_schema
from gcap.store.repository import load_run_records
from tests.harness.factories import make_manifest

# Generous gates: 32 cases x 5 arms x 3 replicates = 480 runs, all through
# FakeLLMAdapter -- this is a correctness/wiring test, not a throughput
# test (see tests/harness/test_integration.py's identical rationale).
_GENEROUS_CONCURRENCY = ConcurrencyConfig(
    min_requests_per_minute_gate=100_000,
    min_input_tokens_per_minute_gate=100_000_000,
    min_output_tokens_per_minute_gate=100_000_000,
)


def test_pilot_end_to_end_from_corpus_to_report(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'pilot.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    corpus = build_pilot_corpus(seed=99)
    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )

    results = run_pilot(harness, corpus, replicates=3, max_workers=16)
    assert len(results) == 32 * len(PILOT_ARMS) * 3
    assert all(inserted for _run_id, inserted in results)

    with session_factory() as session:
        records = load_run_records(session, manifest_hash=manifest.manifest_hash(), split="pilot")
    assert len(records) == len(results)
    assert {r.failure for r in records} == {"none"}
    assert {r.config for r in records} == set(PILOT_ARMS)

    report = build_pilot_report(records, kmin_n_mc=20_000, central_test_n_sims=100)
    assert report.metrics.n_records == len(records)
    # Structural sanity, not a scientific claim (see module docstring): the
    # report must be well-formed even when the fake adapter's hint-free
    # defaults leave no m<=0 D1/D2 runs to condition g on.
    assert 0.0 <= report.central_test_power <= 1.0
    if report.g_point is None:
        assert report.kmin_frozen is False
    else:
        assert -1.0 <= report.g_point <= 1.0


def test_pilot_run_is_idempotent_on_rerun(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'pilot_idem.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    corpus = build_pilot_corpus(seed=7)
    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )

    first = run_pilot(harness, corpus, replicates=1, max_workers=16)
    second = run_pilot(harness, corpus, replicates=1, max_workers=16)
    assert sum(1 for _rid, inserted in first if inserted) == len(first)
    assert sum(1 for _rid, inserted in second if inserted) == 0  # every run_id already existed

    with session_factory() as session:
        records = load_run_records(session, manifest_hash=manifest.manifest_hash(), split="pilot")
    assert len(records) == len(first)  # no duplicate rows from the rerun
