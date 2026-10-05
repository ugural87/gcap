"""End-to-end M2 integration test: harness -> graph -> agents (fake
adapter) -> core -> store (real SQLite DB) -> analysis reading RunRecords
back out. This is the one test that exercises the full pipeline across
every M2 sub-package together rather than each in isolation, closing the
loop back to M1's analysis/ layer exactly as Section 9.6 describes it.
"""

from __future__ import annotations

from decimal import Decimal

from gcap.agents.llm import FakeLLMAdapter
from gcap.analysis.pipeline import run_all_primary_contrasts
from gcap.bench.generator import generate_slice
from gcap.bench.labelling import label_case
from gcap.config.registry import ConcurrencyConfig
from gcap.harness.cases import authoritative_claim_hints
from gcap.harness.runner import Harness, RunSpec
from gcap.store.db import build_engine, get_session_factory, init_schema
from gcap.store.models import ToolResultRow
from gcap.store.repository import load_run_records
from tests.harness.factories import make_manifest

TOLERANCES: dict[str, Decimal] = {}
BAND_ANN = Decimal("0.0050")

# Generous gates: this test is about correctness of the full wiring, not
# about the rate limiter's own throttling behavior (that is covered, with
# an injectable clock, by tests/harness/test_rate_limit.py) — a real
# 20-requests/minute gate would make this test genuinely sleep for real
# wall-clock time given the number of LLM calls a handful of B/C runs
# already needs.
_GENEROUS_CONCURRENCY = ConcurrencyConfig(
    min_requests_per_minute_gate=10_000, min_input_tokens_per_minute_gate=10_000_000,
    min_output_tokens_per_minute_gate=10_000_000,
)


def test_full_loop_from_case_generation_to_analysis_contrast(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'integration.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    cases = generate_slice(seed=21, split="dev", n_per_family=1)
    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )

    specs = [
        RunSpec(
            bench_case=case, config_name=config_name, replicate=replicate, split="dev", block="integration",
            oracle_labels=label_case(case, TOLERANCES, BAND_ANN),
        )
        for case in cases
        for config_name in ("B", "C")
        for replicate in range(3)
    ]
    results = harness.run_batch(specs, max_workers=8)
    assert len(results) == len(specs)
    assert all(inserted for _run_id, inserted in results)

    with session_factory() as session:
        records = load_run_records(session, manifest_hash=manifest.manifest_hash(), split="dev")
    assert len(records) == len(specs)
    assert {r.failure for r in records} == {"none"}
    assert {r.config for r in records} == {"B", "C"}

    outcomes = run_all_primary_contrasts(records, run_log_closed=True, n_boot=200, seed=0)
    assert "numerical_error" in outcomes  # the one contrast (B, C) can supply
    contrast = outcomes["numerical_error"]
    assert contrast.arm_a == "B"
    assert contrast.arm_b == "C"
    assert 0.0 <= contrast.result.p_value <= 1.0


def test_config_c_reconciliation_tool_actually_runs_via_the_production_hint_path(tmp_path):
    """P0-1 regression test. Before the fix, ``gcap run``'s RunSpec never
    populated ``hints_by_proposal``, so ``run_proposal``'s
    ``claim_hints[claim_id]["authoritative_value"]`` was always empty and
    the deterministic reconciliation tool (config C+) never actually ran on
    this path — ``tool_results`` stayed empty and ``ep_numerical_error``
    could never come out True no matter what the case actually planted.
    This test wires ``harness.cases.authoritative_claim_hints`` the same
    way ``harness.cli._cmd_run`` now does (not a hand-crafted test-only
    hint dict) and checks both symptoms the code review reproduced are
    gone: a real ``tool_results`` row gets written, and C's in-graph
    ``ep_numerical_error`` comes out True for a case the oracle also flags.
    """
    engine = build_engine(f"sqlite:///{tmp_path / 'p0_1.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    cases = generate_slice(seed=2, split="dev", n_per_family=1)
    case = next(c for c in cases if c.family == "numerical_reconciliation_error")
    labels = label_case(case, TOLERANCES, BAND_ANN)
    assert labels.ep_numerical_error is True  # sanity: this case genuinely plants an error

    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )
    run_id, _inserted = harness.run_case(
        bench_case=case,
        config_name="C",
        replicate=0,
        split="dev",
        block="p0-1",
        oracle_labels=labels,
        hints_by_proposal=authoritative_claim_hints(case),
    )

    with session_factory() as session:
        records = load_run_records(session, manifest_hash=manifest.manifest_hash(), split="dev")
        tool_rows = session.query(ToolResultRow).filter(ToolResultRow.run_id == run_id).all()

    assert len(records) == 1
    assert records[0].failure == "none"
    assert any(row.tool_name == "reconciliation" for row in tool_rows)
    assert records[0].ep_numerical_error is True


def test_config_c_without_hints_never_flags_numerical_error_documenting_the_pre_fix_gap(tmp_path):
    """The contrapositive of the test above, kept as living documentation
    of exactly the bug P0-1 fixed: a RunSpec built WITHOUT
    ``authoritative_claim_hints`` (the old, broken CLI behaviour) still
    cannot detect the same planted error, because the reconciliation tool
    has no authoritative value to compare against. This is expected,
    intentional 'not measured' behaviour for a caller that genuinely
    supplies no hints (e.g. config A/B, which never call the tool at all)
    — the fix is that the production CLI path is no longer such a caller."""
    engine = build_engine(f"sqlite:///{tmp_path / 'p0_1_gap.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    cases = generate_slice(seed=2, split="dev", n_per_family=1)
    case = next(c for c in cases if c.family == "numerical_reconciliation_error")
    labels = label_case(case, TOLERANCES, BAND_ANN)
    assert labels.ep_numerical_error is True

    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )
    run_id, _inserted = harness.run_case(
        bench_case=case, config_name="C", replicate=0, split="dev", block="p0-1-gap", oracle_labels=labels
        # hints_by_proposal omitted entirely, like the pre-fix CLI path.
    )

    with session_factory() as session:
        tool_rows = session.query(ToolResultRow).filter(ToolResultRow.run_id == run_id).all()
    assert not any(row.tool_name == "reconciliation" for row in tool_rows)


def test_full_loop_is_idempotent_on_rerun(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'integration_idem.db'}")
    init_schema(engine)
    session_factory = get_session_factory(engine)

    case = generate_slice(seed=22, split="dev", n_per_family=1)[0]
    manifest = make_manifest()
    harness = Harness(
        adapter=FakeLLMAdapter(), manifest=manifest, session_factory=session_factory, concurrency=_GENEROUS_CONCURRENCY
    )
    labels = label_case(case, TOLERANCES, BAND_ANN)

    first = harness.run_case(bench_case=case, config_name="B", replicate=0, split="dev", block="blk", oracle_labels=labels)
    second = harness.run_case(bench_case=case, config_name="B", replicate=0, split="dev", block="blk", oracle_labels=labels)
    assert first[0] == second[0]
    assert first[1] is True
    assert second[1] is False

    with session_factory() as session:
        records = load_run_records(session, manifest_hash=manifest.manifest_hash())
    assert len(records) == 1  # no duplicate row from the retried run
