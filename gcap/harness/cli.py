"""CLI entry point (``gcap`` console script, see pyproject.toml's
``[project.scripts]``). Four subcommands cover the corpus-preparation /
execution / analysis / concurrency-selection split the rest of the
codebase is built around:

``gcap generate-cases`` builds a bench-case slice (bench.generator) and
computes each case's oracle labels (oracle.labels) up front, at
authoring time — exactly as Section 7.3 describes. For every split except
``confirm`` it writes both case and labels to one plaintext JSON file, same
as always (Section 7.3's "development and pilot labels are readable").
For ``--split confirm`` (P0-3 fix), it instead requires ``--sealed-store``/
``--seal-key-file`` and writes the ground-truth-bearing case data
(``OracleProposal``, ``OracleClaim.true_value``, ...) AND the oracle labels
into an encrypted ``bench.sealing`` store gated on a committed freeze
manifest (invariant 8); the plaintext ``--out`` file for a confirm split
carries nothing but a bare list of case_ids — no oracle_proposal, no
claim true_values, no labels, so anyone with filesystem access to the
ordinary working file (not just the agent/graph code path) is blind to
confirmatory ground truth until the manifest is committed. It never
touches agents/graph/store.

``gcap run`` loads that file, strips ground truth (harness.cases), runs
every case through the graph under one config, and persists to the store
(P0-03 fix: a plaintext ``--cases`` file is refused outright, before any
graph work, if it contains even one ``split="confirm"`` record --
``_load_open_cases``'s own docstring has the full rationale; only a sealed
manifest file may ever carry confirm-split data into this command),
via harness.runner.Harness — the only place LLM calls happen, always
through ``FakeLLMAdapter`` unless ``GCAP_LLM_ADAPTER=openai`` is set AND a
real key is configured (agents.llm.OpenAIAdapter itself still refuses to
proceed without ``OPENAI_API_KEY``; the CLI does not add a second gate on
top of that, it just defaults to the fake adapter so a bare invocation
never needs a key). ``--cache-db PATH`` makes the D1/D2 shared-call cache
durable across separate invocations of this command instead of the
default in-memory, single-invocation cache. ``--run-log-store`` (P1-03
fix; REQUIRED for ``--split confirm`` since P1-01, v2.1.0 third code
review round) makes a closed run log a real write barrier: a confirm-split
run against a manifest_hash whose log was already closed via `gcap
close-run-log` is refused before any graph work, not merely refused later
when someone tries to `gcap analyze` it -- and a confirm-split run given
no ``--run-log-store`` at all is now refused outright too, rather than
silently proceeding with the write-side closure check skipped.

``gcap analyze`` reads RunRecords back out of the store
(store.repository.load_run_records) and runs every co-primary contrast
they can support (analysis.report.build_confirmatory_report), printing a
plain-text result — the read side of the loop ``gcap run`` writes to.
``--run-log-closed`` must be passed explicitly to read ``split=confirm``
records (invariant 8); no other split needs it.

``gcap select-concurrency`` reads a JSON file of pilot worker-pool
observations (analysis.concurrency.select_worker_pool) and prints the
fastest candidate clearing the frozen reliability gates — advisory only;
it never writes to the registry or a manifest itself.

This is deliberately a thin sandbox CLI, not a deployment tool: a real
freeze manifest is only ever available for ``split=confirm`` runs by
pointing ``--manifest-store``/``--manifest-version`` at a directory where
one has genuinely been committed with ``gcap.config.manifest.FreezeManifestStore.commit``
(a separate, human governance act — Section 9.1's "the freeze is a file,
not a meeting" — deliberately not exposed as a CLI subcommand here, since
committing a manifest is not something this harness should do on its own
initiative). Every other split gets an ad-hoc, uncommitted, in-memory
manifest built fresh per invocation — fine for dev/pilot/audit/roadmap,
and structurally rejected the moment anyone passes ``--split confirm``
without a real committed store (``assert_confirmatory_run_permitted``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from gcap.agents.cache_store import SqliteCacheStore
from gcap.agents.llm import FakeLLMAdapter, LLMClient, OpenAIAdapter, resolve_model_identity
from gcap.analysis.concurrency import ConcurrencyObservation, select_worker_pool
from gcap.analysis.report import build_confirmatory_report
from gcap.bench.generator import generate_slice
from gcap.bench.labelling import label_case
from gcap.bench.schema import BenchCase
from gcap.bench.sealing import SealedCaseStore, SealedLabelStore, generate_key
from gcap.config.flags import ConfigName
from gcap.config.manifest import FreezeManifest, FreezeManifestStore, RunLogStore
from gcap.config.registry import (
    EligibilityThresholds,
    ParameterRegistry,
    ReplicateCounts,
    default_registry,
)
from gcap.errors import ConfirmSplitPlaintextRefusedError
from gcap.harness.cases import authoritative_claim_hints
from gcap.harness.manifest_binding import validate_confirmatory_corpus_complete
from gcap.harness.runner import Harness, RunSpec, assert_confirmatory_run_permitted
from gcap.oracle.schema import OracleLabels
from gcap.store.db import (
    apply_pending_migrations,
    build_engine,
    get_session_factory,
    resolve_database_url,
)
from gcap.store.repository import load_run_records


def _build_ad_hoc_manifest(version: str, *, committed_by: str) -> FreezeManifest:
    """A minimal, internally-consistent manifest for dev/pilot/audit/
    roadmap runs, where no real freeze governance act is expected. NEVER
    used for split='confirm' — Harness.run_case's confirmatory guard
    requires a genuinely committed FreezeManifestStore entry, which this
    ad-hoc manifest (constructed fresh, in memory, every invocation) is
    not and cannot become.

    ``committed_at`` is pinned to a fixed sentinel rather than left to
    ``FreezeManifest``'s own ``datetime.now(UTC)`` default_factory. This is
    NOT a real freeze commit — it is a throwaway object rebuilt from
    scratch on every ``gcap run``/``gcap generate-cases`` invocation with
    no manifest store — so unlike a genuine ``assemble_freeze_manifest``
    call (pilot/freeze.py, which correctly wants real wall-clock time for
    an actual governance act and only overrides it for tests), two ad-hoc
    manifests built from identical ``(version, committed_by)`` inputs must
    hash identically. ``FreezeManifest.manifest_hash()`` hashes the whole
    object including ``committed_at``, and that hash now also seeds the
    D1/D2 shared-call cache key (agents/cache.py, P1-6) and
    ``runner.py``'s ``provider_seed`` derivation — a wall-clock
    ``committed_at`` would make both non-reproducible across separate CLI
    invocations of "the same" ad-hoc dev manifest, exactly the class of
    bug ``tests/harness/factories.py``'s ``make_manifest`` already works
    around for its own fixed test manifest (see its module comment)."""
    registry = default_registry()
    zeros = "0" * 64
    return FreezeManifest(
        manifest_version=version,
        committed_at=datetime(1970, 1, 1, tzinfo=UTC),
        committed_by=committed_by,
        model_snapshots={
            "primary": "fake-luna-sandbox-1.0",
            "lightweight": "fake-luna-sandbox-1.0-mini",
        },
        node_prompt_hashes={},
        config_flags_table_hash=hashlib.sha256(b"gcap-config-table-v1").hexdigest(),
        replicate_counts=ReplicateCounts(),
        eligibility=EligibilityThresholds(),
        nested_simulation_seed=0,
        nested_simulation_parameter_distributions_hash=zeros,
        nested_simulation_monte_carlo_error=Decimal("0"),
        lhc_candidate_pool_seed=0,
        lhc_selected_draws=(),
        lhc_replay_case_ids=(),
        parameter_registry_version=registry.registry_version,
        parameter_registry_hash=registry.content_hash(),
        run_list_hash=zeros,
        budget_row="3",
        concurrency=registry.fixed.concurrency,
        realised_max_in_flight_calls=registry.fixed.concurrency.model_call_semaphore,
        analysis_code_hash=zeros,
        confirmatory_label_store_hash=zeros,
        benchmark_case_files_hash=zeros,
    )


def _resolve_adapter() -> LLMClient:
    if os.environ.get("GCAP_LLM_ADAPTER", "fake").strip().lower() == "openai":
        return OpenAIAdapter(resolve_model_identity("primary"))
    return FakeLLMAdapter()


def _get_or_create_seal_key(path: Path) -> bytes:
    """Reads the Fernet key at ``path``, generating and persisting a fresh
    one on first use. Convenience for this sandbox CLI, not a substitute
    for the real secret-manager-held key a production deployment would use
    (see bench.sealing's module docstring) — but the file is still written
    with owner-only permissions, and losing/rotating it simply means the
    confirm-split cases sealed under the OLD key stop being decryptable
    (a loud failure — bench.sealing.SealedCaseStore/SealedLabelStore raise
    LabelSealError — never a silent wrong answer)."""
    if path.exists():
        return path.read_bytes()
    key = generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return key


def _load_seal_key(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(
            f"--seal-key-file {str(path)!r} does not exist. It must be the SAME key file "
            "`generate-cases --split confirm` created; a missing key file means the confirm-split "
            "cases sealed under it can never be unsealed by this command."
        )
    return path.read_bytes()


def _cmd_generate_cases(args: argparse.Namespace) -> int:
    registry = default_registry()
    # oracle.labels.label_numerical_error falls back to its own hard-coded
    # DEFAULT_TOLERANCE for any metric not listed here; the two percentage-
    # type metrics bench.generator actually plants get the registry's
    # calibrated return_pct tolerance instead of that generic fallback.
    return_pct_tolerance = registry.to_decide.numerical_error_tolerance.return_pct
    tolerances = {
        "revenue_growth_yoy": return_pct_tolerance,
        "portfolio_return_pct": return_pct_tolerance,
    }
    marginal_band_ann = registry.fixed.marginal_band.primary_ann

    cases = generate_slice(seed=args.seed, split=args.split, n_per_family=args.n_per_family)

    if args.split == "confirm":
        if not args.sealed_store or not args.seal_key_file:
            print(
                "--split confirm requires --sealed-store and --seal-key-file (P0-3 / invariant 8): "
                "confirmatory ground truth (oracle_proposal, claim true_values, oracle labels) must "
                "never be written to the plain --out file.",
                file=sys.stderr,
            )
            return 1
        sealed_root = Path(args.sealed_store)
        key = _get_or_create_seal_key(Path(args.seal_key_file))
        # A manifest_store here is only ever used for SealedCaseStore/
        # SealedLabelStore's WRITE path (.put()), which never checks
        # is_committed() — only .get() does (see bench.sealing) — so an
        # uncommitted, locally-rooted store is correct and sufficient for
        # the authoring step; `gcap run` later opens its OWN
        # FreezeManifestStore from --manifest-store (the one a real,
        # committed manifest must live in) to unseal.
        writer_manifest_store = FreezeManifestStore(sealed_root / "manifest")
        case_store = SealedCaseStore(sealed_root / "cases", key, writer_manifest_store)
        label_store = SealedLabelStore(sealed_root / "labels", key, writer_manifest_store)
        case_ids = []
        for case in cases:
            labels = label_case(case, tolerances, marginal_band_ann)
            case_store.put(case)
            label_store.put(case.case_id, case.split, labels)
            case_ids.append(case.case_id)

        out_path = Path(args.out)
        out_path.write_text(
            json.dumps({"sealed": True, "split": "confirm", "case_ids": case_ids}, indent=2),
            encoding="utf-8",
        )
        print(
            f"Wrote {len(case_ids)} confirm-split case ids (observable manifest only — no "
            f"oracle_proposal, claim true_values or labels) to {out_path}; ground truth sealed "
            f"under {sealed_root} (key: {args.seal_key_file})."
        )
        return 0

    records = []
    for case in cases:
        labels = label_case(case, tolerances, marginal_band_ann)
        records.append(
            {
                "case": json.loads(case.model_dump_json()),
                "labels": json.loads(labels.model_dump_json()),
            }
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} cases (split={args.split!r}) to {out_path}")
    return 0


def _load_open_cases(path: Path) -> list[tuple[BenchCase, OracleLabels]]:
    """P0-03 (v2.0.0 code review): this is the plaintext path -- the one
    ``_cmd_run`` takes whenever ``--cases`` does NOT point at a
    ``{"sealed": True, ...}`` manifest file (see the branch in
    ``_cmd_run``). Before this fix, nothing here ever inspected
    ``case.split``: a plaintext file containing a genuine
    ``split="confirm"`` ``BenchCase``/``OracleLabels`` pair -- whether from
    hand-editing a dev/pilot file's split field, an older/buggy version of
    ``generate-cases``, or a case file copied from elsewhere -- would load
    without complaint, putting real confirmatory ground truth
    (``OracleProposal``, ``OracleClaim.true_value``, the oracle labels
    themselves) into this process' memory and onto whatever plaintext disk
    location the caller's ``--cases`` file already lived at. Nothing further
    downstream closes that gap: ``Harness.run_case``'s
    ``assert_confirmatory_run_permitted``/``ParameterRegistry.
    assert_ready_for_split`` guards only ever decide whether the RUN may
    proceed, never whether the ground truth should have touched plaintext
    at all -- by the time either guard runs, the leak has already happened.
    So this function refuses outright, unconditionally, the moment it
    parses a ``split="confirm"`` record -- structurally enforcing invariant
    8's "confirmatory ground truth never exists in plaintext" independent
    of every other guard, exactly mirroring the write-side refusal
    ``_cmd_generate_cases`` already applies for ``--split confirm`` (P0-3).
    The only supported way to load confirm-split data is
    ``_load_sealed_cases`` below, via a ``{"sealed": True, ...}`` manifest
    file and ``--sealed-store``/``--seal-key-file``."""
    records = json.loads(path.read_text(encoding="utf-8"))
    loaded = [
        (BenchCase.model_validate(r["case"]), OracleLabels.model_validate(r["labels"]))
        for r in records
    ]
    confirm_case_ids = [case.case_id for case, _ in loaded if case.split == "confirm"]
    if confirm_case_ids:
        raise ConfirmSplitPlaintextRefusedError(
            f"{path} contains {len(confirm_case_ids)} split='confirm' case(s) in a plaintext "
            f"open-case file (e.g. {confirm_case_ids[0]!r}) -- refusing to load. Confirmatory "
            "ground truth (oracle_proposal, claim true_values, oracle labels) must never be "
            "read from plaintext (invariant 8). Use `gcap generate-cases --split confirm` "
            "(P0-3) to produce a sealed manifest file instead, and pass it to `gcap run` with "
            "--sealed-store/--seal-key-file."
        )
    return loaded


def _load_sealed_cases(
    manifest_payload: dict[str, object],
    *,
    sealed_store: Path,
    seal_key_file: Path,
    manifest_store: FreezeManifestStore,
) -> list[tuple[BenchCase, OracleLabels]]:
    """Unseals the case_ids a confirm-split ``--sealed-store`` manifest file
    names, using the same ``manifest_store`` a confirmatory run is already
    required to supply via ``--manifest-store`` (Harness.run_case's own
    ``assert_confirmatory_run_permitted`` guard) — never a second,
    independently-constructed one, so decrypting a case can never succeed
    against a manifest weaker than the one this run is actually stamped
    with."""
    key = _load_seal_key(seal_key_file)
    case_store = SealedCaseStore(sealed_store / "cases", key, manifest_store)
    label_store = SealedLabelStore(sealed_store / "labels", key, manifest_store)
    case_ids = manifest_payload["case_ids"]
    assert isinstance(case_ids, list)
    return [(case_store.get(cid), label_store.get(cid)) for cid in case_ids]


def _cmd_run(args: argparse.Namespace) -> int:
    manifest_store: FreezeManifestStore | None = None
    if args.manifest_store:
        manifest_store = FreezeManifestStore(args.manifest_store)
        manifest = (
            manifest_store.get(args.manifest_version)
            if args.manifest_version
            else manifest_store.active()
        )
    else:
        manifest = _build_ad_hoc_manifest(
            args.manifest_version or "dev-adhoc", committed_by=args.committed_by
        )

    cases_payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if isinstance(cases_payload, dict) and cases_payload.get("sealed") is True:
        # Fail fast on the invariant-8 guard BEFORE touching the sealed
        # store at all — mirrors assert_confirmatory_run_permitted's own
        # docstring ("before any graph work happens"): a confirm-split
        # cases file with no committed manifest behind it never even
        # attempts to unseal, let alone run.
        assert_confirmatory_run_permitted(manifest, manifest_store)
        if not args.sealed_store or not args.seal_key_file:
            print(
                "This --cases file is a sealed confirm-split manifest (no observable case data "
                "in it); `gcap run` needs --sealed-store and --seal-key-file to unseal it.",
                file=sys.stderr,
            )
            return 1
        assert manifest_store is not None  # guaranteed by assert_confirmatory_run_permitted above
        cases = _load_sealed_cases(
            cases_payload,
            sealed_store=Path(args.sealed_store),
            seal_key_file=Path(args.seal_key_file),
            manifest_store=manifest_store,
        )
    else:
        cases = _load_open_cases(Path(args.cases))

    if not cases:
        print("No cases loaded; nothing to run.", file=sys.stderr)
        return 1

    database_url = resolve_database_url()
    apply_pending_migrations(database_url)
    engine = build_engine(database_url)
    session_factory = get_session_factory(engine)

    # P0-01 (v2.0.0 code review): --registry-file is the only way this CLI
    # can ever produce a registry with calibrated=True on the to-decide
    # fields Harness.run_case now requires for split="confirm"/"audit" --
    # nothing here performs calibration itself (that is real pilot-data
    # analysis, out of scope for this sandbox CLI), it only loads values a
    # caller already calibrated and dumped with registry_to_json.
    registry = (
        ParameterRegistry.model_validate_json(Path(args.registry_file).read_text(encoding="utf-8"))
        if args.registry_file
        else default_registry()
    )
    return_pct_tolerance = registry.to_decide.numerical_error_tolerance.return_pct
    cache_store = SqliteCacheStore(args.cache_db) if args.cache_db else None
    # P1-03 (v2.0.0 code review): wires the same RunLogStore `gcap
    # close-run-log`/`gcap analyze --run-log-store` already read from into
    # the write path too -- see Harness.run_log_store's own docstring.
    run_log_store = RunLogStore(args.run_log_store) if args.run_log_store else None
    harness = Harness(
        adapter=_resolve_adapter(),
        manifest=manifest,
        session_factory=session_factory,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
        # P0-01 (v2.0.0 code review): the SAME default_registry() this CLI
        # command already builds for tolerances/day-count/etc. is now also
        # what gates split="confirm"/"audit" -- so a bare `gcap run
        # --split confirm` correctly refuses today (every to-decide
        # parameter here is still an uncalibrated sandbox placeholder;
        # nothing in this repository has a calibration-loading mechanism
        # yet) until a real, calibrated registry is wired in.
        registry=registry,
        concurrency=registry.fixed.concurrency,
        token_budget=registry.fixed.token_budget,
        cache_store=cache_store,
        # Mirrors _cmd_generate_cases's oracle-labelling tolerances exactly,
        # so the in-graph reconciliation tool (config C+) and the oracle's
        # own post-hoc grading never disagree only because they were
        # configured with two different tolerance tables.
        numerical_tolerances={
            "revenue_growth_yoy": return_pct_tolerance,
            "portfolio_return_pct": return_pct_tolerance,
        },
        default_tolerance=registry.to_decide.numerical_error_tolerance.ratio_generic,
        day_count_basis=registry.to_decide.day_count_basis.days,
        marginal_band_ann=registry.fixed.marginal_band.primary_ann,
        hedge_rp_reduction_pct=registry.to_decide.revised_margin_recomputation.hedge_rp_reduction_pct,
        limit_addon_reduction_pct=registry.to_decide.revised_margin_recomputation.limit_addon_reduction_pct,
        tenor_restructuring_bounds=registry.to_decide.tenor_restructuring_bounds,
    )

    config_name: ConfigName = args.config
    specs = [
        RunSpec(
            bench_case=case,
            config_name=config_name,
            replicate=replicate,
            split=case.split,
            block=args.block,
            oracle_labels=labels,
            # P0-1 fix: without this, ``run_proposal``'s ``hints_for(...)``
            # always returns an empty ``ProposalHints`` on the CLI path, so
            # the deterministic reconciliation tool (config C+) is never
            # actually called by ``gcap run`` — see
            # harness.cases.authoritative_claim_hints's docstring for why
            # this is not a ground-truth leak.
            hints_by_proposal=authoritative_claim_hints(case),
        )
        for case, labels in cases
        for replicate in range(args.replicates)
    ]

    try:
        results = harness.run_batch(specs)
    finally:
        if cache_store is not None:
            cache_store.close()
    inserted = sum(1 for _run_id, was_inserted in results if was_inserted)
    print(
        f"Ran {len(results)} (case, replicate) units under config={config_name!r}: {inserted} new, "
        f"{len(results) - inserted} idempotent no-ops."
    )
    if cache_store is not None:
        print(
            f"Cross-process replay cache: {args.cache_db} "
            f"(hits={harness._shared_cache.hits}, misses={harness._shared_cache.misses})"
        )

    # P1-8 (code review): a node exhausting its rework cycle, a real API
    # error, or any other unexpected exception used to be swallowed into a
    # RunRecord with failure="runtime"/"schema" and nothing else -- `gcap
    # run` always printed only new/idempotent counts and always returned 0,
    # so a run that was silently half-broken looked identical to a clean
    # one. manifest_hash + config scope this query to exactly the records
    # this invocation just wrote (the idempotency key harness.runner
    # already enforces), so a prior invocation's failures under a different
    # manifest/config are never double-counted here.
    with session_factory() as session:
        this_run_records = load_run_records(session, manifest_hash=manifest.manifest_hash(), config=config_name)
    failed = [r for r in this_run_records if r.failure != "none"]
    if failed:
        by_kind: dict[str, int] = {}
        for r in failed:
            by_kind[r.failure] = by_kind.get(r.failure, 0) + 1
        print(
            f"{len(failed)} of {len(this_run_records)} record(s) under this manifest/config failed: "
            f"{by_kind}. First reason: {failed[0].failure_reason!r} (case_id={failed[0].case_id!r}). "
            "See RunRecord.failure_reason for every failed row; `gcap analyze` excludes these from "
            "every contrast rather than counting them as a non-event.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Reads RunRecords back out of the store and runs every co-primary
    contrast they can support (analysis.report.build_confirmatory_report),
    printing (and optionally writing) the plain-text result. This is the
    read side of the same store `gcap run` writes to — the two halves of
    the CLI closing the loop Section 9.6 describes: run, then analyze.

    ``--run-log-closed`` defaults to false, matching invariant 8/Section
    7.3's default posture ("the analysis code must refuse to read
    confirmatory endpoints before the run log is closed"): a dev/pilot/
    audit/roadmap analysis never needs it (those splits are never gated),
    and a confirm-split analysis must pass it explicitly, as a deliberate
    statement that the run log genuinely has been closed -- this CLI does
    not infer that on the caller's behalf.
    """
    database_url = resolve_database_url()
    apply_pending_migrations(database_url)
    engine = build_engine(database_url)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        records = load_run_records(
            session, manifest_hash=args.manifest_hash, split=args.split, config=args.config
        )
    if not records:
        print("No matching run records found; nothing to analyze.", file=sys.stderr)
        return 1

    # P1-5 fix: when --manifest-store/--run-log-store are given, they are
    # authoritative over the bare --run-log-closed flag (see
    # analysis.pipeline's module docstring) — a confirm-split analysis
    # then requires a genuine RunLogStore closure record for the one
    # manifest_hash every confirm-split record shares, validated against
    # FreezeManifestStore, not merely a flag the caller typed.
    manifest_store = FreezeManifestStore(args.manifest_store) if args.manifest_store else None
    run_log_store = RunLogStore(args.run_log_store) if args.run_log_store else None

    report = build_confirmatory_report(
        records,
        run_log_closed=args.run_log_closed,
        alpha=args.alpha,
        n_boot=args.n_boot,
        seed=args.seed,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
    )
    text = report.to_text()
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


def _cmd_close_run_log(args: argparse.Namespace) -> int:
    """The genuine, auditable "close the run log" act invariant 8 refers
    to (P1-5): writes a RunLogStore closure record for the CURRENTLY
    ACTIVE committed manifest's hash — never for a manifest the caller
    merely names, so this command cannot be pointed at a manifest nothing
    was actually run under. Idempotent: closing an already-closed
    manifest_hash is a no-op (RunLogStore.close's own contract) -- checked
    FIRST, before the completeness check below, so re-running this command
    against an already-closed manifest never starts failing on old data
    just because it is invoked again.

    P1-01 (v2.1.0 third code review round): before actually writing the
    closure marker for the first time, the persisted split='confirm'
    RunRecords for this manifest_hash must hash-match the manifest's own
    frozen run_list_hash (gcap.harness.manifest_binding.validate_
    confirmatory_corpus_complete) -- closure is a declaration that the
    confirmatory corpus is COMPLETE, and nothing previously checked that
    before writing the marker.
    """
    manifest_store = FreezeManifestStore(args.manifest_store)
    manifest = manifest_store.active()  # raises ManifestMissingError if nothing is committed
    run_log_store = RunLogStore(args.run_log_store)
    manifest_hash = manifest.manifest_hash()

    database_url = resolve_database_url()
    apply_pending_migrations(database_url)
    engine = build_engine(database_url)
    session_factory = get_session_factory(engine)

    def validate_complete() -> None:
        with session_factory() as session:
            records = load_run_records(session, manifest_hash=manifest_hash, split="confirm")
        validate_confirmatory_corpus_complete(manifest, records)

    run_log_store.close(
        manifest_hash,
        closed_by=args.closed_by,
        validate_before_close=validate_complete,
    )
    print(
        f"Run log closed for manifest_hash={manifest_hash!r} "
        f"(manifest_version={manifest.manifest_version!r}), closed_by={args.closed_by!r}."
    )
    return 0


def _cmd_select_concurrency(args: argparse.Namespace) -> int:
    """Reads a JSON file of measured pilot throughput/reliability per
    worker-pool candidate (one object per candidate; see
    configs/concurrency_pilot.example.json for the shape) and prints the
    selected candidate. This is advisory, not an automatic write: per
    config.manifest's "the freeze is a file, not a meeting" governance,
    a human takes this output and updates
    config.registry.ConcurrencyConfig.model_call_semaphore (and the
    manifest that freezes it) themselves — this subcommand never touches
    either.
    """
    payload = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    observations = tuple(
        ConcurrencyObservation(
            workers=int(row["workers"]),
            p95_latency_s=Decimal(str(row["p95_latency_s"])),
            requests_per_minute=Decimal(str(row["requests_per_minute"])),
            input_tokens_per_minute=Decimal(str(row["input_tokens_per_minute"])),
            output_tokens_per_minute=Decimal(str(row["output_tokens_per_minute"])),
            rate_limit_fraction=Decimal(str(row["rate_limit_fraction"])),
            error_fraction=Decimal(str(row["error_fraction"])),
        )
        for row in payload
    )
    registry = default_registry()
    selected = select_worker_pool(observations, registry.fixed.concurrency)
    print(f"Selected worker pool: {selected}")
    print(
        "This is advisory only — update ConcurrencyConfig.model_call_semaphore and the "
        "freeze manifest yourself; this command does not write either."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcap", description="G-CAP harness CLI (sandbox use — see module docstring)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate-cases", help="Generate a bench-case slice with oracle labels precomputed."
    )
    gen.add_argument(
        "--split", required=True, choices=["dev", "pilot", "confirm", "audit", "roadmap"]
    )
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--n-per-family", type=int, default=1)
    gen.add_argument("--out", required=True)
    gen.add_argument(
        "--sealed-store",
        default=None,
        help="Required for --split confirm (P0-3 / invariant 8): directory root for the encrypted "
        "bench.sealing case+label store. Ignored for every other split, which stays plaintext.",
    )
    gen.add_argument(
        "--seal-key-file",
        default=None,
        help="Required for --split confirm: path to the Fernet key file (auto-created on first use). "
        "`gcap run` must be given the SAME file to unseal these cases later.",
    )
    gen.set_defaults(func=_cmd_generate_cases)

    run = sub.add_parser(
        "run", help="Run a case file through the graph under one config, recording to the store."
    )
    run.add_argument("--cases", required=True)
    run.add_argument("--config", required=True, choices=["A", "B", "C", "D1", "D2", "E"])
    run.add_argument("--replicates", type=int, default=1)
    run.add_argument("--block", required=True)
    run.add_argument(
        "--manifest-store", default=None, help="Path to a committed FreezeManifestStore directory."
    )
    run.add_argument("--manifest-version", default=None)
    run.add_argument("--committed-by", default="sandbox")
    run.add_argument(
        "--cache-db",
        default=None,
        help="Optional path to a SQLite file backing the D1/D2 shared-call cache across "
        "separate `gcap run` invocations (agents.cache_store.SqliteCacheStore). Omit for "
        "the default in-memory, single-invocation cache.",
    )
    run.add_argument(
        "--sealed-store",
        default=None,
        help="Required to run a confirm-split --cases file produced by `generate-cases --split "
        "confirm` (P0-3): the same --sealed-store directory that command used.",
    )
    run.add_argument(
        "--seal-key-file",
        default=None,
        help="Required together with --sealed-store: the same --seal-key-file "
        "`generate-cases --split confirm` used.",
    )
    run.add_argument(
        "--registry-file",
        default=None,
        help="P0-01 (v2.0.0 code review): path to a ParameterRegistry JSON file (as written by "
        "gcap.config.registry.registry_to_json) with every to-decide parameter this split needs "
        "marked calibrated=True. Required for --split confirm/audit -- ParameterRegistry."
        "assert_ready_for_split refuses those splits against the uncalibrated sandbox default. "
        "Omit for dev/pilot/roadmap, which tolerate the default placeholder registry.",
    )
    run.add_argument(
        "--run-log-store",
        default=None,
        help="P1-03 (v2.0.0 code review): path to a RunLogStore directory. A confirm-split run "
        "whose manifest_hash's run log is already closed (`gcap close-run-log`) is refused "
        "before any graph work happens -- makes closure a real write barrier, not just "
        "something `gcap analyze` checks on read. P1-01 (v2.1.0 third code review round): "
        "REQUIRED for --split confirm -- omitting it used to silently skip the write-side "
        "closure check entirely; a confirm-split run with no --run-log-store now refuses "
        "outright (RunLogStoreRequiredError) rather than proceeding unchecked.",
    )
    run.set_defaults(func=_cmd_run)

    analyze = sub.add_parser(
        "analyze", help="Run the four co-primary contrasts over stored RunRecords and report."
    )
    analyze.add_argument("--manifest-hash", default=None)
    analyze.add_argument(
        "--split", default=None, choices=["dev", "pilot", "confirm", "audit", "roadmap"]
    )
    analyze.add_argument("--config", default=None, choices=["A", "B", "C", "D1", "D2", "E"])
    analyze.add_argument(
        "--run-log-closed",
        action="store_true",
        help="Fallback only: required to read split=confirm records (invariant 8) when neither "
        "--manifest-store nor --run-log-store is given. Ignored (superseded by a genuine "
        "RunLogStore closure record) once --run-log-store is supplied.",
    )
    analyze.add_argument(
        "--manifest-store",
        default=None,
        help="Path to a committed FreezeManifestStore directory (P1-5): when given, every "
        "split=confirm record analyzed must share one manifest_hash, validated against it.",
    )
    analyze.add_argument(
        "--run-log-store",
        default=None,
        help="Path to a RunLogStore directory (P1-5): when given, this — not --run-log-closed — "
        "decides whether split=confirm records may be read. Close it first with "
        "`gcap close-run-log`.",
    )
    analyze.add_argument("--alpha", type=float, default=0.05)
    analyze.add_argument("--n-boot", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument(
        "--out", default=None, help="Optional path to also write the text report to."
    )
    analyze.set_defaults(func=_cmd_analyze)

    close_run_log = sub.add_parser(
        "close-run-log",
        help="Record the genuine, auditable closure of the run log for the active committed "
        "manifest (P1-5) — the real invariant-8 act `gcap analyze --run-log-closed` used to let "
        "a caller merely assert with a flag.",
    )
    close_run_log.add_argument(
        "--manifest-store", required=True, help="Path to a committed FreezeManifestStore directory."
    )
    close_run_log.add_argument(
        "--run-log-store", required=True, help="Path to the RunLogStore directory to write the closure record into."
    )
    close_run_log.add_argument("--closed-by", default="sandbox")
    close_run_log.set_defaults(func=_cmd_close_run_log)

    concurrency = sub.add_parser(
        "select-concurrency",
        help="Select a worker-pool size from a JSON file of measured pilot throughput.",
    )
    concurrency.add_argument(
        "observations",
        help="Path to a JSON array of per-candidate observations "
        "(see configs/concurrency_pilot.example.json).",
    )
    concurrency.set_defaults(func=_cmd_select_concurrency)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
