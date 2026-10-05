"""The top-level orchestrator (M2d): runs bench-case slices through
graph/ under a ConfigFlags + FreezeManifest, records into store/, manages
the model-call semaphore/rate limiter, seeds deterministically per
(manifest_hash, case_id, replicate[, draw_id]), and is idempotent by
construction (every write goes through store.repository.save_run, whose
natural-key IntegrityError-catch already makes a retried run a no-op).

Harness is the one layer allowed to import both ``gcap.oracle`` and
``gcap.bench`` (no import-linter contract restricts ``gcap.harness``): it
is where the ground-truth-stripping conversion (harness/cases.py) and the
post-hoc oracle grading (harness/grading.py) both run, strictly AFTER the
graph has already produced its own in-graph decision with zero oracle
access.

Failures are recorded outcomes, never silent crashes (invariant 9): a node
that exhausts its rework cycle, or any other unexpected exception from one
case's graph run, is caught here and written as a RunRecord with
``failure`` set rather than aborting the whole batch or the harness
process.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
from threading import Lock

from sqlalchemy.orm import Session, sessionmaker

from gcap.agents.cache import SharedCallCache
from gcap.agents.cache_store import CacheStore
from gcap.agents.llm import LLMClient
from gcap.bench.schema import BenchCase
from gcap.config.flags import CONFIG_TABLE, ConfigName
from gcap.config.manifest import FreezeManifest, FreezeManifestStore, RunLogStore
from gcap.config.registry import (
    ConcurrencyConfig,
    ParameterRegistry,
    TenorRestructuringBounds,
    TokenBudget,
    default_registry,
)
from gcap.core.alco import AlcoPackageName
from gcap.errors import (
    ManifestMissingError,
    RunLogClosedError,
    RunLogStoreRequiredError,
    SchemaConformanceError,
)
from gcap.graph.context import GraphContext, ProposalHints
from gcap.graph.pipeline import run_case as run_case_graph
from gcap.harness.cases import to_agent_visible_case
from gcap.harness.grading import grade_case_outcome
from gcap.harness.manifest_binding import (
    validate_run_list_against_manifest,
    validate_runtime_against_manifest,
)
from gcap.harness.rate_limit import RateLimiter, ThrottledLLMClient, make_semaphore
from gcap.harness.record import build_run_write
from gcap.harness.seeding import derive_seed
from gcap.oracle.schema import OracleLabels
from gcap.store.db import session_scope
from gcap.store.repository import RunWrite, save_run
from gcap.store.run_record import (
    FailureKind,
    GovernanceComponents,
    RunRecord,
    Split,
    StateComponents,
)


def assert_confirmatory_run_permitted(manifest: FreezeManifest, manifest_store: FreezeManifestStore | None) -> None:
    """Mirrors the M1 analysis-side invariant-8 guard: a confirmatory
    (``split="confirm"``) run must be traceable to a genuinely committed
    freeze manifest, not merely constructed in memory by whatever process
    happens to be running the harness. Raises before any graph work
    happens — nothing about starting a confirmatory run is attempted once
    this passes without a valid committed manifest behind it."""
    if manifest_store is None:
        raise ManifestMissingError(
            "Cannot start a confirmatory (split='confirm') run: no FreezeManifestStore was configured "
            "for this Harness, so there is nothing to check the manifest against."
        )
    active = manifest_store.active()  # raises ManifestMissingError if nothing is committed
    if active.manifest_hash() != manifest.manifest_hash():
        raise ManifestMissingError(
            f"This Harness's manifest ({manifest.manifest_version!r}) does not match the actively "
            f"committed manifest ({active.manifest_version!r}); a confirmatory run's manifest_hash "
            "must equal the committed one (Section 9.2 rule 2)."
        )


_FAILURE_REASON_MAX_LEN = 2000


def _format_failure_reason(exc: BaseException) -> str:
    """P1-8 (code review): ``failure="runtime"``/``"schema"`` alone used to
    carry no detail at all -- a real API error, a rework cycle exhausted on
    a genuinely malformed schema, and any other unexpected exception were
    all indistinguishable in the store. ``{type}: {message}`` is enough to
    triage without a debugger attached; truncated defensively since an
    exception's message is arbitrary, caller-uncontrolled text (e.g. a
    provider's full HTTP error body) and this is a database column, not a
    log file."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) > _FAILURE_REASON_MAX_LEN:
        text = text[: _FAILURE_REASON_MAX_LEN - 3] + "..."
    return text


def _failure_run_write(
    *,
    bench_case: BenchCase,
    config_name: ConfigName,
    split: Split,
    block: str,
    replicate: int,
    draw_id: str | None,
    manifest_hash: str,
    provider_seed: int | None,
    failure: FailureKind,
    model_snapshot: str,
    failure_reason: str | None = None,
) -> RunWrite:
    run_record = RunRecord(
        run_id=uuid.uuid4(),
        config=config_name,
        split=split,
        block=block,
        case_id=bench_case.case_id,
        episode_id=bench_case.case_id if bench_case.kind == "episode" else None,
        replicate=replicate,
        draw_id=draw_id,
        family=bench_case.family,
        materiality=bench_case.materiality,
        tenor_stratum=bench_case.tenor_stratum,
        model_snapshot=model_snapshot,
        policy_version=bench_case.policy_version,
        state_version_in=bench_case.initial_state.version,
        provider_seed=provider_seed,
        manifest_hash=manifest_hash,
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
        failure=failure,
        failure_reason=failure_reason,
    )
    return RunWrite(run_record=run_record)


@dataclass(frozen=True)
class RunSpec:
    """One (case, config, replicate) unit of work for run_batch."""

    bench_case: BenchCase
    config_name: ConfigName
    replicate: int
    split: Split
    block: str
    oracle_labels: OracleLabels
    draw_id: str | None = None
    hints_by_proposal: Mapping[str, ProposalHints] | None = None


@dataclass
class Harness:
    """Owns everything constant across a batch of runs: the LLM adapter,
    the shared D1/D2 call cache (scoped to this Harness instance — see
    agents/cache.py), the concurrency semaphore/rate limiter, and the
    freeze manifest every RunRecord it writes is stamped with. The shared
    cache is wired into the graph context ONLY for config_name in
    {"D1", "D2"} (see ``_graph_context``) — every other config never
    touches it, so a batch that mixes several non-paired configs (e.g. B
    and C) through one Harness instance for the same case/replicate can
    never collide with it, even though research_node/trade_node's prompts
    are identical across ALL configs (they depend only on the proposal,
    never on config flags)."""

    adapter: LLMClient
    manifest: FreezeManifest
    session_factory: sessionmaker[Session]
    manifest_store: FreezeManifestStore | None = None
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    numerical_tolerances: Mapping[str, Decimal] = field(
        default_factory=lambda: {
            "revenue_growth_yoy": Decimal("0.0005"),
            "portfolio_return_pct": Decimal("0.0005"),
        }
    )
    default_tolerance: Decimal = Decimal("0.005")
    day_count_basis: Decimal = Decimal("365")
    marginal_band_ann: Decimal = Decimal("0.0050")
    alco_package: AlcoPackageName = "NORMAL"
    # Section 9.6's revised-margin recomputation fractions (to-decide — see
    # config.registry.RevisedMarginRecomputation); defaults mirror that
    # registry class's own placeholders.
    hedge_rp_reduction_pct: Decimal = Decimal("0.50")
    limit_addon_reduction_pct: Decimal = Decimal("0.50")
    # P1-9 (code review): threaded into GraphContext -> agents.nodes.
    # trade_node's node-level domain check on TradeProposal.tau_days.
    # Default mirrors config/registry.py's TenorRestructuringBounds
    # placeholder.
    tenor_restructuring_bounds: TenorRestructuringBounds = field(default_factory=TenorRestructuringBounds)
    # None (the default) is an in-memory, process-local cache -- exactly
    # the original behaviour. Passing agents.cache_store.SqliteCacheStore
    # makes D1/D2's shared Research/Trade calls replay across separate
    # process invocations instead of only within this one (see
    # agents/cache.py's module docstring and harness/cli.py's
    # ``run --cache-db`` flag).
    cache_store: CacheStore | None = None
    # P0-01 (v2.0.0 code review): the whole parameter registry, so
    # run_case can enforce ``ParameterRegistry.assert_ready_for_split``
    # itself instead of that guard existing only as a method nothing in
    # the real run path ever called. Defaults to ``default_registry()`` --
    # the fully UNCALIBRATED sandbox placeholder set -- deliberately: a
    # caller that runs ``split="confirm"``/``"audit"`` without explicitly
    # passing a registry with every M3-freeze-required to-decide parameter
    # marked ``calibrated=True`` gets refused, exactly like a caller with
    # no ``manifest_store`` gets refused by ``assert_confirmatory_run_
    # permitted`` below. dev/pilot/roadmap runs are unaffected: assert_
    # ready_for_split is a no-op for those splits by its own design (that
    # is what a pilot calibrates against).
    registry: ParameterRegistry = field(default_factory=default_registry)
    # P1-03 (v2.0.0 code review): closing the run log (RunLogStore.close,
    # written by `gcap close-run-log`) is a declaration that the
    # confirmatory corpus for a manifest_hash is COMPLETE --
    # analysis.pipeline/build_confirmatory_report already refuse to read
    # confirmatory endpoints before it (the read-side half of invariant 8).
    # Before this field existed, nothing on the WRITE side ever consulted a
    # RunLogStore at all: a caller could close the log and then still run
    # `gcap run --split confirm` again against the same manifest and
    # successfully insert new rows, silently invalidating the completeness
    # the closure was supposed to attest to.
    #
    # P1-01 (v2.1.0 third code review round): None is STILL the default
    # (every non-confirm split -- dev/pilot/audit/roadmap -- never needs
    # one), but for split="confirm" specifically, ``run_case`` now REFUSES
    # outright (RunLogStoreRequiredError) rather than silently skipping the
    # closure check the way a ``None`` used to. The review's own
    # reproduction was exactly this: a closed marker genuinely existed, but
    # the confirm run succeeded anyway because ``run_log_store=None`` opted
    # the write-side check out entirely.
    run_log_store: RunLogStore | None = None

    def __post_init__(self) -> None:
        self._semaphore = make_semaphore(self.concurrency)
        self._rate_limiter = RateLimiter.from_concurrency_config(self.concurrency)
        self._shared_cache = SharedCallCache(store=self.cache_store)
        self._cache_lock = Lock()

    def _graph_context(
        self,
        *,
        bench_case: BenchCase,
        config_name: ConfigName,
        hints_by_proposal: Mapping[str, ProposalHints],
        manifest_hash: str,
    ) -> GraphContext:
        throttled = ThrottledLLMClient(self.adapter, semaphore=self._semaphore, rate_limiter=self._rate_limiter)
        # The shared cache is wired in ONLY for D1/D2 — the one declared
        # shareable pair (agents/cache.py's SHAREABLE_NODE_NAMES +
        # is_d1_d2_pair). research_node/trade_node's prompts depend only
        # on the proposal, never on config flags, so ANY two configs
        # sharing a (case_id, replicate) would otherwise produce the
        # identical CacheKey and collide — correctly raising
        # UndeclaredCacheHitError for a real batch that runs several
        # non-paired configs (e.g. B and C) through one Harness instance.
        # Passing None for every config except D1/D2 means those other
        # configs simply never touch the cache at all, so the collision
        # can't happen; D1/D2 keep getting the real cost saving this cache
        # exists for.
        shared_cache = self._shared_cache if config_name in ("D1", "D2") else None
        return GraphContext(
            adapter=throttled,
            token_budget=self.token_budget,
            numerical_tolerances=dict(self.numerical_tolerances),
            default_tolerance=self.default_tolerance,
            day_count_basis=self.day_count_basis,
            marginal_band_ann=self.marginal_band_ann,
            policy_rules=bench_case.policy_rules,
            policy_context_values=dict(bench_case.policy_context_values),
            policy_evaluated_at=bench_case.as_of,
            policy_version=bench_case.policy_version,
            alco_package=self.alco_package,
            hedge_rp_reduction_pct=self.hedge_rp_reduction_pct,
            limit_addon_reduction_pct=self.limit_addon_reduction_pct,
            tenor_restructuring_bounds=self.tenor_restructuring_bounds,
            shared_cache=shared_cache,
            hints_by_proposal=dict(hints_by_proposal),
            manifest_hash=manifest_hash,
        )

    def run_case(
        self,
        *,
        bench_case: BenchCase,
        config_name: ConfigName,
        replicate: int,
        split: Split,
        block: str,
        oracle_labels: OracleLabels,
        draw_id: str | None = None,
        hints_by_proposal: Mapping[str, ProposalHints] | None = None,
    ) -> tuple[str, bool]:
        # P0-01 (v2.0.0 code review): a confirm/audit run against a
        # registry that still has an uncalibrated to-decide placeholder
        # must never proceed -- this is the SAME guard
        # ParameterRegistry.assert_ready_for_split has always implemented,
        # now actually invoked on the real run path instead of only its
        # own unit test. Runs BEFORE the confirm-only manifest check below
        # so an uncalibrated registry is rejected even for an "audit" run,
        # which has no manifest to check at all.
        self.registry.assert_ready_for_split(split)
        if split == "confirm":
            assert_confirmatory_run_permitted(self.manifest, self.manifest_store)
            # P0-01 (v2.1.0 third code review round): assert_confirmatory_
            # run_permitted just above only proves the manifest OBJECT is
            # self-consistent with what was committed -- it says nothing
            # about whether the registry/model/prompts/CONFIG_TABLE/
            # analysis-code/concurrency actually in use right now conform
            # to what that manifest's own fields recorded. This is the
            # separate, additional runtime-vs-manifest binding barrier --
            # see gcap.harness.manifest_binding's module docstring.
            validate_runtime_against_manifest(
                self.manifest,
                registry=self.registry,
                model_snapshot=getattr(self.adapter, "model_snapshot", "unknown-model-snapshot"),
                concurrency=self.concurrency,
                token_budget=self.token_budget,
                numerical_tolerances=dict(self.numerical_tolerances),
                default_tolerance=self.default_tolerance,
                day_count_basis=self.day_count_basis,
                marginal_band_ann=self.marginal_band_ann,
                hedge_rp_reduction_pct=self.hedge_rp_reduction_pct,
                limit_addon_reduction_pct=self.limit_addon_reduction_pct,
                tenor_restructuring_bounds=self.tenor_restructuring_bounds,
            )
            validate_run_list_against_manifest(
                self.manifest,
                [(config_name, bench_case.case_id, replicate, draw_id)],
            )
            # P1-01 (v2.1.0 third code review round): run_log_store is no
            # longer optional for split="confirm" -- see RunLogStoreRequired
            # Error's own docstring for why a silently-skipped closure check
            # was itself the bug the review reproduced.
            if self.run_log_store is None:
                raise RunLogStoreRequiredError(
                    "Cannot start a confirmatory (split='confirm') run: no RunLogStore was configured "
                    "for this Harness. Closure is a mandatory barrier for confirmatory runs -- pass "
                    "--run-log-store (CLI) / run_log_store= (Harness) pointing at the RunLogStore this "
                    "manifest's confirmatory corpus is tracked under."
                )
            # P1-03 (v2.0.0 code review): the write-side half of invariant
            # 8's run-log closure barrier -- see RunLogClosedError's own
            # docstring above. Checked here, before any graph work, exactly
            # like the manifest guard just above it -- a fast-fail so a run
            # already known to be closed never wastes an LLM call. P1-01
            # (v2.1.0 third code review round) adds a SECOND check
            # immediately before the DB write below, closing the window
            # this early check alone leaves open (see below).
            if self.run_log_store.is_closed(self.manifest.manifest_hash()):
                raise RunLogClosedError(
                    f"Cannot start a confirmatory (split='confirm') run: the run log for manifest_hash "
                    f"{self.manifest.manifest_hash()!r} is already closed (RunLogStore). Closure "
                    "declares the confirmatory corpus for this manifest complete; no further "
                    "confirmatory runs may be written against it. A genuinely new manifest_version "
                    "is required to run more confirmatory cases."
                )

        config = CONFIG_TABLE[config_name]
        manifest_hash = self.manifest.manifest_hash()
        provider_seed = derive_seed(manifest_hash, bench_case.case_id, replicate, draw_id)

        agent_case = to_agent_visible_case(bench_case)
        # Cache sharing across configs is only ever correct for the
        # declared D1/D2 pair. SharedCallCache.get_or_call is itself now
        # internally single-flight-locked per cache key (P1-6, agents/
        # cache.py) so concurrent workers racing on the SAME key can no
        # longer both miss and both call; this harness-level lock instead
        # just keeps GraphContext construction itself (a cheap, pure
        # dataclass build) from racing on shared harness state while
        # building it.
        with self._cache_lock:
            ctx = self._graph_context(
                bench_case=bench_case,
                config_name=config_name,
                hints_by_proposal=hints_by_proposal or {},
                manifest_hash=manifest_hash,
            )

        failure: FailureKind = "none"
        failure_reason: str | None = None
        run_write: RunWrite | None = None
        try:
            outcome = run_case_graph(
                case=agent_case, config=config, config_name=config_name, ctx=ctx, replicate=replicate
            )
            graded = grade_case_outcome(outcome, config, oracle_labels)
        except SchemaConformanceError as exc:
            failure = "schema"
            failure_reason = _format_failure_reason(exc)
        except Exception as exc:  # invariant 9: caught here, recorded, never a crash
            failure = "runtime"
            failure_reason = _format_failure_reason(exc)
        else:
            run_write = build_run_write(
                case=agent_case,
                outcome=outcome,
                graded=graded,
                oracle_labels=oracle_labels,
                config=config,
                config_name=config_name,
                ctx=ctx,
                split=split,
                block=block,
                replicate=replicate,
                draw_id=draw_id,
                manifest_hash=manifest_hash,
                provider_seed=provider_seed,
                tenor_stratum=bench_case.tenor_stratum,
                failure="none",
            )

        if run_write is None:
            run_write = _failure_run_write(
                bench_case=bench_case,
                config_name=config_name,
                split=split,
                block=block,
                replicate=replicate,
                draw_id=draw_id,
                manifest_hash=manifest_hash,
                provider_seed=provider_seed,
                failure=failure,
                failure_reason=failure_reason,
                model_snapshot=getattr(self.adapter, "model_snapshot", "unknown-model-snapshot"),
            )

        # P1-01 (v2.1.0 third code review round): the TOCTOU-closing re-
        # check. The early check above (before graph work) is a fast-fail
        # only -- it says nothing about whether the run log was closed
        # WHILE this case's graph/LLM work was in flight, which is exactly
        # the review's own reproduction (close the log mid-flight; the
        # already-started run still finishes and writes). Re-checking here,
        # immediately before the DB write and with no I/O of any kind left
        # in between, shrinks that window to as close to zero as a file-
        # backed marker store and a separate SQL transaction can get in
        # this sandbox harness (true cross-process transactional atomicity
        # would need both under one shared transactional resource, which
        # RunLogStore -- deliberately a simple, auditable file marker, see
        # its own docstring -- does not attempt to be).
        if split == "confirm":
            assert self.run_log_store is not None  # guaranteed by the mandatory check above
            # The shared lock is held through session_scope's COMMIT.  A
            # close takes the exclusive counterpart, so it cannot create a
            # marker between this final check and the durable SQL write.
            with self.run_log_store.write_guard(manifest_hash):
                if self.run_log_store.is_closed(manifest_hash):
                    raise RunLogClosedError(
                        f"Cannot persist this confirmatory (split='confirm') run: the run log for "
                        f"manifest_hash {manifest_hash!r} was closed while this case's graph/LLM work "
                        "was in flight. The outcome is discarded and was not written."
                    )
                with session_scope(self.session_factory) as session:
                    return save_run(session, run_write)

        with session_scope(self.session_factory) as session:
            return save_run(session, run_write)

    def run_batch(self, specs: Sequence[RunSpec], *, max_workers: int | None = None) -> list[tuple[str, bool]]:
        """Runs many (case, config, replicate) units concurrently. Worker
        count defaults to the model-call semaphore size: each worker here
        executes a whole case (several LLM calls in sequence), not one
        call each, so sizing the pool any larger would only add workers
        blocked waiting on the same semaphore, never more real
        concurrency."""
        # P0-01 (v2.1.0 third code review round): the batch-level half of
        # the runtime-vs-manifest binding barrier -- only a full batch of
        # planned RunSpecs (never one run_case call in isolation) can
        # reconstruct the run-list hash to compare against
        # manifest.run_list_hash (Section 9.1's frozen confirmatory run
        # plan, which freezes the confirmatory units specifically). Only
        # the confirm-split subset is hashed -- RunSpec.split is per-case
        # (harness.cli._cmd_run sets it from each loaded BenchCase.split),
        # so a single ``run_batch`` call is not structurally guaranteed to
        # be all-confirm-or-none even though every real caller's cases file
        # is homogeneous in practice (a confirm-split cases file can only
        # ever reach here via the sealed-store path -- ``_load_open_cases``
        # already refuses any plaintext confirm record outright). Skipped
        # entirely when no spec in this batch is confirm-split, so dev/
        # pilot/audit/roadmap batches never pay for or are blocked by a
        # check that only confirmatory runs are bound by.
        #
        # Ordered to mirror run_case's own guard sequence (calibration ->
        # manifest-object self-consistency -> ...) rather than jumping
        # straight to the run-list check: an uncalibrated registry or an
        # uncommitted/mismatched manifest OBJECT is refused with its own,
        # more fundamental error first, exactly as it would be if this
        # batch had gone through run_case one spec at a time instead of via
        # run_batch. Without this ordering, a caller with BOTH problems
        # (uncalibrated registry AND a stale run-list) would see only the
        # run-list mismatch and never learn the registry was uncalibrated
        # too until after fixing the first issue and resubmitting.
        confirm_specs = [spec for spec in specs if spec.split == "confirm"]
        if confirm_specs:
            self.registry.assert_ready_for_split("confirm")
            assert_confirmatory_run_permitted(self.manifest, self.manifest_store)
            validate_run_list_against_manifest(
                self.manifest,
                [(spec.config_name, spec.bench_case.case_id, spec.replicate, spec.draw_id) for spec in confirm_specs],
            )

        workers = max_workers or self.concurrency.model_call_semaphore
        results: list[tuple[str, bool] | None] = [None] * len(specs)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            future_to_idx = {
                pool.submit(
                    self.run_case,
                    bench_case=spec.bench_case,
                    config_name=spec.config_name,
                    replicate=spec.replicate,
                    split=spec.split,
                    block=spec.block,
                    oracle_labels=spec.oracle_labels,
                    draw_id=spec.draw_id,
                    hints_by_proposal=spec.hints_by_proposal,
                ): i
                for i, spec in enumerate(specs)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return [r for r in results if r is not None]
