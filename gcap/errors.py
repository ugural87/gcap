"""Explicit error taxonomy for G-CAP.

Every raised error in gcap/ is one of these types (or a subclass). Generic
``ValueError``/``RuntimeError`` are reserved for genuine programming bugs
caught by tests, never for expected domain failures. This lets the harness
and analysis layer distinguish "the run failed as an outcome" (invariant 9 —
failures are recorded, not swallowed) from "the code itself is broken".
"""

from __future__ import annotations


class GCapError(Exception):
    """Base class for every G-CAP domain error."""


# ---------------------------------------------------------------------------
# config/
# ---------------------------------------------------------------------------
class ConfigError(GCapError):
    """Configuration registry or flag errors."""


class InvalidConfigFlagsError(ConfigError):
    """A configuration flag combination outside the frozen A-E table."""


class UncalibratedParameterError(ConfigError):
    """A Section-10 'to decide' parameter was used where calibration is
    required (e.g. starting a confirmatory run) but is still a placeholder.
    """


class ManifestError(ConfigError):
    """Freeze manifest violations."""


class ManifestImmutableError(ManifestError):
    """Attempted to mutate a committed freeze manifest in place."""


class ManifestMissingError(ManifestError):
    """No committed freeze manifest exists where one is required."""


class RuntimeManifestMismatchError(ManifestError):
    """A confirmatory (``split='confirm'``) run's ACTUAL runtime facts --
    the parameter registry in use, the adapter's real model snapshot, the
    node-prompt-building source code, the ``CONFIG_TABLE`` content, the
    ``gcap.analysis`` source tree, the concurrency configuration, or the
    exact set of planned (case, config, replicate) units -- diverge from
    what a committed ``FreezeManifest`` recorded for each of those fields
    (P0-01, v2.1.0 third code review round).

    Distinct from ``ManifestMissingError`` (raised by ``assert_confirmatory_
    run_permitted``): that check only proves the ``FreezeManifest`` OBJECT
    handed to ``Harness`` is byte-identical to the one actually committed --
    self-consistency of the manifest, nothing about whether the code/model/
    prompts/registry/run-plan actually running right now conforms to what
    that manifest recorded. This error is what a genuine divergence between
    "what was frozen" and "what is actually running" raises -- see
    ``gcap.harness.manifest_binding``'s module docstring for the full
    rationale and exactly which fields are compared."""


# ---------------------------------------------------------------------------
# core/
# ---------------------------------------------------------------------------
class CoreValidationError(GCapError):
    """A deterministic engine rejected its inputs (invariant-guarding)."""


class FTPValidationError(CoreValidationError):
    """FTP engine input validation failure (e.g. double-counted term spread,
    TradeAlpha <= -1)."""


class LedgerInfeasibleError(CoreValidationError):
    """apply() was called with an action the current state cannot support."""


class StaleStateError(CoreValidationError):
    """The package's recorded state version differs from the ledger's
    current version at gate time."""


class PolicyPredicateNotRegisteredError(CoreValidationError):
    """A policy rule references a predicate id with no registered callable.
    Text can never become enforceable without this registration."""


# ---------------------------------------------------------------------------
# oracle/
# ---------------------------------------------------------------------------
class OracleError(GCapError):
    """Oracle labelling errors."""


class LabelSealError(GCapError):
    """Sealed confirmatory label store errors."""


class LabelDecryptionRefusedError(LabelSealError):
    """The harness attempted to read confirmatory labels without an active
    committed freeze manifest (invariant 8)."""


class SealedRecordIdentityMismatchError(LabelSealError):
    """A ``bench.sealing`` ``SealedLabelStore``/``SealedCaseStore`` record
    decrypted successfully but its OWN embedded identity (``OracleLabels.
    case_id`` / ``BenchCase.case_id``) does not match the ``case_id`` the
    caller asked for -- i.e. the ``case_id`` used to look up the on-disk
    path (P1-02, v2.1.0 third code review round).

    This is a stronger, more specific check than the whole-store content
    hash comparison ``LabelDecryptionRefusedError`` already raises on: that
    check proves the STORE's aggregate content still matches what the
    freeze manifest recorded, which does catch a swapped/corrupted file
    too (a swap changes the {filename: content} pairing
    ``config.hashing.hash_sealed_store`` hashes) -- but only for as long as
    that check is actually re-run. Binding each record's own decrypted
    content to the ``case_id`` its path was requested under is a second,
    independent line of defence that holds record-by-record regardless of
    when the store was last hash-verified, and gives a far more precise
    diagnostic (which case_id was actually returned) than a generic
    store-wide hash mismatch would."""


# ---------------------------------------------------------------------------
# bench/
# ---------------------------------------------------------------------------
class BenchError(GCapError):
    """Benchmark case generation/verification errors."""


class UnverifiedCaseError(BenchError):
    """A case without a second-reviewer verification id was used in a
    split that requires one."""


# ---------------------------------------------------------------------------
# agents/
# ---------------------------------------------------------------------------
class AgentError(GCapError):
    """Agent-layer errors."""


class SchemaConformanceError(AgentError):
    """Agent output failed strict JSON-schema validation after the one
    bounded rework cycle."""


class TokenBudgetExceededError(AgentError):
    """A call would exceed its hard input/output token cap and is refused
    rather than sent over cap."""


class ModelIdentityMismatchError(AgentError):
    """The configured model id does not match what startup validation
    expects (product name vs. real API model id confusion guard)."""


class CacheSchemaMismatchError(AgentError):
    """A ``SqliteCacheStore`` was pointed at a file whose existing
    ``call_cache`` table has a different column set than this version of
    gcap expects (P1-04, v2.0.0 code review) -- typically a durable
    --cache-db file written by an older version (e.g. pre-P1-6, which
    widened the cache key from 4 columns to 7). Raised once, at
    construction time, instead of surfacing later as an opaque
    ``sqlite3.OperationalError`` from deep inside a real run."""


# ---------------------------------------------------------------------------
# store/
# ---------------------------------------------------------------------------
class StoreError(GCapError):
    """Persistence-layer errors."""


class AppendOnlyViolationError(StoreError):
    """An UPDATE or DELETE was attempted against an append-only entity."""


class RunRecordIncompleteError(StoreError):
    """A required RunRecord field was neither populated nor explicitly
    marked null by design."""


# ---------------------------------------------------------------------------
# harness/
# ---------------------------------------------------------------------------
class HarnessError(GCapError):
    """Execution-harness errors."""


class PairingViolationError(HarnessError):
    """Paired runs do not carry identical upstream hashes."""


class UndeclaredCacheHitError(HarnessError):
    """A cache hit occurred outside a declared shareable pair (Section 8.2).
    This is always a bug and must raise, never silently degrade."""


class CacheIsolationError(HarnessError):
    """A stochastic replicate's LLM cache entry was about to be reused by a
    different replicate."""


class RateLimitBudgetError(HarnessError):
    """Client-side rate limiter or token-meter guard tripped."""


class RequestExceedsRateLimitError(RateLimitBudgetError):
    """A single request's own weight (1 request, or its estimated
    input/output token cost) exceeds a configured per-minute ceiling on
    its own -- no amount of waiting can ever bring it under budget, since
    the sliding window has nothing older left to evict (P2-02, v2.0.0 code
    review). Before this existed, ``harness.rate_limit._Bucket.wait_needed``
    let such a request through immediately as a best-effort exception
    (P2-11's fix for the IndexError this same situation used to raise) --
    correct in that it no longer crashed, but it meant the limiter could
    silently exceed its own configured ceiling with no signal to the
    caller. This raises instead: the caller (or a human reviewing a failed
    run) sees exactly which ceiling was structurally impossible to satisfy
    and can raise it in ``ConcurrencyConfig`` or reduce the request's cost,
    rather than the limiter quietly doing the thing it exists to prevent."""


class RunLogClosedError(HarnessError):
    """A ``split='confirm'`` run was attempted against a manifest_hash whose
    run log (``gcap.config.manifest.RunLogStore``) has already been closed
    (P1-03, v2.0.0 code review). Closing the run log is a declaration that
    the confirmatory corpus for that manifest is COMPLETE — ``analysis.
    pipeline``/``build_confirmatory_report`` trust that declaration and
    refuse to read confirmatory endpoints before it, but nothing previously
    stopped a new confirmatory run from being written to the store AFTER
    it, silently invalidating the completeness the closure was supposed to
    attest to. ``Harness.run_case`` now refuses such a write outright,
    before any graph work happens, making closure a real barrier on both
    the read side (already true) and the write side (this).

    P1-01 (v2.1.0 third code review round): the write-side check above ran
    exactly ONCE, before any graph/LLM work -- a run log closed WHILE a
    case's graph work was in flight (an LLM call in progress, say) still
    wrote its RunRecord successfully once that work finished, since nothing
    re-checked closure between then and the DB write (reproduced by the
    review: close the log mid-flight, let the run finish, the record is
    persisted anyway). ``Harness.run_case`` now re-checks immediately
    before ``store.repository.save_run`` too, not only at the top of the
    method -- the same ``RunLogClosedError`` is raised either way, so a
    caller (or a test) never needs to know which of the two checks fired."""


class RunLogStoreRequiredError(HarnessError):
    """P1-01 (v2.1.0 third code review round): a ``split='confirm'`` run was
    attempted with no ``Harness.run_log_store`` configured. Before this, a
    ``None`` ``run_log_store`` silently meant "the closure check is skipped
    entirely" (the field's own docstring: "the check below is a no-op
    unless a RunLogStore is actually supplied") -- so the very same `gcap
    run --split confirm` invocation that ``RunLogClosedError`` above
    correctly refuses when ``--run-log-store`` IS given could be re-run
    with that one flag simply omitted and would write successfully even
    though the run log for this manifest_hash was, in fact, already closed
    (reproduced by the review: a closed marker exists, but ``run_log_
    store=None`` lets the run through anyway). Closure is now a mandatory
    barrier for every confirmatory run, exactly like ``manifest_store``
    already is via ``ManifestMissingError`` -- there is no more "opt out of
    the check" path for ``split='confirm'``."""


class ConfirmatoryCorpusIncompleteError(HarnessError):
    """P1-01 (v2.1.0 third code review round): ``gcap close-run-log`` was
    asked to write a closure record for a manifest_hash whose PERSISTED
    ``split='confirm'`` RunRecords do not yet hash-match that manifest's
    frozen ``run_list_hash`` (Section 9.1's frozen confirmatory run plan --
    the same set ``gcap.harness.manifest_binding.validate_run_list_against_
    manifest`` checks on the WRITE side, at batch-submission time; this is
    its read-side mirror, checked at CLOSE time). Before this, ``close-run-
    log`` wrote the closure marker unconditionally -- no check that every
    (config, case_id, replicate, draw_id) unit the frozen plan called for
    actually has a persisted outcome (success OR a recorded failure --
    invariant 9; a failed case still counts as "attempted", never as
    missing) before the corpus was declared complete and read-gated
    confirmatory endpoints unlocked."""


class CacheClaimLostError(GCapError):
    """A cache worker tried to publish after its lease had been stolen.

    The stale worker must not overwrite the new owner's result or delete
    the new owner's claim. ``SharedCallCache.get_or_call`` propagates this
    error; its enclosing harness records the attempt as a run failure under
    invariant 9. A higher-level caller may explicitly retry the run later.
    """


class ConfirmSplitPlaintextRefusedError(HarnessError):
    """A ``split='confirm'`` ``BenchCase``/``OracleLabels`` pair was found in
    a plaintext open-case file (P0-03, v2.0.0 code review). Confirmatory
    ground truth must only ever exist on disk inside a ``bench.sealing``
    encrypted store, gated on a committed freeze manifest (invariant 8) --
    never in the ordinary plaintext JSON ``gcap generate-cases`` writes for
    every other split. Raised the moment such a record is loaded, before any
    graph/harness work happens, regardless of what ``--manifest-store``
    state might otherwise have allowed the run to proceed with."""


# ---------------------------------------------------------------------------
# analysis/
# ---------------------------------------------------------------------------
class AnalysisError(GCapError):
    """Statistical analysis layer errors."""


class ConfirmatoryUnsealError(AnalysisError):
    """Analysis code attempted to read confirmatory endpoints before the
    run log was closed."""


class InsufficientLhcCandidatesError(AnalysisError):
    """The LHC replay candidate pool cannot satisfy the requested
    selection: too few candidates overall, or too few in some tenor
    stratum to meet its floor (Section 9.5's ">=6 per tenor stratum")."""


class NoEligibleConcurrencyCandidateError(AnalysisError):
    """None of the measured worker-pool candidates cleared the frozen
    reliability gates (rate-limit fraction, error fraction, p95 latency) --
    analysis.concurrency.select_worker_pool must never silently fall back
    to an unmeasured or ineligible size."""
