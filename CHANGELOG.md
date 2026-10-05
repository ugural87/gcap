# Changelog

## v2.3.1 — independent validation cleanup

An independent validation found no functional regression and confirmed every
v2.3 security/integrity claim. It reported six residual release-hygiene,
coverage and documentation findings, all now closed:

- removed the orphaned, incomplete top-level `alembic/` tree and its release
  allowlist entry; packaged migrations have one canonical home under
  `gcap/migrations/`;
- added a negative test proving a stale cache owner receives
  `CacheClaimLostError` and cannot publish over a stolen lease;
- corrected `CacheClaimLostError` documentation to describe propagation and
  invariant-9 run-failure recording rather than implying an automatic retry;
- added a negative migration test for the partial-`detected_*` defensive
  branch;
- separated and completed the README version paragraphs, and documented why
  v2.3 was a minor invariant-boundary bump rather than a patch or major bump;
- restored the release's explicit net-result summary.

Publication tooling added after validation includes a Python 3.11/3.12 GitHub
Actions quality pipeline, a documented branch/release workflow, and the MIT
License. These additions do not change runtime behavior.

### Net result

475 tests passing, strict mypy clean across 80 source files, Ruff clean, and
all 5 import-linter contracts kept. Wheel installation/migration and release
archive content remain covered by executable tests.

## v2.3 — confirmatory integrity closure round

This release closes the residual findings independently reproduced against
v2.2.  The changes are enforced at the shared boundaries rather than only in
the standard CLI path:

- confirmatory runtime binding is fail-closed for the primary model and every
  required node prompt, and validates all Harness financial/numerical settings
  against the frozen parameter registry;
- both `run_batch` and direct `run_case` enforce frozen run-list membership;
- confirmatory writes and run-log closure use a cross-process reader/writer
  barrier, covering the final SQL commit and close-time completeness check;
- confirmatory analysis requires real manifest and run-log stores, rechecks
  frozen-corpus completeness, and ignores the historical bare boolean as an
  authority;
- versionless legacy databases are stamped at the schema revision their actual
  columns represent, rather than unconditionally at the initial revision;
- Alembic resources ship inside the wheel and migrations work outside a source
  checkout;
- cache leases carry owner tokens, so an expired worker cannot release or
  overwrite a stolen claim;
- governance detection has hard-rule-only scalar semantics in every arm;
- run-list hashing uses canonical typed JSON rather than ambiguous comma joins;
- proxy-sensitive tests are isolated from ambient proxy configuration.

Negative regression coverage includes every reproduced bypass, a realistic
pre-tracking database, and an installed-wheel migration smoke test.

### Net result

473 tests passing (up from 466 in v2.2), strict mypy clean across 80 source
files, Ruff clean, and all 5 import-linter contracts kept. The installed-wheel
migration smoke test and ambient-proxy test run are included in that total.

## v2.2 — third, independent code review fix round

A third independent code review — of the `v2.1` codebase produced by the review
round just below — raised 10 findings (2 P0, 5 P1, 3 P2). Same discipline as the
two rounds before it: each finding was first verified against the running code
(reproduced or traced to the exact lines responsible) before any fix was written,
each fix shipped with its own new regression test, and a full green
`pytest`/`mypy`/`ruff`/`lint-imports` pass ran before moving to the next finding.
All 10 were confirmed accurate; none were false positives.

### P0 — correctness bugs undermining the confirmatory pilot's own guarantees

- **P0-01: a committed freeze manifest was not actually bound to the runtime
  executing under it.** `FreezeManifest` recorded `run_list_hash` (a hash of the
  frozen run plan) but never the plan itself, so nothing at runtime could check a
  given `Harness.run_batch` call's actual units against what was frozen — a
  manifest could be committed over one run plan and a different plan executed
  under its name with no structural check catching the substitution, only a hash
  a caller would have to think to recompute and compare by hand. Fixed by adding
  a new `run_list: tuple[...]` field to `FreezeManifest` itself (additive,
  defaults to `()`, backward compatible with every existing serialized manifest)
  and a new `gcap.harness.manifest_binding` module with barrier functions that
  the real run path now calls to refuse a mismatch before any graph work happens.
- **P0-02: a confirmatory analysis could silently mix records from different
  splits.** Nothing stopped `dev`/`pilot`/`confirm` records sharing one run-log
  store from being read together as if they were one homogeneous confirmatory
  corpus — a `split="confirm"` analysis could be contaminated by earlier
  rehearsal-phase records without any error or warning. Fixed with an explicit
  split check before any record is folded into a confirmatory result.

### P1 — write-barrier, caching, and reporting-parity gaps

- **P1-01: closing a confirmatory run log was bypassable and non-atomic.** The
  prior round's P1-03 fix added a write barrier (`RunLogClosedError`) but the
  closure check itself raced with concurrent writers and the frozen run plan
  wasn't threaded through to what closure actually verified against, so a
  carefully timed write could still land after closure, and closure could
  succeed without confirming every frozen unit had actually run. Fixed by
  threading `run_list` (this round's P0-01 field) through `gcap.pilot.freeze`
  into the manifest, so closure has the real frozen plan to check completeness
  against, not just a hash.
- **P1-02: a sealed store's cached hash-verification result could go stale
  across a single process's own lifetime, and neither sealed store checked a
  decrypted record's own identity against the path used to look it up.** The
  prior round's P1-01 fix cached `hash_sealed_store`'s result per-instance after
  the first successful check — correct at that moment, but a mutation to the
  sealed files *after* that first successful `get()` (from another process, or a
  bug elsewhere in the same one) would go undetected for the rest of that
  instance's life. Separately, `SealedLabelStore.get(case_id)`/
  `SealedCaseStore.get(case_id)` decrypted the file at the path derived from the
  requested `case_id` but never checked that the record's own embedded
  `case_id` matched it — a swapped or misnamed file on disk would be silently
  returned as if it were the requested case. Fixed by removing the cache
  entirely (the hash is now re-verified on every `get()` call) and adding an
  explicit identity check with a new, clearly named
  `SealedRecordIdentityMismatchError`.
- **P1-03: the cross-process single-flight cache claim was orphaned on
  exception, and the existing "cross-process" test wasn't actually testing
  separate OS processes.** If `call_fn` or the store's own `set()` raised after a
  `ClaimableCacheStore` claim was successfully taken, nothing released it —
  every subsequent caller for that key would block until the claim's lease
  timeout expired, turning a transient failure into a manufactured outage for
  every other caller sharing the cache. Separately, the test claiming to cover
  cross-process behaviour used two threads within one interpreter, which cannot
  exercise real OS-level process boundaries at all. Fixed with a `finally`-style
  release-on-exception in `get_or_call` and a new test using
  `multiprocessing.get_context("fork")` with genuinely separate worker
  processes synchronized via on-disk flag files.
- **P1-04: the CLI never applied pending Alembic migrations.** Every `gcap`
  subcommand that touches the database called the test-only `init_schema`
  (`Base.metadata.create_all`) instead of running the real Alembic migration
  chain — a fresh database got today's full schema by accident (create_all
  doesn't know about migrations, it just builds whatever the current ORM models
  say), but an existing database created before a later migration was added
  would never actually be migrated by the CLI at all. Fixed with a new
  `apply_pending_migrations` function that runs the real Alembic chain (stamping
  a legacy create_all-only database at the known initial revision first, when
  one is detected, so it can then upgrade normally), wired into all three
  database-touching CLI commands in place of `init_schema`.
- **P1-05: the governance endpoint's cross-arm contrast was narrowed at the
  report layer without saying so.** The prior round's P0-02 fix correctly
  narrowed the D2-E governance comparison to the one component (`hard_rule`)
  both arms can genuinely detect, but the report-facing key and label were left
  reading plain `"governance_failure"` — indistinguishable, to a reader of the
  analysis output alone, from the full five-component oracle definition used
  everywhere else `"governance_failure"` appears. Fixed by renaming the
  report-facing key/label to `"governance_failure_hard_rule_only"`
  (`gcap.analysis.endpoints`); the underlying `RunRecord.ep_governance_failure`
  database column is unchanged — this is a reporting-layer clarification, not a
  schema change.

### P2 — hermeticity and packaging gaps

- **P2-01: two `OpenAIAdapter` tests were not hermetic.** The prior round's
  P2-01 fix made the adapter's client injectable and tested via
  `httpx.MockTransport`, but two tests exercising the adapter's upstream-context
  prompt rendering still used a global `monkeypatch.setattr(httpx.Client,
  "post", ...)` instead — a class-level patch that bypasses httpx's real
  transport pipeline and affects every `httpx.Client` instance for the duration
  of the test, not just the one under test. Fixed by rewriting both to use the
  same injected-`MockTransport` pattern as the rest of the suite.
- **P2-02: `detected_governance_failure` carried two different meanings under
  one column with no explanation of why.** For an oracle-graded arm it is the
  full five-component composite (matching `ep_governance_failure`'s own
  scope for that arm); for the in-graph-graded arm it can only ever be the
  `hard_rule` component alone (the same structural asymmetry P0-02 in the round
  before this one already established as correct and permanent, not a bug to
  fix). Decided to document rather than split the column, since it is a
  per-arm corpus-composition diagnostic only and never compared across arms
  anywhere in `analysis/`. Fixed with an explicit field-level docstring on
  `RunRecord.detected_governance_failure` and a strengthened existing
  regression test that now asserts the in-graph value directly.
- **P2-03: the release archive packaged test/lint cache state alongside the
  real source.** The previously delivered `v2.1.0.dev0` archive was built by
  hand — effectively "tar up the whole repository directory" — and shipped
  every test-run cache/state directory alongside the real source: 165 of that
  archive's 356 files (`.hypothesis/`, `.import_linter_cache/`, and friends)
  were pytest/hypothesis/mypy/ruff/import-linter scratch state that regenerates
  on the next test run and was never meant to ship. Fixed with a new
  `scripts/package_release.sh` that builds the release archive from an explicit
  source-path allowlist (never a bare "tar everything except a remembered
  exclude list"), a new `.gitignore` (this repo had none before), and a new
  `tests/test_release_packaging.py` that runs the real script as a subprocess
  against this repository's actual working tree and asserts the resulting
  archives contain zero cache-marker paths and exactly the real `.py` file set.

### Net result

466 tests passing, mypy strict clean, ruff clean, 5/5 import-linter contracts
kept. One additive, non-DB schema change (P0-01's `FreezeManifest.run_list`
field — a pydantic model field defaulting to `()`, not a database migration; no
new Alembic revision was added this round) — see this project's
`README.md#versioning` for why that keeps this round a minor version bump
(`2.1` -> `2.2`) rather than a major one.

## v2.1 — second, independent code review fix round

A second independent code review — of the `v2.0` codebase produced by the review
round just below — raised 12 findings (3 P0, 5 P1, 4 P2). Same discipline as
`v2.0`'s own round: each finding was first verified against the running code
(reproduced or traced to the exact lines responsible) before any fix was written,
each fix shipped with its own new regression test, and a full green
`pytest`/`mypy`/`ruff`/`lint-imports` pass ran before moving to the next finding.
All 12 were confirmed accurate; none were false positives.

### P0 — correctness bugs undermining the confirmatory pilot's own guarantees

- **P0-01: the confirmatory calibration guard was never actually called on a real
  run.** `UncalibratedParameterError` existed and was unit-tested in isolation, but
  nothing on the real `gcap run --split confirm`/`gcap run --split audit` CLI path
  ever invoked the check — an uncalibrated Section-10 parameter could silently
  reach a confirmatory or audit run despite the guard's own docstring promising
  otherwise. Fixed by wiring the guard into the real CLI run path, not just a
  standalone unit test.
- **P0-02: D2 vs E's governance endpoint definition disagreed on what a "governance
  failure" even means.** `oracle.labels.label_governance`'s ground truth is an OR
  of five components (`hard_rule`, `unsupported_claim`, `unauthorised_exception`,
  `approval_chain`, `tool_action`); D2 (oracle-graded) compared against the full
  five-component OR while E (in-graph-graded) can only ever detect one of them
  (`hard_rule`, via its own policy engine — the other four are oracle-only ground
  truth invariant 6 forbids `graph/` from ever seeing). The D2-E co-primary
  governance contrast was silently comparing two different questions. Fixed by
  narrowing the oracle-graded branch to `hard_rule` alone — the one component both
  arms can genuinely, comparably detect today.
- **P0-03: a `split=confirm` case could reach the CLI's plaintext case-loading path
  at all.** Nothing rejected a confirmatory-split `BenchCase`/`OracleLabels` pair
  found in an ordinary plaintext open-case file — confirmatory ground truth must
  only ever exist inside a `bench.sealing` encrypted store gated on a committed
  freeze manifest (invariant 8), never in plaintext JSON, regardless of what
  `--manifest-store` state might otherwise have allowed the run to proceed with.
  Fixed with a new `ConfirmSplitPlaintextRefusedError`, raised the moment such a
  record is loaded, before any graph/harness work happens.

### P1 — governance, integrity, and reproducibility gaps

- **P1-01: a sealed confirmatory store's on-disk content was never checked against
  the freeze manifest's own committed hash.** `SealedLabelStore`/`SealedCaseStore`
  verified the manifest was committed, but never that the sealed files actually on
  disk still matched `confirmatory_label_store_hash`/`benchmark_case_files_hash` —
  a store silently edited, corrupted, or swapped after freeze would be read as if
  nothing had changed. Fixed with a new `hash_sealed_store` helper and a real
  content-hash check (cached per-instance) in both stores' `get()`, raising
  `LabelDecryptionRefusedError` on any mismatch.
- **P1-02: PM's recommendation had no causal path into Treasury's own prompt.**
  `ProposalOutcome.pm` carried PM's recommendation, but `treasury_prompt` never
  rendered it into the text a real model receives, so a real Treasury LLM call
  would never actually see what PM recommended — the multi-agent chain would be
  causally broken between PM and Treasury the moment a real adapter is plugged in,
  the same class of gap P0-4's earlier fix (in the prior round) closed for
  Research/Trade but missed for PM/Treasury. Fixed by threading `pm` through
  `treasury_node`/`treasury_prompt` into the rendered prompt text.
- **P1-03: closing a confirmatory run log was not a write barrier, only a read-side
  check.** `analysis.pipeline` already refused to read confirmatory endpoints
  before a run log's closure, but nothing stopped a *new* confirmatory run from
  being written to the store *after* closure, silently invalidating the
  completeness that closure was supposed to attest to. Fixed: `Harness.run_case`
  now refuses such a write outright (`RunLogClosedError`), before any graph work,
  via a new `run_log_store` field and `--run-log-store` CLI flag.
- **P1-04: an older-schema durable SQLite cache file failed with an opaque,
  unrelated error instead of a clear one.** A `--cache-db` file written before
  the prior round's P1-6 fix (which widened the cache key from 4 columns to 7) was
  silently accepted at construction time and only failed later, deep inside a real
  run, as a confusing `sqlite3.OperationalError`. Fixed with a schema check at
  construction time and a new, clearly named `CacheSchemaMismatchError`.
- **P1-05: the durable SQLite cache had no cross-process single-flight
  guarantee.** `SharedCallCache`'s single-flight guarantee (never make the same
  call twice) held within one process's threads but not across separate OS
  processes sharing a `--cache-db` file — two harness processes racing on the same
  key could both make the real call. Fixed with an atomic SQLite-level claim/lease
  mechanism (`ClaimableCacheStore` protocol, a new `call_cache_claim` table, a
  configurable lease timeout to auto-steal an abandoned claim from a crashed
  process) and a claim-loop in `get_or_call`, including a TOCTOU re-check
  immediately after a successful claim to close a race the initial design missed.

### P2 — hermeticity, ceiling-violation, observability, and documentation gaps

- **P2-01: `OpenAIAdapter` was not hermetically testable and silently inherited
  ambient proxy configuration.** No way to inject a test transport meant no real
  test coverage of the adapter's own HTTP-handling logic, and a `trust_env`
  default of `True` (`httpx.Client`'s own default) meant it would silently pick up
  this sandbox's own pre-configured agent proxy — never appropriate for real
  OpenAI API traffic. Fixed with an injectable `httpx.Client` (tested via
  `httpx.MockTransport`, fully hermetic, zero real network) and an explicit,
  secure-by-default `trust_env: bool = False`; the client is now a lazily built,
  reused singleton rather than one-per-call.
- **P2-02: a single oversized request still silently exceeded the rate limiter's
  own configured ceiling.** The prior round's P2-11 fix replaced a crash
  (`IndexError` on a cold bucket) with letting an over-limit single request through
  unthrottled — no longer a crash, but the limiter could silently violate the very
  ceiling it exists to enforce, with no signal to the caller. Fixed: `_Bucket.
  wait_needed` now raises a clear, named, catchable `RequestExceedsRateLimitError`
  instead.
- **P2-03: the raw, ungated detection diagnostic was asymmetric between arms and
  never persisted.** `GradedEndpoints.detected_*` is meant to answer "did the
  mechanism notice the problem at all", ungated by `approval_family` — genuinely
  true for an oracle-graded arm, but for an in-graph-graded arm it had nothing
  ungated to read and silently fell back to the already-gated `ep_*` itself; the
  fields were also never persisted to `RunRecord`/the store, so the (already
  incomplete) diagnostic couldn't survive past the run that computed it either.
  Fixed by threading the raw pre-gate signal into four new `ProposalOutcome.
  detected_*` fields (populated by `graph/pipeline.py` alongside each `ep_*`
  field, from the same underlying boolean before the `approval_family` gate),
  reading those in `harness/grading.py`'s in-graph branch instead of falling back
  to `ep_*`, and persisting them through to `RunRecord`/`RunRow` (migration
  `1a2b3c4d5e6f`, four new nullable-free `NOT NULL DEFAULT false`-backfilled
  columns on `runs`).
- **P2-04: quality-gate and version documentation were inconsistent with a clean
  install.** Three separate issues: (1) the README's status line, `pyproject.
  toml`'s version, `VERSION`, and the README's own "Versioning" section prose
  (still reading a pre-`2.0`-bump `1.1.0-dev`) all disagreed with each other; (2)
  every dependency was declared with a broad lower bound only, so a clean
  resolution could silently pick a different dependency set — including a NumPy
  release whose stubs require Python 3.12 syntax, which stops `mypy gcap/` before
  it ever reaches this project's own code, on 3.11 — than the one the documented
  "zero issues" gate result actually verified. Fixed: the Versioning section now
  explains all three version strings and how they relate; a new `constraints.txt`
  (generated via `pip freeze` against a known-good `.venv`) pins the full resolved
  dependency graph the quality gates are actually verified against, installed with
  `pip install -c constraints.txt -e ".[dev]"`.

### Net result

All tests passing, mypy strict clean, ruff clean, 5/5 import-linter contracts
kept. One additive schema change (P2-03's four new `runs` columns) — see this
project's `README.md#versioning` for why that keeps this round a minor version
bump (`2.0` -> `2.1`) rather than another major one.

## v2.0 — code review fix round

An independent code review of the merged `v1.1` repository raised 11 findings (4
P0, 5 P1, 2 P2). Each was first verified against the running code — reproduced or
traced to the exact lines responsible — before any fix was written, and each fix
shipped with its own new regression test and a full green `pytest`/`mypy`/`ruff`/
`lint-imports` pass before moving to the next finding. All 11 were confirmed
accurate; none were false positives.

### P0 — correctness bugs that silently undermined the pilot's own design

- **P0-1: arm C's numeric reconciliation tool never actually ran.** The CLI path
  never threaded a proposal's authoritative claim values through to config C's
  tool-calling agent, so its "check the numbers" mechanism had nothing to check
  against and could never fire. Fixed by wiring the real values through
  end to end from `gcap run` down to the tool call.
- **P0-2: the primary endpoint was defined differently per arm.** Configs being
  compared for the confirmatory analysis computed "was this a numerical error"
  two different ways depending on config, which would have silently biased any
  cross-arm contrast. Unified into one definition used by every arm.
- **P0-3: confirmatory labels leaked into CLI case files in cleartext.** The
  oracle's ground-truth labels and the "true" proposal, meant to be sealed until
  freeze, were written unsealed into the case files the CLI path produced — an
  agent (or a careless script) could read the answer key. Fixed so the CLI path
  never writes sealed content unsealed.
- **P0-4: agents in the real LLM path never actually saw upstream results.**
  `FakeLLMAdapter`'s test-only `default_responder` reads `request.metadata`
  directly, which masked the fact that upstream agent outputs and tool findings
  were never rendered into the prompt TEXT a real model receives — the multi-agent
  chain would have been six independent single-shot calls sharing a proposal_id,
  not a real decision chain, the moment a real adapter was plugged in. Fixed by
  wiring upstream outputs into prompt text; regression tests use a
  prompt-text-reading adapter (not metadata) to prove the chain is causally real.

### P1 — governance, reproducibility, and observability gaps

- **P1-5: confirmatory analysis trusted a CLI flag instead of verifying closure.**
  `gcap analyze --confirmatory` took the caller's word that a run log was closed
  and its manifest committed; now it independently validates both before treating
  any result as confirmatory.
- **P1-6: the shared call cache wasn't atomic per key and its key omitted model/
  manifest/schema versioning.** A concurrent cache miss on the same key could
  double-call the adapter, and a cache key that didn't include the model snapshot,
  freeze manifest hash, or response schema name could silently answer a lookup
  made under a different one of those. Fixed with a per-key lock and a widened
  7-tuple cache key; ad-hoc (non-frozen) manifests used outside a real freeze now
  get a pinned `committed_at` so `manifest_hash()` — and therefore the cache key
  and `provider_seed` derived from it — is reproducible across separate CLI
  invocations with identical inputs, not just within one process.
- **P1-7: `executed` was read as implying README's "execution only happens in
  config E" claim, which was never true of it.** D1/D2 have a real, working
  ledger and `executed` is a genuine, consequential state transition in both of
  them, not a simulation — that was correct, tested, deliberate behaviour, not a
  bug. The actual claim invariant 5 makes (full dual-authorisation: business
  approval ∧ control clearance ∧ policy pass) is structurally possible only in
  config E, since only E runs an IC node and policy engine at all. Resolved by
  adding a new `dual_authorised: bool` field (`ProposalOutcome`,
  `ExecutionEventRow`) distinct from `executed`, and rewriting README's invariant
  3 and 5 text to describe each field accurately instead of leaving `executed`
  carrying a claim it never made.
- **P1-8: a runtime failure inside a run produced a silently successful-looking
  result.** An exception during `run_case` was recorded as a bare `failure` kind
  with no detail, `gcap run` always exited 0 regardless, and the analysis layer
  folded failed records in as if they were clean non-events, biasing rates
  downward. Fixed: `RunRecord.failure_reason` captures the exception's own
  `type(exc).__name__: str(exc)` (bounded to 2000 chars); `gcap run` now exits 1
  and prints a failure summary when any run in the batch failed;
  `extract_cluster_observations` now excludes failed records entirely instead of
  counting them as fake non-events; `ConfirmatoryReport.n_failed_records` surfaces
  the count.
- **P1-9: LLM output was validated for shape only, never for request-response
  identity or domain correctness.** A schema-valid response for the wrong
  `proposal_id`/`claim_id`, or a `trade_alpha_horizon`/`tau_days` a real model
  simply made up outside any contractual bound, would have sailed through
  unchallenged. Fixed with a per-node `extra_check` hook that runs after schema
  parsing and raises the same `SchemaConformanceError` a shape violation already
  does — so a semantic mismatch gets the identical one-bounded-rework-cycle
  treatment. `trade_node` additionally checks `tau_days` against a new
  `TenorRestructuringBounds` to-decide registry parameter (a 0.5x–2.0x band
  around the proposal's own contractual tenor) and rejects
  `trade_alpha_horizon <= -1`.
- **P2-10: physical API calls, cache hits, and rework retries were all
  conflated in the usage record.** Three separate bugs: (1) a rework attempt's
  real tokens/latency were silently dropped the moment a later attempt
  succeeded; (2) a cache hit's tokens/latency were re-recorded as if it were a
  fresh physical call, double-counting cost across D1/D2; (3)
  `RunRecord.prompt_hashes` overwrote earlier claims' hashes whenever a node
  makes more than one call (research_node makes one per claim). Fixed with a new
  `CallEvent` (`cache_hit`, `tokens_in`, `tokens_out`, `latency_ms`) recording
  every attempt, and `NodeCall` redesigned to hold `events: tuple[CallEvent,
  ...]` with `tokens_in`/`tokens_out`/`latency_ms`/`physical_call_count`/
  `cache_hit_count` as computed properties that sum only the non-cache-hit
  events. **This changes `RunRecord.prompt_hashes`'s value type from `dict[str,
  str]` to `dict[str, list[str]]`** and adds a new `RunRecord.cache_hits: dict[str,
  int]` field (new `runs.cache_hits` column, migration `05632a0d2768`) — the
  schema change behind this release's `2.0` version bump.
- **P2-11: the rate limiter raised `IndexError` on a cold bucket given an
  oversized single request.** `_Bucket.wait_needed` assumed its internal deque
  was non-empty whenever the incoming event would exceed the limit; false when a
  single request's own weight (e.g. one very large prompt's estimated token
  count) exceeds the whole per-minute limit on an empty or freshly-evicted
  bucket — indexing `self._events[0]` then raised on the empty deque. Fixed: an
  empty window with an over-limit single event is let through immediately (there
  is nothing older to evict, so waiting could never help and would otherwise
  block the caller forever), while still being recorded so the *next* request
  correctly queues behind it.

### Net result

389 tests passing (up from 351), mypy strict clean, ruff clean, 5/5 import-linter
contracts kept.

## v1.1 — merge with an independent parallel implementation

An independent implementation of the same research proposal ("G-CAP v1.0", 49
Python files, 1222 statements) was compared against this codebase (then 76 files,
3638 statements) module by module. Both were actually run — venv installed, full
`pytest`/`mypy`/`ruff`/`lint-imports` — rather than trusting either project's own
documentation. Both were legitimate: the independent build's own delivery report
(32 tests, 91.30% coverage, strict mypy clean, 3/3 import-linter contracts) checked
out exactly as claimed.

### What this codebase had that the other did not

- An action-field schema guard on the FTP engine's output, enforced at import
  time (`core.ftp.assert_no_action_field`).
- Hysteresis in ALCO package transitions (de-escalation passes through RECOVERY
  before NORMAL, so one noisy clean observation cannot flip STRESS straight back
  to NORMAL) — the other implementation's ALCO selection was a stateless function
  of the current observation alone.
- The Section 9.6 revised-margin recomputation for `APPROVE_WITH_LIMIT`/
  `APPROVE_WITH_HEDGE` — no analogue existed in the other graph at all.
- Fail-loud cache isolation (`CacheIsolationError`/`UndeclaredCacheHitError`) for
  cross-config cache reuse; the other implementation had a full cache/rate-limit/
  replay stack (`ResponseCache`, `SlidingWindowRateLimiter`, `ReplayRuntime`) that
  was never actually imported from its own CLI or engine — dead code.
- A token-aware sliding-window rate limiter tracking requests/input-tokens/
  output-tokens independently, versus a bare `asyncio.Semaphore` restricted to
  exactly `{32, 64, 128}` (unusable for local debugging at concurrency=1).
- A substantially deeper statistics/pilot layer: a tiered contrast estimator
  (Bayesian mixed GLM → GEE → sign-flip, auto-selected per frozen trigger rules),
  ICC estimation, nested-Monte-Carlo `kmin`, a power simulator, a conformance-gap
  bootstrap LCB, Krippendorff's alpha with a real coincidence matrix and bootstrap
  CI, and an oracle-eligibility-filtered, tenor-stratified LHC replay pool. The
  other implementation used a single sign-flip test for every contrast and a
  generic LHC sampler with no eligibility filter or pool expansion.
- A freeze manifest with real governance mechanics: immutable-on-commit, an
  `ACTIVE` pointer, and `validate_conservative_revision` rejecting a later
  manifest that lowers `kmin` or the C-D1 replicate floor. The other
  implementation's manifest was schema-only — no store, no commit/immutability
  enforcement.
- ORM-level append-only enforcement on all seven store entities, FK enforcement
  even on SQLite, and an idempotent `save_run` (unique-constraint + `IntegrityError`
  catch). The other implementation had no idempotency check anywhere — a retried
  run could double-insert — and stored `economic_alpha` as `Float`, not `Decimal`.
- A genuinely seeded, varied benchmark generator (`random.Random(seed)`); the
  other implementation's 393-case corpus used no randomness at all — every field
  was a fixed literal or a deterministic function of the case index.
- An oracle that computes every label from `core` functions alone; the other
  implementation's oracle read `package.research.evidence_ids`, itself an
  agent-produced field — a weaker independence guarantee despite passing the same
  import-linter contract.

### What the other implementation had that this codebase did not (now adopted)

- **`gcap.analysis.concurrency.select_worker_pool`** — a real, programmatic
  worker-pool selection procedure: takes measured pilot throughput/error/latency
  per candidate and picks the fastest one clearing frozen reliability gates. This
  codebase previously only hard-coded `model_call_semaphore = 64` with a dated
  comment. Wired into a new `gcap select-concurrency` CLI subcommand (advisory
  only — it prints a recommendation, it never writes to the registry or a
  manifest itself, consistent with "the freeze is a file, not a meeting").
- **`gcap.core.validator.validate_integrated_package`** — a standalone, named,
  independently testable function for config E's final governance gate. This
  codebase had the same logic, correct, but inlined into `graph.pipeline`'s E
  branch as five separate booleans rather than one auditable unit. Extracted and
  wired in so `graph.pipeline` now derives `executed` from this function's
  verdict — not merely present alongside the old logic.
- **A durable, cross-process replay cache** — the other implementation's
  `ReplayRuntime`/`ResponseCache` persisted to disk, unlike this codebase's
  in-memory-only `SharedCallCache` (lost between separate CLI invocations), even
  though that mechanism was never wired into its own CLI. Adopted as
  `gcap.agents.cache_store.SqliteCacheStore`, a pluggable backing store for the
  existing `SharedCallCache` (`agents.cache_store.CacheStore` protocol;
  `InMemoryCacheStore` remains the default), exposed via `gcap run --cache-db
  PATH`. The isolation checks in `SharedCallCache` apply identically regardless
  of which store answers `get`/`set`.

### Bugs found

- **Pre-existing, this codebase:** `config.registry.ParameterRegistry.content_hash`
  contained a dead branch (`exclude={"registry_version"} if False else set()`) —
  harmless (always evaluated to `set()`) but misleading. Fixed.
- **Independent implementation:** `agents/fake.py`'s deterministic runtime compared
  `economic_alpha` as a raw string (`state.get("economic_alpha", "1") <= "0"`),
  which is wrong at exact zero depending on decimal formatting (e.g.
  `"0.00" <= "0"` is `False` as strings, `True` numerically). Not carried into this
  codebase — `FakeLLMAdapter` here never re-derives numeric thresholds from
  strings; all comparisons happen once, in `Decimal`, in `core`/`graph`.
- **Introduced and caught during this merge:** `SqliteCacheStore`'s first draft
  opened a plain `sqlite3.connect()` in the constructor. `Harness.run_batch` runs
  many (case, config, replicate) units concurrently under a `ThreadPoolExecutor`,
  and a single `sqlite3.Connection` is not safe for concurrent use from multiple
  threads without `check_same_thread=False` plus external locking. Caught by
  actually running a `run_batch` call through it (not just a single-threaded unit
  test) — it surfaced as a silently-recorded `RunRecord.failure == "runtime"`
  rather than a raised exception, which is its own small lesson about how easy an
  invariant-9 "failures are recorded, never silent" design can mask a bug that
  should have been loud during development. Fixed with an internal lock plus
  `check_same_thread=False`; a dedicated concurrency regression test
  (`tests/agents/test_cache_store.py::test_sqlite_store_survives_concurrent_access_from_a_thread_pool`)
  now guards against a regression.

### Net result

351 tests passing (up from 326), mypy strict clean, ruff clean, 5/5 import-linter
contracts kept, ~94% coverage maintained. `gcap.core.validator`,
`gcap.analysis.concurrency`, and `gcap.agents.cache_store` are all at 96–100%
coverage on their own.
