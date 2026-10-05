# G-CAP — Governed Agentic AI Architecture for Capital Allocation, Treasury and Portfolio Decision-Making

G-CAP is a reference implementation of the research proposal *"A Governed Agentic AI
Architecture for Capital Allocation, Treasury, and Portfolio Decision-Making"*
(Uğur Alkan, September 2026). It builds a multi-agent decision pipeline for bank
capital-allocation proposals — research, trade structuring, portfolio management,
treasury (funds-transfer pricing), a deterministic ledger, and a policy/governance
layer with dual authorisation — instrumented so that every claim the proposal makes
about the *system* (which layer can do what, what is deterministic vs. LLM-authored,
what must be reproducible) is a property this codebase actually enforces and tests,
not just a design intention.

Equity is the primary instrument class; fixed income is a first-class, pluggable
asset class through the same `InstrumentSpec`/FTP/ledger/policy path (see
[Architecture](#architecture) — no module branches on `asset_class` except the one
policy predicate that is explicitly about it).

**Status:** `2.3.1.dev0` (see [Versioning](#versioning) for how this relates to the
`VERSION` file's `v2.3` and `ParameterRegistry.registry_version`'s `2.0.0-dev`).
Milestones M1 (deterministic core + oracle + bench + analysis
statistics), M2 (agents + graph + harness + store), and M3 (pilot rehearsal, LHC
replay pool, Table 5 budget-row selection, freeze-manifest assembly) are built and
tested end to end. See [What's built, milestone by milestone](#whats-built-milestone-by-milestone).

**v2.3.1** is the independent validation cleanup for v2.3. The reviewer found
no functional regression and confirmed all v2.3 claims, then reported six
residual packaging, test-coverage and documentation findings. All six are
closed; the project now passes 475 tests, strict mypy, Ruff and all 5
import-linter contracts.

**v2.3** closes 10 residual confirmatory-integrity findings reproduced after
the third independent review of v2.2. The fixes tighten the public execution
contract at the manifest, run-log, analysis and cache boundaries, and ship with
negative regression tests. That release passed 473 tests, strict mypy, Ruff and
all 5 import-linter contracts. See `CHANGELOG.md` for the complete list.

**v2.2** was a THIRD, independent code-review pass over `v2.1` (a different
reviewer again): 10 findings (2 P0, 5 P1, 3 P2) were verified against the running
code one by one, then all 10 fixed, each with its own regression test, exactly the
same discipline the two rounds before it used. One fix (P0-01) adds a new
`run_list` field to `FreezeManifest` — purely additive (a pydantic model field
defaulting to `()`, not a database migration or table; no new Alembic revision was
added this round), so this stays a minor bump (`2.1` -> `2.2`) rather than a major
one, matching the same "additive, not breaking" reasoning the `v2.1` paragraph
below gives for its own bump. See `CHANGELOG.md` for the full list of this round's
findings, the fix for each, and the new regression test that guards it.

**v2.1** is a SECOND, independent code-review pass over `v2.0` (a different
reviewer than the one behind `v2.0` below): 12 findings (3 P0, 5 P1, 4 P2) were
verified against the running code one by one, then all 12 fixed, each with its own
regression test, exactly the same discipline `v2.0`'s own round used. One fix
(P2-03) adds four new columns to the `runs` table (`RunRecord.detected_numerical_
error`/`detected_state_infeasible`/`detected_pricing_inconsistent`/`detected_
governance_failure`, migration `1a2b3c4d5e6f`) — purely additive (every new column
is nullable-free but `NOT NULL DEFAULT false`-backfilled, no existing column
changed type, was renamed, or was removed), so this stays a minor bump (`2.0` ->
`2.1`) rather than another major one: this project's "a schema change bumps the
major version" rule (see [Versioning](#versioning)) is about a change that breaks
an existing consumer's assumptions, which an additive, backward-compatible column
is not — `v2.0`'s own bump above involved an actual field *type* change
(`prompt_hashes`), which is the kind of change that rule is for. See
`CHANGELOG.md` for the full list of this round's findings, the fix for each, and
the new regression test that guards it.

**v2.0** is a full code-review pass over the merged `v1.1` codebase: 11 findings (4
P0, 5 P1, 2 P2) from an independent review were verified against the running code
one by one, then all 11 fixed, each with its own regression test. Two of the fixes
change the public schema — `RunRecord.prompt_hashes` is now `dict[str, list[str]]`
(one node can make several physical calls, e.g. one Research call per claim, and
every call's own hash is now kept, not just the last one) and `RunRecord` gained
`cache_hits`/`failure_reason` fields and `ExecutionEventRow` gained
`dual_authorised` — which is what moved the version to `2.0` under this project's
own rule that a schema change bumps the major version. See `CHANGELOG.md` for the
full list of findings, the fix for each, and the new regression test that guards
it.

**v1.1** merges in three features adopted, with attribution, from an independent
parallel implementation of this same proposal, after a module-by-module comparison
of the two codebases (see `CHANGELOG.md`): a standalone, named, independently
testable Integrated Validator (`gcap.core.validator`), a programmatic worker-pool
selection procedure (`gcap.analysis.concurrency`, wired into a new `gcap select-
concurrency` CLI command), and a durable, cross-process D1/D2 shared-call cache
(`gcap.agents.cache_store.SqliteCacheStore`). All three are fully wired into the
graph/CLI/harness, not merely present — see the comparison note in `CHANGELOG.md`
for why that distinction mattered.

## Read this first: what this build is, and is not

- **No real model is ever called.** Every LLM-touching code path runs through
  `gcap.agents.llm.FakeLLMAdapter`, a deterministic, hint-driven stand-in. A real
  `OpenAIAdapter` exists (`gcap/agents/llm.py`) for a future production deployment,
  but nothing in this repository's tests, CLI defaults, or CI ever constructs or
  calls it. There is no API key anywhere in this project.
- **"ChatGPT 5.6 Luna"** is a product-facing model name that appears only in
  human-readable strings (e.g. `model_snapshots={"primary": "fake-luna-sandbox-1.0"}`);
  it is kept deliberately separate from any real API model identifier, which never
  appears in code.
- **This is a rehearsal of the governance machinery, not a calibrated system.**
  Every "to-decide" parameter the proposal leaves open (Section 10's six, plus two
  more this build introduced — see [To-decide parameters](#to-decide-parameters))
  ships with a literature-consistent placeholder and `calibrated=False`. The pilot
  statistics (conformance gap, kmin, ICC, Table 5's budget row) are real, correct
  implementations of the proposal's own procedures, but every number they produce
  from this build's synthetic data and `FakeLLMAdapter` is a rehearsal value, not a
  scientific finding — see `gcap/pilot/report.py`'s module docstring for a worked
  example of a genuine finding this distinction actually surfaced.
- **The ALM/treasury side (FTP/funding curves) will never have real data in this
  project** — only equity-side "fundamental alpha" might possibly become real one
  day. The pilot/freeze machinery is designed to run in rehearsal mode permanently
  on that side, not as a temporary limitation.
- **This is not a deployment pipeline.** There is no orchestration, scheduler,
  container build, or secrets manager here — `gcap/harness/cli.py` is described in
  its own docstring as "a thin sandbox CLI, not a deployment tool." Committing a
  freeze manifest (`FreezeManifestStore.commit`) is treated throughout as a human
  governance act and is deliberately never called by any library code, only by
  tests and, in a real deployment, by a person.

## Architecture

Ten packages, each with a single clear responsibility and import-linter-enforced
boundaries (`pyproject.toml`'s `[tool.importlinter]`, checked by `lint-imports`):

| Package | Responsibility |
|---|---|
| `gcap.core` | Deterministic engines only: FTP pricing (`ftp.py`), the ledger (`ledger.py`), ALCO package transitions (`alco.py`), the policy predicate registry (`policy.py`), portfolio contribution math (`portfolio.py`), numerics helpers, reconciliation, and the Integrated Validator (`validator.py` — config E's final governance gate as one named, independently testable function, not scattered booleans). No LLM or agent import is possible here — enforced by an import-linter contract, not just convention. |
| `gcap.oracle` | Ground-truth labelling for bench cases (`labels.py`, `schema.py`) computed from the case definition alone, via the exact same `core` engines the graph uses — so a label and a live tool result can never disagree because of two different implementations. |
| `gcap.bench` | The synthetic case generator (`generator.py`, 8 scenario families, equity/fixed-income mixed), oracle-label sealing for the confirmatory split (`sealing.py`, Fernet-encrypted, gated on a committed freeze manifest), and second-reviewer verification bookkeeping (`verification.py`). |
| `gcap.config` | The parameter registry (`registry.py` — every fixed and to-decide constant, typed and versioned), the config-flags table A–E (`flags.py`), the freeze manifest schema and store (`manifest.py`), and generic content-hashing utilities (`hashing.py`). |
| `gcap.agents` | The LLM client abstraction (`llm.py` — `FakeLLMAdapter` and `OpenAIAdapter`), per-node prompts and schemas, the Research/Trade/PM/Single-agent node implementations (`nodes.py`), deterministic tool wrappers (`tools.py`), the D1/D2 shared-call cache (`cache.py`), and its pluggable backing stores (`cache_store.py` — the default in-memory store, plus `SqliteCacheStore` for a durable cache that survives across separate process invocations). |
| `gcap.graph` | Wires one proposal through the right nodes for a given config (A–E) and produces a `ProposalOutcome` (`pipeline.py`, `outcomes.py`, `context.py`) — the revised-margin recomputation for `APPROVE_WITH_HEDGE`/`APPROVE_WITH_LIMIT` lives here, and config E's final decision is delegated to `gcap.core.validator.validate_integrated_package`. Never imports `oracle` or `bench`: the decision path is structurally blind to ground truth. |
| `gcap.harness` | Runs cases through the graph and persists results (`runner.py`, `record.py`), a token-bucket rate limiter (`rate_limit.py`), deterministic per-run seeding (`seeding.py`), and the `gcap` CLI (`cli.py`: `generate-cases` / `run` / `analyze` / `select-concurrency`). |
| `gcap.store` | The seven-table SQLAlchemy schema (`models.py`), the append-only write/read bridge to the flat `RunRecord` view (`repository.py`, `run_record.py`), and engine/session setup (`db.py`). |
| `gcap.analysis` | Every statistical procedure the proposal specifies: mixed-effects/marginal/sign-flip contrast estimation (`mixed_model.py`), Holm correction (`multiplicity.py`), the four-contrast pipeline (`pipeline.py`) and its human-readable report (`report.py`), the pilot conformance gap and its bootstrap CI (`conformance.py`), the worst-boundary nested simulation deriving kmin (`nested_simulation.py`), power simulation (`power.py`), the frozen-seed LHC draw pool (`lhc.py`), ICC (`icc.py`), the Table 5 C-D1 budget-row selector (`budget_row.py`), and a programmatic worker-pool selection procedure (`concurrency.py` — turns real pilot throughput/error/latency measurements into a `model_call_semaphore` choice, advisory only). Reads recorded data only — cannot import `agents`, `graph`, or `harness`. |
| `gcap.pilot` | Ties the above together for the M3 pilot: corpus generation (`corpus.py`), execution across all five arms (`run.py`), descriptive metrics (`metrics.py`), the conformance-gap/kmin report (`report.py`), the 20-case LHC replay-pool selection and oracle-eval function (`lhc_pool.py`), and freeze-manifest assembly (`freeze.py` — assembles, never commits). |

### The nine structural invariants

Carried through every layer and directly tested (see `tests/`, especially
`tests/graph/test_pipeline.py` and `tests/harness/test_integration.py`):

1. An LLM never computes a number, transitions ledger state, or enforces a rule.
2. The FTP engine never emits an action label (`gcap.core.ftp.assert_no_action_field`
   runs at import time).
3. Ledger state changes only on an *executed* action, never speculatively.
   `executed` (`ProposalOutcome.executed`/`RunRecord` via the event log) is a
   real, consequential transition in D1, D2 and E alike — a later proposal in
   the same episode genuinely sees the capital/liquidity/etc. an earlier
   D1/D2 proposal's execution used. It is `None` only where there is no
   ledger mechanism at all (A/B/C).
4. A validator cannot modify what it validates.
5. Execution requires `BUSINESS_APPROVAL ∧ CONTROL_CLEARANCE ∧ POLICY_PASS`
   (`gcap.core.validator.validate_integrated_package`) to count as *dually
   authorised*, and only config E has the mechanisms (IC vote, policy
   engine) that could ever satisfy it — tracked by its own field,
   `ProposalOutcome.dual_authorised` / `ExecutionEventRow.dual_authorised`,
   which is structurally `False` for every D1/D2 event even though those
   configs do execute (invariant 3). D1/D2's execution gate is single-gate
   ledger-feasibility only (`approval_family and ledger_ok`, no IC/policy
   review at all) — this is a deliberate, tested contrast with E's stricter
   gate (see `gcap/graph/pipeline.py`'s module docstring, "P1-7"), not an
   invariant-5 violation: invariant 5 constrains `dual_authorised`, not
   `executed`.
6. The oracle is computed independently of the system under test (same engines,
   separate call path, `gcap.oracle` cannot import `gcap.agents`/`gcap.graph`/`gcap.harness`).
7. Every run is reproducible from its manifest (`gcap.harness.seeding`, the freeze
   manifest's hash, `FreezeManifestStore.assert_run_is_confirmatory_valid`).
8. Confirmatory labels stay sealed until freeze (`gcap.bench.sealing`), and the
   analysis layer refuses to read confirmatory endpoints before the run log is
   closed (`gcap.analysis.pipeline._assert_confirmatory_readable`).
9. Failures are recorded outcomes, never silent crashes (`RunRecord.failure`).

### Config table (A–E)

| Config | Agents | Deterministic tools | Ledger/Treasury | FTP addons | Policy engine | Dual authorisation |
|---|---|---|---|---|---|---|
| A | single generalist | – | – | – | – | – |
| B | Research/Trade/PM | – | – | – | – | – |
| C | Research/Trade/PM | ✓ | – | – | – | – |
| D1 | Research/Trade/PM | ✓ | ✓ | forced to zero (pricing placebo) | – | – |
| D2 | Research/Trade/PM | ✓ | ✓ | full | – | – |
| E | Research/Trade/PM | ✓ | ✓ | full | ✓ | ✓ |

The four co-primary contrasts are exactly the adjacent pairs: B–C (numerical
reliability), C–D1 (state feasibility), D1–D2 (pricing consistency), D2–E
(governance failure) — `gcap.analysis.endpoints.PRIMARY_CONTRASTS`.

### To-decide parameters

`gcap.config.registry.ToDecideParameters` — every one ships `calibrated=False`, a
`needed_by` milestone, and a rationale; `ParameterRegistry.assert_ready_for_split`
refuses a confirmatory run while a milestone-due parameter is still uncalibrated:

- `NumericalErrorTolerance`, `MaterialityLevels`, `PortfolioContributionWeights`
  (λ, γ), `AlcoTransitionThresholds`, `DayCountBasis`, `FalseBlockTolerance` — the
  proposal's own six.
- `RevisedMarginRecomputation` — this build's own addition, closing a real M2 spec
  gap: Section 9.6 requires `APPROVE_WITH_LIMIT`/`APPROVE_WITH_HEDGE` to trigger a
  deterministic margin recomputation but does not specify its magnitude.
- `LhcAddonBounds` — the per-component sampling bounds the frozen-seed LHC draw
  needs (Section 8.4/9.5), for the same reason.

## What's built, milestone by milestone

- **M1** — the deterministic core, the oracle, the bench case generator, and every
  statistical procedure the confirmatory analysis needs, proven on synthetic
  `RunRecord`s with a planted effect (`tests/analysis/test_pipeline_synthetic.py`).
- **M2** — the full agent graph across all six configs, the harness (idempotent,
  rate-limited, seeded), the seven-table store, and the closed loop from case
  generation through a real SQLite-backed run to the analysis layer
  (`tests/harness/test_integration.py`). Includes the Section 9.6 revised-margin fix.
- **M3** — the 32-case development pilot corpus and its execution across all five
  arms (`gcap.pilot.corpus`/`run`/`metrics`/`report`), the LHC replay pool's 20-case
  selection and oracle-eval wiring (`gcap.pilot.lhc_pool`), the Table 5 C-D1
  budget-row selector (`gcap.analysis.budget_row`), and freeze-manifest assembly
  (`gcap.pilot.freeze`) tying all of the above into a schema-valid, committable
  `FreezeManifest`.
- **Closing the loop** — `tests/test_whole_project_loop.py` chains every one of
  the above, once, in the order a real deployment would: pilot statistics → LHC
  pool → budget row → freeze-manifest assembly → a genuinely committed
  `FreezeManifestStore` → a confirm-split harness run gated on that manifest →
  `gcap analyze` reading the sealed split back out, refused before the run log is
  closed and accepted once it is.

Deliberately left as a human step, never automated: actually committing a freeze
manifest. Deliberately out of scope, per the proposal's own budgeting: authoring
G-CAP Bench v1's real, human-reviewed 393-case confirmatory corpus (`gcap.bench.generator`'s
own module docstring calls this "a 10–14 person-week data-authoring effort... not
a software task").

## Running it

```bash
pip install -e ".[dev]"

# 1. Generate a case slice with oracle labels precomputed.
gcap generate-cases --split dev --seed 1 --n-per-family 2 --out cases.json

# 2. Run it through the graph under a config, persisting to the store
#    (GCAP_SQLITE_PATH picks the SQLite file; unset it and it's :memory:).
export GCAP_SQLITE_PATH=./gcap.db
gcap run --cases cases.json --config B --replicates 3 --block dev-1
gcap run --cases cases.json --config C --replicates 3 --block dev-1

# 3. Read the results back and run every co-primary contrast they support.
gcap analyze --split dev --n-boot 2000 --out report.txt

# Optional: a durable D1/D2 shared-call cache across separate `gcap run`
# invocations (default is in-memory, lost when the process exits).
gcap run --cases cases.json --config D1 --replicates 3 --block dev-1 --cache-db cache.sqlite3
gcap run --cases cases.json --config D2 --replicates 3 --block dev-1 --cache-db cache.sqlite3

# Optional: turn measured pilot worker-pool throughput into a selection
# (advisory only — see configs/concurrency_pilot.example.json for the input shape).
gcap select-concurrency configs/concurrency_pilot.example.json
```

A `split=confirm` run additionally needs `--manifest-store`/`--manifest-version`
pointing at a directory holding a manifest someone has actually committed with
`gcap.config.manifest.FreezeManifestStore.commit` (see `gcap.pilot.freeze` to
assemble one, and `tests/test_whole_project_loop.py` for the full sequence) —
without one, `gcap run --split confirm` and a `run_log_closed=False` `gcap analyze`
both refuse, by design (invariants 7 and 8).

## Quality gates

```bash
pip install -c constraints.txt -e ".[dev]"   # reproducible install -- see note below
pytest                      # ~94% branch+line coverage over gcap/; exact count drifts with the suite, don't hardcode it
mypy gcap/                  # strict, zero issues
ruff check gcap/ tests/     # zero issues
lint-imports                # 5/5 import-linter contracts kept
```

GitHub Actions runs these gates for every push and pull request on Python 3.11
and 3.12, then builds clean release archives after both environments pass. See
[`DEVELOPMENT.md`](DEVELOPMENT.md) for the branch, definition-of-done, migration,
and release workflow.

The project is distributed under the [MIT License](LICENSE).

P2-04 (v2.0.0 code review): every dependency in `pyproject.toml` is declared with a
broad lower bound only (`pydantic>=2.6,<3`, `numpy>=1.26`, etc.), so a `pip install`
against those bounds alone can resolve a different, later dependency set on two
different days or two different machines — silently changing what "zero issues"
above actually verified. `constraints.txt` (generated via `pip freeze` against this
project's own known-good `.venv`, and re-generated whenever dependencies are
deliberately upgraded) pins every transitive dependency to the exact versions the
quality gates above are actually verified against; `-c constraints.txt` on top of the
normal editable install makes the resolution reproducible without turning
`pyproject.toml`'s own ranges into hard pins (which would fight downstream projects
depending on `gcap` with their own constraints). This is a constraints file, not a
full lockfile with hashes — proportionate to a research-reference codebase that is
explicitly not an MLOps/deployment pipeline (see the top-level scope note), not a
substitute for one where a real institutional deployment would want hash-pinned,
platform-specific lockfiles instead.

One specific resolution hazard this guards against: a clean dependency resolution
can select a NumPy release whose bundled type stubs use Python 3.12-only syntax,
while this project's `mypy` target (`[tool.mypy] python_version = "3.11"`,
matching `requires-python = ">=3.11"`) is 3.11 — `mypy gcap/` then fails while
parsing NumPy's own stubs, before it ever reaches this project's code, which reads
as a mypy regression in *this* codebase until traced back to the dependency
resolution. `constraints.txt` pins NumPy (and everything else) to versions already
known to satisfy the 3.11 mypy target; if you deliberately upgrade NumPy, re-check
`mypy gcap/` against the new version's stubs before updating the pin.

Concurrency default: the model-call semaphore is 64 in-flight calls
(`ConcurrencyConfig.model_call_semaphore`), underneath a client-side rate limiter
set to the account's real per-minute request/token ceilings — a 2026-09-21 pilot
decision overriding the proposal's more conservative 8-way default; the rate
limiter still enforces the real ceilings regardless of the semaphore's width. A
single request whose own estimated cost exceeds a configured per-minute ceiling
on its own (e.g. a very large `max_output_tokens`) is refused outright with
`RequestExceedsRateLimitError` (P2-02, v2.0.0 code review) rather than silently
let through over budget — raise the relevant ceiling in `ConcurrencyConfig`, or
reduce the request's cost, rather than retrying as-is.

## Versioning

Three independent version strings exist in this repo, deliberately not collapsed
into one (they answer different questions and are consumed by different things),
but which P2-04 (v2.0.0 code review) found had drifted out of sync with each other
after the `1.1` -> `2.0` breaking-change bump — this section is the single place
that defines how they relate, so a future bump can keep them honest:

- `pyproject.toml`'s `version` (currently `2.3.1.dev0`) is the PEP 440-valid
  package version `pip`/`build` actually consume; the `.devN` suffix marks it as a
  pre-release while every Section-10 to-decide parameter remains uncalibrated —
  note this is unaffected by a code-review fix round completing; only
  Section-10 parameter calibration drops it.
- `VERSION` (currently `v2.3`) is the human-readable major.minor project milestone
  tag this README's own prose refers to (e.g. "moves the version to `2.0`" above) —
  it deliberately drops the `.dev0` pre-release qualifier and the patch digit,
  since it tracks the coarser milestone/breaking-change lineage, not the exact
  package build.
- `ParameterRegistry.registry_version` (currently `2.0.0-dev`, `gcap/config/
  registry.py`) is a *third*, independent version: the calibration parameter set's
  own version, with its own content hash, recorded in every freeze manifest and
  `RunRecord` so a parameter change is always traceable to the exact run(s) it
  affected. It moves on its own schedule (a to-decide parameter getting calibrated),
  not the package version's own schedule — unaffected by the `v2.1`/`v2.2`/`v2.3` bumps
  above since no Section-10 parameter's actual value changed in either round, only
  code — and is not expected to equal the package version in general. The two
  merely coincided at `2.0.0-dev`-shaped strings for one review round because that
  round happened to touch both; that was a coincidence of timing, not a coupling
  this project maintains going forward.

Small, backward-compatible changes bump `pyproject.toml`'s patch version while
`VERSION` keeps the same major.minor milestone (e.g. `2.1.0.dev0` ->
`2.1.1.dev0` / `VERSION` remains `v2.1`, or off pre-release entirely once
calibrated). Additive schema fields or a material tightening of a public
invariant boundary may bump the minor version in both. A change that incompatibly reinterprets
persisted schema, changes a scientific invariant's definition, or changes the
experiment's config-table semantics bumps the major version in both (-> `3.0`),
matching the `1.1` -> `2.0` precedent recorded above — `2.0` -> `2.1` and `2.1` -> `2.2` above
are both the minor-bump side of that same rule: each round's one schema-shaped
change (the `v2.1` paragraph near the top of this README for that round's four new
`runs` columns; the `v2.2` paragraph for this round's `FreezeManifest.run_list`
field) was additive and backward-compatible, not breaking, which is what that rule
actually keys on. Before the `v2.1` fix, the Versioning section above still read
its pre-`2.0`-bump text (`1.1.0-dev`) verbatim, contradicting both `VERSION` and
`pyproject.toml` — an editorial oversight (nothing here enforces this at review
time), not a discovered code defect, but confusing enough on its own to mislead a
reader about which version the rest of this README was actually describing.

`v2.3` is also a minor bump even though it adds no database or Pydantic field.
It hardens enforcement at existing invariant boundaries without changing the
scientific invariant definitions or the valid confirmatory workflow: a bare
`run_log_closed=True` can no longer impersonate persisted closure, confirmatory
reads require both stores, and incomplete bindings are rejected. Those formerly
accepted patterns were unsafe bypasses rather than supported valid workflows.
The minor bump makes this public boundary tightening visible; a patch bump would
understate it, while a major bump would incorrectly imply persisted-data or
experiment-semantics incompatibility.

## Merge history

`v1.1` folds in three features from an independently developed implementation of
the same research proposal, after reading both codebases module by module,
running both test suites for real (not trusting either project's own claims), and
listing concrete strengths/weaknesses on each side rather than a numeric score.
See `CHANGELOG.md` for the full comparison notes, what was adopted and why, and
the two correctness bugs the merge process caught (one pre-existing in this
codebase — a harmless but misleading dead branch in `ParameterRegistry.content_hash`
— and one introduced and caught during the merge itself — a `sqlite3.Connection`
thread-safety bug in the new `SqliteCacheStore`, only surfaced once it was actually
exercised through `Harness.run_batch`'s real thread pool, not a single-threaded
call).
