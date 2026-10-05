"""The runtime-vs-manifest binding barrier (P0-01, v2.1.0 third code
review round).

``harness.runner.assert_confirmatory_run_permitted`` proves exactly one
thing: the in-memory ``FreezeManifest`` object handed to ``Harness`` is
byte-identical (via ``FreezeManifest.manifest_hash()``) to whatever a
``FreezeManifestStore`` has committed and marked active. That is
self-consistency of the manifest OBJECT -- it says nothing about whether
the code/model/prompts/registry/run-plan actually executing right now
conforms to what that manifest's own individual fields recorded. Before
this module existed, every one of those fields was write-only: set once at
manifest assembly time (``pilot.freeze.assemble_freeze_manifest``) and
never read back for comparison against a live runtime value anywhere in
``gcap.harness``. A caller could run ``split="confirm"`` with a registry
whose actual VALUES (and therefore ``content_hash()``) had drifted from
what was frozen, a different real model snapshot, prompt-building code
that had since changed, a ``CONFIG_TABLE``/``gcap.analysis`` source tree
edited after freeze, different concurrency limits, or an entirely
different set of (case, config, replicate) units than the frozen run
plan -- and ``assert_confirmatory_run_permitted`` alone would never
notice, because none of that touches the manifest object's own hash.

This module is the fix: one function per manifest field group that can
actually be recomputed from something live (``validate_runtime_against_
manifest`` for everything a single ``Harness`` instance already has in
hand; ``validate_run_list_against_manifest`` separately, since only a full
batch of planned ``RunSpec``s -- never one ``Harness.run_case`` call in
isolation -- can reconstruct the run-list hash), raising one clear,
itemised ``RuntimeManifestMismatchError`` naming every field that diverged
rather than stopping at the first. Both are wired into the confirm-only
gate exactly where ``assert_confirmatory_run_permitted`` already runs
(``Harness.run_case``/``Harness.run_batch``): AFTER that check (so a
non-committed or mismatched manifest object is still rejected first, by
the narrower, cheaper check) and BEFORE any graph/LLM work.

What this module deliberately does NOT attempt: per-call prompt-TEXT
comparison. ``manifest.node_prompt_hashes`` is one hash per NODE NAME, not
per rendered call -- a rendered prompt varies with every proposal's own
content (claim values, tenor, action, ...), so there is no single "the
runtime prompt hash" to compare against a frozen per-node value; nothing
in this codebase has ever computed one (every existing ``node_prompt_
hashes`` value in tests/freeze callers is an arbitrary placeholder string,
never a real hash of anything). What CAN be verified, and is verified
here, is that the CODE which builds every node's prompts
(``gcap/agents/prompts.py``) has not changed since freeze -- the same
"hash the source, not a specific rendering" choice ``analysis_code_hash``
already makes for ``gcap.analysis``. Hashed as ONE WHOLE-FILE content hash
(not per public function via ``inspect.getsource``) deliberately: several
node-prompt functions share private helper blocks
(``_research_findings_block``, ``_pm_recommendation_block``, ...), and a
change to a shared helper would not show up in a per-function
``getsource`` diff of the public function that merely calls it --
whole-file hashing can never silently miss that. The cost is coarser
granularity (a mismatch says "prompts.py changed", not which node's
prompt), which is the right trade for a tamper/drift DETECTION barrier
(never miss a real change) over a diagnostic tool (localise exactly where).
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import gcap.analysis as _analysis_package
from gcap.agents import prompts as _agent_prompts
from gcap.config.flags import CONFIG_TABLE, ConfigName
from gcap.config.hashing import hash_bytes_map, hash_config_table, hash_directory, hash_run_list
from gcap.config.manifest import FreezeManifest
from gcap.config.registry import (
    ConcurrencyConfig,
    ParameterRegistry,
    TenorRestructuringBounds,
    TokenBudget,
)
from gcap.errors import ConfirmatoryCorpusIncompleteError, RuntimeManifestMismatchError
from gcap.store.run_record import RunRecord

# Every node whose prompt-building logic lives in gcap/agents/prompts.py
# (research_prompt, trade_prompt, pm_prompt, single_agent_prompt,
# treasury_prompt, ic_prompt) maps to the SAME whole-file hash -- see this
# module's own docstring for why per-function hashing would silently miss
# a shared-helper change.
_NODE_NAMES: tuple[str, ...] = ("research", "trade", "pm", "treasury", "ic", "single_agent")


def hash_node_prompt_source() -> str:
    """Content hash of ``gcap/agents/prompts.py``'s entire source file."""
    path = Path(inspect.getfile(_agent_prompts))
    return hash_bytes_map({"prompts.py": path.read_bytes()})


def hash_node_prompts() -> dict[str, str]:
    """Every node name mapped to the SAME whole-``prompts.py`` content
    hash -- see this module's docstring for why one file-level hash,
    repeated per node, is the safe (never-silently-miss-a-change) choice
    over per-function hashing."""
    whole_file_hash = hash_node_prompt_source()
    return {name: whole_file_hash for name in _NODE_NAMES}


def hash_config_table_default() -> str:
    """``hash_config_table`` applied to this build's actual ``CONFIG_TABLE``
    (``gcap.config.flags``) -- the live, frozen-dataclass VALUES every arm
    actually runs under, not its Python source text.

    Rebuilt as a plain ``dict[str, ConfigFlags]`` before the call: ``hash_
    config_table``'s own parameter type (``Mapping[str, DataclassInstance]``)
    is deliberately generic (see its docstring), and ``Mapping``'s key type
    is invariant, so ``CONFIG_TABLE``'s own ``dict[ConfigName, ConfigFlags]``
    (``ConfigName`` a ``Literal[...]`` of ``str``s, narrower than ``str``
    itself) does not satisfy it directly under strict mypy -- this is a
    pure typing-level reshape, not a semantic one; the keys and values are
    identical."""
    return hash_config_table({str(name): flags for name, flags in CONFIG_TABLE.items()})


def hash_analysis_code() -> str:
    """Content hash of the ``gcap.analysis`` package's own source tree --
    the SAME ``hash_directory`` call ``pilot.freeze``'s caller is expected
    to have used to produce ``FreezeManifest.analysis_code_hash`` in the
    first place, so the two sides can never drift apart over which glob or
    which directory counts."""
    root = Path(inspect.getfile(_analysis_package)).parent
    return hash_directory(root)


def validate_runtime_against_manifest(
    manifest: FreezeManifest,
    *,
    registry: ParameterRegistry,
    model_snapshot: str,
    concurrency: ConcurrencyConfig,
    token_budget: TokenBudget,
    numerical_tolerances: dict[str, object],
    default_tolerance: object,
    day_count_basis: object,
    marginal_band_ann: object,
    hedge_rp_reduction_pct: object,
    limit_addon_reduction_pct: object,
    tenor_restructuring_bounds: TenorRestructuringBounds,
) -> None:
    """The per-``Harness``-instance half of the P0-01 barrier: every
    manifest field that can be recomputed from what a single ``Harness``
    already has in hand (the registry it was constructed with, its
    adapter's real model snapshot, its own concurrency config) plus the
    two pieces of source-derived state that need no caller input at all
    (node-prompt source hash, ``CONFIG_TABLE`` hash, ``gcap.analysis``
    source hash). Deliberately collects EVERY mismatch before raising --
    a confirm run with three drifted fields should report all three in one
    refusal, not force three separate fix-and-retry cycles to discover
    them one at a time.

    Called from ``Harness.run_case`` for every ``split="confirm"`` case,
    immediately after ``assert_confirmatory_run_permitted`` -- see that
    function's own docstring for why this is a SEPARATE, additional check
    rather than folded into it."""
    problems: list[str] = []

    actual_registry_hash = registry.content_hash()
    if actual_registry_hash != manifest.parameter_registry_hash:
        problems.append(
            f"parameter registry content_hash {actual_registry_hash!r} does not match "
            f"manifest.parameter_registry_hash {manifest.parameter_registry_hash!r} -- the registry "
            "actually in use has different VALUES than the one frozen, even though it may share the "
            "same registry_version string."
        )
    if registry.registry_version != manifest.parameter_registry_version:
        problems.append(
            f"parameter registry_version {registry.registry_version!r} does not match "
            f"manifest.parameter_registry_version {manifest.parameter_registry_version!r}."
        )

    frozen_primary_snapshot = manifest.model_snapshots.get("primary")
    if not frozen_primary_snapshot:
        problems.append("manifest.model_snapshots['primary'] is required and must be non-empty.")
    elif frozen_primary_snapshot != model_snapshot:
        problems.append(
            f"adapter model_snapshot {model_snapshot!r} does not match "
            f"manifest.model_snapshots['primary'] {frozen_primary_snapshot!r}."
        )

    actual_node_prompt_hashes = hash_node_prompts()
    for node, actual_hash in actual_node_prompt_hashes.items():
        frozen_hash = manifest.node_prompt_hashes.get(node)
        if not frozen_hash:
            problems.append(f"manifest.node_prompt_hashes[{node!r}] is required and must be non-empty.")
        elif actual_hash != frozen_hash:
            problems.append(
                f"node {node!r}'s prompt-building source (gcap/agents/prompts.py) has changed since "
                f"freeze: current hash {actual_hash!r} != manifest.node_prompt_hashes[{node!r}] "
                f"{frozen_hash!r}."
            )

    expected_tolerances: dict[str, object] = {
        "revenue_growth_yoy": registry.to_decide.numerical_error_tolerance.return_pct,
        "portfolio_return_pct": registry.to_decide.numerical_error_tolerance.return_pct,
    }
    runtime_bindings = (
        ("token_budget", token_budget, registry.fixed.token_budget),
        ("numerical_tolerances", numerical_tolerances, expected_tolerances),
        ("default_tolerance", default_tolerance, registry.to_decide.numerical_error_tolerance.ratio_generic),
        ("day_count_basis", day_count_basis, registry.to_decide.day_count_basis.days),
        ("marginal_band_ann", marginal_band_ann, registry.fixed.marginal_band.primary_ann),
        (
            "hedge_rp_reduction_pct",
            hedge_rp_reduction_pct,
            registry.to_decide.revised_margin_recomputation.hedge_rp_reduction_pct,
        ),
        (
            "limit_addon_reduction_pct",
            limit_addon_reduction_pct,
            registry.to_decide.revised_margin_recomputation.limit_addon_reduction_pct,
        ),
        (
            "tenor_restructuring_bounds",
            (tenor_restructuring_bounds.min_multiple, tenor_restructuring_bounds.max_multiple),
            (
                registry.to_decide.tenor_restructuring_bounds.min_multiple,
                registry.to_decide.tenor_restructuring_bounds.max_multiple,
            ),
        ),
    )
    for name, actual, expected in runtime_bindings:
        if actual != expected:
            problems.append(
                f"runtime {name} {actual!r} does not match the value {expected!r} "
                "derived from the frozen parameter registry."
            )

    actual_config_table_hash = hash_config_table_default()
    if actual_config_table_hash != manifest.config_flags_table_hash:
        problems.append(
            f"CONFIG_TABLE content hash {actual_config_table_hash!r} does not match "
            f"manifest.config_flags_table_hash {manifest.config_flags_table_hash!r} -- one or more "
            "arms' flags have changed since freeze."
        )

    actual_analysis_hash = hash_analysis_code()
    if actual_analysis_hash != manifest.analysis_code_hash:
        problems.append(
            f"gcap.analysis source-tree hash {actual_analysis_hash!r} does not match "
            f"manifest.analysis_code_hash {manifest.analysis_code_hash!r} -- the statistical analysis "
            "code has changed since freeze."
        )

    if concurrency != manifest.concurrency:
        problems.append(
            f"runtime concurrency config {concurrency!r} does not match manifest.concurrency "
            f"{manifest.concurrency!r}."
        )

    if problems:
        raise RuntimeManifestMismatchError(
            "Cannot start a confirmatory (split='confirm') run: the runtime does not conform to the "
            f"committed freeze manifest {manifest.manifest_version!r} in {len(problems)} way(s):\n- "
            + "\n- ".join(problems)
        )


def validate_run_list_against_manifest(
    manifest: FreezeManifest,
    specs: Sequence[tuple[ConfigName, str, int, str | None]],
) -> None:
    """The batch-level half of the P0-01 barrier: the planned (config,
    case_id, replicate, draw_id) units about to run must be sanctioned by
    the frozen confirmatory run plan (Section 9.1).

    Two modes, chosen by whether ``manifest.run_list`` (the actual identity
    tuples, not just their hash) was populated at freeze time:

    * ``manifest.run_list`` non-empty: every one of ``specs`` must be a
      MEMBER of it (subset check). This is the realistic case -- a genuine
      confirmatory corpus is built through several ``gcap run --config X``
      invocations (one ``ConfigName`` per invocation is baked into the CLI
      itself), each submitting only PART of the frozen plan; a subset
      check is the only way to validate a partial submission against a
      plan that spans more than one invocation's worth of units.
    * ``manifest.run_list`` empty (the field's default -- every manifest
      built before it existed, or one that never populates it): falls back
      to the ORIGINAL exact-hash-equality check against ``manifest.
      run_list_hash``, using the SAME ``hash_run_list`` function whatever
      produced the frozen value is expected to have used. This only ever
      accepts a batch that submits the WHOLE frozen plan in one call.

    Takes plain identity tuples rather than ``harness.runner.RunSpec``
    objects so this module never needs to import ``runner`` (which already
    imports this module) -- ``Harness.run_batch`` is what extracts the
    tuples from its own ``RunSpec`` sequence before calling this."""
    if manifest.run_list:
        frozen_units = set(manifest.run_list)
        not_sanctioned = [spec for spec in specs if spec not in frozen_units]
        if not_sanctioned:
            raise RuntimeManifestMismatchError(
                "Cannot start this batch of confirmatory (split='confirm') runs: "
                f"{len(not_sanctioned)} of {len(specs)} planned unit(s) are not part of the committed "
                f"freeze manifest {manifest.manifest_version!r}'s frozen run list, e.g. "
                f"{not_sanctioned[0]!r}. Every (config, case_id, replicate, draw_id) unit about to run "
                "must be one the frozen confirmatory run plan actually called for."
            )
        return

    actual_hash = hash_run_list(specs)
    if actual_hash != manifest.run_list_hash:
        raise RuntimeManifestMismatchError(
            "Cannot start this batch of confirmatory (split='confirm') runs: the actual planned run "
            f"list hashes to {actual_hash!r}, which does not match the committed freeze manifest "
            f"{manifest.manifest_version!r}'s run_list_hash {manifest.run_list_hash!r}. The set of "
            "(config, case_id, replicate, draw_id) units about to run differs from the frozen "
            "confirmatory run plan (manifest.run_list was not populated, so only an exact whole-plan "
            "match is accepted -- see this function's docstring)."
        )


def validate_confirmatory_corpus_complete(manifest: FreezeManifest, records: Sequence[RunRecord]) -> None:
    """P1-01 (v2.1.0 third code review round): the READ-side mirror of
    ``validate_run_list_against_manifest`` above -- that function checks,
    at batch-SUBMISSION time, that the units about to run match the frozen
    plan; this checks, at run-log CLOSE time, that the units actually
    PERSISTED do. Called from ``harness.cli._cmd_close_run_log`` with every
    ``split="confirm"`` ``RunRecord`` stored under the manifest_hash about
    to be closed.

    Before this existed, ``RunLogStore.close`` (a bare file-marker write,
    see its own docstring) had no way to know whether the confirmatory
    corpus it was about to declare COMPLETE actually was -- a case still in
    flight, one that crashed the harness process before ``save_run`` ever
    ran, or simply never submitted, would leave a gap ``close-run-log``
    could not see. A recorded FAILURE (``RunRecord.failure != "none"``)
    still counts as complete here -- invariant 9 treats a failure as an
    attempted, recorded outcome, never as missing data; only a genuinely
    ABSENT record for a planned unit fails this check."""
    actual_hash = hash_run_list([(r.config, r.case_id, r.replicate, r.draw_id) for r in records])
    if actual_hash != manifest.run_list_hash:
        raise ConfirmatoryCorpusIncompleteError(
            f"Refusing to close the run log for manifest_hash {manifest.manifest_hash()!r}: the "
            f"{len(records)} persisted split='confirm' RunRecord(s) hash to {actual_hash!r}, which does "
            f"not match this manifest's frozen run_list_hash {manifest.run_list_hash!r}. Every "
            "(config, case_id, replicate, draw_id) unit the frozen confirmatory run plan calls for must "
            "have a persisted RunRecord (a success OR a recorded failure -- invariant 9) before this "
            "manifest's run log may be closed."
        )
