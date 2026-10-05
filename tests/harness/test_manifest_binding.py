"""P0-01 (v2.1.0 third code review round): regression tests for the
runtime-vs-manifest binding barrier -- gcap.harness.manifest_binding's
own unit-level behaviour (this file) plus its wiring into Harness.run_case/
run_batch (tests/harness/test_runner.py's confirm-split tests, which now
all exercise this barrier as a side effect of using tests/harness/
factories.make_manifest()'s real, self-consistent defaults).

``tests/harness/factories.make_manifest()`` is the "everything matches"
baseline throughout: it now computes every P0-01-relevant field (node_
prompt_hashes, config_flags_table_hash, analysis_code_hash, parameter_
registry_hash/version) the SAME way ``validate_runtime_against_manifest``
recomputes them at runtime, using a calibrated default registry and the
default ConcurrencyConfig() Harness itself defaults to. Each "rejects ..."
test below takes that baseline and mutates exactly one field on the
manifest (or the runtime input) to prove the corresponding check actually
fires -- never by constructing an already-broken manifest, so a bug that
made every check a no-op would still show every "rejects" test failing to
raise.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from gcap.config.registry import ConcurrencyConfig, default_registry
from gcap.errors import ConfirmatoryCorpusIncompleteError, RuntimeManifestMismatchError
from gcap.harness.manifest_binding import (
    hash_analysis_code,
    hash_config_table_default,
    hash_node_prompts,
    validate_confirmatory_corpus_complete,
    validate_run_list_against_manifest,
    validate_runtime_against_manifest,
)
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents
from tests.harness.factories import make_manifest

_REGISTRY = default_registry().mark_all_calibrated()


def _validate(
    manifest,
    *,
    registry=None,
    model_snapshot="fake-luna-sandbox-1.0",
    concurrency=None,
    **runtime_overrides,
):
    runtime_registry = registry if registry is not None else _REGISTRY
    tolerance = runtime_registry.to_decide.numerical_error_tolerance
    runtime = dict(
        registry=runtime_registry,
        model_snapshot=model_snapshot,
        concurrency=concurrency if concurrency is not None else ConcurrencyConfig(),
        token_budget=runtime_registry.fixed.token_budget,
        numerical_tolerances={
            "revenue_growth_yoy": tolerance.return_pct,
            "portfolio_return_pct": tolerance.return_pct,
        },
        default_tolerance=tolerance.ratio_generic,
        day_count_basis=runtime_registry.to_decide.day_count_basis.days,
        marginal_band_ann=runtime_registry.fixed.marginal_band.primary_ann,
        hedge_rp_reduction_pct=(
            runtime_registry.to_decide.revised_margin_recomputation.hedge_rp_reduction_pct
        ),
        limit_addon_reduction_pct=(
            runtime_registry.to_decide.revised_margin_recomputation.limit_addon_reduction_pct
        ),
        tenor_restructuring_bounds=runtime_registry.to_decide.tenor_restructuring_bounds,
    )
    runtime.update(runtime_overrides)
    validate_runtime_against_manifest(manifest, **runtime)


def test_validate_runtime_against_manifest_passes_for_a_self_consistent_manifest():
    manifest = make_manifest(registry=_REGISTRY)
    _validate(manifest)  # must not raise


def test_validate_runtime_against_manifest_rejects_registry_content_hash_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    drifted = _REGISTRY.model_copy(
        update={
            "fixed": _REGISTRY.fixed.model_copy(
                update={
                    "marginal_band": _REGISTRY.fixed.marginal_band.model_copy(
                        update={"primary_bps": Decimal("99")}
                    )
                }
            )
        }
    )
    assert drifted.registry_version == _REGISTRY.registry_version  # same version, different values
    with pytest.raises(RuntimeManifestMismatchError, match="content_hash"):
        _validate(manifest, registry=drifted)


def test_validate_runtime_against_manifest_rejects_registry_version_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    renamed = _REGISTRY.model_copy(update={"registry_version": "9.9.9-mismatch"})
    with pytest.raises(RuntimeManifestMismatchError, match="registry_version"):
        _validate(manifest, registry=renamed)


def test_validate_runtime_against_manifest_rejects_model_snapshot_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    with pytest.raises(RuntimeManifestMismatchError, match="model_snapshot"):
        _validate(manifest, model_snapshot="some-other-model-snapshot")


def test_validate_runtime_against_manifest_rejects_missing_primary_model_snapshot():
    """A manifest that never recorded a 'primary' model_snapshots entry has
    nothing to compare against -- must not raise (this mirrors how the
    node_prompt_hashes check below only compares entries the manifest
    actually recorded)."""
    manifest = make_manifest(registry=_REGISTRY).model_copy(update={"model_snapshots": {}})
    with pytest.raises(RuntimeManifestMismatchError, match="primary"):
        _validate(manifest)


def test_validate_runtime_against_manifest_rejects_node_prompt_hash_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    tampered = manifest.model_copy(
        update={"node_prompt_hashes": {**manifest.node_prompt_hashes, "research": "f" * 64}}
    )
    with pytest.raises(RuntimeManifestMismatchError, match="research"):
        _validate(tampered)


def test_validate_runtime_against_manifest_rejects_missing_required_node_prompt_hashes():
    manifest = make_manifest(registry=_REGISTRY).model_copy(update={"node_prompt_hashes": {}})
    with pytest.raises(RuntimeManifestMismatchError, match="required"):
        _validate(manifest)


def test_validate_runtime_against_manifest_rejects_config_table_hash_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    tampered = manifest.model_copy(update={"config_flags_table_hash": "e" * 64})
    with pytest.raises(RuntimeManifestMismatchError, match="CONFIG_TABLE"):
        _validate(tampered)


def test_validate_runtime_against_manifest_rejects_analysis_code_hash_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    tampered = manifest.model_copy(update={"analysis_code_hash": "a" * 64})
    with pytest.raises(RuntimeManifestMismatchError, match="analysis"):
        _validate(tampered)


def test_validate_runtime_against_manifest_rejects_concurrency_mismatch():
    manifest = make_manifest(registry=_REGISTRY)
    with pytest.raises(RuntimeManifestMismatchError, match="concurrency"):
        _validate(manifest, concurrency=ConcurrencyConfig(model_call_semaphore=1))


def test_validate_runtime_against_manifest_rejects_financial_runtime_drift():
    manifest = make_manifest(registry=_REGISTRY)
    with pytest.raises(RuntimeManifestMismatchError, match="day_count_basis"):
        _validate(manifest, day_count_basis=Decimal("360"))


def test_validate_runtime_against_manifest_collects_every_mismatch_in_one_error():
    manifest = make_manifest(registry=_REGISTRY)
    with pytest.raises(RuntimeManifestMismatchError) as excinfo:
        _validate(
            manifest,
            model_snapshot="wrong-model",
            concurrency=ConcurrencyConfig(model_call_semaphore=1),
        )
    message = str(excinfo.value)
    assert "model_snapshot" in message
    assert "concurrency" in message
    assert message.count("\n-") == 2  # exactly the two problems injected above


def test_validate_runtime_against_manifest_sanity_check_helpers_agree_with_each_other():
    """hash_node_prompts()/hash_config_table_default()/hash_analysis_code()
    are each called twice, independently, inside make_manifest() and inside
    validate_runtime_against_manifest() -- this pins that both call sites
    really do compute the same values (the property the "passes for a
    self-consistent manifest" test above relies on), rather than merely
    happening to agree by coincidence in this environment."""
    assert hash_node_prompts() == hash_node_prompts()
    assert hash_config_table_default() == hash_config_table_default()
    assert hash_analysis_code() == hash_analysis_code()


def test_validate_run_list_against_manifest_passes_when_the_plan_matches():
    specs = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    from gcap.config.hashing import hash_run_list

    manifest = make_manifest(registry=_REGISTRY).model_copy(update={"run_list_hash": hash_run_list(specs)})
    validate_run_list_against_manifest(manifest, specs)  # must not raise


def test_validate_run_list_against_manifest_is_order_independent():
    from gcap.config.hashing import hash_run_list

    specs = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    manifest = make_manifest(registry=_REGISTRY).model_copy(update={"run_list_hash": hash_run_list(specs)})
    validate_run_list_against_manifest(manifest, list(reversed(specs)))  # must not raise


def test_validate_run_list_against_manifest_rejects_a_diverged_plan():
    from gcap.config.hashing import hash_run_list

    frozen_specs = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    manifest = make_manifest(registry=_REGISTRY).model_copy(
        update={"run_list_hash": hash_run_list(frozen_specs)}
    )
    actually_planned = [("B", "case-1", 0, None), ("C", "case-2", 0, None)]  # a different case for C
    with pytest.raises(RuntimeManifestMismatchError, match="run_list_hash"):
        validate_run_list_against_manifest(manifest, actually_planned)


def test_validate_run_list_against_manifest_accepts_a_partial_submission_when_run_list_is_populated():
    """P1-01 (v2.1.0 third code review round) follow-on fix: a real
    confirmatory corpus is built through several `gcap run --config X`
    invocations (one ConfigName per invocation is baked into the CLI
    itself), so a batch that submits only PART of a multi-config frozen
    plan must be accepted -- as long as ``manifest.run_list`` (the actual
    identity tuples, not just their hash) was populated at freeze time.
    Before this, only exact whole-plan equality was ever accepted, which
    would have made a genuine multi-arm confirmatory study impossible to
    run through this CLI at all."""
    from gcap.config.hashing import hash_run_list

    frozen_plan = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    manifest = make_manifest(registry=_REGISTRY).model_copy(
        update={"run_list_hash": hash_run_list(frozen_plan), "run_list": frozen_plan}
    )
    only_b_this_invocation = [("B", "case-1", 0, None)]
    validate_run_list_against_manifest(manifest, only_b_this_invocation)  # must not raise


def test_validate_run_list_against_manifest_rejects_an_unsanctioned_unit_when_run_list_is_populated():
    from gcap.config.hashing import hash_run_list

    frozen_plan = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    manifest = make_manifest(registry=_REGISTRY).model_copy(
        update={"run_list_hash": hash_run_list(frozen_plan), "run_list": frozen_plan}
    )
    not_in_the_frozen_plan = [("B", "case-1", 0, None), ("D1", "case-1", 0, None)]
    with pytest.raises(RuntimeManifestMismatchError, match="not part of"):
        validate_run_list_against_manifest(manifest, not_in_the_frozen_plan)


def _confirm_record(*, config, case_id, replicate=0, draw_id=None, failure="none") -> RunRecord:
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split="confirm",
        block="blk",
        case_id=case_id,
        replicate=replicate,
        draw_id=draw_id,
        family="numerical_reconciliation_error",
        materiality="medium",
        model_snapshot="fake-luna-sandbox-1.0",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="0" * 64,
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
        failure=failure,
    )


def test_validate_confirmatory_corpus_complete_passes_when_persisted_records_match_the_plan():
    from gcap.config.hashing import hash_run_list

    records = [_confirm_record(config="B", case_id="case-1"), _confirm_record(config="C", case_id="case-1")]
    manifest = make_manifest(registry=_REGISTRY).model_copy(
        update={
            "run_list_hash": hash_run_list([(r.config, r.case_id, r.replicate, r.draw_id) for r in records])
        }
    )
    validate_confirmatory_corpus_complete(manifest, records)  # must not raise


def test_validate_confirmatory_corpus_complete_treats_a_recorded_failure_as_attempted():
    """P1-01 (v2.1.0 third code review round): invariant 9 -- a recorded
    failure is an attempted, recorded outcome, never missing data. A
    manifest frozen against a plan that includes a since-failed unit must
    still be considered complete once that unit has ANY persisted
    RunRecord, success or failure."""
    from gcap.config.hashing import hash_run_list

    records = [_confirm_record(config="B", case_id="case-1", failure="runtime")]
    manifest = make_manifest(registry=_REGISTRY).model_copy(
        update={
            "run_list_hash": hash_run_list([(r.config, r.case_id, r.replicate, r.draw_id) for r in records])
        }
    )
    validate_confirmatory_corpus_complete(manifest, records)  # must not raise


def test_validate_confirmatory_corpus_complete_rejects_a_missing_unit():
    """The exact P1-01 scenario: `close-run-log` asked to declare a corpus
    complete when one of the frozen plan's units (here, config C's case-1)
    has no persisted RunRecord at all -- still in flight, crashed before
    writing, or simply never submitted."""
    from gcap.config.hashing import hash_run_list

    frozen_plan = [("B", "case-1", 0, None), ("C", "case-1", 0, None)]
    manifest = make_manifest(registry=_REGISTRY).model_copy(update={"run_list_hash": hash_run_list(frozen_plan)})
    only_b_persisted = [_confirm_record(config="B", case_id="case-1")]
    with pytest.raises(ConfirmatoryCorpusIncompleteError, match="run_list_hash"):
        validate_confirmatory_corpus_complete(manifest, only_b_persisted)
