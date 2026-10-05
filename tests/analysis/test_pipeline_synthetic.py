"""Milestone 1 acceptance test (Section 11): "analysis/: Full pipeline on
synthetic RunRecords with a known planted effect recovers it; separation
fallback fires on a zero-event cell; Holm, Clopper-Pearson and bootstrap
outputs match reference values." The Holm/Clopper-Pearson/bootstrap
reference-value checks live in their own dedicated test modules; this
module is the end-to-end planted-effect recovery test plus the invariant-8
analysis-side guard.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from gcap.analysis.pipeline import run_all_primary_contrasts
from gcap.config.manifest import FreezeManifestStore, RunLogStore
from gcap.errors import ConfirmatoryUnsealError, ManifestMissingError
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents
from tests.harness.factories import make_manifest

CONTRASTS = (
    ("ep_numerical_error", "B", "C"),
    ("ep_state_infeasible", "C", "D1"),
    ("ep_pricing_inconsistent", "D1", "D2"),
    ("ep_governance_failure", "D2", "E"),
)


def _outcome_key(endpoint_field: str) -> str:
    """P1-05 (v2.1.0 third code review round): run_all_primary_contrasts's
    outcome dict is keyed by analysis.endpoints.PRIMARY_CONTRASTS's own
    report-facing identifiers, which is a plain ``removeprefix("ep_")`` of
    the RunRecord field name for three of the four contrasts but NOT for
    governance -- that one is deliberately keyed
    "governance_failure_hard_rule_only", not bare "governance_failure" (see
    analysis.endpoints.ENDPOINT_FIELDS's own comment on why). This mirrors
    that one exception explicitly rather than hard-coding a now-wrong
    removeprefix assumption into every call site below."""
    if endpoint_field == "ep_governance_failure":
        return "governance_failure_hard_rule_only"
    return endpoint_field.removeprefix("ep_")


def _run_record(config, case_id, replicate, endpoint_field, value, split="dev"):
    endpoints = {
        "ep_numerical_error": False,
        "ep_state_infeasible": False,
        "ep_pricing_inconsistent": False,
        "ep_governance_failure": False,
    }
    endpoints[endpoint_field] = value
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split=split,
        block=f"{config}-block",
        case_id=case_id,
        replicate=replicate,
        family="numerical_reconciliation_error",
        materiality="medium",
        model_snapshot="gpt-luna-test",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="0" * 64,
        ep_numerical_error=endpoints["ep_numerical_error"],
        ep_state_infeasible=endpoints["ep_state_infeasible"],
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=endpoints["ep_pricing_inconsistent"],
        ep_governance_failure=endpoints["ep_governance_failure"],
        ep_governance_components=GovernanceComponents(),
    )


def _generate_contrast_records(endpoint_field, arm_a, arm_b, p_a, p_b, n_clusters, n_reps, rng):
    records = []
    for s in range(n_clusters):
        scenario_effect = rng.normal(0, 0.4)
        for config, base_p in ((arm_a, p_a), (arm_b, p_b)):
            logit = np.log(base_p / (1 - base_p)) + scenario_effect
            p = 1 / (1 + np.exp(-logit))
            p = min(max(p, 1e-6), 1 - 1e-6)
            for r in range(n_reps):
                y = bool(rng.binomial(1, p))
                records.append(
                    _run_record(
                        config=config,
                        case_id=f"{endpoint_field}-s{s}",
                        replicate=r,
                        endpoint_field=endpoint_field,
                        value=y,
                    )
                )
    return records


def test_full_pipeline_recovers_planted_effects_on_all_four_contrasts():
    rng = np.random.default_rng(2026)
    all_records = []
    # Every contrast is given a strong, unambiguous planted reduction in
    # the treatment arm (roughly matching the proposal's 15-point-plus
    # planning effect, scaled up here so a single synthetic draw reliably
    # recovers it without needing thousands of clusters).
    planted = {
        "ep_numerical_error": (0.40, 0.10),
        "ep_state_infeasible": (0.35, 0.08),
        "ep_pricing_inconsistent": (0.45, 0.12),
        "ep_governance_failure": (0.30, 0.06),
    }
    for endpoint_field, arm_a, arm_b in CONTRASTS:
        p_a, p_b = planted[endpoint_field]
        all_records.extend(
            _generate_contrast_records(endpoint_field, arm_a, arm_b, p_a, p_b, n_clusters=35, n_reps=5, rng=rng)
        )

    outcomes = run_all_primary_contrasts(all_records, run_log_closed=True, alpha=0.05, n_boot=1500, seed=1)

    assert set(outcomes) == {_outcome_key(e) for e, _, _ in CONTRASTS}
    for endpoint_field, _arm_a, _arm_b in CONTRASTS:
        outcome = outcomes[_outcome_key(endpoint_field)]
        assert outcome.result.absolute_risk_difference < 0, f"{endpoint_field}: expected a negative ARD"
        assert outcome.holm_reject is True, f"{endpoint_field}: expected Holm-adjusted significance"
        lower, upper = outcome.result.ard_ci
        assert lower <= outcome.result.absolute_risk_difference <= upper


def test_pipeline_refuses_confirmatory_records_when_run_log_not_closed(tmp_path):
    # P1-5: a confirmatory analysis also requires all four co-primary
    # contrasts to have pairable data (never a partial confirmatory
    # result), so this test's confirm-split records must cover all four,
    # not just the one under direct test.
    rng = np.random.default_rng(1)
    all_records = []
    for endpoint_field, arm_a, arm_b in CONTRASTS:
        all_records.extend(
            _generate_contrast_records(endpoint_field, arm_a, arm_b, 0.3, 0.1, n_clusters=10, n_reps=3, rng=rng)
        )
    confirm_records = [r.model_copy(update={"split": "confirm"}) for r in all_records]
    with pytest.raises(ConfirmatoryUnsealError):
        run_all_primary_contrasts(confirm_records, run_log_closed=False)

    identities = [(r.config, r.case_id, r.replicate, r.draw_id) for r in confirm_records]
    from gcap.config.hashing import hash_run_list

    manifest = make_manifest("closed-analysis").model_copy(
        update={"run_list": tuple(identities), "run_list_hash": hash_run_list(identities)}
    )
    confirm_records = [
        r.model_copy(update={"manifest_hash": manifest.manifest_hash()}) for r in confirm_records
    ]
    manifest_store = FreezeManifestStore(tmp_path / "manifest_store")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log_store")
    run_log_store.close(manifest.manifest_hash(), closed_by="test")
    outcomes = run_all_primary_contrasts(
        confirm_records,
        run_log_closed=False,
        n_boot=200,
        seed=1,
        manifest_store=manifest_store,
        run_log_store=run_log_store,
    )
    assert "numerical_error" in outcomes
    assert set(outcomes) == {_outcome_key(e) for e, _, _ in CONTRASTS}


def _full_confirm_record_set(rng, manifest_hash="0" * 64):
    all_records = []
    for endpoint_field, arm_a, arm_b in CONTRASTS:
        all_records.extend(
            _generate_contrast_records(endpoint_field, arm_a, arm_b, 0.3, 0.1, n_clusters=10, n_reps=3, rng=rng)
        )
    return [r.model_copy(update={"split": "confirm", "manifest_hash": manifest_hash}) for r in all_records]


def test_pipeline_rejects_a_record_set_mixing_confirm_and_non_confirm_splits():
    """P0-02 (v2.1.0 third code review round): before this fix, only the
    split='confirm' SUBSET was checked for manifest/closure validity --
    ``run_all_primary_contrasts`` then fed the FULL, unfiltered record set
    (including these dev-split rows, sharing the confirm set's B/C config
    arm) straight into every contrast's cluster extraction, silently
    pooling rehearsal data into a result ``is_confirmatory_analysis`` still
    called confirmatory off the strength of the confirm rows alone. This
    reproduces the review's own scenario: a `gcap analyze` invocation with
    no --split filter against a store holding both splits."""
    rng = np.random.default_rng(7)
    confirm_records = _full_confirm_record_set(rng)
    dev_records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=5, n_reps=2, rng=rng
    )
    with pytest.raises(ConfirmatoryUnsealError, match="mixes split='confirm'"):
        run_all_primary_contrasts(confirm_records + dev_records, run_log_closed=True, n_boot=200, seed=1)


def test_pipeline_rejects_mixed_splits_even_when_a_manifest_store_is_supplied(tmp_path):
    """Same guard, exercised on the P1-5 store-authoritative path (the one
    `gcap analyze --manifest-store --run-log-store` actually uses) rather
    than the bare-bool path above -- the split-homogeneity check runs
    before either branch, so both must refuse identically."""
    rng = np.random.default_rng(8)
    manifest = make_manifest("mixed-split-1.0.0")
    confirm_records = _full_confirm_record_set(rng, manifest_hash=manifest.manifest_hash())
    dev_records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=5, n_reps=2, rng=rng
    )
    manifest_store = FreezeManifestStore(tmp_path / "manifest_store")
    manifest_store.commit(manifest)
    with pytest.raises(ConfirmatoryUnsealError, match="mixes split='confirm'"):
        run_all_primary_contrasts(
            confirm_records + dev_records,
            run_log_closed=True,
            n_boot=200,
            seed=1,
            manifest_store=manifest_store,
        )


def test_pipeline_rejects_mixed_manifest_hashes_when_a_manifest_store_is_supplied(tmp_path):
    """P1-5: with a manifest_store supplied, a confirm-split record set
    stamped with more than one manifest_hash must be refused outright —
    mixing runs from two different manifests into one confirmatory
    analysis is exactly the apples-to-oranges mistake Section 9.2 rule 2
    exists to prevent."""
    rng = np.random.default_rng(3)
    records = _full_confirm_record_set(rng, manifest_hash="1" * 64)
    records[0] = records[0].model_copy(update={"manifest_hash": "2" * 64})
    manifest_store = FreezeManifestStore(tmp_path / "manifest_store")
    with pytest.raises(ConfirmatoryUnsealError):
        run_all_primary_contrasts(records, run_log_closed=True, n_boot=200, seed=1, manifest_store=manifest_store)


def test_pipeline_rejects_manifest_hash_that_is_not_the_active_committed_one(tmp_path):
    rng = np.random.default_rng(4)
    manifest_store = FreezeManifestStore(tmp_path / "manifest_store")
    manifest_store.commit(make_manifest("committed-1.0.0"))
    records = _full_confirm_record_set(rng, manifest_hash="not-the-committed-hash")
    run_log_store = RunLogStore(tmp_path / "run_log_store")
    with pytest.raises(ManifestMissingError):
        run_all_primary_contrasts(
            records,
            run_log_closed=True,
            n_boot=200,
            seed=1,
            manifest_store=manifest_store,
            run_log_store=run_log_store,
        )


def test_pipeline_run_log_store_is_authoritative_over_the_bare_flag(tmp_path):
    """P1-5's core fix: once a RunLogStore is supplied, --run-log-closed
    (here passed as True) is NOT enough on its own -- there must be a
    genuine closure record for this exact manifest_hash."""
    rng = np.random.default_rng(5)
    manifest = make_manifest("run-log-test-1.0.0")
    records = _full_confirm_record_set(rng)
    from gcap.config.hashing import hash_run_list

    identities = [(r.config, r.case_id, r.replicate, r.draw_id) for r in records]
    manifest = manifest.model_copy(
        update={"run_list": tuple(identities), "run_list_hash": hash_run_list(identities)}
    )
    records = [r.model_copy(update={"manifest_hash": manifest.manifest_hash()}) for r in records]
    manifest_store = FreezeManifestStore(tmp_path / "manifest_store")
    manifest_store.commit(manifest)
    run_log_store = RunLogStore(tmp_path / "run_log_store")

    with pytest.raises(ConfirmatoryUnsealError):
        run_all_primary_contrasts(
            records, run_log_closed=True, n_boot=200, seed=1,
            manifest_store=manifest_store, run_log_store=run_log_store,
        )

    run_log_store.close(manifest.manifest_hash(), closed_by="test")
    outcomes = run_all_primary_contrasts(
        records, run_log_closed=False, n_boot=200, seed=1,
        manifest_store=manifest_store, run_log_store=run_log_store,
    )
    assert set(outcomes) == {_outcome_key(e) for e, _, _ in CONTRASTS}


def test_pipeline_refuses_a_partial_confirmatory_result_even_when_run_log_closed():
    """P1-5 regression test: even with the run log genuinely closed, a
    confirm-split record set missing one of the four co-primary contrasts
    must raise, never quietly report the other three as if that were the
    complete confirmatory result."""
    rng = np.random.default_rng(1)
    records = _generate_contrast_records("ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=10, n_reps=3, rng=rng)
    confirm_records = [r.model_copy(update={"split": "confirm"}) for r in records]
    with pytest.raises(ConfirmatoryUnsealError):
        run_all_primary_contrasts(confirm_records, run_log_closed=True, n_boot=200, seed=1)


def test_pipeline_holm_correction_is_applied_jointly_not_per_contrast():
    """A borderline-significant single contrast should NOT survive Holm
    correction once bundled with three flat (no-effect) contrasts, unless
    its uncorrected p-value already clears the tightened per-contrast
    Holm threshold — this test just confirms the adjusted p-value stored
    on the outcome is >= the contrast's own uncorrected p-value, which
    must always hold for Holm's step-down adjustment."""
    rng = np.random.default_rng(5)
    all_records = []
    all_records.extend(
        _generate_contrast_records("ep_numerical_error", "B", "C", 0.25, 0.20, n_clusters=20, n_reps=3, rng=rng)
    )
    for endpoint_field, arm_a, arm_b in CONTRASTS[1:]:
        all_records.extend(
            _generate_contrast_records(endpoint_field, arm_a, arm_b, 0.2, 0.2, n_clusters=20, n_reps=3, rng=rng)
        )
    outcomes = run_all_primary_contrasts(all_records, run_log_closed=True, n_boot=500, seed=2)
    for outcome in outcomes.values():
        assert outcome.holm_adjusted_p >= outcome.result.p_value - 1e-12
