from __future__ import annotations

import uuid

import numpy as np
import pytest

from gcap.analysis.report import build_confirmatory_report
from gcap.errors import ConfirmatoryUnsealError
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents

CONTRASTS = (
    ("ep_numerical_error", "B", "C"),
    ("ep_state_infeasible", "C", "D1"),
    ("ep_pricing_inconsistent", "D1", "D2"),
    ("ep_governance_failure", "D2", "E"),
)


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
        model_snapshot="fake-luna-sandbox-1.0",
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


def _generate_contrast_records(
    endpoint_field, arm_a, arm_b, p_a, p_b, n_clusters, n_reps, rng, split="dev"
):
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
                        split=split,
                    )
                )
    return records


def _all_four_contrast_records(rng):
    planted = {
        "ep_numerical_error": (0.40, 0.10),
        "ep_state_infeasible": (0.35, 0.08),
        "ep_pricing_inconsistent": (0.45, 0.12),
        "ep_governance_failure": (0.30, 0.06),
    }
    records = []
    for endpoint_field, arm_a, arm_b in CONTRASTS:
        p_a, p_b = planted[endpoint_field]
        records.extend(
            _generate_contrast_records(
                endpoint_field, arm_a, arm_b, p_a, p_b, n_clusters=25, n_reps=4, rng=rng
            )
        )
    return records


def test_report_summarizes_all_four_contrasts_with_planted_effects():
    rng = np.random.default_rng(2026)
    records = _all_four_contrast_records(rng)
    report = build_confirmatory_report(records, run_log_closed=True, n_boot=500, seed=1)

    assert report.n_records == len(records)
    assert len(report.contrasts) == 4
    assert report.missing_contrasts == ()
    endpoints_seen = {c.endpoint for c in report.contrasts}
    assert endpoints_seen == {
        "numerical_error",
        "state_infeasible",
        "pricing_inconsistent",
        # P1-05: this contrast's report-facing identifier is
        # "governance_failure_hard_rule_only", not bare "governance_failure"
        # -- see analysis.endpoints.ENDPOINT_FIELDS's own comment.
        "governance_failure_hard_rule_only",
    }
    for c in report.contrasts:
        assert c.absolute_risk_difference < 0
        assert c.holm_reject is True
        assert c.ard_ci[0] <= c.absolute_risk_difference <= c.ard_ci[1]
        assert c.n_clusters > 0


def test_missing_contrasts_are_reported_when_data_is_incomplete():
    rng = np.random.default_rng(3)
    records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.30, 0.10, n_clusters=20, n_reps=3, rng=rng
    )
    report = build_confirmatory_report(records, run_log_closed=True, n_boot=300, seed=1)

    assert len(report.contrasts) == 1
    assert report.contrasts[0].endpoint == "numerical_error"
    assert set(report.missing_contrasts) == {
        "state_infeasible",
        "pricing_inconsistent",
        "governance_failure_hard_rule_only",  # P1-05: see ENDPOINT_FIELDS's own comment
    }


def test_failed_records_are_counted_and_surfaced_but_excluded_from_contrasts():
    """P1-8 (code review): failure!='none' rows must (a) never be folded
    into a contrast as a fake non-event (analysis.endpoints's own fix) and
    (b) still be visibly counted in the report, so a failure-heavy run does
    not look identical to a clean one just because its contrasts still
    computed from whatever clean rows remained."""
    rng = np.random.default_rng(3)
    records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.30, 0.10, n_clusters=20, n_reps=3, rng=rng
    )
    n_clean = len(records)
    failed = [
        _run_record("C", "extra-failed-case", 0, "ep_numerical_error", True).model_copy(
            update={"failure": "runtime"}
        )
        for _ in range(3)
    ]
    report = build_confirmatory_report(records + failed, run_log_closed=True, n_boot=300, seed=1)

    assert report.n_records == n_clean + 3
    assert report.n_failed_records == 3
    # The failed rows' planted ep_numerical_error=True never reaches the
    # numerical_error contrast -- n_clusters is unaffected by adding them.
    numerical = next(c for c in report.contrasts if c.endpoint == "numerical_error")
    unfailed_only_report = build_confirmatory_report(records, run_log_closed=True, n_boot=300, seed=1)
    assert numerical.n_clusters == unfailed_only_report.contrasts[0].n_clusters
    assert "3 record(s) had failure != 'none'" in report.to_text()


def test_to_text_includes_every_contrast_and_a_holm_verdict():
    rng = np.random.default_rng(2026)
    records = _all_four_contrast_records(rng)
    report = build_confirmatory_report(records, run_log_closed=True, n_boot=500, seed=1)
    text = report.to_text()
    assert "numerical_error" in text
    assert "REJECT H0 (Holm)" in text
    assert str(len(records)) in text


def test_to_text_handles_no_contrasts_gracefully():
    from gcap.analysis.report import ConfirmatoryReport

    report = ConfirmatoryReport(n_records=0, contrasts=(), missing_contrasts=("numerical_error",))
    text = report.to_text()
    assert "No primary contrast had pairable data." in text
    assert "numerical_error" in text


def test_report_still_enforces_the_confirmatory_seal_guard():
    rng = np.random.default_rng(1)
    records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=10, n_reps=3, rng=rng, split="confirm"
    )
    with pytest.raises(ConfirmatoryUnsealError):
        build_confirmatory_report(records, run_log_closed=False)


def test_report_rejects_a_record_set_mixing_confirm_and_dev_splits():
    """P0-02 (v2.1.0 third code review round) acceptance test, at the
    build_confirmatory_report level `gcap analyze` actually calls: a
    caller who forgets --split confirm (both --split and --manifest-hash
    are optional on that command) and gets back a store's full mix of
    dev and confirm rows must be refused, not handed a confirmatory report
    silently contaminated with dev-split rehearsal data pooled in under
    the SAME config arms (B vs C here)."""
    rng = np.random.default_rng(9)
    confirm_records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=10, n_reps=3, rng=rng, split="confirm"
    )
    dev_records = _generate_contrast_records(
        "ep_numerical_error", "B", "C", 0.3, 0.1, n_clusters=5, n_reps=2, rng=rng, split="dev"
    )
    with pytest.raises(ConfirmatoryUnsealError, match="mixes split='confirm'"):
        build_confirmatory_report(confirm_records + dev_records, run_log_closed=True)
