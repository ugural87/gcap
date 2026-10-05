"""Unit tests for the Integrated Validator (core.validator), independent of
the graph -- exercising every reason code against a synthetic
FTPOutput/PolicyEvaluationResult pair rather than running a whole case
through run_proposal (that end-to-end wiring is covered by
tests/graph/test_pipeline.py's config-E tests instead).
"""

from __future__ import annotations

from decimal import Decimal

from gcap.core.ftp import FTPAddonVector, FTPInputs, compute_ftp
from gcap.core.policy import PolicyEvaluationResult
from gcap.core.validator import validate_integrated_package


def _ftp_output(trade_alpha_horizon=Decimal("0.02"), base_rate=Decimal("0.01")):
    return compute_ftp(
        FTPInputs(
            tau_days=Decimal("180"),
            day_count_basis=Decimal("365"),
            base_funding_curve_rate_ann=base_rate,
            addons=FTPAddonVector(),
            trade_alpha_horizon=trade_alpha_horizon,
        )
    )


def _policy(verdict="PASS"):
    return PolicyEvaluationResult(
        verdict=verdict,
        triggering_rule_ids=() if verdict == "PASS" else ("R1",),
        policy_version="policy-v1",
        evaluated_rule_ids=("R1",),
    )


def _validate(**overrides):
    defaults = dict(
        ledger_stale=False,
        ledger_violations=(),
        pricing_ftp_output=_ftp_output(),
        pricing_margin=Decimal("0.01"),
        business_approval=True,
        policy_result=_policy(),
    )
    defaults.update(overrides)
    return validate_integrated_package(**defaults)


def test_clean_package_is_approved_with_no_reasons():
    decision = _validate()
    assert decision.approved is True
    assert decision.reasons == ()


def test_stale_state_is_flagged():
    decision = _validate(ledger_stale=True)
    assert decision.approved is False
    assert "STALE_STATE" in decision.reasons


def test_ledger_violations_flag_infeasible():
    decision = _validate(ledger_violations=("funding",))
    assert decision.approved is False
    assert "INFEASIBLE" in decision.reasons


def test_non_positive_pricing_margin_is_flagged():
    decision = _validate(pricing_margin=Decimal("0"))
    assert decision.approved is False
    assert "NON_POSITIVE_ECONOMIC_ALPHA" in decision.reasons

    decision_negative = _validate(pricing_margin=Decimal("-0.001"))
    assert "NON_POSITIVE_ECONOMIC_ALPHA" in decision_negative.reasons


def test_hard_policy_fail_blocks_regardless_of_everything_else():
    decision = _validate(policy_result=_policy("HARD_FAIL"))
    assert decision.approved is False
    assert "HARD_POLICY_FAIL" in decision.reasons


def test_policy_escalated_is_its_own_reason_distinct_from_hard_fail():
    decision = _validate(policy_result=_policy("ESCALATE"))
    assert decision.approved is False
    assert decision.reasons == ("POLICY_ESCALATED",)


def test_missing_business_approval_is_flagged():
    decision = _validate(business_approval=False)
    assert decision.approved is False
    assert "MISSING_BUSINESS_APPROVAL" in decision.reasons


def test_ftp_reconciliation_error_is_flagged_for_a_tampered_output():
    # compute_ftp guarantees internal consistency; construct a tampered
    # copy the same way a corrupted package would arrive at the gate.
    honest = _ftp_output()
    tampered = honest.model_copy(update={"economic_alpha_horizon": honest.economic_alpha_horizon + Decimal("1")})
    decision = _validate(pricing_ftp_output=tampered, pricing_margin=tampered.economic_alpha_horizon)
    assert decision.approved is False
    assert "FTP_RECONCILIATION_ERROR" in decision.reasons


def test_multiple_failures_all_reported_not_just_the_first():
    decision = _validate(ledger_stale=True, business_approval=False, policy_result=_policy("HARD_FAIL"))
    assert decision.approved is False
    assert set(decision.reasons) == {"STALE_STATE", "HARD_POLICY_FAIL", "MISSING_BUSINESS_APPROVAL"}
