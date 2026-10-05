from datetime import UTC, datetime

import pytest

from gcap.core.policy import (
    PolicyContext,
    PolicyRule,
    evaluate,
    get_predicate,
    register_predicate,
    registered_predicate_ids,
)
from gcap.errors import PolicyPredicateNotRegisteredError

NOW = datetime(2026, 9, 21, tzinfo=UTC)


def test_unregistered_predicate_raises():
    with pytest.raises(PolicyPredicateNotRegisteredError):
        get_predicate("not.a.real.predicate")


def test_reference_predicates_are_registered():
    ids = registered_predicate_ids()
    assert "hard.mandate_asset_class_breach" in ids
    assert "soft.concentration_corridor" in ids


def _rule(rule_id, severity, predicate_id) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        owner="cro",
        scope="mandate",
        severity=severity,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        version="v1",
        predicate_id=predicate_id,
    )


def test_hard_mandate_breach_passing_and_failing_case():
    rule = _rule("R1", "hard", "hard.mandate_asset_class_breach")

    passing_ctx = PolicyContext(values={"mandate_permitted_asset_classes": {"equity"}, "instrument_asset_class": "equity"})
    result = evaluate([rule], passing_ctx, NOW, "policy-v1")
    assert result.verdict == "PASS"

    failing_ctx = PolicyContext(values={"mandate_permitted_asset_classes": {"equity"}, "instrument_asset_class": "fixed_income"})
    result = evaluate([rule], failing_ctx, NOW, "policy-v1")
    assert result.verdict == "HARD_FAIL"
    assert result.triggering_rule_ids == ("R1",)


def test_soft_corridor_passing_and_failing_case():
    rule = _rule("R2", "soft", "soft.concentration_corridor")

    passing_ctx = PolicyContext(values={"concentration_used_pct": 0.5, "concentration_corridor_pct": 0.85})
    assert evaluate([rule], passing_ctx, NOW, "policy-v1").verdict == "PASS"

    failing_ctx = PolicyContext(values={"concentration_used_pct": 0.9, "concentration_corridor_pct": 0.85})
    result = evaluate([rule], failing_ctx, NOW, "policy-v1")
    assert result.verdict == "ESCALATE"
    assert result.triggering_rule_ids == ("R2",)


def test_hard_fail_takes_precedence_over_escalate():
    hard = _rule("R1", "hard", "hard.mandate_asset_class_breach")
    soft = _rule("R2", "soft", "soft.concentration_corridor")
    ctx = PolicyContext(
        values={
            "mandate_permitted_asset_classes": {"equity"},
            "instrument_asset_class": "fixed_income",
            "concentration_used_pct": 0.9,
            "concentration_corridor_pct": 0.85,
        }
    )
    result = evaluate([hard, soft], ctx, NOW, "policy-v1")
    assert result.verdict == "HARD_FAIL"
    assert result.triggering_rule_ids == ("R1",)


def test_rule_outside_effective_interval_is_ignored():
    rule = _rule("R1", "hard", "hard.mandate_asset_class_breach").model_copy(
        update={"effective_from": datetime(2027, 1, 1, tzinfo=UTC)}
    )
    ctx = PolicyContext(values={"mandate_permitted_asset_classes": {"equity"}, "instrument_asset_class": "fixed_income"})
    result = evaluate([rule], ctx, NOW, "policy-v1")
    assert result.verdict == "PASS"
    assert result.evaluated_rule_ids == ()


def test_custom_predicate_registration_roundtrip():
    @register_predicate("test.always_breach")
    def _always(ctx: PolicyContext) -> bool:
        return True

    rule = _rule("R9", "hard", "test.always_breach")
    ctx = PolicyContext(values={})
    result = evaluate([rule], ctx, NOW, "policy-v1")
    assert result.verdict == "HARD_FAIL"
