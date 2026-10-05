from __future__ import annotations

import pytest

from gcap.agents.schemas import ClaimReview, TreasuryDecision, parse_structured_output
from gcap.errors import SchemaConformanceError


def test_parse_structured_output_returns_typed_model():
    out = parse_structured_output("claim_review", {"claim_id": "c1", "flagged": True}, node_name="research")
    assert isinstance(out, ClaimReview)
    assert out.flagged is True


def test_parse_structured_output_rejects_unknown_schema_name():
    with pytest.raises(SchemaConformanceError):
        parse_structured_output("not_a_schema", {}, node_name="research")


def test_parse_structured_output_rejects_malformed_payload():
    with pytest.raises(SchemaConformanceError):
        parse_structured_output("claim_review", {"claim_id": "c1"}, node_name="research")  # missing 'flagged'


def test_treasury_decision_coerces_string_action_to_enum():
    out = parse_structured_output(
        "treasury_decision", {"proposal_id": "p1", "treasury_action": "REJECT"}, node_name="treasury"
    )
    assert isinstance(out, TreasuryDecision)
    assert out.treasury_action.value == "REJECT"
