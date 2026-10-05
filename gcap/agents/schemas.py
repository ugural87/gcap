"""Structured agent-output schemas, one per node, plus the strict
parse-and-validate step every node result goes through. Invariant 1 (LLMs
never compute/transition state/enforce) means every field on these models
is a *proposal* or *claim about the world* — never a resource balance,
never a policy verdict, never a committed action; the graph is the only
thing that turns an agent's stated intent into an executed action, and
only after the deterministic gates (ledger, FTP, policy) agree.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from gcap.errors import SchemaConformanceError
from gcap.store.run_record import TreasuryAction

T = TypeVar("T", bound=BaseModel)


class ClaimReview(BaseModel):
    """Research node output: the agent's OWN verdict on one claim — never
    ground truth, since the agent never sees ground truth."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    flagged: bool
    rationale: str = ""


class TradeProposal(BaseModel):
    """Trade Structuring node output."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    trade_alpha_horizon: Decimal
    tau_days: Decimal
    rationale: str = ""


class PMRecommendation(BaseModel):
    """Portfolio Manager node output: a recommendation, never an action —
    only Treasury (D1/D2/E) or, in A/B/C, this recommendation itself
    stands in as the system's final output since there is no Treasury
    node in those configs."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    recommend: bool
    rationale: str = ""


class TreasuryDecision(BaseModel):
    """Treasury node output (D1/D2/E only): a PROPOSED treasury action.
    Note this is still not "executed" — invariant 5 requires the dual
    BUSINESS_APPROVAL/CONTROL_CLEARANCE/POLICY_PASS gate in config E
    before anything here is treated as committed."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    treasury_action: TreasuryAction
    rationale: str = ""


class ICVote(BaseModel):
    """Investment Committee node output (config E only) — the
    BUSINESS_APPROVAL half of invariant 5's conjunction."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str
    business_approval: bool
    rationale: str = ""


_SCHEMA_BY_NAME: dict[str, type[BaseModel]] = {
    "claim_review": ClaimReview,
    "trade_proposal": TradeProposal,
    "pm_recommendation": PMRecommendation,
    "treasury_decision": TreasuryDecision,
    "ic_vote": ICVote,
}


def parse_structured_output(schema_name: str, parsed: dict[str, Any], *, node_name: str) -> BaseModel:
    """Strict validation of one node's raw parsed JSON. Raises
    SchemaConformanceError (never a bare pydantic ValidationError) so the
    caller's one-bounded-rework-cycle logic (agents/nodes.py) has a single
    exception type to catch, per the rework_reserve config's max_cycles=1
    contract."""
    model_cls = _SCHEMA_BY_NAME.get(schema_name)
    if model_cls is None:
        raise SchemaConformanceError(f"{node_name}: no registered schema named {schema_name!r}.")
    try:
        return model_cls.model_validate(parsed)
    except ValidationError as exc:
        raise SchemaConformanceError(f"{node_name}: output failed schema validation for {schema_name!r}: {exc}") from exc
