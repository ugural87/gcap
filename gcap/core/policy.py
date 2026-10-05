"""Deterministic policy engine (Section 4.3 / proposal Section 6.2).

Each rule is a record with id, owner, scope, severity (hard or soft),
effective interval, version, required evidence, and a predicate id.
Predicates are Python callables registered by id; text never becomes
enforceable without a registered predicate (invariant guard: an LLM can
draft candidate rule text, but nothing it writes is enforceable until a
human wires a predicate_id to a function registered here).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gcap.errors import PolicyPredicateNotRegisteredError

Severity = Literal["hard", "soft"]
PolicyVerdict = Literal["PASS", "HARD_FAIL", "ESCALATE"]

# A predicate returns True iff the rule is BREACHED for this context.
PredicateFn = Callable[["PolicyContext"], bool]

_PREDICATE_REGISTRY: dict[str, PredicateFn] = {}


def register_predicate(predicate_id: str) -> Callable[[PredicateFn], PredicateFn]:
    """Decorator registering a predicate under a stable id. Re-registering
    the same id with a different function is allowed only to support test
    isolation; production predicate ids are expected to be registered
    exactly once at import time."""

    def decorator(fn: PredicateFn) -> PredicateFn:
        _PREDICATE_REGISTRY[predicate_id] = fn
        return fn

    return decorator


def get_predicate(predicate_id: str) -> PredicateFn:
    if predicate_id not in _PREDICATE_REGISTRY:
        raise PolicyPredicateNotRegisteredError(
            f"No predicate registered for id {predicate_id!r}. Rule text can never "
            "become enforceable without an explicitly registered predicate."
        )
    return _PREDICATE_REGISTRY[predicate_id]


def registered_predicate_ids() -> tuple[str, ...]:
    return tuple(sorted(_PREDICATE_REGISTRY))


class PolicyContext(BaseModel):
    """A generic, typed bag of values predicates read from. Concrete
    predicates document which keys they require; the policy engine itself
    stays agnostic of their meaning."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    values: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class PolicyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    owner: str
    scope: str
    severity: Severity
    effective_from: datetime
    effective_to: datetime | None = None
    version: str
    required_evidence: tuple[str, ...] = ()
    predicate_id: str

    def is_effective(self, at: datetime) -> bool:
        if at < self.effective_from:
            return False
        return self.effective_to is None or at < self.effective_to


class PolicyEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: PolicyVerdict
    triggering_rule_ids: tuple[str, ...]
    policy_version: str
    evaluated_rule_ids: tuple[str, ...]


def evaluate(
    rules: Sequence[PolicyRule],
    context: PolicyContext,
    at: datetime,
    policy_version: str,
) -> PolicyEvaluationResult:
    """PASS, HARD_FAIL or ESCALATE, with the list of triggering rule ids and
    the policy version. Hard failures take precedence over soft escalations,
    since execution requires the conjunction BUSINESS_APPROVAL AND
    CONTROL_CLEARANCE AND POLICY_PASS — a single hard breach must block
    regardless of how many soft corridors also triggered.
    """
    hard_triggers: list[str] = []
    soft_triggers: list[str] = []
    evaluated: list[str] = []

    for rule in rules:
        if not rule.is_effective(at):
            continue
        evaluated.append(rule.rule_id)
        predicate = get_predicate(rule.predicate_id)
        breached = predicate(context)
        if breached:
            (hard_triggers if rule.severity == "hard" else soft_triggers).append(rule.rule_id)

    if hard_triggers:
        verdict: PolicyVerdict = "HARD_FAIL"
        triggers = tuple(hard_triggers)
    elif soft_triggers:
        verdict = "ESCALATE"
        triggers = tuple(soft_triggers)
    else:
        verdict = "PASS"
        triggers = ()

    return PolicyEvaluationResult(
        verdict=verdict,
        triggering_rule_ids=triggers,
        policy_version=policy_version,
        evaluated_rule_ids=tuple(evaluated),
    )


# ---------------------------------------------------------------------------
# Reference predicates. Real policy content is data (loaded from configs/),
# these are the small set of generically useful, reusable predicates that
# ship with the engine plus the ones needed by the M1 golden oracle tests.
# ---------------------------------------------------------------------------
@register_predicate("hard.mandate_asset_class_breach")
def _hard_mandate_asset_class_breach(ctx: PolicyContext) -> bool:
    """Breached if the proposal's instrument asset class is outside the
    mandate's permitted set."""
    permitted = ctx.get("mandate_permitted_asset_classes", set())
    asset_class = ctx.get("instrument_asset_class")
    return asset_class is not None and asset_class not in permitted


@register_predicate("hard.unauthorised_exception")
def _hard_unauthorised_exception(ctx: PolicyContext) -> bool:
    """Breached if an exception to a hard limit was applied without a
    recorded authorising approval id."""
    exception_applied = bool(ctx.get("exception_applied", False))
    authorising_approval_id = ctx.get("authorising_approval_id")
    return exception_applied and not authorising_approval_id


@register_predicate("hard.stale_state_gate")
def _hard_stale_state(ctx: PolicyContext) -> bool:
    return bool(ctx.get("state_is_stale", False))


@register_predicate("hard.unsupported_material_claim")
def _hard_unsupported_material_claim(ctx: PolicyContext) -> bool:
    """Breached if the agent graph's OWN research review (never oracle
    ground truth — this predicate is evaluated only from
    ``unsupported_claim_present``, a flag the graph derives from Research
    node ClaimReview outputs) flagged a material claim as unsupported.
    Wiring the policy engine to Research's own findings (config E only)
    closes the loop between evidence review and governance, so a config-E
    run's ep_governance_failure detection is not limited to
    mandate/exception/staleness checks alone."""
    return bool(ctx.get("unsupported_claim_present", False))


@register_predicate("soft.concentration_corridor")
def _soft_concentration_corridor(ctx: PolicyContext) -> bool:
    used = ctx.get("concentration_used_pct")
    corridor = ctx.get("concentration_corridor_pct")
    if used is None or corridor is None:
        return False
    return bool(used > corridor)


@register_predicate("soft.liquidity_buffer_corridor")
def _soft_liquidity_buffer(ctx: PolicyContext) -> bool:
    buffer_pct = ctx.get("liquidity_buffer_pct")
    target_pct = ctx.get("liquidity_buffer_target_pct")
    if buffer_pct is None or target_pct is None:
        return False
    return bool(buffer_pct < target_pct)
