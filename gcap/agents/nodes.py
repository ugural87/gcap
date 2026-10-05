"""Node functions: each wraps one LLM call (or, for research, one call per
claim) plus the one-bounded-rework-cycle-on-schema-failure policy
(``ReworkReserveConfig.max_cycles=1``, Section 9.9). A node function never
touches core/ directly and never decides whether an action executes — it
returns a typed *proposal*; the graph (M2c) is the only thing that
consults core/ and turns proposals into decisions.

``metadata`` on every call is caller-supplied and opaque to this module —
see agents/llm.py's module docstring: only FakeLLMAdapter ever reads it,
and it exists so a test harness can simulate realistic model behavior
deterministically without needing an actual model. Passing it through
unexamined keeps this module identical whether the underlying adapter is
fake or real.

P1-9 (code review): schema validation alone (agents/schemas.py) checks
only *shape* — that a response has a ``proposal_id`` string, a
``tau_days`` Decimal, etc. — never that the returned identity actually
matches what was asked, or that an economic value is in a domain-sensible
range. A model that answers with the WRONG proposal's id, or a wildly
wrong tenor, used to sail through unnoticed: the id would silently
mislabel a downstream write, and ``TradeProposal.tau_days`` fed straight
into ``core.ftp.compute_ftp`` (graph/pipeline.py) unchecked before this
fix — invariant 1 says an LLM never computes, but an unchecked LLM number
driving a deterministic engine's input is the same failure by another
route. ``_call_with_rework``'s new ``extra_check`` hook lets each node
raise ``SchemaConformanceError`` for a request-response identity mismatch
or a domain-rule violation, exactly as it already does for a bare shape
violation — routing both through the SAME one-bounded-rework-cycle path
(Section 9.9) rather than letting a bad identity/value quietly propagate
or crash unrelated code many calls later.

P2-10 (code review): usage/cost accounting used to conflate three
different things into one number. (1) A rework attempt that failed
schema/domain validation still made a real physical call — costing real
tokens and latency — but its usage was silently dropped the moment a
later attempt succeeded, so a RunRecord's tokens/latency undercounted
whenever a node reworked. (2) A cache hit (D1/D2's shared Research/Trade
calls, agents/cache.py) replays a response some EARLIER call already
paid for; recording that replay's tokens/latency again, as if this
RunRecord made a fresh physical call, double-(or N-times-)counted the
same physical cost across every config that ever hit the cache. (3)
``RunRecord.prompt_hashes`` was one hash per node_name, so a multi-claim
node (research: one call per claim) silently lost every claim's hash but
the last. All three are fixed together: ``NodeCall`` now carries the full
per-attempt event list (``CallEvent``, below) rather than one flattened
number, with ``tokens_in``/``tokens_out``/``latency_ms`` properties that
sum ONLY the physical (non-cache-hit) events — so every rework attempt's
real cost is counted once, and a cache hit contributes zero, never a
second (duplicate) charge; ``RunRecord.prompt_hashes`` is now
``dict[str, list[str]]``, one entry per physical+cached call in order,
so no claim's hash is lost; and a new ``RunRecord.cache_hits`` field
(same per-node_name-dict shape as tokens_in/out) makes cache reuse
visible in the store itself, not just in ``SharedCallCache``'s own
process-local hit/miss counters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from gcap.agents.cache import SharedCallCache
from gcap.agents.llm import LLMClient, LLMRequest, LLMResponse, enforce_token_budget
from gcap.agents.prompts import (
    ic_prompt,
    pm_prompt,
    research_prompt,
    single_agent_prompt,
    trade_prompt,
    treasury_prompt,
)
from gcap.agents.schemas import (
    ClaimReview,
    ICVote,
    PMRecommendation,
    TradeProposal,
    TreasuryDecision,
    parse_structured_output,
)
from gcap.agents.view import AgentVisibleProposal
from gcap.config.flags import ConfigName
from gcap.config.registry import TenorRestructuringBounds, TokenBudget
from gcap.errors import SchemaConformanceError


@dataclass(frozen=True)
class CallEvent:
    """P2-10: one attempt within a node's (possibly rework-retried) call
    sequence. ``cache_hit=True`` means this attempt replayed a response an
    EARLIER call already paid for (agents/cache.py's SharedCallCache) —
    zero physical cost to THIS run, however plausible its tokens/latency
    numbers look; ``cache_hit=False`` means the adapter was actually
    invoked and these numbers are real spend. Kept as its own event
    (never pre-summed away) so a rework attempt's cost is never silently
    dropped and a cache hit's cost is never silently duplicated."""

    cache_hit: bool
    tokens_in: int
    tokens_out: int
    latency_ms: int


@dataclass(frozen=True)
class NodeCall:
    """One LLM round-trip's bookkeeping — feeds RunRecord.tokens_in/out,
    latency_ms, cache_hits, prompt_hashes, retries, and the
    tool_results-adjacent audit trail. ``tokens_in``/``tokens_out``/
    ``latency_ms``/``cache_hit_count``/``physical_call_count`` are derived
    from ``events`` (P2-10) rather than stored redundantly, so they can
    never drift out of sync with it."""

    node_name: str
    prompt_hash: str
    reworked: bool
    events: tuple[CallEvent, ...]

    @property
    def tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.events if not e.cache_hit)

    @property
    def tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.events if not e.cache_hit)

    @property
    def latency_ms(self) -> int:
        return sum(e.latency_ms for e in self.events if not e.cache_hit)

    @property
    def physical_call_count(self) -> int:
        return sum(1 for e in self.events if not e.cache_hit)

    @property
    def cache_hit_count(self) -> int:
        return sum(1 for e in self.events if e.cache_hit)


def _make_call_fn(
    adapter: LLMClient,
    *,
    shared_cache: SharedCallCache | None,
    config_name: ConfigName | None,
    case_id: str | None,
    replicate: int | None,
    manifest_hash: str = "",
) -> Callable[[LLMRequest], tuple[LLMResponse, bool]]:
    """Routes a node's calls through the D1/D2 shared cache (agents/cache.py)
    when one is supplied, else calls the adapter directly. Only
    research_node/trade_node ever pass a non-None ``shared_cache`` — those
    are the only two node names in ``SHAREABLE_NODE_NAMES``; passing one
    for any other node would raise ``CacheIsolationError`` on the first
    D1/D2 cross-config hit, which is exactly the point (see that module's
    docstring: an accidental share must raise, never silently serve a
    wrong answer).

    Returns ``(response, was_cache_hit)`` uniformly (P2-10) — even the
    no-cache path wraps ``adapter.complete`` to always report
    ``was_cache_hit=False`` — so ``_call_with_rework`` never has to guess
    whether a given attempt was a real physical call.

    ``manifest_hash`` (P1-6) is threaded straight through to
    ``SharedCallCache.get_or_call``, which folds it — plus the adapter's
    own ``model_snapshot`` — into the cache key, so a cache built under one
    manifest/model can never silently answer a lookup made under another
    (see agents/cache.py's module docstring)."""
    if shared_cache is None:
        return lambda request: (adapter.complete(request), False)
    assert config_name is not None and case_id is not None and replicate is not None
    model_snapshot = getattr(adapter, "model_snapshot", "unknown-model-snapshot")

    def call_fn(request: LLMRequest) -> tuple[LLMResponse, bool]:
        return shared_cache.get_or_call(
            config=config_name,
            case_id=case_id,
            replicate=replicate,
            request=request,
            call_fn=adapter.complete,
            model_snapshot=model_snapshot,
            manifest_hash=manifest_hash,
        )

    return call_fn


def _call_with_rework(
    adapter: LLMClient,
    request: LLMRequest,
    *,
    max_cycles: int = 1,
    call_fn: Callable[[LLMRequest], tuple[LLMResponse, bool]] | None = None,
    extra_check: Callable[[BaseModel], None] | None = None,
) -> tuple[LLMResponse, NodeCall]:
    """``extra_check`` (P1-9): called with the already-schema-valid parsed
    model; raising ``SchemaConformanceError`` from it (a request-response
    identity mismatch, a tenor outside its contractual band, ...) is
    treated exactly like a shape violation — caught here, counted as a
    rework, and retried up to ``max_cycles`` before propagating. Each
    node below supplies its own ``extra_check`` closure since only the
    node knows what it asked for (the requested proposal/claim id, the
    proposal's own tau_days, ...)."""
    invoke = call_fn if call_fn is not None else (lambda req: (adapter.complete(req), False))
    events: list[CallEvent] = []
    last_error: SchemaConformanceError | None = None
    for _attempt in range(max_cycles + 1):
        response, was_cache_hit = invoke(request)
        # P2-10: every attempt's own cost is recorded as its own event
        # BEFORE validation runs, so a rework attempt that turns out to be
        # invalid still keeps its real physical tokens/latency (previously
        # discarded entirely the moment a later attempt succeeded) —
        # unless it was itself a cache hit, in which case it cost nothing
        # physically and is recorded as such.
        events.append(
            CallEvent(
                cache_hit=was_cache_hit,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                latency_ms=response.latency_ms,
            )
        )
        try:
            parsed_model = parse_structured_output(request.response_schema_name, response.parsed, node_name=request.node_name)
            if extra_check is not None:
                extra_check(parsed_model)
        except SchemaConformanceError as exc:
            last_error = exc
            continue
        call = NodeCall(
            node_name=request.node_name,
            prompt_hash=request.prompt_hash(),
            reworked=len(events) > 1,
            events=tuple(events),
        )
        return response, call
    assert last_error is not None
    raise last_error


def _proposal_id_check(node_name: str, expected_proposal_id: str) -> Callable[[BaseModel], None]:
    """P1-9: every node schema below carries its own ``proposal_id`` field
    (see agents/schemas.py) precisely so this check is possible — a
    response whose ``proposal_id`` does not echo back the one this node
    was actually asked about must never be silently attributed to the
    right proposal downstream."""

    def _check(model: BaseModel) -> None:
        got = getattr(model, "proposal_id", None)
        if got != expected_proposal_id:
            raise SchemaConformanceError(
                f"{node_name}: response proposal_id {got!r} does not match the requested "
                f"proposal_id {expected_proposal_id!r}."
            )

    return _check


def _claim_id_check(claim_id: str) -> Callable[[BaseModel], None]:
    def _check(model: BaseModel) -> None:
        got = getattr(model, "claim_id", None)
        if got != claim_id:
            raise SchemaConformanceError(
                f"research: response claim_id {got!r} does not match the requested claim_id {claim_id!r}."
            )

    return _check


def _trade_domain_check(
    proposal: AgentVisibleProposal, *, tenor_bounds: TenorRestructuringBounds
) -> Callable[[BaseModel], None]:
    """P1-9: beyond the shared identity check, TradeProposal carries two
    economic values that feed straight into core.ftp.compute_ftp
    (graph/pipeline.py) with no further validation — tau_days and
    trade_alpha_horizon. tau_days must stay within the contractually
    acceptable band around the proposal's own tenor (see
    TenorRestructuringBounds's docstring: a genuine restructuring is fine,
    an off-by-orders-of-magnitude or near-zero value is not);
    trade_alpha_horizon must clear the same "-100%-or-worse is not a real
    holding-period return" floor core.ftp.FTPInputs itself already
    enforces (Section 4.1) — checked here too so a violation is a
    node-level rework, not an FTPValidationError several calls later deep
    in the graph."""
    lo = proposal.tau_days * tenor_bounds.min_multiple
    hi = proposal.tau_days * tenor_bounds.max_multiple
    id_check = _proposal_id_check("trade", proposal.proposal_id)

    def _check(model: BaseModel) -> None:
        id_check(model)
        assert isinstance(model, TradeProposal)
        if not (lo <= model.tau_days <= hi):
            raise SchemaConformanceError(
                f"trade: response tau_days={model.tau_days} is outside the contractually acceptable "
                f"band [{lo}, {hi}] around proposal.tau_days={proposal.tau_days} "
                f"(TenorRestructuringBounds: {tenor_bounds.min_multiple}x-{tenor_bounds.max_multiple}x)."
            )
        if model.trade_alpha_horizon <= Decimal("-1"):
            raise SchemaConformanceError(
                f"trade: response trade_alpha_horizon={model.trade_alpha_horizon} <= -1: a holding-period "
                "return cannot be -100% or worse (Section 4.1 validation rule, mirrored from core.ftp)."
            )

    return _check


def research_node(
    *,
    adapter: LLMClient,
    proposal: AgentVisibleProposal,
    token_budget: TokenBudget,
    metadata_by_claim: dict[str, dict[str, Any]] | None = None,
    shared_cache: SharedCallCache | None = None,
    config_name: ConfigName | None = None,
    case_id: str | None = None,
    replicate: int | None = None,
    manifest_hash: str = "",
) -> tuple[list[ClaimReview], list[NodeCall]]:
    metadata_by_claim = metadata_by_claim or {}
    call_fn = _make_call_fn(
        adapter, shared_cache=shared_cache, config_name=config_name, case_id=case_id, replicate=replicate,
        manifest_hash=manifest_hash,
    )
    reviews: list[ClaimReview] = []
    calls: list[NodeCall] = []
    for claim in proposal.claims:
        system, user = research_prompt(proposal, claim)
        request = LLMRequest(
            node_name="research",
            system_prompt=system,
            user_prompt=user,
            response_schema_name="claim_review",
            max_output_tokens=token_budget.output_cap_lightweight,
            metadata={"claim_id": claim.claim_id, **metadata_by_claim.get(claim.claim_id, {})},
        )
        enforce_token_budget(request, input_cap=token_budget.input_cap_evidence, output_cap=token_budget.output_cap_lightweight)
        response, call = _call_with_rework(
            adapter, request, call_fn=call_fn, extra_check=_claim_id_check(claim.claim_id)
        )
        review = parse_structured_output("claim_review", response.parsed, node_name="research")
        assert isinstance(review, ClaimReview)
        reviews.append(review)
        calls.append(call)
    return reviews, calls


def single_agent_node(
    *, adapter: LLMClient, proposal: AgentVisibleProposal, token_budget: TokenBudget, metadata: dict[str, Any] | None = None
) -> tuple[TreasuryDecision, NodeCall]:
    """Config A only: one generalist LLM call standing in for the entire
    Research/Trade/PM/Treasury decomposition later configs use. Reuses the
    ``treasury_decision`` schema (the same six-action vocabulary) so A's
    outcome is directly comparable to every later config's final decision
    — see agents/prompts.py's single_agent_prompt docstring."""
    system, user = single_agent_prompt(proposal)
    request = LLMRequest(
        node_name="single_agent",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="treasury_decision",
        max_output_tokens=token_budget.output_cap_critical,
        metadata={"proposal_id": proposal.proposal_id, **(metadata or {})},
    )
    enforce_token_budget(request, input_cap=token_budget.input_cap_total, output_cap=token_budget.output_cap_critical)
    response, call = _call_with_rework(
        adapter, request, extra_check=_proposal_id_check("single_agent", proposal.proposal_id)
    )
    out = parse_structured_output("treasury_decision", response.parsed, node_name="single_agent")
    assert isinstance(out, TreasuryDecision)
    return out, call


def trade_node(
    *,
    adapter: LLMClient,
    proposal: AgentVisibleProposal,
    token_budget: TokenBudget,
    metadata: dict[str, Any] | None = None,
    shared_cache: SharedCallCache | None = None,
    config_name: ConfigName | None = None,
    case_id: str | None = None,
    replicate: int | None = None,
    claim_reviews: list[ClaimReview] | None = None,
    manifest_hash: str = "",
    tenor_bounds: TenorRestructuringBounds | None = None,
) -> tuple[TradeProposal, NodeCall]:
    tenor_bounds = tenor_bounds or TenorRestructuringBounds()
    call_fn = _make_call_fn(
        adapter, shared_cache=shared_cache, config_name=config_name, case_id=case_id, replicate=replicate,
        manifest_hash=manifest_hash,
    )
    system, user = trade_prompt(proposal, claim_reviews)
    request = LLMRequest(
        node_name="trade",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="trade_proposal",
        max_output_tokens=token_budget.output_cap_critical,
        # P1-9: FakeLLMAdapter.default_responder's own "180"-day fallback
        # for tau_days (agents/llm.py) is a bare stand-in with no notion of
        # THIS proposal's actual contractual tenor -- fine as long as
        # nothing validated the value, but now that trade_node's own
        # domain check enforces a band around proposal.tau_days, a test (or
        # a real case) whose tenor genuinely differs from 180 days needs a
        # correct default here, not the fake adapter's generic one. A
        # caller-supplied hint (``metadata``, test-only) still overrides it
        # via the dict spread below.
        metadata={"proposal_id": proposal.proposal_id, "tau_days": str(proposal.tau_days), **(metadata or {})},
    )
    enforce_token_budget(request, input_cap=token_budget.input_cap_total, output_cap=token_budget.output_cap_critical)
    response, call = _call_with_rework(
        adapter, request, call_fn=call_fn, extra_check=_trade_domain_check(proposal, tenor_bounds=tenor_bounds)
    )
    out = parse_structured_output("trade_proposal", response.parsed, node_name="trade")
    assert isinstance(out, TradeProposal)
    return out, call


def pm_node(
    *,
    adapter: LLMClient,
    proposal: AgentVisibleProposal,
    token_budget: TokenBudget,
    metadata: dict[str, Any] | None = None,
    claim_reviews: list[ClaimReview] | None = None,
    trade: TradeProposal | None = None,
) -> tuple[PMRecommendation, NodeCall]:
    system, user = pm_prompt(proposal, claim_reviews, trade)
    request = LLMRequest(
        node_name="pm",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="pm_recommendation",
        max_output_tokens=token_budget.output_cap_lightweight,
        metadata={"proposal_id": proposal.proposal_id, **(metadata or {})},
    )
    enforce_token_budget(
        request, input_cap=token_budget.input_cap_state_policy_instructions, output_cap=token_budget.output_cap_lightweight
    )
    response, call = _call_with_rework(adapter, request, extra_check=_proposal_id_check("pm", proposal.proposal_id))
    out = parse_structured_output("pm_recommendation", response.parsed, node_name="pm")
    assert isinstance(out, PMRecommendation)
    return out, call


def treasury_node(
    *,
    adapter: LLMClient,
    proposal: AgentVisibleProposal,
    token_budget: TokenBudget,
    metadata: dict[str, Any] | None = None,
    trade: TradeProposal | None = None,
    pm: PMRecommendation | None = None,
) -> tuple[TreasuryDecision, NodeCall]:
    # P1-02 (v2.0.0 code review): pm is new -- see treasury_prompt's
    # docstring. Deliberately NOT read by FakeLLMAdapter.default_responder
    # (metadata's "suggested_action" hint remains the sole deterministic-
    # adapter lever, unchanged by this fix — see agents/llm.py's module
    # docstring on why metadata and prompt text are two separate channels);
    # this only ever matters to an adapter that reads user_prompt text,
    # i.e. a real model.
    system, user = treasury_prompt(proposal, trade, pm)
    request = LLMRequest(
        node_name="treasury",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="treasury_decision",
        max_output_tokens=token_budget.output_cap_critical,
        metadata={"proposal_id": proposal.proposal_id, **(metadata or {})},
    )
    enforce_token_budget(request, input_cap=token_budget.input_cap_total, output_cap=token_budget.output_cap_critical)
    response, call = _call_with_rework(
        adapter, request, extra_check=_proposal_id_check("treasury", proposal.proposal_id)
    )
    out = parse_structured_output("treasury_decision", response.parsed, node_name="treasury")
    assert isinstance(out, TreasuryDecision)
    return out, call


def ic_node(
    *,
    adapter: LLMClient,
    proposal: AgentVisibleProposal,
    token_budget: TokenBudget,
    metadata: dict[str, Any] | None = None,
    treasury: TreasuryDecision | None = None,
    pricing_margin: Decimal | None = None,
    pricing_in_band: bool | None = None,
    ledger_ok: bool = True,
    ledger_violations: tuple[str, ...] = (),
) -> tuple[ICVote, NodeCall]:
    system, user = ic_prompt(
        proposal,
        treasury,
        pricing_margin=pricing_margin,
        pricing_in_band=pricing_in_band,
        ledger_ok=ledger_ok,
        ledger_violations=ledger_violations,
    )
    request = LLMRequest(
        node_name="ic",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="ic_vote",
        max_output_tokens=token_budget.output_cap_lightweight,
        metadata={"proposal_id": proposal.proposal_id, **(metadata or {})},
    )
    enforce_token_budget(
        request, input_cap=token_budget.input_cap_state_policy_instructions, output_cap=token_budget.output_cap_lightweight
    )
    response, call = _call_with_rework(adapter, request, extra_check=_proposal_id_check("ic", proposal.proposal_id))
    out = parse_structured_output("ic_vote", response.parsed, node_name="ic")
    assert isinstance(out, ICVote)
    return out, call
