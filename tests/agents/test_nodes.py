from __future__ import annotations

import json
from decimal import Decimal

import pytest

from gcap.agents.llm import FakeLLMAdapter, LLMRequest, LLMResponse
from gcap.agents.nodes import ic_node, pm_node, research_node, trade_node, treasury_node
from gcap.agents.schemas import ClaimReview, PMRecommendation, TradeProposal
from gcap.agents.view import AgentVisibleClaim, AgentVisibleInstrument, AgentVisibleProposal
from gcap.config.registry import TokenBudget
from gcap.errors import SchemaConformanceError


def _proposal(**claim_overrides) -> AgentVisibleProposal:
    claim = AgentVisibleClaim(
        claim_id="c1",
        claim_type="calculation",
        metric="portfolio_return_pct",
        stated_value=Decimal("0.15"),
        **claim_overrides,
    )
    return AgentVisibleProposal(
        proposal_id="p1",
        instrument=AgentVisibleInstrument(asset_class="equity", fields={"ticker": "SYN0001"}),
        claims=(claim,),
        tau_days=Decimal("180"),
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=Decimal("0.03"),
    )


def test_research_node_reviews_every_claim():
    adapter = FakeLLMAdapter()
    proposal = _proposal()
    reviews, calls = research_node(
        adapter=adapter,
        proposal=proposal,
        token_budget=TokenBudget(),
        metadata_by_claim={"c1": {"tool_flagged": True}},
    )
    assert len(reviews) == 1
    assert reviews[0].claim_id == "c1"
    assert reviews[0].flagged is True
    assert len(calls) == 1
    assert calls[0].node_name == "research"
    assert calls[0].reworked is False


def test_trade_node_returns_typed_proposal():
    adapter = FakeLLMAdapter()
    proposal = _proposal()
    out, call = trade_node(adapter=adapter, proposal=proposal, token_budget=TokenBudget())
    assert out.proposal_id == "p1"
    assert call.node_name == "trade"


def test_pm_and_treasury_and_ic_nodes_return_typed_output():
    adapter = FakeLLMAdapter()
    proposal = _proposal()
    budget = TokenBudget()

    pm_out, _ = pm_node(adapter=adapter, proposal=proposal, token_budget=budget, metadata={"recommend": True})
    assert pm_out.recommend is True

    treasury_out, _ = treasury_node(
        adapter=adapter, proposal=proposal, token_budget=budget, metadata={"suggested_action": "REJECT"}
    )
    assert treasury_out.treasury_action.value == "REJECT"

    ic_out, _ = ic_node(adapter=adapter, proposal=proposal, token_budget=budget, metadata={"business_approval": False})
    assert ic_out.business_approval is False


class _ContextReadingAdapter:
    """Stands in for a real LLM that actually reads its prompt text (unlike
    FakeLLMAdapter's default_responder, which only ever consults
    request.metadata — see agents/llm.py's module docstring). Used to prove
    the P0-4 wiring causally matters: this adapter's decision depends on
    facts appearing in user_prompt TEXT alone, exactly the channel the real
    OpenAIAdapter actually transmits."""

    def __init__(self) -> None:
        self.seen_prompts: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_prompts.append(request.user_prompt)
        if request.response_schema_name == "pm_recommendation":
            # A "careful" PM declines whenever Research's text says a
            # claim was flagged -- decided purely by reading the prompt,
            # never by a metadata shortcut.
            recommend = "flagged=True" not in request.user_prompt
            parsed = {"proposal_id": "p1", "recommend": recommend, "rationale": "read from prompt text"}
        elif request.response_schema_name == "treasury_decision":
            # Treasury sizes down (APPROVE_WITH_LIMIT) whenever it can read
            # a high trade_alpha_horizon in Trade's own upstream summary,
            # and REJECTs outright whenever it can read that the Portfolio
            # Manager recommended against the deal (P1-02: only possible at
            # all once treasury_prompt actually carries PM's recommendation
            # as text — see agents.prompts.treasury_prompt / _pm_
            # recommendation_block).
            if "recommend=False" in request.user_prompt:
                action = "REJECT"
            elif "trade_alpha_horizon=0.09" in request.user_prompt:
                action = "APPROVE_WITH_LIMIT"
            else:
                action = "APPROVE"
            parsed = {"proposal_id": "p1", "treasury_action": action, "rationale": "read from prompt text"}
        else:
            raise AssertionError(f"unexpected schema in this test: {request.response_schema_name}")
        raw_text = json.dumps(parsed, sort_keys=True)
        return LLMResponse(
            raw_text=raw_text, parsed=parsed, tokens_in=10, tokens_out=5, latency_ms=1,
            model_snapshot="context-reading-test",
        )


def test_pm_decision_changes_when_upstream_research_finding_changes():
    """P0-4 regression test: PM's decision must be able to change purely
    because Research's upstream finding, carried as prompt TEXT (not
    metadata), changed -- proving the multi-agent workflow is a real
    decision chain over the wire, not six independent single-shot calls
    that happen to share a proposal_id."""
    adapter = _ContextReadingAdapter()
    proposal = _proposal()
    budget = TokenBudget()

    clean_review = [ClaimReview(claim_id="c1", flagged=False, rationale="looks fine")]
    flagged_review = [ClaimReview(claim_id="c1", flagged=True, rationale="tool caught a mismatch")]

    pm_when_clean, _ = pm_node(adapter=adapter, proposal=proposal, token_budget=budget, claim_reviews=clean_review)
    pm_when_flagged, _ = pm_node(adapter=adapter, proposal=proposal, token_budget=budget, claim_reviews=flagged_review)

    assert pm_when_clean.recommend is True
    assert pm_when_flagged.recommend is False


def test_treasury_decision_changes_when_upstream_trade_alpha_changes():
    """Same proof for Treasury reading Trade's upstream trade_alpha_horizon
    (only ever available to a real adapter as prompt text, via
    agents.prompts.treasury_prompt's trade summary block)."""
    adapter = _ContextReadingAdapter()
    proposal = _proposal()
    budget = TokenBudget()

    modest_trade = TradeProposal(proposal_id="p1", trade_alpha_horizon=Decimal("0.02"), tau_days=Decimal("180"))
    rich_trade = TradeProposal(proposal_id="p1", trade_alpha_horizon=Decimal("0.09"), tau_days=Decimal("180"))

    modest_decision, _ = treasury_node(adapter=adapter, proposal=proposal, token_budget=budget, trade=modest_trade)
    rich_decision, _ = treasury_node(adapter=adapter, proposal=proposal, token_budget=budget, trade=rich_trade)

    assert modest_decision.treasury_action.value == "APPROVE"
    assert rich_decision.treasury_action.value == "APPROVE_WITH_LIMIT"


def test_treasury_decision_changes_when_upstream_pm_recommendation_changes():
    """P1-02 (v2.0.0 code review) regression test: Treasury's decision must
    be able to change purely because PM's upstream recommendation, carried
    as prompt TEXT (not metadata — see agents/llm.py's module docstring on
    why metadata alone doesn't count), changed. Before this fix,
    treasury_prompt never mentioned PM's recommendation at all: a real
    model deciding Treasury had no way to be causally influenced by PM,
    regardless of what the graph's Python objects held — the same class of
    gap P0-4 already fixed for Research->Trade/PM and Trade->Treasury."""
    adapter = _ContextReadingAdapter()
    proposal = _proposal()
    budget = TokenBudget()

    pm_in_favour = PMRecommendation(proposal_id="p1", recommend=True, rationale="clean evidence")
    pm_against = PMRecommendation(proposal_id="p1", recommend=False, rationale="unsupported claim")
    modest_trade = TradeProposal(proposal_id="p1", trade_alpha_horizon=Decimal("0.02"), tau_days=Decimal("180"))

    decision_pm_in_favour, _ = treasury_node(
        adapter=adapter, proposal=proposal, token_budget=budget, trade=modest_trade, pm=pm_in_favour
    )
    decision_pm_against, _ = treasury_node(
        adapter=adapter, proposal=proposal, token_budget=budget, trade=modest_trade, pm=pm_against
    )

    assert decision_pm_in_favour.treasury_action.value == "APPROVE"
    assert decision_pm_against.treasury_action.value == "REJECT"


def test_node_rework_cycle_succeeds_after_one_retry():
    """First call returns a schema-invalid payload; the bounded rework
    cycle (max_cycles=1) retries once and the second call succeeds."""
    attempts = {"n": 0}

    class FlakyAdapter:
        def complete(self, request: LLMRequest) -> LLMResponse:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return LLMResponse(
                    raw_text="{}", parsed={"not_a_valid_field": True}, tokens_in=10, tokens_out=5,
                    latency_ms=1, model_snapshot="flaky-test",
                )
            return LLMResponse(
                raw_text="{}",
                parsed={"proposal_id": "p1", "trade_alpha_horizon": "0.02", "tau_days": "180", "rationale": ""},
                tokens_in=10, tokens_out=5, latency_ms=1, model_snapshot="flaky-test",
            )

    out, call = trade_node(adapter=FlakyAdapter(), proposal=_proposal(), token_budget=TokenBudget())
    assert out.proposal_id == "p1"
    assert call.reworked is True
    assert attempts["n"] == 2


def test_node_raises_after_exhausting_rework_cycle():
    class AlwaysBrokenAdapter:
        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                raw_text="{}", parsed={"garbage": True}, tokens_in=10, tokens_out=5, latency_ms=1,
                model_snapshot="broken-test",
            )

    with pytest.raises(SchemaConformanceError):
        trade_node(adapter=AlwaysBrokenAdapter(), proposal=_proposal(), token_budget=TokenBudget())


# ---------------------------------------------------------------------------
# P1-9 (code review): shape-valid but semantically wrong output -- a
# mismatched request-response identity, or a domain-rule violation -- must
# raise SchemaConformanceError (and therefore go through the same
# one-bounded-rework-cycle path a bare shape violation already uses), not
# sail through silently or crash unrelated code many calls later.
# ---------------------------------------------------------------------------
class _FixedResponder:
    """Always returns the same parsed payload, however many times it's
    called -- unlike the FlakyAdapter fixtures above, this never recovers
    on rework, so a test using it proves the mismatch is caught on EVERY
    attempt, not merely detected and silently accepted on retry."""

    def __init__(self, parsed: dict) -> None:
        self._parsed = parsed
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            raw_text=json.dumps(self._parsed), parsed=self._parsed, tokens_in=10, tokens_out=5,
            latency_ms=1, model_snapshot="fixed-test",
        )


def test_trade_node_rejects_a_response_for_the_wrong_proposal_id():
    adapter = _FixedResponder(
        {"proposal_id": "WRONG-ID", "trade_alpha_horizon": "0.02", "tau_days": "180", "rationale": ""}
    )
    with pytest.raises(SchemaConformanceError, match="proposal_id"):
        trade_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())
    assert adapter.call_count == 2  # one attempt, one rework retry, both wrong


def test_trade_node_rejects_a_tau_days_outside_the_contractual_band():
    """_proposal()'s own tau_days is 180; the default TenorRestructuringBounds
    is 0.5x-2x, so 180*3=540 is well outside [90, 360]."""
    adapter = _FixedResponder(
        {"proposal_id": "p1", "trade_alpha_horizon": "0.02", "tau_days": "540", "rationale": ""}
    )
    with pytest.raises(SchemaConformanceError, match="tau_days"):
        trade_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


def test_trade_node_accepts_a_genuine_restructuring_within_the_band():
    """A tenor extension well inside the band (180 -> 300 days, < 2x) is a
    legitimate trade-structuring decision, not a violation -- the check
    must not be so tight it blocks real restructuring."""
    adapter = _FixedResponder(
        {"proposal_id": "p1", "trade_alpha_horizon": "0.02", "tau_days": "300", "rationale": ""}
    )
    out, call = trade_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())
    assert out.tau_days == Decimal("300")
    assert call.reworked is False


def test_trade_node_rejects_trade_alpha_horizon_at_or_below_minus_one():
    adapter = _FixedResponder(
        {"proposal_id": "p1", "trade_alpha_horizon": "-1.5", "tau_days": "180", "rationale": ""}
    )
    with pytest.raises(SchemaConformanceError, match="trade_alpha_horizon"):
        trade_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


def test_pm_node_rejects_a_response_for_the_wrong_proposal_id():
    adapter = _FixedResponder({"proposal_id": "WRONG-ID", "recommend": True, "rationale": ""})
    with pytest.raises(SchemaConformanceError, match="proposal_id"):
        pm_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


def test_treasury_node_rejects_a_response_for_the_wrong_proposal_id():
    adapter = _FixedResponder({"proposal_id": "WRONG-ID", "treasury_action": "APPROVE", "rationale": ""})
    with pytest.raises(SchemaConformanceError, match="proposal_id"):
        treasury_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


def test_ic_node_rejects_a_response_for_the_wrong_proposal_id():
    adapter = _FixedResponder({"proposal_id": "WRONG-ID", "business_approval": True, "rationale": ""})
    with pytest.raises(SchemaConformanceError, match="proposal_id"):
        ic_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


def test_research_node_rejects_a_response_for_the_wrong_claim_id():
    adapter = _FixedResponder({"claim_id": "WRONG-CLAIM", "flagged": True, "rationale": ""})
    with pytest.raises(SchemaConformanceError, match="claim_id"):
        research_node(adapter=adapter, proposal=_proposal(), token_budget=TokenBudget())


# ---------------------------------------------------------------------------
# P2-10 (code review): a cache hit must not be recorded as if it were a
# fresh physical call. trade_node/research_node are the two node names
# SHAREABLE_NODE_NAMES (agents/cache.py) lets the D1/D2 shared cache
# actually replay, so a real SharedCallCache + a second, distinct
# (config_name) call against the identical request is what triggers a real
# cache hit -- not a mock.
# ---------------------------------------------------------------------------
def test_trade_node_cache_hit_does_not_double_count_tokens_but_is_recorded():
    from gcap.agents.cache import SharedCallCache

    adapter = FakeLLMAdapter()
    shared_cache = SharedCallCache()
    proposal = _proposal()
    budget = TokenBudget()

    _out_d1, call_d1 = trade_node(
        adapter=adapter, proposal=proposal, token_budget=budget,
        shared_cache=shared_cache, config_name="D1", case_id="case-1", replicate=0,
    )
    assert call_d1.cache_hit_count == 0
    assert call_d1.physical_call_count == 1
    assert call_d1.tokens_in > 0

    # D1/D2 is the one declared shareable pair (agents/cache.py's
    # is_d1_d2_pair) -- an identical request under D2 replays D1's cached
    # response rather than calling the adapter again.
    _out_d2, call_d2 = trade_node(
        adapter=adapter, proposal=proposal, token_budget=budget,
        shared_cache=shared_cache, config_name="D2", case_id="case-1", replicate=0,
    )
    assert call_d2.cache_hit_count == 1
    assert call_d2.physical_call_count == 0
    # The whole point of the fix: a cache hit must not add spend on top of
    # the physical call that already paid for it.
    assert call_d2.tokens_in == 0
    assert call_d2.tokens_out == 0
    assert call_d2.latency_ms == 0
