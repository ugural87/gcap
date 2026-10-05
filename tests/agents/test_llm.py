from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from gcap.agents.llm import (
    FakeLLMAdapter,
    LLMRequest,
    OpenAIAdapter,
    enforce_token_budget,
    resolve_model_identity,
)
from gcap.agents.prompts import ic_prompt, pm_prompt
from gcap.agents.schemas import ClaimReview, TradeProposal, TreasuryDecision
from gcap.agents.view import AgentVisibleInstrument, AgentVisibleProposal, ResourceFootprint
from gcap.errors import ModelIdentityMismatchError, SchemaConformanceError, TokenBudgetExceededError
from gcap.store.run_record import TreasuryAction


def test_resolve_model_identity_requires_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL_PRIMARY", raising=False)
    with pytest.raises(ModelIdentityMismatchError):
        resolve_model_identity("primary")


def test_resolve_model_identity_separates_product_name_from_real_id(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    identity = resolve_model_identity("primary")
    assert identity.real_model_id == "gpt-5.6-luna-2026-09-01"
    assert "Luna" in identity.product_name
    assert identity.product_name != identity.real_model_id


def test_fake_adapter_is_deterministic_for_identical_requests():
    adapter = FakeLLMAdapter()
    request = LLMRequest(
        node_name="research",
        system_prompt="sys",
        user_prompt="user",
        response_schema_name="claim_review",
        max_output_tokens=250,
        metadata={"claim_id": "c1", "tool_flagged": True},
    )
    r1 = adapter.complete(request)
    r2 = adapter.complete(request)
    assert r1.parsed == r2.parsed
    assert r1.parsed["flagged"] is True


def test_fake_adapter_heeds_tool_flag_over_seeded_default():
    adapter = FakeLLMAdapter()
    flagged_request = LLMRequest(
        node_name="research",
        system_prompt="s",
        user_prompt="u",
        response_schema_name="claim_review",
        max_output_tokens=250,
        metadata={"claim_id": "c1", "tool_flagged": True},
    )
    not_flagged_request = LLMRequest(
        node_name="research",
        system_prompt="s",
        user_prompt="u2",
        response_schema_name="claim_review",
        max_output_tokens=250,
        metadata={"claim_id": "c1", "tool_flagged": False},
    )
    assert adapter.complete(flagged_request).parsed["flagged"] is True
    assert adapter.complete(not_flagged_request).parsed["flagged"] is False


def test_enforce_token_budget_rejects_oversized_input():
    request = LLMRequest(
        node_name="trade",
        system_prompt="x" * 100_000,
        user_prompt="y",
        response_schema_name="trade_proposal",
        max_output_tokens=100,
        metadata={},
    )
    with pytest.raises(TokenBudgetExceededError):
        enforce_token_budget(request, input_cap=6000, output_cap=700)


def test_enforce_token_budget_rejects_oversized_output_request():
    request = LLMRequest(
        node_name="trade",
        system_prompt="short",
        user_prompt="short",
        response_schema_name="trade_proposal",
        max_output_tokens=9999,
        metadata={},
    )
    with pytest.raises(TokenBudgetExceededError):
        enforce_token_budget(request, input_cap=6000, output_cap=700)


def test_openai_adapter_refuses_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    identity = resolve_model_identity("primary")
    adapter = OpenAIAdapter(identity)
    request = LLMRequest(
        node_name="trade",
        system_prompt="s",
        user_prompt="u",
        response_schema_name="trade_proposal",
        max_output_tokens=100,
        metadata={},
    )
    with pytest.raises(ModelIdentityMismatchError):
        adapter.complete(request)


def _sample_proposal() -> AgentVisibleProposal:
    return AgentVisibleProposal(
        proposal_id="p-1",
        instrument=AgentVisibleInstrument(asset_class="equity", fields={}),
        claims=(),
        tau_days=Decimal("180"),
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=Decimal("0.03"),
        resource_footprint=ResourceFootprint(),
    )


def test_openai_adapter_request_body_carries_upstream_agent_context(monkeypatch):
    """P0-4 mock-HTTP-body test: builds a PM prompt the way graph/pipeline.py
    actually does (Research findings + Trade's proposal folded into the
    prompt text by agents.prompts.pm_prompt), sends it through the REAL
    OpenAIAdapter, and asserts the upstream facts are present in the literal
    JSON body that would go over the wire — not merely in
    LLMRequest.metadata, which OpenAIAdapter never transmits (see this
    module's docstring).

    P2-01 (v2.1.0 third code review round): this test (and
    test_openai_adapter_request_body_carries_ic_controls_context below) used
    to intercept the request via ``monkeypatch.setattr(httpx.Client, "post",
    fake_post)`` — a patch of the CLASS's ``post`` method, reaching into
    EVERY ``httpx.Client`` instance in the process for the duration of the
    test, not just this adapter's own client, and bypassing httpx's real
    request-building/transport pipeline entirely (a hand-written
    ``fake_post(self, path, *, headers, json)`` signature that could
    silently drift from httpx's own calling convention across a version
    upgrade, with nothing to catch it). Rewritten to inject an
    ``httpx.MockTransport``-backed client instead — the SAME hermetic
    pattern ``test_openai_adapter_client_is_injectable_via_a_mock_transport``
    already established: zero real sockets touched, zero process-wide
    monkeypatching, and the full real httpx request/response pipeline
    (URL joining, header/body construction, the transport layer any ambient
    proxy configuration would actually flow through) genuinely exercised
    against a MockTransport handler instead of skipped."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    identity = resolve_model_identity("primary")

    proposal = _sample_proposal()
    claim_reviews = [ClaimReview(claim_id="c-flagged-claim", flagged=True, rationale="tool caught a mismatch")]
    trade = TradeProposal(proposal_id="p-1", trade_alpha_horizon=Decimal("0.045"), tau_days=Decimal("180"))
    system, user = pm_prompt(proposal, claim_reviews, trade)

    request = LLMRequest(
        node_name="pm",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="pm_recommendation",
        max_output_tokens=300,
        metadata={"proposal_id": "p-1"},  # deliberately NOT carrying the upstream facts under test
    )

    captured: dict[str, Any] = {}

    def handler(mock_request: httpx.Request) -> httpx.Response:
        captured["path"] = mock_request.url.path
        captured["json"] = json.loads(mock_request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"proposal_id": "p-1", "recommend": true, "rationale": "ok"}'}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    adapter = OpenAIAdapter(identity, client=mock_client)

    response = adapter.complete(request)
    assert response.parsed["recommend"] is True

    sent_messages = captured["json"]["messages"]
    assert sent_messages == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    sent_user_text = sent_messages[1]["content"]
    # The upstream facts must be literal text in the wire payload, not just
    # available via request.metadata (which this test's LLMRequest above
    # deliberately does NOT carry them in).
    assert "c-flagged-claim" in sent_user_text
    assert "flagged=True" in sent_user_text
    assert "0.045" in sent_user_text  # Trade's proposed trade_alpha_horizon
    assert "metadata" not in json.dumps(captured["json"])  # never sent over the wire at all

    adapter.close()
    mock_client.close()


def test_openai_adapter_request_body_carries_ic_controls_context(monkeypatch):
    """Same mock-HTTP-body proof for IC: Treasury's action plus the
    deterministic FTP/ledger findings must be literal text in the wire
    payload for the ic prompt, matching what graph/pipeline.py now passes
    into ic_node after computing them.

    P2-01 (v2.1.0 third code review round): see the sibling test above for
    why this is now an ``httpx.MockTransport``-backed injected client
    instead of a global ``monkeypatch.setattr(httpx.Client, "post", ...)``."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    identity = resolve_model_identity("primary")

    proposal = _sample_proposal()
    treasury = TreasuryDecision(proposal_id="p-1", treasury_action=TreasuryAction.APPROVE_WITH_HEDGE)
    system, user = ic_prompt(
        proposal,
        treasury,
        pricing_margin=Decimal("0.0012"),
        pricing_in_band=False,
        ledger_ok=False,
        ledger_violations=("capital", "liquidity"),
    )
    request = LLMRequest(
        node_name="ic",
        system_prompt=system,
        user_prompt=user,
        response_schema_name="ic_vote",
        max_output_tokens=300,
        metadata={"proposal_id": "p-1"},
    )

    captured: dict[str, Any] = {}

    def handler(mock_request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(mock_request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"proposal_id": "p-1", "business_approval": false, "rationale": "ok"}'}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    adapter = OpenAIAdapter(identity, client=mock_client)
    adapter.complete(request)

    sent_user_text = captured["json"]["messages"][1]["content"]
    assert "APPROVE_WITH_HEDGE" in sent_user_text
    assert "0.0012" in sent_user_text
    assert "capital" in sent_user_text and "liquidity" in sent_user_text

    adapter.close()
    mock_client.close()


def test_openai_adapter_defaults_to_not_trusting_ambient_env_proxy_config(monkeypatch):
    """P2-01 (v2.0.0 code review): httpx's own default is trust_env=True
    (silently read HTTP_PROXY/HTTPS_PROXY/NO_PROXY/~/.netrc from the
    process environment) -- this adapter must make that choice explicit
    and default to NOT trusting it, so a real deployment never silently
    routes a real API call (bearer token included) through an ambient
    proxy nobody reviewed for this purpose. Inspects the lazily-built
    client's own trust_env setting directly rather than calling
    complete() -- constructing an httpx.Client performs no network I/O by
    itself, so this stays fully hermetic."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    identity = resolve_model_identity("primary")

    default_adapter = OpenAIAdapter(identity)
    assert default_adapter._get_client()._trust_env is False

    explicit_adapter = OpenAIAdapter(identity, trust_env=True)
    assert explicit_adapter._get_client()._trust_env is True


def test_openai_adapter_client_is_injectable_via_a_mock_transport(monkeypatch):
    """P2-01 (v2.0.0 code review) regression test: a caller can inject its
    own httpx.Client (here, one built on httpx.MockTransport) so
    complete()'s full request-shaping/response-parsing logic is exercised
    with ZERO real sockets touched and no process-wide
    monkeypatch.setattr(httpx.Client, "post", ...) reaching into every
    httpx.Client instance. P2-01 (v2.1.0 third code review round) later
    converted the two other request-body tests above
    (test_openai_adapter_request_body_carries_upstream_agent_context /
    _ic_controls_context) to this SAME MockTransport pattern -- they used
    to predate this fix and still use the global-monkeypatch style; all
    three now share one hermetic approach, this test's own."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    identity = resolve_model_identity("primary")

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"proposal_id": "p1", "recommend": true, "rationale": "ok"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    adapter = OpenAIAdapter(identity, client=mock_client)
    request = LLMRequest(
        node_name="pm",
        system_prompt="s",
        user_prompt="u",
        response_schema_name="pm_recommendation",
        max_output_tokens=100,
        metadata={},
    )

    response = adapter.complete(request)
    assert response.parsed["recommend"] is True
    assert captured["url"].endswith("/chat/completions")
    assert captured["authorization"] == "Bearer sk-test-not-real"

    # The adapter never owns/closes a client it did not construct itself.
    adapter.close()
    assert mock_client.is_closed is False
    mock_client.close()


def test_openai_adapter_reuses_one_lazily_built_client_across_calls(monkeypatch):
    """P2-01: complete() used to construct a brand-new httpx.Client on
    EVERY call (no connection-pool reuse, nothing to inject). Now it
    builds one lazily, once, and reuses it."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    identity = resolve_model_identity("primary")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"proposal_id": "p1", "recommend": true, "rationale": "ok"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    adapter = OpenAIAdapter(
        identity, client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    )
    request = LLMRequest(
        node_name="pm", system_prompt="s", user_prompt="u", response_schema_name="pm_recommendation",
        max_output_tokens=100, metadata={},
    )

    adapter.complete(request)
    first_client = adapter._get_client()
    adapter.complete(request)
    second_client = adapter._get_client()
    assert first_client is second_client


def test_openai_adapter_close_only_closes_a_client_it_constructed_itself(monkeypatch):
    """P2-01: close()/the context-manager protocol must release a client
    this adapter built lazily for itself, but never an injected one --
    that client's lifecycle belongs to whoever constructed and passed it
    in."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-5.6-luna-2026-09-01")
    identity = resolve_model_identity("primary")

    injected_client = httpx.Client(trust_env=False)
    adapter_with_injected = OpenAIAdapter(identity, client=injected_client)
    adapter_with_injected.close()
    assert injected_client.is_closed is False
    injected_client.close()

    with OpenAIAdapter(identity) as owning_adapter:
        owned_client = owning_adapter._get_client()
        assert owned_client.is_closed is False
    assert owned_client.is_closed is True


def test_default_responder_raises_for_unknown_schema():
    adapter = FakeLLMAdapter()
    request = LLMRequest(
        node_name="mystery",
        system_prompt="s",
        user_prompt="u",
        response_schema_name="no_such_schema",
        max_output_tokens=100,
        metadata={},
    )
    with pytest.raises(SchemaConformanceError):
        adapter.complete(request)
