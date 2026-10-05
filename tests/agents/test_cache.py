from __future__ import annotations

import pytest

from gcap.agents.cache import SharedCallCache
from gcap.agents.llm import FakeLLMAdapter, LLMRequest
from gcap.errors import CacheIsolationError, UndeclaredCacheHitError


def _request(node_name: str, case_id_marker: str) -> LLMRequest:
    return LLMRequest(
        node_name=node_name,
        system_prompt="s",
        user_prompt=f"identical prompt for {case_id_marker}",
        response_schema_name="trade_proposal" if node_name == "trade" else "claim_review",
        max_output_tokens=250,
        metadata={},
    )


def test_d1_d2_pair_shares_research_and_trade_calls():
    cache = SharedCallCache()
    adapter = FakeLLMAdapter()
    request = _request("trade", "case-1")

    response_d1, hit_d1 = cache.get_or_call(
        config="D1", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    response_d2, hit_d2 = cache.get_or_call(
        config="D2", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )

    assert hit_d1 is False
    assert hit_d2 is True
    assert response_d1 is response_d2
    assert adapter.call_count == 1  # the second config never actually re-invoked the LLM
    assert cache.hits == 1
    assert cache.misses == 1


def test_non_d1_d2_pair_raises_undeclared_cache_hit():
    cache = SharedCallCache()
    adapter = FakeLLMAdapter()
    request = _request("trade", "case-1")

    cache.get_or_call(
        config="B", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    with pytest.raises(UndeclaredCacheHitError):
        cache.get_or_call(
            config="C", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )


def test_non_shareable_node_raises_cache_isolation_error_even_for_d1_d2():
    cache = SharedCallCache()
    adapter = FakeLLMAdapter()
    # "treasury" legitimately differs between D1 and D2 (the FTP addon
    # vector differs), so a coincidental prompt-hash collision there must
    # never be silently served from cache.
    request = LLMRequest(
        node_name="treasury",
        system_prompt="s",
        user_prompt="identical by coincidence",
        response_schema_name="treasury_decision",
        max_output_tokens=250,
        metadata={},
    )
    cache.get_or_call(
        config="D1", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    with pytest.raises(CacheIsolationError):
        cache.get_or_call(
            config="D2", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
            model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
        )


def test_different_replicates_never_share_cache():
    cache = SharedCallCache()
    adapter = FakeLLMAdapter()
    request = _request("trade", "case-1")

    cache.get_or_call(
        config="D1", case_id="case-1", replicate=0, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    _, hit = cache.get_or_call(
        config="D1", case_id="case-1", replicate=1, request=request, call_fn=adapter.complete,
        model_snapshot="fake-luna-sandbox-1.0", manifest_hash="manifest-hash-fixture",
    )
    assert hit is False
    assert adapter.call_count == 2
