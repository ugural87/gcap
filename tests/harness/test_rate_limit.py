from __future__ import annotations

import threading

import pytest

from gcap.agents.llm import FakeLLMAdapter, LLMRequest
from gcap.config.registry import ConcurrencyConfig
from gcap.errors import RequestExceedsRateLimitError
from gcap.harness.rate_limit import RateLimiter, ThrottledLLMClient, _Bucket, make_semaphore


class _FakeClock:
    """Deterministic, injectable clock/sleep pair: sleep() advances the
    clock itself rather than actually blocking, so tests run instantly
    while still exercising the real wait-computation logic."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_allows_calls_under_the_limit_without_sleeping():
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=5, input_tokens_per_minute=10_000, output_tokens_per_minute=10_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    for _ in range(5):
        limiter.acquire(input_tokens=10, output_tokens=10)
    assert fc.slept == []


def test_rate_limiter_sleeps_once_request_budget_is_exhausted():
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=2, input_tokens_per_minute=10_000, output_tokens_per_minute=10_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # third call within the same 60s window must wait
    assert len(fc.slept) >= 1
    assert fc.now > 0


def test_rate_limiter_sleeps_when_token_budget_exhausted():
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=1_000, input_tokens_per_minute=100, output_tokens_per_minute=1_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    limiter.acquire(input_tokens=60)
    limiter.acquire(input_tokens=60)  # 120 > 100 -> must wait for the window to age out
    assert fc.slept


def test_rate_limiter_window_expiry_frees_budget():
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=1, input_tokens_per_minute=10_000, output_tokens_per_minute=10_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    limiter.acquire()
    fc.now += 61  # past the 60s window
    limiter.acquire()  # must not sleep: the first event has aged out
    assert fc.slept == []


def test_from_concurrency_config_reads_the_min_gates():
    limiter = RateLimiter.from_concurrency_config(ConcurrencyConfig())
    assert limiter.requests_per_minute == ConcurrencyConfig().min_requests_per_minute_gate
    assert limiter.input_tokens_per_minute == ConcurrencyConfig().min_input_tokens_per_minute_gate
    assert limiter.output_tokens_per_minute == ConcurrencyConfig().min_output_tokens_per_minute_gate


def test_throttled_llm_client_delegates_to_inner_and_respects_semaphore():
    inner = FakeLLMAdapter()
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=1_000, input_tokens_per_minute=1_000_000, output_tokens_per_minute=1_000_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    semaphore = make_semaphore(ConcurrencyConfig(model_call_semaphore=2))
    throttled = ThrottledLLMClient(inner, semaphore=semaphore, rate_limiter=limiter)
    assert throttled.model_snapshot == inner.model_snapshot

    request = LLMRequest(
        node_name="pm", system_prompt="s", user_prompt="u", response_schema_name="pm_recommendation",
        max_output_tokens=100, metadata={"proposal_id": "p1", "recommend": True},
    )
    response = throttled.complete(request)
    assert response.parsed["proposal_id"] == "p1"
    assert inner.call_count == 1


# ---------------------------------------------------------------------------
# History: a single request whose own weight exceeds the whole per-minute
# limit originally crashed (IndexError on the empty deque). P2-11 replaced
# that crash with silently letting the request through unthrottled. P2-02
# (v2.0.0 code review) replaces THAT with the correct end state: raise a
# clear, named, catchable RequestExceedsRateLimitError instead of either
# crashing or silently exceeding the configured ceiling. These tests
# supersede the old P2-11 tests of the same names' behaviour (their names
# below describe the CURRENT, correct behaviour, not the superseded one).
# ---------------------------------------------------------------------------
def test_bucket_wait_needed_raises_on_a_cold_bucket_with_an_oversized_request():
    bucket = _Bucket(limit=100, name="requests_per_minute")
    with pytest.raises(RequestExceedsRateLimitError) as exc_info:
        bucket.wait_needed(now=0.0, incoming_weight=150)
    # The error names the actual ceiling and the request's own cost, not
    # just "something went wrong".
    assert "requests_per_minute" in str(exc_info.value)
    assert "150" in str(exc_info.value)
    assert "100" in str(exc_info.value)


def test_bucket_wait_needed_raises_once_the_window_empties_out():
    """Same situation, reached a different way: the bucket had events, but
    by the time this call runs every one of them has aged out of the 60s
    window, leaving it empty again right before the oversized check."""
    bucket = _Bucket(limit=100)
    bucket.record(now=0.0, weight=50)
    with pytest.raises(RequestExceedsRateLimitError):
        bucket.wait_needed(now=61.0, incoming_weight=150)


def test_bucket_wait_needed_does_not_raise_when_under_limit_even_with_prior_events():
    """Sanity check that the new raise is scoped correctly: a request that
    fits within the remaining budget must still be let through normally,
    oversized-request handling must never fire for it."""
    bucket = _Bucket(limit=100)
    bucket.record(now=0.0, weight=50)
    assert bucket.wait_needed(now=0.0, incoming_weight=10) == 0.0


def test_rate_limiter_acquire_raises_when_a_single_call_exceeds_the_token_gate():
    """End-to-end through RateLimiter.acquire (not just the bucket): a
    conservative token gate smaller than one real request's own estimated
    cost must raise a clear, named error rather than crash with an
    unrelated exception or silently exceed the configured ceiling."""
    fc = _FakeClock()
    limiter = RateLimiter(
        requests_per_minute=1_000, input_tokens_per_minute=100, output_tokens_per_minute=1_000,
        clock=fc.clock, sleep=fc.sleep,
    )
    with pytest.raises(RequestExceedsRateLimitError):
        limiter.acquire(input_tokens=500)  # exceeds the 100-token gate all by itself
    assert fc.slept == []
    # The failed call recorded nothing -- a normal-sized call right after
    # it must not find itself unexpectedly queued behind a phantom charge.
    limiter.acquire(input_tokens=10)
    assert fc.slept == []


def test_make_semaphore_caps_concurrent_acquisitions():
    semaphore = make_semaphore(ConcurrencyConfig(model_call_semaphore=1))
    acquired_second = threading.Event()
    released_first = threading.Event()

    def hold_then_release() -> None:
        with semaphore:
            released_first.wait(timeout=2)

    t = threading.Thread(target=hold_then_release)
    t.start()
    try:
        # The semaphore has capacity 1 and thread t is holding it; a
        # non-blocking acquire from this thread must fail until t releases.
        assert semaphore.acquire(blocking=False) is False
        acquired_second.set()
    finally:
        released_first.set()
        t.join(timeout=2)
