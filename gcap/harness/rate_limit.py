"""Concurrency control (Section 8's ConcurrencyConfig): a worker-pool-wide
cap on in-flight model calls (the semaphore) plus a client-side sliding-
window rate limiter enforcing the account's real per-minute ceilings
underneath it. See ``gcap.config.registry.ConcurrencyConfig``'s docstring
for the two-layer design this implements: a large async/thread task pool
prepares many independent calls, ``model_call_semaphore`` caps how many of
them may be genuinely in flight at once, and this rate limiter caps the
requests/tokens actually leaving the account per minute even if the
semaphore alone would allow a burst.

Deliberately simple (one lock held for the duration of a wait, rather than
a lock-free or per-bucket-released design): Section-level guidance is
explicit that this codebase is NOT an MLOps/deployment pipeline, so a
correct, easy-to-audit limiter is preferred over a higher-throughput one.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from gcap.agents.llm import LLMClient, LLMRequest, LLMResponse
from gcap.config.registry import ConcurrencyConfig
from gcap.errors import RequestExceedsRateLimitError

_WINDOW_SECONDS = 60.0


@dataclass
class _Bucket:
    """One sliding-60s-window budget: a deque of (timestamp, weight)
    events. ``weight`` is 1 per event for the requests-per-minute bucket
    and the token count itself for the two token buckets."""

    limit: int
    # P2-02 (v2.0.0 code review): only used to name which ceiling a
    # RequestExceedsRateLimitError below refers to -- purely diagnostic,
    # never consulted by any budget logic.
    name: str = ""
    _events: deque[tuple[float, int]] = field(default_factory=deque, repr=False, compare=False)

    def _evict(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= _WINDOW_SECONDS:
            self._events.popleft()

    def wait_needed(self, now: float, incoming_weight: int) -> float:
        """How long to wait before an event of ``incoming_weight`` may be
        recorded without pushing the window's total over ``limit``."""
        self._evict(now)
        if self.limit <= 0:
            return 0.0
        total = sum(weight for _ts, weight in self._events)
        if total + incoming_weight <= self.limit:
            return 0.0
        if not self._events:
            # The window is empty (a cold bucket, or every prior event has
            # just aged out) yet this one event's own weight already
            # exceeds ``limit`` by itself -- e.g. a single request's
            # estimated token cost exceeds a conservative per-minute token
            # gate on its own. There is nothing older in the window to
            # evict, so no amount of waiting can ever bring this request
            # under budget.
            #
            # History: the original code fell through to
            # ``self._events[0]`` here and raised a bare IndexError on the
            # empty deque (a crash, not a domain error). P2-11 replaced
            # that with letting the request through immediately as a
            # best-effort exception -- no longer a crash, but it meant this
            # limiter could silently exceed its own configured ceiling with
            # no signal to the caller. P2-02 (v2.0.0 code review) replaces
            # THAT with the correct end state: raise a clear, named,
            # catchable error instead of either crashing or silently
            # violating the budget it exists to enforce. This is a
            # deliberately unrecoverable-by-waiting condition (invariant 9:
            # Harness.run_case still catches it and records a `failure`
            # outcome rather than crashing the batch), not a transient one.
            raise RequestExceedsRateLimitError(
                f"A single request's {self.name or 'weight'} ({incoming_weight}) exceeds the "
                f"configured per-minute ceiling ({self.limit}) on its own; the sliding window is "
                "already empty, so no amount of waiting can bring it under budget. Raise this "
                "ceiling in ConcurrencyConfig, or reduce the request's cost (e.g. a smaller "
                "max_output_tokens), before retrying."
            )
        oldest_ts = self._events[0][0]
        return max(0.0, _WINDOW_SECONDS - (now - oldest_ts))

    def record(self, now: float, weight: int) -> None:
        self._events.append((now, weight))


@dataclass
class RateLimiter:
    """A sliding-60s-window limiter over three independent budgets
    (requests, input tokens, output tokens). ``acquire`` blocks (sleeping
    via the injectable ``sleep`` hook, so tests never need a real clock)
    until all three budgets have room, then records the event."""

    requests_per_minute: int
    input_tokens_per_minute: int
    output_tokens_per_minute: int
    clock: Callable[[], float] = field(default=time.monotonic)
    sleep: Callable[[float], None] = field(default=time.sleep)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._requests = _Bucket(self.requests_per_minute, name="requests_per_minute")
        self._input_tokens = _Bucket(self.input_tokens_per_minute, name="input_tokens_per_minute")
        self._output_tokens = _Bucket(self.output_tokens_per_minute, name="output_tokens_per_minute")

    def acquire(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            while True:
                now = self.clock()
                wait = max(
                    self._requests.wait_needed(now, 1),
                    self._input_tokens.wait_needed(now, input_tokens),
                    self._output_tokens.wait_needed(now, output_tokens),
                )
                if wait <= 0:
                    self._requests.record(now, 1)
                    self._input_tokens.record(now, input_tokens)
                    self._output_tokens.record(now, output_tokens)
                    return
                self.sleep(wait)

    @classmethod
    def from_concurrency_config(
        cls,
        config: ConcurrencyConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> RateLimiter:
        return cls(
            requests_per_minute=config.min_requests_per_minute_gate,
            input_tokens_per_minute=config.min_input_tokens_per_minute_gate,
            output_tokens_per_minute=config.min_output_tokens_per_minute_gate,
            clock=clock,
            sleep=sleep,
        )


class ThrottledLLMClient:
    """Wraps any LLMClient (fake or real) with the semaphore + rate
    limiter, so graph/agents code never has to know throttling exists —
    the harness constructs one of these and hands it to GraphContext as
    ``adapter``. Exposes ``model_snapshot`` when the wrapped client does,
    so RunRecord assembly (harness/record.py) can read it through the
    wrapper transparently."""

    def __init__(self, inner: LLMClient, *, semaphore: threading.Semaphore, rate_limiter: RateLimiter) -> None:
        self._inner = inner
        self._semaphore = semaphore
        self._rate_limiter = rate_limiter

    @property
    def model_snapshot(self) -> str:
        return str(getattr(self._inner, "model_snapshot", "unknown-model-snapshot"))

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._rate_limiter.acquire(input_tokens=request.input_token_estimate(), output_tokens=request.max_output_tokens)
        with self._semaphore:
            return self._inner.complete(request)


def make_semaphore(config: ConcurrencyConfig) -> threading.Semaphore:
    return threading.Semaphore(config.model_call_semaphore)
