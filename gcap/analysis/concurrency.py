"""Programmatic worker-pool selection.

ConcurrencyConfig.worker_pool_candidates names the three sizes a real
pilot must benchmark (32, 64, 128) and ConcurrencyConfig.model_call_semaphore
is today a hand-picked, dated, commented constant (see registry.py: "2026-
09-21 pilot decision"). This module is what turns a real pilot's measured
throughput/error/latency into that constant instead -- select_worker_pool
never invents or silently substitutes a candidate; it only ever returns one
of the sizes it was actually given a measurement for, and only when that
measurement clears the frozen reliability gates.

Selecting a worker pool this way does not, by itself, change
model_call_semaphore: per this project's freeze-manifest governance
("the freeze is a file, not a meeting" -- config/manifest.py), a human
reviews this function's output and is the one who updates the registry and
commits the manifest, exactly as harness/cli.py's ``select-concurrency``
subcommand's docstring describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gcap.config.registry import ConcurrencyConfig
from gcap.errors import NoEligibleConcurrencyCandidateError


@dataclass(frozen=True)
class ConcurrencyObservation:
    """One worker-pool candidate's measured pilot throughput and
    reliability, over some fixed observation window."""

    workers: int
    p95_latency_s: Decimal
    requests_per_minute: Decimal
    input_tokens_per_minute: Decimal
    output_tokens_per_minute: Decimal
    rate_limit_fraction: Decimal
    error_fraction: Decimal


def select_worker_pool(
    observations: tuple[ConcurrencyObservation, ...], config: ConcurrencyConfig
) -> int:
    """Among candidates measured at exactly ``config.worker_pool_candidates``,
    returns the fastest (highest observed requests/minute, ties broken by
    lower p95 latency) that clears every reliability gate
    (``max_rate_limit_fraction``, ``max_error_fraction``, ``max_p95_latency_s``).

    Raises ValueError if the observations don't cover exactly the
    configured candidate sizes (a partial or padded pilot is a pilot that
    was not actually run as specified), and
    NoEligibleConcurrencyCandidateError if none clears the gates -- both
    fail loud rather than falling back to a default, per this project's
    "failures are recorded outcomes, never silent" ethos.
    """
    observed = {o.workers for o in observations}
    required = set(config.worker_pool_candidates)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            "pilot must evaluate exactly the configured worker-pool candidates "
            f"{sorted(required)}; missing={missing}, unexpected={extra}"
        )

    eligible = [
        o
        for o in observations
        if o.rate_limit_fraction <= config.max_rate_limit_fraction
        and o.error_fraction <= config.max_error_fraction
        and o.p95_latency_s <= config.max_p95_latency_s
    ]
    if not eligible:
        raise NoEligibleConcurrencyCandidateError(
            "no worker-pool candidate cleared the frozen reliability gates "
            f"(max_rate_limit_fraction={config.max_rate_limit_fraction}, "
            f"max_error_fraction={config.max_error_fraction}, "
            f"max_p95_latency_s={config.max_p95_latency_s})."
        )
    best = max(eligible, key=lambda o: (o.requests_per_minute, -o.p95_latency_s))
    return best.workers
