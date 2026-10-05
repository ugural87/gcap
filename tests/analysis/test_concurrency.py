"""Tests for analysis.concurrency.select_worker_pool -- the pilot-throughput
-driven replacement for config.registry's hand-picked model_call_semaphore
constant.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from gcap.analysis.concurrency import ConcurrencyObservation, select_worker_pool
from gcap.config.registry import ConcurrencyConfig
from gcap.errors import NoEligibleConcurrencyCandidateError

_CONFIG = ConcurrencyConfig()


def _obs(workers, *, p95=5, rpm=100, error=Decimal("0"), rate_limited=Decimal("0")):
    return ConcurrencyObservation(
        workers=workers,
        p95_latency_s=Decimal(p95),
        requests_per_minute=Decimal(rpm),
        input_tokens_per_minute=Decimal("50000"),
        output_tokens_per_minute=Decimal("5000"),
        rate_limit_fraction=rate_limited,
        error_fraction=error,
    )


def test_selects_the_fastest_eligible_candidate():
    observations = (
        _obs(32, p95=4, rpm=90),
        _obs(64, p95=6, rpm=150),
        _obs(128, p95=8, rpm=140),
    )
    assert select_worker_pool(observations, _CONFIG) == 64


def test_ineligible_candidate_is_excluded_even_if_fastest():
    observations = (
        _obs(32, p95=4, rpm=90),
        _obs(64, p95=6, rpm=300, error=Decimal("0.5")),  # fastest but unreliable
        _obs(128, p95=8, rpm=140),
    )
    assert select_worker_pool(observations, _CONFIG) == 128


def test_ties_on_throughput_broken_by_lower_latency():
    observations = (
        _obs(32, p95=10, rpm=100),
        _obs(64, p95=3, rpm=100),
        _obs(128, p95=20, rpm=100),
    )
    assert select_worker_pool(observations, _CONFIG) == 64


def test_missing_a_required_candidate_raises_value_error():
    observations = (_obs(32), _obs(64))
    with pytest.raises(ValueError, match="missing"):
        select_worker_pool(observations, _CONFIG)


def test_unexpected_candidate_raises_value_error():
    observations = (_obs(32), _obs(64), _obs(128), _obs(256))
    with pytest.raises(ValueError, match="unexpected"):
        select_worker_pool(observations, _CONFIG)


def test_no_eligible_candidate_raises_dedicated_error():
    observations = (
        _obs(32, error=Decimal("1")),
        _obs(64, error=Decimal("1")),
        _obs(128, error=Decimal("1")),
    )
    with pytest.raises(NoEligibleConcurrencyCandidateError):
        select_worker_pool(observations, _CONFIG)


def test_p95_latency_gate_is_enforced():
    observations = (
        _obs(32, p95=1, rpm=50),
        _obs(64, p95=1, rpm=60),
        _obs(128, p95=31, rpm=500),  # fastest, but breaches the 30s gate
    )
    assert select_worker_pool(observations, _CONFIG) == 64
