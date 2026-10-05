"""Descriptive pilot analytics (Section 9.7): "estimates baseline event
rates, both paired discordance directions, intra-cluster correlation,
between-run variance, ordering variance, token distributions, rework
incidence and length, per-call latency quantiles, achieved requests and
tokens per minute, rate-limit incidence, retry burden, and the proportion
of units informative for each endpoint."

Scope note: every number below is computed from RunRecord's own flat
fields only — the same boundary every other analysis/ module respects.
Two items from the Section 9.7 list are therefore left out rather than
approximated: "ordering variance" needs more than one scenario ordering
per cluster, which a 3-replicate pilot with no ordering diagnostic does
not exercise (Section 9.7's ordering-permutation diagnostic is scoped to
eight PRE-SELECTED episodes at confirmatory scale, not the pilot), and
"rate-limit incidence" has no RunRecord field recording whether a call was
throttled (harness.rate_limit.RateLimiter enforces the gate but does not
write a per-run marker) — inventing a number for either here would not be
grounded in anything the run log actually recorded. The informative-unit
fraction is likewise reported only for the one endpoint (pricing) that
RunRecord's own shown_margin field cleanly answers "was this unit even
eligible to show the endpoint" for; the other three would need a join
against decision_packages/evidence_claims this RunRecord-only module does
not perform.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from gcap.analysis.endpoints import ENDPOINT_FIELDS, PRIMARY_CONTRASTS, cluster_id_of
from gcap.analysis.icc import one_way_icc
from gcap.config.flags import ConfigName
from gcap.store.run_record import RunRecord


@dataclass(frozen=True)
class PilotMetrics:
    n_records: int
    event_rate: dict[str, dict[str, float]]  # endpoint -> config -> mean(ep_*)
    # Descriptive discordance directions per primary contrast (Section 9.7:
    # "the pilot's discordance estimates are diagnostic rather than precise
    # probability estimates"), from a per-cluster majority-vote coding —
    # p10: fraction of paired clusters favouring arm_a (endpoint fires under
    # arm_a's majority vote but not arm_b's); p01: the reverse.
    discordance: dict[str, tuple[float, float]]
    icc: dict[str, dict[str, float | None]]  # endpoint -> config -> ICC, one arm's own clusters at a time
    pricing_informative_fraction: float | None  # fraction of D1/D2 runs with shown_margin <= 0
    tokens_in_total: dict[str, int] = field(default_factory=dict)
    tokens_out_total: dict[str, int] = field(default_factory=dict)
    latency_quantiles_ms: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # (p50, p90, mean)
    rework_rate: float = 0.0
    mean_retries: float = 0.0
    achieved_requests_per_minute: float | None = None
    achieved_tokens_per_minute: float | None = None


def _event_rates(records: Sequence[RunRecord]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for endpoint, field_name in ENDPOINT_FIELDS.items():
        by_config: dict[str, list[bool]] = {}
        for r in records:
            by_config.setdefault(r.config, []).append(bool(getattr(r, field_name)))
        out[endpoint] = {config: float(np.mean(values)) for config, values in by_config.items() if values}
    return out


def _majority_vote_by_cluster(records: Sequence[RunRecord], config: ConfigName, field_name: str) -> dict[str, int]:
    by_cluster: dict[str, list[bool]] = {}
    for r in records:
        if r.config != config:
            continue
        by_cluster.setdefault(cluster_id_of(r), []).append(bool(getattr(r, field_name)))
    return {cluster_id: (1 if np.mean(values) >= 0.5 else 0) for cluster_id, values in by_cluster.items()}


def _discordance(records: Sequence[RunRecord]) -> dict[str, tuple[float, float]]:
    discordance: dict[str, tuple[float, float]] = {}
    for endpoint, arm_a, arm_b in PRIMARY_CONTRASTS:
        field_name = ENDPOINT_FIELDS[endpoint]
        votes_a = _majority_vote_by_cluster(records, arm_a, field_name)
        votes_b = _majority_vote_by_cluster(records, arm_b, field_name)
        shared = sorted(set(votes_a) & set(votes_b))
        if shared:
            p10 = sum(1 for c in shared if votes_a[c] == 1 and votes_b[c] == 0) / len(shared)
            p01 = sum(1 for c in shared if votes_a[c] == 0 and votes_b[c] == 1) / len(shared)
            discordance[endpoint] = (p10, p01)
    return discordance


def _icc_by_config(records: Sequence[RunRecord]) -> dict[str, dict[str, float | None]]:
    """ICC per (endpoint, config) — deliberately NOT pooled across a
    contrast's two arms: pooling arm_a and arm_b into the same cluster
    groups would count exactly the systematic difference the contrast
    exists to measure as within-cluster noise, biasing the estimate
    (typically downward, even negative) whenever the contrast has a real
    effect. Each arm's own replicate structure is what Section 9.6's
    crossed random intercept for "base scenario" is actually estimated
    within, so ICC belongs to one (endpoint, config) pair, matching
    event_rate's own dict shape."""
    out: dict[str, dict[str, float | None]] = {}
    for endpoint, field_name in ENDPOINT_FIELDS.items():
        by_config_cluster: dict[str, dict[str, list[float]]] = {}
        for r in records:
            clusters = by_config_cluster.setdefault(r.config, {})
            clusters.setdefault(cluster_id_of(r), []).append(1.0 if bool(getattr(r, field_name)) else 0.0)
        out[endpoint] = {config: one_way_icc(clusters) for config, clusters in by_config_cluster.items()}
    return out


def _pricing_informative_fraction(records: Sequence[RunRecord]) -> float | None:
    margins = [r.shown_margin for r in records if r.config in ("D1", "D2") and r.shown_margin is not None]
    if not margins:
        return None
    return sum(1 for m in margins if m <= 0) / len(margins)


def _token_and_latency_stats(
    records: Sequence[RunRecord],
) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[float, float, float]]]:
    tokens_in_total: dict[str, int] = {}
    tokens_out_total: dict[str, int] = {}
    latency_by_node: dict[str, list[float]] = {}
    for r in records:
        for node, v in r.tokens_in.items():
            tokens_in_total[node] = tokens_in_total.get(node, 0) + v
        for node, v in r.tokens_out.items():
            tokens_out_total[node] = tokens_out_total.get(node, 0) + v
        for node, v in r.latency_ms.items():
            latency_by_node.setdefault(node, []).append(float(v))
    latency_quantiles_ms = {
        node: (float(np.quantile(vals, 0.5)), float(np.quantile(vals, 0.9)), float(np.mean(vals)))
        for node, vals in latency_by_node.items()
    }
    return tokens_in_total, tokens_out_total, latency_quantiles_ms


def compute_pilot_metrics(
    records: Sequence[RunRecord],
    *,
    wall_clock_seconds: float | None = None,
) -> PilotMetrics:
    """``wall_clock_seconds``, when supplied by the caller (the actual
    elapsed time of the run_pilot() call these records came from), is what
    achieved_requests_per_minute / achieved_tokens_per_minute are computed
    against; omitted, both stay None rather than silently reporting a rate
    against an unknown duration."""
    if not records:
        return PilotMetrics(
            n_records=0, event_rate={}, discordance={}, icc={}, pricing_informative_fraction=None
        )

    event_rate = _event_rates(records)
    discordance = _discordance(records)
    icc = _icc_by_config(records)
    pricing_informative_fraction = _pricing_informative_fraction(records)
    tokens_in_total, tokens_out_total, latency_quantiles_ms = _token_and_latency_stats(records)
    rework_rate = float(np.mean([r.rework for r in records]))
    mean_retries = float(np.mean([r.retries for r in records]))

    achieved_requests_per_minute = None
    achieved_tokens_per_minute = None
    if wall_clock_seconds and wall_clock_seconds > 0:
        total_calls = sum(len(r.tokens_in) for r in records)  # one call per populated node per run
        total_tokens = sum(tokens_in_total.values()) + sum(tokens_out_total.values())
        minutes = wall_clock_seconds / 60.0
        achieved_requests_per_minute = total_calls / minutes
        achieved_tokens_per_minute = total_tokens / minutes

    return PilotMetrics(
        n_records=len(records),
        event_rate=event_rate,
        discordance=discordance,
        icc=icc,
        pricing_informative_fraction=pricing_informative_fraction,
        tokens_in_total=tokens_in_total,
        tokens_out_total=tokens_out_total,
        latency_quantiles_ms=latency_quantiles_ms,
        rework_rate=rework_rate,
        mean_retries=mean_retries,
        achieved_requests_per_minute=achieved_requests_per_minute,
        achieved_tokens_per_minute=achieved_tokens_per_minute,
    )
