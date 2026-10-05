from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.pilot.metrics import compute_pilot_metrics
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents


def _rr(
    *,
    config,
    case_id,
    replicate,
    ep_numerical_error=False,
    ep_state_infeasible=False,
    ep_pricing_inconsistent=False,
    ep_governance_failure=False,
    shown_margin=None,
    approval_family=None,
    tokens_in=None,
    tokens_out=None,
    latency_ms=None,
    rework=False,
    retries=0,
):
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split="pilot",
        block="pilot",
        case_id=case_id,
        replicate=replicate,
        family="alpha_eliminated_by_funding_cost",
        materiality="high",
        model_snapshot="fake-luna-sandbox-1.0",
        policy_version="policy-v1",
        state_version_in=0,
        manifest_hash="0" * 64,
        shown_margin=shown_margin,
        approval_family=approval_family,
        ep_numerical_error=ep_numerical_error,
        ep_state_infeasible=ep_state_infeasible,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=ep_pricing_inconsistent,
        ep_governance_failure=ep_governance_failure,
        ep_governance_components=GovernanceComponents(),
        tokens_in=tokens_in or {},
        tokens_out=tokens_out or {},
        latency_ms=latency_ms or {},
        rework=rework,
        retries=retries,
    )


def test_empty_records_returns_zeroed_metrics():
    metrics = compute_pilot_metrics([])
    assert metrics.n_records == 0
    assert metrics.event_rate == {}
    assert metrics.pricing_informative_fraction is None


def test_event_rate_per_config():
    records = [
        _rr(config="B", case_id="c1", replicate=0, ep_numerical_error=True),
        _rr(config="B", case_id="c2", replicate=0, ep_numerical_error=False),
        _rr(config="C", case_id="c1", replicate=0, ep_numerical_error=False),
    ]
    metrics = compute_pilot_metrics(records)
    assert metrics.event_rate["numerical_error"]["B"] == 0.5
    assert metrics.event_rate["numerical_error"]["C"] == 0.0


def test_discordance_from_majority_vote_by_cluster():
    # D1: c1 fires (2/2 replicates); c2 does not fire (0/2).
    # D2: c1 does not fire (0/2); c2 fires (2/2) -> a fully discordant pair
    # in BOTH directions across the two clusters (p10 and p01 each 0.5).
    records = [
        _rr(config="D1", case_id="c1", replicate=0, ep_pricing_inconsistent=True),
        _rr(config="D1", case_id="c1", replicate=1, ep_pricing_inconsistent=True),
        _rr(config="D1", case_id="c2", replicate=0, ep_pricing_inconsistent=False),
        _rr(config="D1", case_id="c2", replicate=1, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c1", replicate=0, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c1", replicate=1, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c2", replicate=0, ep_pricing_inconsistent=True),
        _rr(config="D2", case_id="c2", replicate=1, ep_pricing_inconsistent=True),
    ]
    metrics = compute_pilot_metrics(records)
    p10, p01 = metrics.discordance["pricing_inconsistent"]
    assert p10 == 0.5
    assert p01 == 0.5


def test_icc_is_computed_per_config_not_pooled_across_a_contrasts_two_arms():
    # Within EACH config, every replicate in a cluster agrees perfectly
    # (D1's c1 is always True, c2 always False; D2 is the mirror image) ->
    # each arm's OWN ICC is 1.0, even though D1 and D2 disagree with each
    # other completely -- pooling them together (the discarded design)
    # would instead show that cross-arm disagreement as noise.
    records = [
        _rr(config="D1", case_id="c1", replicate=0, ep_pricing_inconsistent=True),
        _rr(config="D1", case_id="c1", replicate=1, ep_pricing_inconsistent=True),
        _rr(config="D1", case_id="c2", replicate=0, ep_pricing_inconsistent=False),
        _rr(config="D1", case_id="c2", replicate=1, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c1", replicate=0, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c1", replicate=1, ep_pricing_inconsistent=False),
        _rr(config="D2", case_id="c2", replicate=0, ep_pricing_inconsistent=True),
        _rr(config="D2", case_id="c2", replicate=1, ep_pricing_inconsistent=True),
    ]
    metrics = compute_pilot_metrics(records)
    assert metrics.icc["pricing_inconsistent"]["D1"] == 1.0
    assert metrics.icc["pricing_inconsistent"]["D2"] == 1.0


def test_pricing_informative_fraction_restricted_to_d1_d2_with_margin():
    records = [
        _rr(config="D1", case_id="c1", replicate=0, shown_margin=Decimal("-0.01")),
        _rr(config="D1", case_id="c2", replicate=0, shown_margin=Decimal("0.02")),
        _rr(config="D2", case_id="c1", replicate=0, shown_margin=Decimal("-0.01")),
        _rr(config="B", case_id="c3", replicate=0, shown_margin=None),  # excluded: not D1/D2
    ]
    metrics = compute_pilot_metrics(records)
    assert metrics.pricing_informative_fraction == 2 / 3


def test_token_latency_rework_retry_aggregation():
    records = [
        _rr(
            config="B", case_id="c1", replicate=0,
            tokens_in={"research": 100}, tokens_out={"research": 20},
            latency_ms={"research": 10}, rework=True, retries=1,
        ),
        _rr(
            config="B", case_id="c2", replicate=0,
            tokens_in={"research": 200}, tokens_out={"research": 40},
            latency_ms={"research": 30}, rework=False, retries=0,
        ),
    ]
    metrics = compute_pilot_metrics(records)
    assert metrics.tokens_in_total == {"research": 300}
    assert metrics.tokens_out_total == {"research": 60}
    p50, p90, mean = metrics.latency_quantiles_ms["research"]
    assert mean == 20.0
    assert metrics.rework_rate == 0.5
    assert metrics.mean_retries == 0.5


def test_achieved_rates_computed_only_when_wall_clock_given():
    records = [_rr(config="B", case_id="c1", replicate=0, tokens_in={"research": 60}, tokens_out={"research": 0})]
    no_clock = compute_pilot_metrics(records)
    assert no_clock.achieved_requests_per_minute is None
    assert no_clock.achieved_tokens_per_minute is None

    with_clock = compute_pilot_metrics(records, wall_clock_seconds=60.0)
    assert with_clock.achieved_requests_per_minute == 1.0  # one populated node -> one call, in one minute
    assert with_clock.achieved_tokens_per_minute == 60.0
