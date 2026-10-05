from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.pilot.report import build_pilot_report
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents

# Keep the nested simulation fast in tests; correctness of its own math is
# already covered by tests/analysis/test_nested_simulation.py.
_FAST_KMIN_N_MC = 20_000
_FAST_CENTRAL_TEST_SIMS = 100


def _rr(*, config, case_id, replicate, shown_margin, approval_family):
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
        ep_numerical_error=False,
        ep_state_infeasible=False,
        ep_state_components=StateComponents(),
        ep_pricing_inconsistent=False,
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
    )


def test_report_with_no_records_has_no_g_or_kmin():
    report = build_pilot_report([], kmin_n_mc=_FAST_KMIN_N_MC, central_test_n_sims=_FAST_CENTRAL_TEST_SIMS)
    assert report.n_conformance_clusters == 0
    assert report.g_point is None
    assert report.g_lower_bound is None
    assert report.kmin is None
    assert report.kmin_frozen is False
    assert 0.0 <= report.central_test_power <= 1.0  # computed from the fixed D1-D2 floor regardless of pilot data


def test_report_with_no_m_leq_0_runs_has_no_g_or_kmin():
    # D1/D2 runs all have a positive shown margin -> nothing is m<=0 eligible.
    records = [
        _rr(config="D1", case_id="c1", replicate=0, shown_margin=Decimal("0.05"), approval_family=True),
        _rr(config="D2", case_id="c1", replicate=0, shown_margin=Decimal("0.05"), approval_family=True),
    ]
    report = build_pilot_report(records, kmin_n_mc=_FAST_KMIN_N_MC, central_test_n_sims=_FAST_CENTRAL_TEST_SIMS)
    assert report.n_conformance_clusters == 0
    assert report.g_point is None
    assert report.kmin_frozen is False


def _perfect_d2_over_d1_records(n_clusters: int = 40) -> list[RunRecord]:
    """40 clusters, every m<=0: D1 always (wrongly) approves; D2 always
    (correctly) declines -> g_point = 1.0, a strong positive lower bound."""
    records = []
    for i in range(n_clusters):
        cid = f"case-{i}"
        records.append(_rr(config="D1", case_id=cid, replicate=0, shown_margin=Decimal("-0.01"), approval_family=True))
        records.append(_rr(config="D2", case_id=cid, replicate=0, shown_margin=Decimal("-0.01"), approval_family=False))
    return records


def test_report_finds_no_frozen_kmin_at_the_spec_default_central_test_floor():
    """A genuine finding from wiring this module up for the first time,
    not a test artifact: at Section 9.7's own pre-registered D1-D2 floor
    (121 clusters, p10=0.20, p01=0.05, Holm alpha=0.0125),
    analysis.power.simulate_paired_binary_power's sign-flip permutation
    estimate comes out to roughly 0.75-0.76, a few points under the 80%
    the proposal's planning narrative targets for those inputs. Since
    analysis.nested_simulation.derive_kmin requires
    central_test_power * stability_power >= 0.80, and central_test_power
    alone already caps that product below 0.80, NO k can ever qualify at
    these exact floor inputs — regardless of how strong g is. This is
    correct, intentional behaviour (Section 9.7: "Blinded pilot
    simulations may raise this floor but may not lower it" — precisely the
    situation this surfaces), not a bug in derive_kmin or in this test."""
    records = _perfect_d2_over_d1_records()
    report = build_pilot_report(
        records, g_n_boot=1000, kmin_n_mc=_FAST_KMIN_N_MC, central_test_n_sims=_FAST_CENTRAL_TEST_SIMS
    )
    assert report.n_conformance_clusters == 40
    assert report.g_point == 1.0
    assert report.g_lower_bound is not None and report.g_lower_bound > 0
    assert report.central_test_power < 0.80
    assert report.kmin is None
    assert report.kmin_frozen is False
    assert len(report.kmin_candidates) > 0  # every candidate was still evaluated and reported


def test_report_derives_a_frozen_kmin_once_central_test_power_clears_its_own_floor():
    """The mirror case: raising central_test_p10 (as a REAL pilot revising
    the D1-D2 floor's planning discordance upward would) pushes
    central_test_power comfortably above 0.80 on its own, and combined
    with a strong g the joint condition is then satisfiable."""
    records = _perfect_d2_over_d1_records()
    report = build_pilot_report(
        records,
        g_n_boot=1000,
        central_test_p10=0.25,
        central_test_n_sims=_FAST_CENTRAL_TEST_SIMS,
        kmin_n_mc=_FAST_KMIN_N_MC,
    )
    assert report.central_test_power >= 0.80
    assert report.kmin is not None
    assert report.kmin_frozen is True


def test_report_metrics_field_is_populated_regardless_of_conformance_data():
    records = [_rr(config="B", case_id="c1", replicate=0, shown_margin=None, approval_family=True)]
    report = build_pilot_report(records, kmin_n_mc=_FAST_KMIN_N_MC, central_test_n_sims=_FAST_CENTRAL_TEST_SIMS)
    assert report.metrics.n_records == 1
