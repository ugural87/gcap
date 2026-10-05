from __future__ import annotations

import uuid
from decimal import Decimal

from gcap.analysis.conformance_extraction import extract_conformance_gap_inputs
from gcap.store.run_record import GovernanceComponents, RunRecord, StateComponents


def _rr(*, config, case_id, replicate, shown_margin, approval_family, split="pilot"):
    return RunRecord(
        run_id=uuid.uuid4(),
        config=config,
        split=split,
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
        ep_pricing_inconsistent=False,  # not exercised by this extraction function
        ep_governance_failure=False,
        ep_governance_components=GovernanceComponents(),
    )


def test_extracts_paired_rates_restricted_to_negative_shown_margin():
    records = [
        # cluster "case-1": D1 approves (nonapproval=0) both m<=0 replicates;
        # D2 correctly declines (nonapproval=1) both -> D1 rate 0.0, D2 rate 1.0.
        _rr(config="D1", case_id="case-1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=True),
        _rr(config="D1", case_id="case-1", replicate=1, shown_margin=Decimal("-0.002"), approval_family=True),
        _rr(config="D2", case_id="case-1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=False),
        _rr(config="D2", case_id="case-1", replicate=1, shown_margin=Decimal("-0.002"), approval_family=False),
        # a positive-margin run in the same cluster must be excluded from
        # the m<=0-conditioned rate.
        _rr(config="D1", case_id="case-1", replicate=2, shown_margin=Decimal("0.01"), approval_family=True),
    ]
    d1_rates, d2_rates = extract_conformance_gap_inputs(records)
    assert d1_rates == [0.0]
    assert d2_rates == [1.0]


def test_excludes_clusters_missing_from_either_arm():
    records = [
        _rr(config="D1", case_id="case-only-d1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=True),
    ]
    d1_rates, d2_rates = extract_conformance_gap_inputs(records)
    assert d1_rates == []
    assert d2_rates == []


def test_ignores_non_d1_d2_configs_and_missing_shown_margin():
    records = [
        _rr(config="B", case_id="case-1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=True),
        _rr(config="D1", case_id="case-1", replicate=0, shown_margin=None, approval_family=True),
        _rr(config="D2", case_id="case-1", replicate=0, shown_margin=None, approval_family=False),
    ]
    d1_rates, d2_rates = extract_conformance_gap_inputs(records)
    assert d1_rates == []
    assert d2_rates == []


def test_pairs_by_episode_id_not_case_id_for_episodes():
    r1 = _rr(config="D1", case_id="ep-1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=True)
    r1 = r1.model_copy(update={"episode_id": "ep-1"})
    r2 = _rr(config="D2", case_id="ep-1", replicate=0, shown_margin=Decimal("-0.001"), approval_family=False)
    r2 = r2.model_copy(update={"episode_id": "ep-1"})
    d1_rates, d2_rates = extract_conformance_gap_inputs([r1, r2])
    assert d1_rates == [0.0]
    assert d2_rates == [1.0]
