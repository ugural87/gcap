from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from gcap.bench.generator import generate_lhc_candidate_pool
from gcap.bench.schema import BenchCase, EquityFundamentals, InstrumentSpec, ProposalSpec
from gcap.core.ftp import FTPAddonVector
from gcap.core.ledger import LedgerAction, LedgerState
from gcap.errors import InsufficientLhcCandidatesError
from gcap.oracle.schema import OracleProposal
from gcap.pilot.lhc_pool import build_oracle_eval_fn, select_lhc_replay_cases

_AS_OF = datetime(2026, 9, 21, tzinfo=UTC)


def _state() -> LedgerState:
    return LedgerState(
        capital_available=Decimal("5000000"),
        liquidity_available=Decimal("5000000"),
        funding_capacity=Decimal("5000000"),
        collateral_available=Decimal("2000000"),
        risk_budget_available=Decimal("1000000"),
    )


def _case(
    case_id: str,
    *,
    tenor_stratum: str,
    tau_days: Decimal = Decimal("365"),
    base_funding_curve_rate_ann: Decimal = Decimal("0.03"),
    trade_alpha_horizon: Decimal = Decimal("0.032"),
    capital_usage: Decimal = Decimal("50000"),
    asset_class: str = "equity",
) -> BenchCase:
    """A minimal, precisely-controlled LHC-shaped case. With
    tau_days == day_count_basis == 365, FTP's horizon conversion is the
    identity ((1+f_ann)^(365/365)-1 == f_ann), so
    economic_alpha_horizon == trade_alpha_horizon - base_rate - aggregate_addon
    exactly, which is what lets the sign-flip tests below use round numbers.
    """
    action = LedgerAction(
        action_id=case_id, instrument_id=case_id, state_version_seen=0, capital_usage=capital_usage
    )
    if asset_class == "fixed_income":
        from gcap.bench.schema import FixedIncomeAnalytics

        instrument = InstrumentSpec(
            asset_class="fixed_income",
            fixed_income=FixedIncomeAnalytics(
                isin=f"SYNFI-{case_id}",
                coupon_pct=Decimal("0.04"),
                maturity_date="2031-06-30",
                clean_price=Decimal("100"),
                yield_pct=Decimal("0.05"),
                spread_bps=Decimal("100"),
                modified_duration=Decimal("3"),
                dv01=Decimal("500"),
            ),
        )
    else:
        instrument = InstrumentSpec(
            asset_class="equity",
            equity=EquityFundamentals(ticker=f"SYN-{case_id}", as_of="2026-09-21"),
        )
    oracle_proposal = OracleProposal(
        proposal_id=case_id,
        tau_days=tau_days,
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=base_funding_curve_rate_ann,
        addons=FTPAddonVector(),
        trade_alpha_horizon=trade_alpha_horizon,
        ledger_action=action,
    )
    proposal = ProposalSpec(
        proposal_id=case_id, instrument=instrument, oracle_proposal=oracle_proposal
    )
    return BenchCase(
        case_id=case_id,
        split="confirm",
        family="alpha_eliminated_by_funding_cost",
        strata=frozenset({"pricing"}),
        materiality="high",
        tenor_stratum=tenor_stratum,
        kind="single",
        proposals=(proposal,),
        initial_state=_state(),
        policy_version="policy-v1",
        as_of=_AS_OF,
        generator_version="test-1.0.0",
        generator_seed=0,
    )


# ---------------------------------------------------------------------------
# select_lhc_replay_cases
# ---------------------------------------------------------------------------
def test_select_returns_n_target_cases_drawn_from_the_candidates():
    candidates = generate_lhc_candidate_pool(seed=1, n=60)
    selected = select_lhc_replay_cases(candidates, seed=7)
    assert len(selected) == 20
    candidate_ids = {c.case_id for c in candidates}
    assert all(c.case_id in candidate_ids for c in selected)
    assert len({c.case_id for c in selected}) == 20  # no duplicates


def test_select_is_deterministic_given_seed():
    candidates = generate_lhc_candidate_pool(seed=1, n=60)
    a = select_lhc_replay_cases(candidates, seed=7)
    b = select_lhc_replay_cases(candidates, seed=7)
    assert [c.case_id for c in a] == [c.case_id for c in b]


def test_select_different_seed_can_change_the_selection():
    candidates = generate_lhc_candidate_pool(seed=1, n=60)
    a = select_lhc_replay_cases(candidates, seed=7)
    b = select_lhc_replay_cases(candidates, seed=99)
    assert [c.case_id for c in a] != [c.case_id for c in b]


def test_select_respects_the_per_tenor_stratum_floor():
    candidates = generate_lhc_candidate_pool(seed=3, n=60)  # 20 per stratum
    selected = select_lhc_replay_cases(candidates, seed=1, min_per_tenor_stratum=6)
    counts: dict[str, int] = {}
    for c in selected:
        counts[c.tenor_stratum] = counts.get(c.tenor_stratum, 0) + 1
    assert all(n >= 6 for n in counts.values())
    assert sum(counts.values()) == 20


def test_select_skips_the_extension_phase_when_the_floors_exactly_fill_n_target():
    # 7 per stratum * 3 strata == 21 == n_target exactly -> no slots left
    # over for the cross-stratum maximin extension phase.
    candidates = generate_lhc_candidate_pool(seed=3, n=60)  # 20 per stratum
    selected = select_lhc_replay_cases(candidates, seed=1, n_target=21, min_per_tenor_stratum=7)
    assert len(selected) == 21


def test_select_raises_when_pool_smaller_than_n_target():
    candidates = generate_lhc_candidate_pool(seed=1, n=10)
    with pytest.raises(InsufficientLhcCandidatesError):
        select_lhc_replay_cases(candidates, n_target=20)


def test_select_raises_when_one_stratum_is_short():
    # Only <1m and 1-12m populated -> >12m stratum has zero candidates.
    candidates = [_case(f"c{i}", tenor_stratum="<1m", tau_days=Decimal("10")) for i in range(15)]
    candidates += [
        _case(f"d{i}", tenor_stratum="1-12m", tau_days=Decimal("180")) for i in range(15)
    ]
    with pytest.raises(InsufficientLhcCandidatesError):
        select_lhc_replay_cases(candidates, n_target=20, min_per_tenor_stratum=6)


def test_select_raises_when_floor_times_strata_exceeds_n_target():
    candidates = generate_lhc_candidate_pool(seed=1, n=60)
    with pytest.raises(InsufficientLhcCandidatesError):
        select_lhc_replay_cases(candidates, n_target=10, min_per_tenor_stratum=6)  # 6*3=18 > 10


def test_select_raises_when_a_candidate_has_no_tenor_stratum():
    candidates = generate_lhc_candidate_pool(seed=1, n=25)
    stripped = candidates[0].model_copy(update={"tenor_stratum": None})
    with pytest.raises(InsufficientLhcCandidatesError):
        select_lhc_replay_cases([stripped, *candidates[1:]])


# ---------------------------------------------------------------------------
# build_oracle_eval_fn
# ---------------------------------------------------------------------------
def test_oracle_eval_fn_counts_sign_flips_correctly():
    cases = [
        # Marginal, zero-addon margin = +0.002 -> flips to -0.001 under a 30bp draw.
        _case(
            "flip-1",
            tenor_stratum="1-12m",
            base_funding_curve_rate_ann=Decimal("0.03"),
            trade_alpha_horizon=Decimal("0.032"),
        ),
        # Comfortably positive, zero-addon margin = +0.02 -> stays positive (0.017) under the same draw.
        _case(
            "no-flip-1",
            tenor_stratum="1-12m",
            base_funding_curve_rate_ann=Decimal("0.03"),
            trade_alpha_horizon=Decimal("0.05"),
        ),
        # Already non-positive at zero addons (-0.005) -> excluded from the count regardless of the draw.
        _case(
            "already-negative",
            tenor_stratum="1-12m",
            base_funding_curve_rate_ann=Decimal("0.03"),
            trade_alpha_horizon=Decimal("0.025"),
        ),
    ]
    oracle_eval_fn = build_oracle_eval_fn(cases)
    draw = {
        "lp": Decimal("0.001"),
        "tp": Decimal("0"),
        "cp": Decimal("0.002"),
        "rp": Decimal("0"),
        "cc": Decimal("0"),
    }
    median_addon_bps, sign_flip_count = oracle_eval_fn(draw)
    assert sign_flip_count == 1
    assert median_addon_bps == Decimal("30")  # 0.001+0.002 = 0.003 = 30bp, identical for every case


def test_oracle_eval_fn_zero_draw_never_flips_anything():
    cases = [
        _case("a", tenor_stratum="<1m", trade_alpha_horizon=Decimal("0.05")),
        _case("b", tenor_stratum="1-12m", trade_alpha_horizon=Decimal("0.031")),
    ]
    oracle_eval_fn = build_oracle_eval_fn(cases)
    zero_draw = {
        "lp": Decimal("0"),
        "tp": Decimal("0"),
        "cp": Decimal("0"),
        "rp": Decimal("0"),
        "cc": Decimal("0"),
    }
    median_addon_bps, sign_flip_count = oracle_eval_fn(zero_draw)
    assert sign_flip_count == 0
    assert median_addon_bps == Decimal("0")


def test_oracle_eval_fn_median_is_case_independent_constant():
    cases = [
        _case(
            "a", tenor_stratum="<1m", tau_days=Decimal("10"), trade_alpha_horizon=Decimal("0.05")
        ),
        _case(
            "b", tenor_stratum="1-12m", tau_days=Decimal("180"), trade_alpha_horizon=Decimal("0.06")
        ),
        _case(
            "c", tenor_stratum=">12m", tau_days=Decimal("540"), trade_alpha_horizon=Decimal("0.07")
        ),
    ]
    oracle_eval_fn = build_oracle_eval_fn(cases)
    draw = {
        "lp": Decimal("0.0010"),
        "tp": Decimal("0.0005"),
        "cp": Decimal("0.0025"),
        "rp": Decimal("0.0010"),
        "cc": Decimal("0.0005"),
    }
    median_addon_bps, _ = oracle_eval_fn(draw)
    assert median_addon_bps == Decimal("55")  # (0.0010+0.0005+0.0025+0.0010+0.0005)*10000


# ---------------------------------------------------------------------------
# End-to-end: real generator -> select -> oracle_eval_fn -> analysis.lhc
# ---------------------------------------------------------------------------
def test_selected_replay_cases_produce_oracle_eval_fn_usable_by_build_eligible_pool():
    from gcap.analysis.lhc import build_eligible_pool
    from gcap.config.registry import default_registry

    candidates = generate_lhc_candidate_pool(seed=11, n=80)
    replay_cases = select_lhc_replay_cases(candidates, seed=5)
    assert len(replay_cases) == 20

    oracle_eval_fn = build_oracle_eval_fn(replay_cases)
    bounds_cfg = default_registry().to_decide.lhc_addon_bounds
    param_names, bounds = bounds_cfg.as_param_bounds()

    selected, all_evals = build_eligible_pool(
        seed=2,
        n_dims=len(param_names),
        bounds=bounds,
        param_names=param_names,
        oracle_eval_fn=oracle_eval_fn,
        addon_floor_bps=Decimal("25"),
        kmin=1,  # generous kmin so this sandbox-scale check does not depend on a real freeze value
        n_target=5,
        initial_pool_size=40,
        max_expansions=3,
    )
    assert len(all_evals) > 0
    for draw_eval in selected:
        assert draw_eval.median_addon_bps >= Decimal("25")
        assert draw_eval.sign_flip_count >= 1
