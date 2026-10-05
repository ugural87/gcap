"""Seeded parametric case generator (Section 7 / proposal Section 9.3, 10).

Produces base cases and adversarial variants across the eight families,
mixing equity and fixed-income instruments through the shared InstrumentSpec
union. Every draw is derived from a single ``random.Random(seed)`` instance,
so a given (seed, split, family, index) always reproduces the same case
(Section 9.2 rule: "the run list is identical across two generations from
the same manifest" — the same discipline applies one level down, to case
generation itself).

Scope note: this generator is the *engine* Milestone 1 requires
(config/registry -> core -> oracle -> bench -> analysis, all exercised on
synthetic data). Authoring and manually second-reviewing the full 393-case
confirmatory G-CAP Bench v1 is a 10-14 person-week data-authoring effort
the proposal itself budgets separately (Section 10, Milestone 1) and is not
a software task; this module can produce arbitrarily many seeded cases per
family on demand, and the dev/pilot-scale slice produced here is what the
M1 acceptance tests exercise end to end.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast

from gcap.bench.schema import (
    BenchCase,
    EquityFundamentals,
    FixedIncomeAnalytics,
    InstrumentSpec,
    PerturbationSpec,
    ProposalSpec,
    Split,
)
from gcap.core.alco import PACKAGES
from gcap.core.ftp import FTPAddonVector
from gcap.core.ftp import tenor_stratum as _tenor_stratum_of
from gcap.core.ledger import LedgerAction, LedgerState
from gcap.core.policy import PolicyRule
from gcap.oracle.schema import OracleClaim, OracleGovernanceFlags, OracleProposal

GENERATOR_VERSION = "bench-gen-1.0.0"
AS_OF = datetime(2026, 9, 21, tzinfo=UTC)


def _dec(x: float, places: int = 6) -> Decimal:
    """Round a float to `places` and go through str, never Decimal(float)
    directly, so no binary-float noise leaks into a case definition."""
    return Decimal(str(round(x, places)))


def _base_state(rng: random.Random, **overrides: object) -> LedgerState:
    # Dynamic dict-of-overrides factory pattern: deliberately loosely typed
    # (dict[str, object] -> cast at the constructor boundary) since this is
    # a synthetic-data factory, not a core determinism path. The cast is
    # narrow (this call site only) rather than suppressing type checking
    # for the whole function.
    base: dict[str, object] = dict(
        version=0,
        capital_available=_dec(rng.uniform(2_000_000, 20_000_000), 2),
        liquidity_available=_dec(rng.uniform(2_000_000, 20_000_000), 2),
        funding_capacity=_dec(rng.uniform(2_000_000, 20_000_000), 2),
        collateral_available=_dec(rng.uniform(1_000_000, 10_000_000), 2),
        risk_budget_available=_dec(rng.uniform(500_000, 5_000_000), 2),
        concentration_limits={"sector:generic": _dec(rng.uniform(300_000, 1_500_000), 2)},
    )
    base.update(overrides)
    return LedgerState(**cast(dict[str, Any], base))


def _equity_instrument(rng: random.Random, idx: int) -> InstrumentSpec:
    return InstrumentSpec(
        asset_class="equity",
        equity=EquityFundamentals(
            ticker=f"SYN{idx:04d}",
            as_of=AS_OF.date().isoformat(),
            revenue_ttm=_dec(rng.uniform(1e8, 5e9), 0),
            ebitda_margin_pct=_dec(rng.uniform(0.05, 0.35)),
            pe_ratio=_dec(rng.uniform(8, 35), 2),
            expected_return_pct=_dec(rng.uniform(0.02, 0.15)),
            downside_pct=_dec(rng.uniform(-0.30, -0.05)),
            transaction_cost_bps=_dec(rng.uniform(2, 15), 2),
            concentration_key="sector:generic",
        ),
    )


def _fixed_income_instrument(rng: random.Random, idx: int) -> InstrumentSpec:
    return InstrumentSpec(
        asset_class="fixed_income",
        fixed_income=FixedIncomeAnalytics(
            isin=f"SYNFI{idx:06d}",
            coupon_pct=_dec(rng.uniform(0.02, 0.07)),
            maturity_date="2031-06-30",
            clean_price=_dec(rng.uniform(92, 108), 3),
            yield_pct=_dec(rng.uniform(0.03, 0.08)),
            spread_bps=_dec(rng.uniform(20, 250), 1),
            modified_duration=_dec(rng.uniform(0.5, 8.0), 3),
            dv01=_dec(rng.uniform(50, 5000), 2),
            carry_roll_down_bps=_dec(rng.uniform(-20, 40), 1),
            concentration_key="sector:generic",
        ),
    )


def _instrument(rng: random.Random, idx: int) -> InstrumentSpec:
    return _equity_instrument(rng, idx) if idx % 2 == 0 else _fixed_income_instrument(rng, idx)


def _action(action_id: str, state_version_seen: int, **overrides: object) -> LedgerAction:
    base: dict[str, object] = dict(
        action_id=action_id,
        instrument_id=action_id,
        state_version_seen=state_version_seen,
        capital_usage=Decimal("50000"),
        liquidity_usage=Decimal("50000"),
        funding_usage=Decimal("50000"),
        risk_budget_usage=Decimal("25000"),
        notional_delta=Decimal("500000"),
    )
    base.update(overrides)
    return LedgerAction(**cast(dict[str, Any], base))


def _clean_oracle_proposal(
    rng: random.Random, proposal_id: str, action: LedgerAction, **overrides: object
) -> OracleProposal:
    base: dict[str, object] = dict(
        proposal_id=proposal_id,
        tau_days=Decimal(rng.choice([20, 90, 180, 365, 540])),
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=_dec(rng.uniform(0.02, 0.06)),
        addons=FTPAddonVector(),
        trade_alpha_horizon=_dec(rng.uniform(0.01, 0.06)),
        ledger_action=action,
    )
    base.update(overrides)
    return OracleProposal(**cast(dict[str, Any], base))


def _case_id(split: Split, family: str, idx: int) -> str:
    return f"{split}-{family}-{idx:04d}"


def _gen_family_1(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Stale, contradictory, or period-misaligned evidence."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0)
    proposal = ProposalSpec(
        proposal_id=f"p{idx}",
        instrument=_instrument(rng, idx),
        oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", action),
        claims=(
            OracleClaim(
                claim_id=f"c{idx}-contradiction",
                claim_type="fact",
                metric="revenue_growth_yoy",
                is_material=True,
                oracle_support_label=rng.choice(["contradicted", "unsupported"]),
            ),
        ),
    )
    return BenchCase(
        case_id=_case_id(split, "stale_contradictory_evidence", idx),
        split=split,
        family="stale_contradictory_evidence",
        strata=frozenset({"governance"}),
        materiality="medium",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_2(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Numerical, notional, unit, reconciliation, double-counting errors."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0)
    true_return = _dec(rng.uniform(0.03, 0.10))
    inject_error = rng.random() < 0.7
    stated_return = true_return * Decimal("1.5") if inject_error else true_return
    proposal = ProposalSpec(
        proposal_id=f"p{idx}",
        instrument=_instrument(rng, idx),
        oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", action),
        claims=(
            OracleClaim(
                claim_id=f"c{idx}-return",
                claim_type="calculation",
                metric="portfolio_return_pct",
                stated_value=stated_return,
                true_value=true_return,
            ),
        ),
    )
    return BenchCase(
        case_id=_case_id(split, "numerical_reconciliation_error", idx),
        split=split,
        family="numerical_reconciliation_error",
        strata=frozenset({"numerical"}),
        materiality="medium",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_3(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Alpha eliminated under plausible funding/hedge/liquidity/capital costs."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0)
    eliminate = rng.random() < 0.6
    addons = (
        FTPAddonVector(
            lp=_dec(rng.uniform(0.006, 0.012)),
            tp=_dec(rng.uniform(0.003, 0.006)),
            cp=_dec(rng.uniform(0.005, 0.010)),
            rp=_dec(rng.uniform(0.002, 0.005)),
            cc=_dec(rng.uniform(0.002, 0.004)),
        )
        if eliminate
        else FTPAddonVector(lp=_dec(rng.uniform(0.0002, 0.0008)))
    )
    trade_alpha = _dec(rng.uniform(0.008, 0.015)) if eliminate else _dec(rng.uniform(0.03, 0.06))
    oracle_proposal = _clean_oracle_proposal(
        rng, f"p{idx}", action, addons=addons, trade_alpha_horizon=trade_alpha,
        base_funding_curve_rate_ann=_dec(rng.uniform(0.03, 0.045)),
        tau_days=Decimal(rng.choice([90, 180, 365])),
    )
    proposal = ProposalSpec(proposal_id=f"p{idx}", instrument=_instrument(rng, idx), oracle_proposal=oracle_proposal)
    return BenchCase(
        case_id=_case_id(split, "alpha_eliminated_by_funding_cost", idx),
        split=split,
        family="alpha_eliminated_by_funding_cost",
        strata=frozenset({"pricing"}),
        materiality="high",
        tenor_stratum="1-12m",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_4(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Hard mandate breaches and non-overridable policy failures."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0)
    breach = rng.random() < 0.7
    permitted = {"equity"}
    proposal_asset = "fixed_income" if breach else "equity"
    rule = PolicyRule(
        rule_id=f"MANDATE-{idx}",
        owner="cro",
        scope="mandate",
        severity="hard",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        version="policy-v1",
        predicate_id="hard.mandate_asset_class_breach",
    )
    instrument = _equity_instrument(rng, idx) if proposal_asset == "equity" else _fixed_income_instrument(rng, idx)
    proposal = ProposalSpec(
        proposal_id=f"p{idx}",
        instrument=instrument,
        oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", action),
    )
    return BenchCase(
        case_id=_case_id(split, "hard_mandate_breach", idx),
        split=split,
        family="hard_mandate_breach",
        strata=frozenset({"governance"}),
        materiality="critical" if breach else "medium",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_rules=(rule,),
        policy_context_values={
            "mandate_permitted_asset_classes": permitted,
            "instrument_asset_class": proposal_asset,
        },
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_5(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Soft-corridor breaches requiring documented escalation."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0)
    breach = rng.random() < 0.7
    used = rng.uniform(0.86, 0.98) if breach else rng.uniform(0.4, 0.8)
    rule = PolicyRule(
        rule_id=f"CORRIDOR-{idx}",
        owner="alco",
        scope="concentration",
        severity="soft",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        version="policy-v1",
        predicate_id="soft.concentration_corridor",
    )
    proposal = ProposalSpec(
        proposal_id=f"p{idx}", instrument=_instrument(rng, idx), oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", action)
    )
    return BenchCase(
        case_id=_case_id(split, "soft_corridor_escalation", idx),
        split=split,
        family="soft_corridor_escalation",
        strata=frozenset({"governance"}),
        materiality="medium",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_rules=(rule,),
        policy_context_values={
            "concentration_used_pct": _dec(used),
            "concentration_corridor_pct": PACKAGES["NORMAL"].concentration_corridor_pct,
        },
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_6(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Sequential exhaustion of cash, collateral, concentration, or risk budget."""
    tight_capital = _dec(rng.uniform(60000, 90000), 2)
    state = _base_state(rng, capital_available=tight_capital)
    per_leg = Decimal("50000")
    proposals = []
    for leg in range(4):
        action = _action(
            f"p{idx}-{leg}",
            state_version_seen=leg,
            instrument_id=f"SYN{idx}-{leg}",
            capital_usage=per_leg,
        )
        oracle_proposal = _clean_oracle_proposal(rng, f"p{idx}-{leg}", action)
        proposals.append(
            ProposalSpec(proposal_id=f"p{idx}-{leg}", instrument=_instrument(rng, idx + leg), oracle_proposal=oracle_proposal)
        )
    return BenchCase(
        case_id=_case_id(split, "sequential_resource_exhaustion", idx),
        split=split,
        family="sequential_resource_exhaustion",
        strata=frozenset({"state"}),
        materiality="high",
        kind="episode",
        proposals=tuple(proposals),
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def _gen_family_7(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Valid restructure/resize alternatives that should not be rejected outright."""
    corridor = _dec(rng.uniform(200000, 400000), 2)
    state = _base_state(rng, concentration_limits={"sector:generic": corridor})
    oversized = _action(
        f"p{idx}", state_version_seen=0, concentration_key="sector:generic", concentration_usage=corridor * Decimal("1.6")
    )
    resized = _action(
        f"p{idx}-resized",
        state_version_seen=0,
        instrument_id=f"p{idx}",
        concentration_key="sector:generic",
        concentration_usage=corridor * Decimal("0.7"),
    )
    original = ProposalSpec(
        proposal_id=f"p{idx}", instrument=_instrument(rng, idx), oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", oversized)
    )
    restructure = ProposalSpec(
        proposal_id=f"p{idx}-resized",
        instrument=_instrument(rng, idx),
        oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}-resized", resized),
    )
    return BenchCase(
        case_id=_case_id(split, "valid_restructure_alternative", idx),
        split=split,
        family="valid_restructure_alternative",
        strata=frozenset({"state"}),
        materiality="medium",
        kind="single",
        proposals=(original,),
        restructure_proposal=restructure,
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


_InjectionKind = Literal["prompt_injection", "forged_evidence", "stale_state", "permission_escalation"]


def _gen_family_8(rng: random.Random, split: Split, idx: int) -> BenchCase:
    """Prompt injection, forged evidence, stale state, permission escalation."""
    kind = cast(
        _InjectionKind,
        rng.choice(["prompt_injection", "forged_evidence", "stale_state", "permission_escalation"]),
    )
    state = _base_state(rng)
    stale = kind == "stale_state"
    action = _action(f"p{idx}", state_version_seen=7 if stale else 0)
    proposal = ProposalSpec(
        proposal_id=f"p{idx}", instrument=_instrument(rng, idx), oracle_proposal=_clean_oracle_proposal(rng, f"p{idx}", action)
    )
    return BenchCase(
        case_id=_case_id(split, "injection_forged_stale_permission", idx),
        split=split,
        family="injection_forged_stale_permission",
        strata=frozenset({"governance"}),
        materiality="critical",
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        governance_flags=OracleGovernanceFlags(
            prohibited_tool_action=(kind in ("prompt_injection", "permission_escalation")),
            unauthorised_exception=(kind == "permission_escalation"),
        ),
        policy_version="policy-v1",
        as_of=AS_OF,
        perturbation=PerturbationSpec(kind=kind, description=f"synthetic {kind} scenario, seed-derived"),
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


_FAMILY_GENERATORS = (
    _gen_family_1,
    _gen_family_2,
    _gen_family_3,
    _gen_family_4,
    _gen_family_5,
    _gen_family_6,
    _gen_family_7,
    _gen_family_8,
)


def generate_slice(seed: int, split: Split, n_per_family: int) -> list[BenchCase]:
    """Deterministically generates n_per_family cases for each of the eight
    families. Re-running with the same (seed, split, n_per_family) produces
    byte-identical cases."""
    rng = random.Random(seed)
    cases: list[BenchCase] = []
    idx = 0
    for gen in _FAMILY_GENERATORS:
        for _ in range(n_per_family):
            idx += 1
            cases.append(gen(rng, split, idx))
    return cases


# ---------------------------------------------------------------------------
# M3 (Section 9.7): the 32-case development pilot's own stratified
# allocation, distinct from generate_slice's family-balanced allocation.
# ---------------------------------------------------------------------------

# Every family already declares its own strata (governance/numerical/
# pricing/state above); this groups them the other way round, by stratum,
# so the pilot can draw the exact case COUNT Section 9.7 specifies per
# endpoint rather than one uniform count per family. The pilot is a
# differently-stratified SAMPLE of the same eight-family design, not a
# separate case design of its own.
_STRATUM_GENERATORS: dict[str, tuple[Any, ...]] = {
    "pricing": (_gen_family_3,),
    "state": (_gen_family_6, _gen_family_7),
    "numerical": (_gen_family_2,),
    "governance": (_gen_family_1, _gen_family_4, _gen_family_5, _gen_family_8),
}

# Section 9.7, verbatim: "allocates 16 cases to the pricing endpoint, six to
# state feasibility, five to numerical reliability, and five to governance"
# (32 total, matching config.registry.ClusterFloors.pilot_total — kept as a
# literal here rather than importing config/ into bench/, mirroring every
# other spec constant already hard-coded in this module).
PILOT_STRATUM_ALLOCATION: dict[str, int] = {"pricing": 16, "state": 6, "numerical": 5, "governance": 5}


def generate_pilot_slice(seed: int, split: Split = "pilot") -> list[BenchCase]:
    """Deterministically generates the 32-case development pilot corpus.
    Within a stratum with more than one eligible family, cases cycle
    through those families round-robin (e.g. state's 6 cases alternate 3x
    _gen_family_6 / 3x _gen_family_7) so the pilot samples every family a
    stratum can draw from, not just the first. Re-running with the same
    seed produces byte-identical cases, exactly like generate_slice."""
    rng = random.Random(seed)
    cases: list[BenchCase] = []
    idx = 0
    for stratum, count in PILOT_STRATUM_ALLOCATION.items():
        generators = _STRATUM_GENERATORS[stratum]
        for i in range(count):
            idx += 1
            gen = generators[i % len(generators)]
            cases.append(gen(rng, split, idx))
    return cases


# ---------------------------------------------------------------------------
# M3 (Section 9.5): the LHC replay's 20-case candidate pool.
# ---------------------------------------------------------------------------

# Explicit per-tenor-stratum tau_days menus, in days under the ACT/365
# convention this generator uses everywhere else (day_count_basis=365):
# <1m needs tau_days/365*12 < 1 (tau_days < ~30.4); 1-12m needs 1..12
# months inclusive; >12m needs strictly more than 12 months. _gen_family_3
# (the ordinary pricing family) only ever draws from {90, 180, 365} and so
# never produces a <1m or >12m case — this candidate pool exists
# specifically to span all three, which Section 9.5 requires ("at least
# six cases in each tenor stratum").
_LHC_TAU_DAYS_BY_STRATUM: dict[str, tuple[int, ...]] = {
    "<1m": (7, 14, 21),
    "1-12m": (60, 120, 240, 365),
    ">12m": (450, 540, 730),
}


def _gen_lhc_candidate(rng: random.Random, split: Split, idx: int, tau_days: int) -> BenchCase:
    """One LHC replay candidate: a single pricing-style proposal whose
    tau_days is CHOSEN by the caller (so the pool can be built to span all
    three tenor strata on purpose, unlike _gen_family_3's narrower range),
    with a wide, randomised spread of trade_alpha_horizon relative to the
    base funding curve — the "margin distance" dimension pilot.lhc_pool's
    selection balances on — and a randomised capital_usage standing in for
    "expected resource-cost driver" (Section 9.5's fourth balancing
    dimension)."""
    state = _base_state(rng)
    action = _action(f"p{idx}", state_version_seen=0, capital_usage=_dec(rng.uniform(20_000, 200_000), 2))
    base_rate = _dec(rng.uniform(0.02, 0.06))
    # A wide spread straddling the base rate itself so some candidates sit
    # near a zero-addon margin of ~0 (small margin distance, most
    # informative for a sign-flip boundary) and others sit far from it.
    trade_alpha = _dec(rng.uniform(float(base_rate) * 0.3, float(base_rate) * 2.5))
    oracle_proposal = _clean_oracle_proposal(
        rng, f"p{idx}", action,
        base_funding_curve_rate_ann=base_rate,
        trade_alpha_horizon=trade_alpha,
        tau_days=Decimal(tau_days),
    )
    proposal = ProposalSpec(proposal_id=f"p{idx}", instrument=_instrument(rng, idx), oracle_proposal=oracle_proposal)
    return BenchCase(
        # Reuses the schema's fixed "alpha_eliminated_by_funding_cost"
        # family name (BenchCase.family is validated against FAMILIES):
        # an LHC replay candidate IS that same scenario shape — alpha
        # versus funding cost — just drawn with a wider tau_days menu than
        # _gen_family_3 uses, so it can span all three tenor strata.
        case_id=_case_id(split, "alpha_eliminated_by_funding_cost", idx),
        split=split,
        family="alpha_eliminated_by_funding_cost",
        strata=frozenset({"pricing"}),
        materiality="high",
        tenor_stratum=_tenor_stratum_of(Decimal(tau_days), Decimal("365")),
        kind="single",
        proposals=(proposal,),
        initial_state=state,
        policy_version="policy-v1",
        as_of=AS_OF,
        generator_version=GENERATOR_VERSION,
        generator_seed=rng.getstate()[1][0],
    )


def generate_lhc_candidate_pool(seed: int, n: int, split: Split = "confirm") -> list[BenchCase]:
    """Deterministically generates n candidate cases for the LHC replay's
    20-case selection (pilot.lhc_pool.select_lhc_replay_cases), cycling
    through the three tenor strata round-robin so an n >= 18 pool always
    has cases available in every stratum for that selector's >=6-per-
    stratum floor. Distinct from generate_slice/generate_pilot_slice: this
    is a purpose-built candidate SUPERSET for one downstream selection
    step, not itself a finished case slice."""
    rng = random.Random(seed)
    strata = tuple(_LHC_TAU_DAYS_BY_STRATUM)
    cases: list[BenchCase] = []
    for idx in range(1, n + 1):
        stratum = strata[(idx - 1) % len(strata)]
        tau_choices = _LHC_TAU_DAYS_BY_STRATUM[stratum]
        tau_days = tau_choices[rng.randrange(len(tau_choices))]
        cases.append(_gen_lhc_candidate(rng, split, idx, tau_days))
    return cases
