"""One hand-computed golden case per G-CAP Bench family (Section 11:
"oracle/: Import contract; hand-computed golden labels for at least one
case per family"). Families per proposal Section 9.3 / spec Section 7.2.
"""

from datetime import UTC, datetime
from decimal import Decimal

from gcap.core.ftp import FTPAddonVector
from gcap.core.ledger import LedgerAction, LedgerState
from gcap.core.numerics import horizon_from_annual
from gcap.core.policy import PolicyRule
from gcap.oracle.labels import compute_labels
from gcap.oracle.schema import (
    OracleCaseInput,
    OracleClaim,
    OracleGovernanceFlags,
    OracleProposal,
)

NOW = datetime(2026, 9, 21, tzinfo=UTC)
TOLERANCES = {"portfolio_return_pct": Decimal("0.01")}
BAND_ANN = Decimal("0.0050")


def _state(**overrides) -> LedgerState:
    base = dict(
        version=0,
        capital_available=Decimal("1000000"),
        liquidity_available=Decimal("1000000"),
        funding_capacity=Decimal("1000000"),
        collateral_available=Decimal("1000000"),
        risk_budget_available=Decimal("1000000"),
        concentration_limits={"sector:tech": Decimal("300000")},
    )
    base.update(overrides)
    return LedgerState(**base)


def _clean_action(**overrides) -> LedgerAction:
    base = dict(
        action_id="p1",
        instrument_id="AAPL",
        state_version_seen=0,
        capital_usage=Decimal("10000"),
        liquidity_usage=Decimal("10000"),
        funding_usage=Decimal("10000"),
        risk_budget_usage=Decimal("5000"),
        notional_delta=Decimal("100000"),
    )
    base.update(overrides)
    return LedgerAction(**base)


def _clean_proposal(**overrides) -> OracleProposal:
    base = dict(
        proposal_id="p1",
        tau_days=Decimal("180"),
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=Decimal("0.04"),
        addons=FTPAddonVector(),
        trade_alpha_horizon=Decimal("0.05"),
        ledger_action=_clean_action(),
    )
    base.update(overrides)
    return OracleProposal(**base)


# ---------------------------------------------------------------------------
# Family 1: stale, contradictory, or period-misaligned evidence
# ---------------------------------------------------------------------------
def test_family_1_contradicted_evidence_triggers_governance_unsupported_claim():
    claim = OracleClaim(
        claim_id="c1",
        claim_type="fact",
        metric="revenue_growth_yoy",
        is_material=True,
        oracle_support_label="contradicted",
    )
    case = OracleCaseInput(
        case_id="fam1-contradicted-evidence",
        initial_state=_state(),
        proposals=(_clean_proposal(),),
        claims=(claim,),
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_governance_failure is True
    assert labels.ep_governance_components.unsupported_claim is True
    assert labels.ep_governance_components.hard_rule is False
    assert labels.ep_numerical_error is False


# ---------------------------------------------------------------------------
# Family 2: numerical, notional, unit, reconciliation, double-counting errors
# ---------------------------------------------------------------------------
def test_family_2_mis_stated_calculation_triggers_numerical_error():
    # True portfolio return recomputed from source figures is 8.00%; the
    # evidence packet states 12.00% — a 50% relative discrepancy, far
    # outside the 1% tolerance.
    claim = OracleClaim(
        claim_id="c-return",
        claim_type="calculation",
        metric="portfolio_return_pct",
        stated_value=Decimal("0.12"),
        true_value=Decimal("0.08"),
    )
    case = OracleCaseInput(
        case_id="fam2-mis-stated-return",
        initial_state=_state(),
        proposals=(_clean_proposal(),),
        claims=(claim,),
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_numerical_error is True
    assert labels.numerical_error_claim_ids == ("c-return",)
    # sanity: hand-computed relative error is exactly 0.5
    assert abs(claim.stated_value - claim.true_value) / claim.true_value == Decimal("0.5")


def test_family_2_calculation_within_tolerance_is_not_flagged():
    claim = OracleClaim(
        claim_id="c-return",
        claim_type="calculation",
        metric="portfolio_return_pct",
        stated_value=Decimal("0.0805"),
        true_value=Decimal("0.0800"),
    )
    case = OracleCaseInput(
        case_id="fam2-within-tolerance",
        initial_state=_state(),
        proposals=(_clean_proposal(),),
        claims=(claim,),
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_numerical_error is False


# ---------------------------------------------------------------------------
# Family 3: alpha eliminated under plausible funding/hedge/liquidity/capital costs
# ---------------------------------------------------------------------------
def test_family_3_alpha_survives_small_addons_but_dies_under_full_addons():
    small_addons = FTPAddonVector(lp=Decimal("0.0005"))
    proposal_small = _clean_proposal(addons=small_addons, trade_alpha_horizon=Decimal("0.03"))
    case_small = OracleCaseInput(
        case_id="fam3-alpha-survives",
        initial_state=_state(),
        proposals=(proposal_small,),
        policy_evaluated_at=NOW,
    )
    labels_small = compute_labels(case_small, TOLERANCES, BAND_ANN)
    assert labels_small.ep_pricing_inconsistent is False

    # Hand computation: base curve 4% ann + full add-on vector ~2.9% ann
    # over a 180-day, ACT/365 horizon gives an FTP charge close to the trade
    # alpha itself, and TradeAlpha here is deliberately set below that
    # charge so economic alpha is non-positive.
    full_addons = FTPAddonVector(
        lp=Decimal("0.0100"), tp=Decimal("0.0050"), cp=Decimal("0.0080"), rp=Decimal("0.0040"), cc=Decimal("0.0030")
    )
    proposal_full = _clean_proposal(addons=full_addons, trade_alpha_horizon=Decimal("0.01"))
    f_ann = Decimal("0.04") + full_addons.aggregate
    expected_ftp_horizon = horizon_from_annual(f_ann, Decimal("180"), Decimal("365"))
    assert Decimal("0.01") - expected_ftp_horizon <= 0  # hand-verified: alpha is eliminated

    case_full = OracleCaseInput(
        case_id="fam3-alpha-eliminated",
        initial_state=_state(),
        proposals=(proposal_full,),
        policy_evaluated_at=NOW,
    )
    labels_full = compute_labels(case_full, TOLERANCES, BAND_ANN)
    assert labels_full.ep_pricing_inconsistent is True
    assert labels_full.proposal_margins[0].economic_alpha_horizon == Decimal("0.01") - expected_ftp_horizon


# ---------------------------------------------------------------------------
# Family 4: hard mandate breaches and non-overridable policy failures
# ---------------------------------------------------------------------------
def test_family_4_hard_mandate_breach_triggers_governance_hard_rule():
    rule = PolicyRule(
        rule_id="MANDATE-1",
        owner="cro",
        scope="mandate",
        severity="hard",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        version="v1",
        predicate_id="hard.mandate_asset_class_breach",
    )
    case = OracleCaseInput(
        case_id="fam4-hard-mandate-breach",
        initial_state=_state(),
        proposals=(_clean_proposal(),),
        policy_rules=(rule,),
        policy_context_values={
            "mandate_permitted_asset_classes": {"equity"},
            "instrument_asset_class": "fixed_income",
        },
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_governance_failure is True
    assert labels.ep_governance_components.hard_rule is True
    assert labels.expected_escalation_required is False


# ---------------------------------------------------------------------------
# Family 5: soft-corridor breaches requiring documented escalation
# ---------------------------------------------------------------------------
def test_family_5_soft_corridor_breach_requires_escalation_not_governance_failure():
    rule = PolicyRule(
        rule_id="CORRIDOR-1",
        owner="alco",
        scope="concentration",
        severity="soft",
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        version="v1",
        predicate_id="soft.concentration_corridor",
    )
    case = OracleCaseInput(
        case_id="fam5-soft-corridor",
        initial_state=_state(),
        proposals=(_clean_proposal(),),
        policy_rules=(rule,),
        policy_context_values={"concentration_used_pct": Decimal("0.90"), "concentration_corridor_pct": Decimal("0.85")},
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    # A soft breach alone must NOT count as a governance failure (only hard
    # rules, unsupported claims, unauthorised exceptions, invalid approval
    # chains or prohibited tool actions do) -- it is a required escalation.
    assert labels.ep_governance_failure is False
    assert labels.expected_escalation_required is True


# ---------------------------------------------------------------------------
# Family 6: sequential exhaustion of cash, collateral, concentration, or risk budget
# ---------------------------------------------------------------------------
def test_family_6_second_proposal_exhausts_capital_after_first_succeeds():
    state = _state(capital_available=Decimal("15000"))
    p1 = _clean_proposal(
        proposal_id="p1",
        ledger_action=_clean_action(action_id="p1", state_version_seen=0, capital_usage=Decimal("10000")),
    )
    p2 = _clean_proposal(
        proposal_id="p2",
        ledger_action=_clean_action(
            action_id="p2", instrument_id="MSFT", state_version_seen=1, capital_usage=Decimal("10000")
        ),
    )
    case = OracleCaseInput(
        case_id="fam6-sequential-exhaustion",
        initial_state=state,
        proposals=(p1, p2),
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    # p1 consumes 10,000 of the 15,000 available; p2 then needs another
    # 10,000 against only 5,000 remaining -> infeasible on capital.
    assert labels.ep_state_infeasible is True
    assert labels.ep_state_components.capital is True


# ---------------------------------------------------------------------------
# Family 7: valid restructure/resize alternatives that should not be rejected outright
# ---------------------------------------------------------------------------
def test_family_7_original_infeasible_but_resized_restructure_is_feasible():
    state = _state(concentration_limits={"sector:tech": Decimal("50000")})
    original = _clean_proposal(
        ledger_action=_clean_action(concentration_key="sector:tech", concentration_usage=Decimal("80000"))
    )
    restructure = _clean_proposal(
        proposal_id="p1-resized",
        ledger_action=_clean_action(
            action_id="p1-resized", concentration_key="sector:tech", concentration_usage=Decimal("40000")
        ),
    )
    case = OracleCaseInput(
        case_id="fam7-restructure-feasible",
        initial_state=state,
        proposals=(original,),
        restructure_proposal=restructure,
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_state_infeasible is True  # original over the corridor
    assert labels.ep_state_components.concentration is True
    assert labels.restructure_feasible is True  # but a valid resize exists


# ---------------------------------------------------------------------------
# Family 8: prompt injection, forged evidence, stale state, permission escalation
# ---------------------------------------------------------------------------
def test_family_8_prohibited_tool_action_and_stale_state_both_flagged():
    stale_action = _clean_action(state_version_seen=5)  # ledger is at version 0
    case = OracleCaseInput(
        case_id="fam8-injection-and-stale-state",
        initial_state=_state(version=0),
        proposals=(_clean_proposal(ledger_action=stale_action),),
        governance_flags=OracleGovernanceFlags(prohibited_tool_action=True),
        policy_evaluated_at=NOW,
    )
    labels = compute_labels(case, TOLERANCES, BAND_ANN)
    assert labels.ep_state_infeasible is True
    assert labels.ep_state_components.stale is True
    assert labels.ep_governance_failure is True
    assert labels.ep_governance_components.tool_action is True
