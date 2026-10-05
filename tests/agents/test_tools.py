from __future__ import annotations

from decimal import Decimal

from gcap.agents.tools import (
    ToolCallCache,
    ftp_tool,
    ledger_feasibility_tool,
    portfolio_tool,
    reconciliation_tool,
)
from gcap.core.ftp import FTPAddonVector, FTPInputs
from gcap.core.ledger import LedgerAction, LedgerState
from gcap.core.portfolio import MarginalContributionInputs


def test_reconciliation_tool_flags_material_discrepancy():
    result = reconciliation_tool(
        claim_id="c1",
        metric="portfolio_return_pct",
        stated_value=Decimal("0.15"),
        authoritative_value=Decimal("0.10"),
        tolerance=Decimal("0.005"),
    )
    assert result.outputs_summary["flagged"] is True
    assert result.tool_name == "reconciliation"


def test_reconciliation_tool_does_not_flag_within_tolerance():
    result = reconciliation_tool(
        claim_id="c1",
        metric="portfolio_return_pct",
        stated_value=Decimal("0.1001"),
        authoritative_value=Decimal("0.1000"),
        tolerance=Decimal("0.005"),
    )
    assert result.outputs_summary["flagged"] is False


def test_ftp_tool_matches_core_compute_ftp():
    inputs = FTPInputs(
        tau_days=Decimal("180"),
        day_count_basis=Decimal("365"),
        base_funding_curve_rate_ann=Decimal("0.03"),
        addons=FTPAddonVector(lp=Decimal("0.001")),
        trade_alpha_horizon=Decimal("0.02"),
    )
    result, output = ftp_tool(inputs)
    assert result.tool_name == "ftp"
    assert Decimal(result.outputs_summary["economic_alpha_horizon"]) == output.economic_alpha_horizon


def test_ledger_feasibility_tool_reports_staleness_and_violations():
    state = LedgerState(
        capital_available=Decimal("1000"),
        liquidity_available=Decimal("1000"),
        funding_capacity=Decimal("1000"),
        collateral_available=Decimal("1000"),
        risk_budget_available=Decimal("1000"),
    )
    action = LedgerAction(
        action_id="a1", instrument_id="i1", state_version_seen=5, capital_usage=Decimal("2000")
    )
    result, stale, violations = ledger_feasibility_tool(state, action)
    assert stale is True
    assert "capital" in violations
    assert result.outputs_summary["stale"] is True


def test_portfolio_tool_matches_core_marginal_contribution():
    inputs = MarginalContributionInputs(
        delta_expected_return=Decimal("0.05"),
        delta_risk=Decimal("0.02"),
        delta_concentration=Decimal("0.01"),
        lam=Decimal("0.5"),
        gamma=Decimal("0.25"),
    )
    result, output = portfolio_tool(inputs)
    assert Decimal(result.outputs_summary["delta_u"]) == output.delta_u


def test_tool_call_cache_avoids_recomputation():
    cache: ToolCallCache = ToolCallCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return reconciliation_tool(
            claim_id="c1",
            metric="m",
            stated_value=Decimal("1"),
            authoritative_value=Decimal("1"),
            tolerance=Decimal("0.01"),
        )

    r1, hit1 = cache.get_or_compute("hash-1", compute)
    r2, hit2 = cache.get_or_compute("hash-1", compute)
    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1
    assert r1 is r2
