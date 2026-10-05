from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gcap.core.ledger import (
    LedgerAction,
    LedgerEvent,
    LedgerState,
    apply,
    check_feasibility,
    fold,
    is_stale,
)
from gcap.errors import LedgerInfeasibleError, StaleStateError


def _state(**overrides) -> LedgerState:
    base = dict(
        version=0,
        capital_available=Decimal("1000000"),
        liquidity_available=Decimal("1000000"),
        funding_capacity=Decimal("1000000"),
        collateral_available=Decimal("1000000"),
        risk_budget_available=Decimal("1000000"),
        concentration_limits={"sector:tech": Decimal("500000")},
    )
    base.update(overrides)
    return LedgerState(**base)


def _action(**overrides) -> LedgerAction:
    base = dict(
        action_id="a1",
        instrument_id="AAPL",
        state_version_seen=0,
        capital_usage=Decimal("10000"),
        liquidity_usage=Decimal("10000"),
        funding_usage=Decimal("10000"),
        collateral_usage=Decimal("0"),
        risk_budget_usage=Decimal("5000"),
        notional_delta=Decimal("100000"),
    )
    base.update(overrides)
    return LedgerAction(**base)


def test_feasible_action_applies_and_increments_version():
    state = _state()
    action = _action()
    new_state = apply(state, action)
    assert new_state.version == 1
    assert new_state.capital_available == Decimal("990000")
    assert new_state.positions["AAPL"] == Decimal("100000")


def test_infeasible_action_raises_and_state_unaffected():
    state = _state(capital_available=Decimal("5000"))
    action = _action(capital_usage=Decimal("10000"))
    assert check_feasibility(state, action) == ["capital"]
    with pytest.raises(LedgerInfeasibleError):
        apply(state, action)
    assert state.version == 0  # frozen model, never mutated


def test_stale_action_raises_before_touching_capacity():
    state = _state(version=2)
    action = _action(state_version_seen=1)
    assert is_stale(state, action) is True
    with pytest.raises(StaleStateError):
        apply(state, action)


def test_concentration_limit_enforced():
    state = _state()
    action = _action(concentration_key="sector:tech", concentration_usage=Decimal("600000"))
    assert "concentration" in check_feasibility(state, action)


def test_fold_equals_incremental_application():
    state0 = _state()
    a1 = _action(action_id="a1", state_version_seen=0, notional_delta=Decimal("10000"))
    state1 = apply(state0, a1)
    a2 = _action(action_id="a2", instrument_id="MSFT", state_version_seen=1, notional_delta=Decimal("20000"))
    state2 = apply(state1, a2)

    events = [
        LedgerEvent(event_id="e1", action=a1, resulting_version=1),
        LedgerEvent(event_id="e2", action=a2, resulting_version=2),
    ]
    folded = fold(state0, events)
    assert folded == state2


def test_capacity_never_goes_negative_in_a_feasible_chain():
    state = _state(capital_available=Decimal("100"))
    action = _action(capital_usage=Decimal("100"), liquidity_usage=Decimal("0"), funding_usage=Decimal("0"), risk_budget_usage=Decimal("0"))
    new_state = apply(state, action)
    assert new_state.capital_available == Decimal("0")
    # One more unit is now infeasible.
    action2 = _action(action_id="a2", capital_usage=Decimal("1"), state_version_seen=1, liquidity_usage=Decimal("0"), funding_usage=Decimal("0"), risk_budget_usage=Decimal("0"))
    with pytest.raises(LedgerInfeasibleError):
        apply(new_state, action2)


usage = st.decimals(min_value="0", max_value="1000", places=2, allow_nan=False, allow_infinity=False)


@given(cap=usage, usg=usage)
@settings(max_examples=50)
def test_feasibility_never_allows_negative_capital(cap, usg):
    state = _state(capital_available=cap)
    action = _action(capital_usage=usg, liquidity_usage=Decimal("0"), funding_usage=Decimal("0"), risk_budget_usage=Decimal("0"))
    violations = check_feasibility(state, action)
    if usg > cap:
        assert "capital" in violations
    else:
        new_state = apply(state, action)
        assert new_state.capital_available >= 0
