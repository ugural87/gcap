from decimal import Decimal

from gcap.analysis.cost_frontier import breakeven_loss, loss_sweep


def test_breakeven_loss_matches_closed_form():
    delta_c = Decimal("500")
    delta_p_crit = Decimal("0.02")
    assert breakeven_loss(delta_c, delta_p_crit) == Decimal("25000")


def test_breakeven_loss_rejects_zero_effect():
    import pytest

    with pytest.raises(ValueError):
        breakeven_loss(Decimal("500"), Decimal("0"))


def test_loss_sweep_spans_the_prespecified_range():
    points = loss_sweep(Decimal("500"), Decimal("0.02"), n_points=41)
    assert len(points) == 41
    assert points[0].loss == Decimal(points[0].loss)  # sanity: valid Decimal
    assert float(points[0].loss) == 1000.0
    assert abs(float(points[-1].loss) - 1e7) / 1e7 < 1e-6


def test_net_benefit_crosses_zero_near_breakeven():
    delta_c = Decimal("500")
    delta_p_crit = Decimal("0.02")
    l_star = breakeven_loss(delta_c, delta_p_crit)
    points = loss_sweep(delta_c, delta_p_crit, low=Decimal("1e3"), high=Decimal("1e7"), n_points=200)
    below = [p for p in points if p.loss < l_star]
    above = [p for p in points if p.loss > l_star]
    assert all(p.net_benefit < 0 for p in below)
    assert all(p.net_benefit > 0 for p in above)
