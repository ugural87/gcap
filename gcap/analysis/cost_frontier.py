"""H4 cost-effectiveness frontier (Section 9.8): "Let Delta C be
incremental inference plus human-review cost per decision and Delta p_crit
the absolute reduction in critical-failure probability. The break-even loss
is L* = Delta C / Delta p_crit. The report sweeps the assumed monetary loss
per critical breach over a log-spaced range from 10^3 to 10^7 currency
units and reports the threshold L* rather than asserting an unobservable
universal breach cost."
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np


@dataclass(frozen=True)
class LossSweepPoint:
    loss: Decimal
    net_benefit: Decimal  # delta_p_crit * loss - delta_c


def breakeven_loss(delta_c: Decimal, delta_p_crit: Decimal) -> Decimal:
    if delta_p_crit == 0:
        raise ValueError("delta_p_crit must be non-zero to compute a break-even loss.")
    return delta_c / delta_p_crit


def loss_sweep(
    delta_c: Decimal,
    delta_p_crit: Decimal,
    low: Decimal = Decimal("1e3"),
    high: Decimal = Decimal("1e7"),
    n_points: int = 41,
    log_spaced: bool = True,
) -> list[LossSweepPoint]:
    if log_spaced:
        losses = np.logspace(float(np.log10(float(low))), float(np.log10(float(high))), n_points)
    else:
        losses = np.linspace(float(low), float(high), n_points)
    out: list[LossSweepPoint] = []
    for loss_f in losses:
        loss = Decimal(str(loss_f))
        net_benefit = delta_p_crit * loss - delta_c
        out.append(LossSweepPoint(loss=loss, net_benefit=net_benefit))
    return out
