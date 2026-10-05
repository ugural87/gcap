"""Endpoint-specific power simulation for the paired-binary planning case
(Section 9.7): "a 15-percentage-point effect, discordance probability
psi = p01 + p10 near 0.15, 80% power, and the tightest two-sided Holm level
0.05/4 = 0.0125."

Monte Carlo: simulate n_clusters paired binary outcomes with discordance
probabilities (p10, p01), run the same sign-flip test the contrast pipeline
would use, and report the empirical rejection rate as power.
"""

from __future__ import annotations

import numpy as np


def _fast_sign_flip_pvalue(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    n = diffs.shape[0]
    observed = abs(float(diffs.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    stats_ = np.abs((signs * diffs).mean(axis=1))
    return float(np.mean(stats_ >= observed - 1e-15))


def simulate_paired_binary_power(
    n_clusters: int,
    p10: float,
    p01: float,
    alpha: float = 0.05,
    n_sims: int = 1000,
    n_perm: int = 500,
    seed: int | None = None,
) -> float:
    """p10 = P(cluster favours arm A, i.e. paired diff = -1); p01 = P(favours
    arm B, diff = +1); remaining mass is concordant (diff = 0). Returns the
    empirical rejection rate (power) of the sign-flip test at level alpha.
    """
    if p10 < 0 or p01 < 0 or p10 + p01 > 1:
        raise ValueError(f"Invalid discordance probabilities p10={p10}, p01={p01}.")
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(n_sims):
        u = rng.random(n_clusters)
        diffs = np.where(u < p10, -1.0, np.where(u < p10 + p01, 1.0, 0.0))
        p_value = _fast_sign_flip_pvalue(diffs, n_perm, rng)
        if p_value <= alpha:
            rejects += 1
    return rejects / n_sims


def minimum_detectable_discordance(
    n_clusters: int,
    p01: float,
    alpha: float,
    target_power: float = 0.80,
    p10_grid: tuple[float, ...] | None = None,
    n_sims: int = 500,
    n_perm: int = 400,
    seed: int | None = None,
) -> float | None:
    """Smallest p10 (holding p01 fixed) in `p10_grid` at which simulated
    power reaches `target_power`, or None if no grid point reaches it.
    `p10_grid` defaults to 0.02..0.58 in steps of 0.02, computed fresh on
    each call (never as a mutable/import-time-evaluated default)."""
    if p10_grid is None:
        p10_grid = tuple(round(float(x), 3) for x in np.arange(0.02, 0.60, 0.02))
    for p10 in p10_grid:
        if p10 + p01 > 1:
            continue
        power = simulate_paired_binary_power(n_clusters, p10, p01, alpha, n_sims, n_perm, seed)
        if power >= target_power:
            return p10
    return None
