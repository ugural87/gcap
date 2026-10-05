"""Worst-boundary nested simulation deriving kmin (Section 9.5).

Modelling note (documented approximation): the spec pins down the
*selection criterion* precisely — smallest k in {5..20} such that, under
the least-favourable calibration where all 30 simulated replay draws sit
exactly at the k-of-20 boundary, the complete H2b conjunction has >=80%
simulated joint power and P(at most one sampling-induced sign reversal
across 30 draws) >= 80% — but the exact per-case detection-noise model is
an empirical pilot output this build does not have (no live pilot ran).
This module implements the selection procedure exactly as specified, with
the per-case detection-noise layer modelled as: each of a draw's k truly
sign-flipping cases is independently and correctly detected by the
stochastic decision process with probability ``g_lower_bound`` (the pilot's
one-sided 95% LCB on the conformance gap, computed in conformance.py); each
of the (20-k) non-flipping cases independently produces a false
sign-flip signal with a small fixed ``false_positive_rate``. A draw
"preserves sign" iff its net detected-flip count is positive. This model is
swapped for a pilot-calibrated one at M3 freeze without changing the
selection procedure below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class KminCandidateResult:
    k: int
    draw_success_probability: float
    stability_power: float  # P(>=29 of 30 draws preserve sign)
    joint_power_product: float  # descriptive only (Section 9.7: "the product is descriptive")


def draw_success_probability(
    k: int,
    n_cases: int,
    g_lower_bound: float,
    false_positive_rate: float = 0.02,
    n_mc: int = 200_000,
    seed: int | None = None,
) -> float:
    """P(a single replay draw's net detected sign-flip count is positive)
    under the worst-boundary model described in the module docstring."""
    if not (0 <= k <= n_cases):
        raise ValueError(f"k={k} must be in [0, {n_cases}].")
    rng = np.random.default_rng(seed)
    true_flip_hits = np.zeros(n_mc, dtype=int)
    if k > 0:
        true_flip_hits = (rng.random((n_mc, k)) < g_lower_bound).sum(axis=1)
    non_flip = n_cases - k
    false_flip_hits = np.zeros(n_mc, dtype=int)
    if non_flip > 0:
        false_flip_hits = (rng.random((n_mc, non_flip)) < false_positive_rate).sum(axis=1)
    net = true_flip_hits - false_flip_hits
    return float(np.mean(net > 0))


def stability_power_for_k(
    k: int,
    n_cases: int,
    g_lower_bound: float,
    of_draws: int = 30,
    min_sign_preserving: int = 29,
    false_positive_rate: float = 0.02,
    n_mc: int = 200_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Returns (draw_success_probability, P(>=min_sign_preserving of
    of_draws draws succeed)) via the exact binomial tail — the 30 draws are
    modelled as iid at the worst boundary, per the spec's least-favourable
    calibration instruction."""
    p_draw = draw_success_probability(k, n_cases, g_lower_bound, false_positive_rate, n_mc, seed)
    power = float(stats.binom.sf(min_sign_preserving - 1, of_draws, p_draw))
    return p_draw, power


def derive_kmin(
    candidates: range,
    n_cases: int,
    g_lower_bound: float,
    central_test_power: float,
    min_stability_power: float = 0.80,
    min_joint_power: float = 0.80,
    false_positive_rate: float = 0.02,
    n_mc: int = 200_000,
    seed: int | None = None,
) -> tuple[int | None, list[KminCandidateResult]]:
    """Smallest candidate k satisfying both the stability-power floor and
    the descriptive joint-power product floor, scanning ascending (Section
    9.5: "the selected kmin is the smallest candidate for which..."). If no
    candidate qualifies, returns (None, results) — Section 9.5: "Failure of
    every candidate triggers redesign rather than discretionary selection."
    """
    results: list[KminCandidateResult] = []
    for k in candidates:
        p_draw, stability_power = stability_power_for_k(
            k, n_cases, g_lower_bound, false_positive_rate=false_positive_rate, n_mc=n_mc, seed=seed
        )
        joint = central_test_power * stability_power
        results.append(
            KminCandidateResult(
                k=k, draw_success_probability=p_draw, stability_power=stability_power, joint_power_product=joint
            )
        )
        if stability_power >= min_stability_power and joint >= min_joint_power:
            return k, results
    return None, results
