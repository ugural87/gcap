"""Clopper-Pearson exact binomial confidence intervals. Used both directly
(the D1-D2 stability rule, Section 9.5/10) and as a building block for the
nested kmin simulation.
"""

from __future__ import annotations

from scipy import stats


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided exact interval."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"Invalid successes={successes}, n={n}.")
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return (lower, upper)


def clopper_pearson_lower(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided lower bound only (used for the pilot conformance-gap g
    lower bound, Section 9.5)."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"Invalid successes={successes}, n={n}.")
    if n == 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, n - successes + 1))
