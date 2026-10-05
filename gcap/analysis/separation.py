"""Separation detection and the exact/Monte Carlo sign-flip permutation
fallback (Section 9.6): "Separation is declared if either paired
configuration has zero events, the optimiser flags a singular boundary, or
the absolute log-odds coefficient exceeds 10."
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

EXACT_ENUMERATION_LIMIT = 20  # 2**20 ~= 1e6 sign patterns, still fast


def detect_separation(
    events_a: int,
    n_a: int,
    events_b: int,
    n_b: int,
    fitted_log_odds: float | None = None,
    singular_boundary: bool = False,
    abs_log_odds_threshold: float = 10.0,
) -> bool:
    zero_or_all = (
        events_a in (0, n_a) if n_a > 0 else True
    ) or (
        events_b in (0, n_b) if n_b > 0 else True
    )
    log_odds_extreme = fitted_log_odds is not None and abs(fitted_log_odds) > abs_log_odds_threshold
    return bool(zero_or_all or singular_boundary or log_odds_extreme)


def sign_flip_permutation_test(
    paired_diffs: Sequence[float],
    n_perm: int = 20000,
    seed: int | None = None,
) -> float:
    """Two-sided p-value for H0: the paired differences are exchangeable in
    sign (i.e. mean is 0), via exact enumeration when the number of
    clusters is small enough, else Monte Carlo sign-flipping.
    """
    diffs = np.asarray(paired_diffs, dtype=float)
    n = diffs.shape[0]
    if n == 0:
        raise ValueError("sign_flip_permutation_test requires at least one paired difference.")
    observed = abs(float(np.mean(diffs)))

    if n <= EXACT_ENUMERATION_LIMIT:
        count_extreme = 0
        total = 0
        for signs in itertools.product([1, -1], repeat=n):
            total += 1
            stat = abs(float(np.mean(diffs * np.array(signs))))
            if stat >= observed - 1e-15:
                count_extreme += 1
        return count_extreme / total

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([1, -1], size=n)
        stat = abs(float(np.mean(diffs * signs)))
        if stat >= observed - 1e-15:
            count_extreme += 1
    return count_extreme / n_perm
