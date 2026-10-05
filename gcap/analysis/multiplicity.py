"""Holm (1979) step-down multiplicity correction (Section 9.6 / proposal
[12]). Controls the family-wise error rate across the four co-primary
contrasts at the pre-specified alpha (0.05).
"""

from __future__ import annotations

from collections.abc import Sequence


def holm_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Returns (reject flags, adjusted p-values), both in the original
    input order.

    Step-down procedure: sort p-values ascending; reject H_(1)..H_(k) where
    k is the largest index such that p_(i) <= alpha/(m-i+1) for every
    i <= k (equivalently: stop at the first p_(i) > alpha/(m-i+1)). Adjusted
    p-values are the usual running-max of (m-rank)*p_(rank), capped at 1.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]

    adjusted_sorted: list[float] = []
    running_max = 0.0
    for rank in range(m):
        factor = m - rank
        candidate = sorted_p[rank] * factor
        running_max = max(running_max, candidate)
        adjusted_sorted.append(min(running_max, 1.0))

    k = 0
    for rank in range(m):
        threshold = alpha / (m - rank)
        if sorted_p[rank] <= threshold:
            k = rank + 1
        else:
            break

    reject = [False] * m
    for rank in range(k):
        reject[order[rank]] = True

    adjusted = [0.0] * m
    for rank, i in enumerate(order):
        adjusted[i] = adjusted_sorted[rank]

    return reject, adjusted
