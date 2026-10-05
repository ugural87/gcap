"""Pilot conformance gap g and its one-sided 95% lower confidence bound
(Section 9.5): "g = Pr(nonapproval | D2, m<=0) - Pr(nonapproval | D1, m<=0).
The nested simulation uses the one-sided 95% lower confidence bound for g,
computed under the pre-specified paired scenario-cluster simulation."
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def conformance_gap_point(
    nonapproval_rate_d1: Sequence[float],
    nonapproval_rate_d2: Sequence[float],
) -> float:
    """Paired per-cluster nonapproval rates (restricted to m<=0 runs) for
    D1 and D2; returns the mean paired difference (D2 - D1)."""
    d1 = np.asarray(nonapproval_rate_d1, dtype=float)
    d2 = np.asarray(nonapproval_rate_d2, dtype=float)
    if d1.shape != d2.shape or d1.size == 0:
        raise ValueError("nonapproval_rate_d1 and _d2 must be equal-length, non-empty, paired by cluster.")
    return float(np.mean(d2 - d1))


def conformance_gap_one_sided_lower_bound(
    nonapproval_rate_d1: Sequence[float],
    nonapproval_rate_d2: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int | None = None,
) -> float:
    """One-sided lower confidence bound via percentile bootstrap over
    clusters: the (1-confidence) quantile of the bootstrap distribution of
    the mean paired difference."""
    d1 = np.asarray(nonapproval_rate_d1, dtype=float)
    d2 = np.asarray(nonapproval_rate_d2, dtype=float)
    if d1.shape != d2.shape or d1.size == 0:
        raise ValueError("nonapproval_rate_d1 and _d2 must be equal-length, non-empty, paired by cluster.")
    diffs = d2 - d1
    n = diffs.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, 1.0 - confidence))
