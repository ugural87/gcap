"""Cluster and crossed (scenario x draw) bootstrap confidence intervals
(Section 9.6 / 9.5). Percentile method throughout — simple, distribution-
free, and adequate at the cluster counts this design uses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def cluster_bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Resamples clusters (rows of `values`) with replacement, n_boot times,
    and returns (point_estimate, lower, upper) as the percentile interval.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cluster_bootstrap_ci requires at least one cluster value.")
    point = float(statistic(arr))
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic(arr[idx])
    alpha = 1.0 - confidence
    lower = float(np.quantile(boot_stats, alpha / 2))
    upper = float(np.quantile(boot_stats, 1 - alpha / 2))
    return point, lower, upper


def crossed_bootstrap_ci(
    matrix: Sequence[Sequence[float]],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Two-way (crossed) bootstrap over a scenario x draw matrix (rows =
    scenarios, columns = draws), used for the LHC replay's parameter-
    averaged H2c estimand (Section 9.6: "uncertainty is obtained by a
    crossed bootstrap over scenario and parameter draw"). Each bootstrap
    replicate resamples scenario rows AND draw columns independently with
    replacement.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("crossed_bootstrap_ci requires a non-empty 2D matrix.")
    n_rows, n_cols = arr.shape
    point = float(statistic(arr.reshape(-1)))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        row_idx = rng.integers(0, n_rows, size=n_rows)
        col_idx = rng.integers(0, n_cols, size=n_cols)
        resampled = arr[np.ix_(row_idx, col_idx)]
        boot_stats[b] = statistic(resampled.reshape(-1))
    alpha = 1.0 - confidence
    lower = float(np.quantile(boot_stats, alpha / 2))
    upper = float(np.quantile(boot_stats, 1 - alpha / 2))
    return point, lower, upper
