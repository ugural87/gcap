"""Krippendorff's alpha (nominal metric) and its scenario-bootstrap 95% CI
(Section 9.4: "Raw inter-rater agreement, Krippendorff's alpha, and its
scenario-bootstrap 95% confidence interval are reported before any
consensus discussion.")

Reference implementation of the standard coincidence-matrix formula for
nominal data, assuming a complete (no missing) raters x units matrix, which
matches the design ("Three practitioners independently review... at least
60 stratified decisions").
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def _coincidence_matrix(
    data: npt.NDArray[np.object_], categories: list[str]
) -> tuple[npt.NDArray[np.float64], int]:
    """data: shape (n_raters, n_units), entries are category labels (may
    include np.nan for missing, though this module assumes complete data).
    Returns (coincidence matrix over categories, n = total pairable values).
    """
    n_units = data.shape[1]
    n_cats = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}
    coincidence = np.zeros((n_cats, n_cats), dtype=float)
    n_total = 0
    for u in range(n_units):
        values = [v for v in data[:, u] if not (isinstance(v, float) and np.isnan(v))]
        m_u = len(values)
        if m_u < 2:
            continue
        n_total += m_u
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                coincidence[cat_index[values[i]], cat_index[values[j]]] += 1.0 / (m_u - 1)
    return coincidence, n_total


def krippendorff_alpha_nominal(reliability_data: Sequence[Sequence[str]]) -> float:
    """reliability_data[r][u] = category assigned by rater r to unit u."""
    data = np.array(reliability_data, dtype=object)
    categories = sorted({v for row in reliability_data for v in row if v is not None})
    if len(categories) < 2:
        return 1.0  # no disagreement possible
    coincidence, n = _coincidence_matrix(data, categories)
    if n < 2:
        raise ValueError("Not enough pairable ratings to compute Krippendorff's alpha.")

    n_c = coincidence.sum(axis=1)  # marginal counts per category
    n_cats = len(categories)

    do = 0.0
    de = 0.0
    for c in range(n_cats):
        for k in range(n_cats):
            if c == k:
                continue
            delta = 1.0  # nominal metric
            do += coincidence[c, k] * delta
            de += n_c[c] * n_c[k] * delta
    do /= n
    de /= n * (n - 1)

    if de == 0:
        return 1.0  # every rater agrees on every unit
    return 1.0 - do / de


def krippendorff_alpha_bootstrap_ci(
    reliability_data: Sequence[Sequence[str]],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Resamples UNITS (columns) with replacement — "scenario-bootstrap" —
    and recomputes alpha each time. Returns (point, lower, upper)."""
    data = np.array(reliability_data, dtype=object)
    n_units = data.shape[1]
    point = krippendorff_alpha_nominal(reliability_data)
    rng = np.random.default_rng(seed)
    boot_alphas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        resampled = data[:, idx]
        try:
            boot_alphas[b] = krippendorff_alpha_nominal(resampled.tolist())
        except ValueError:
            boot_alphas[b] = np.nan
    valid = boot_alphas[~np.isnan(boot_alphas)]
    alpha = 1.0 - confidence
    lower = float(np.quantile(valid, alpha / 2))
    upper = float(np.quantile(valid, 1 - alpha / 2))
    return point, lower, upper
