"""One-way random-effects ANOVA intraclass correlation, ICC(1) (Fisher's
ANOVA-based estimator, e.g. Shrout & Fleiss 1979 §1). Section 9.7 lists
"intra-cluster correlation" among the pilot's own deliverables (alongside
between-run and ordering variance); this is the generic estimator the
pilot layer applies per endpoint, grouping runs by base-scenario/episode
cluster (analysis.endpoints.cluster_id_of).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def one_way_icc(groups: Mapping[str, Sequence[float]]) -> float | None:
    """ICC(1): the fraction of total outcome variance attributable to
    between-cluster differences. Uses the standard unequal-group-size
    correction (n0, the harmonic-mean-like adjustment below) so an
    unbalanced pilot sample (clusters with different replicate counts) is
    handled without approximation.

    Returns None rather than a degenerate number when the estimate is not
    well-defined: fewer than two groups, or no group with more than one
    observation (no within-group degrees of freedom to estimate MSW from).
    The estimator can legitimately return a negative value (a known
    property of the ANOVA-based ICC when the observed between-group
    variance is smaller than chance would predict) — callers should not
    clip it to zero, since that would silently hide a genuinely
    uninformative or overdispersed pilot stratum.
    """
    usable = {k: np.asarray(v, dtype=float) for k, v in groups.items() if len(v) > 0}
    k = len(usable)
    if k < 2:
        return None
    n_i = np.array([arr.size for arr in usable.values()], dtype=float)
    n = float(n_i.sum())
    if not np.any(n_i > 1):
        return None

    all_values = np.concatenate(list(usable.values()))
    grand_mean = float(all_values.mean())

    ssb = float(sum(arr.size * (float(arr.mean()) - grand_mean) ** 2 for arr in usable.values()))
    df_b = k - 1
    msb = ssb / df_b

    ssw = float(sum(float(np.sum((arr - arr.mean()) ** 2)) for arr in usable.values()))
    df_w = n - k
    if df_w <= 0:
        return None
    msw = ssw / df_w

    n0 = (n - float(np.sum(n_i**2)) / n) / df_b
    denom = msb + (n0 - 1) * msw
    if denom == 0:
        return None
    return float((msb - msw) / denom)
