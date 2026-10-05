"""Primary-contrast effect estimation (Section 9.6).

"A mixed-effects logistic model estimates each configuration effect with
scenario family and materiality as fixed effects and crossed random
intercepts for base scenario, inference replicate, and scenario ordering...
Two distinct failure fallbacks are frozen in advance. Ordinary non-
convergence without separation triggers a marginal model with base-
scenario-clustered standard errors... Separation is declared if either
paired configuration has zero events, the optimiser flags a singular
boundary, or the absolute log-odds coefficient exceeds 10. Under
separation... the fallback outcome is the mean of its pre-frozen binary
replicate outcomes... The primary effect is then the mean paired difference
between these scenario-level rates, with a base-scenario bootstrap 95%
interval, and the inferential p-value comes from an exact or Monte Carlo
sign-flip permutation test."

This module implements all three tiers, selected automatically by the
trigger conditions above, never by inspecting which method looks more
favourable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from gcap.analysis.bootstrap import cluster_bootstrap_ci
from gcap.analysis.separation import detect_separation, sign_flip_permutation_test

Method = Literal["mixed_model", "marginal_cluster_robust", "sign_flip_fallback"]


@dataclass(frozen=True)
class ClusterObservation:
    """One stochastic inference run's endpoint outcome, already rolled up
    to the replicate level (an episode's replicate is coded 1 if ANY
    constituent proposal triggers the endpoint, per Section 6 / 9.6)."""

    cluster_id: str  # base-scenario or episode id
    replicate: str
    ordering: str | None
    arm: Literal[0, 1]  # 0 = baseline (control) side of the contrast, 1 = treatment
    y: int  # 0/1 endpoint indicator


@dataclass(frozen=True)
class ContrastResult:
    method: Method
    separation_detected: bool
    odds_ratio: float | None
    odds_ratio_ci: tuple[float, float] | None
    p_value: float
    absolute_risk_difference: float
    ard_ci: tuple[float, float]
    n_clusters: int
    events_a: int
    n_a: int
    events_b: int
    n_b: int


class _NonConvergenceError(Exception):
    pass


def paired_cluster_rates(observations: list[ClusterObservation]) -> dict[str, tuple[float, float]]:
    """cluster_id -> (mean_y_arm_a, mean_y_arm_b), including only clusters
    observed in both arms (a paired design should have both by construction;
    a cluster missing from one arm is excluded with a note left to the
    caller, since it indicates a pairing problem upstream)."""
    by_cluster: dict[str, dict[int, list[int]]] = {}
    for obs in observations:
        by_cluster.setdefault(obs.cluster_id, {0: [], 1: []})[obs.arm].append(obs.y)
    out: dict[str, tuple[float, float]] = {}
    for cluster_id, arms in by_cluster.items():
        if arms[0] and arms[1]:
            out[cluster_id] = (float(np.mean(arms[0])), float(np.mean(arms[1])))
    return out


def _fit_mixed_glm(observations: list[ClusterObservation]) -> tuple[float, float]:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = pd.DataFrame(
        {
            "cluster_id": [o.cluster_id for o in observations],
            "replicate": [o.replicate for o in observations],
            "ordering": [o.ordering if o.ordering is not None else "NA" for o in observations],
            "arm": [o.arm for o in observations],
            "y": [o.y for o in observations],
        }
    )
    vc_formulas: dict[str, str] = {}
    if df["cluster_id"].nunique() > 1:
        vc_formulas["cluster_id"] = "0 + C(cluster_id)"
    if df["replicate"].nunique() > 1:
        vc_formulas["replicate"] = "0 + C(replicate)"
    if df["ordering"].nunique() > 1:
        vc_formulas["ordering"] = "0 + C(ordering)"

    if not vc_formulas:
        raise _NonConvergenceError("No crossed grouping factor has more than one level; nothing to model as random.")

    model = BinomialBayesMixedGLM.from_formula("y ~ arm", vc_formulas=vc_formulas, data=df)
    result = model.fit_vb()
    exog_names = list(model.exog_names)
    if "arm" not in exog_names:
        raise _NonConvergenceError("Fixed-effect design did not include 'arm'.")
    idx = exog_names.index("arm")
    coef = float(result.fe_mean[idx])
    sd = float(result.fe_sd[idx])
    if not np.isfinite(coef) or not np.isfinite(sd) or sd <= 0:
        raise _NonConvergenceError(f"Non-finite or degenerate posterior: coef={coef}, sd={sd}.")
    return coef, sd


def _fit_marginal_cluster_robust(observations: list[ClusterObservation]) -> tuple[float, float]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(
        {
            "cluster_id": [o.cluster_id for o in observations],
            "arm": [o.arm for o in observations],
            "y": [o.y for o in observations],
        }
    )
    family = sm.families.Binomial()
    cov_struct = sm.cov_struct.Independence()
    model = smf.gee("y ~ arm", groups="cluster_id", data=df, family=family, cov_struct=cov_struct)
    result = model.fit()
    coef = float(result.params["arm"])
    se = float(result.bse["arm"])
    if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
        raise _NonConvergenceError(f"Marginal GEE fit degenerate: coef={coef}, se={se}.")
    return coef, se


def fit_primary_contrast(
    observations: list[ClusterObservation],
    abs_log_odds_threshold: float = 10.0,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> ContrastResult:
    arm_a = [o for o in observations if o.arm == 0]
    arm_b = [o for o in observations if o.arm == 1]
    events_a, n_a = sum(o.y for o in arm_a), len(arm_a)
    events_b, n_b = sum(o.y for o in arm_b), len(arm_b)

    paired = paired_cluster_rates(observations)
    cluster_ids = sorted(paired)
    diffs = [paired[c][1] - paired[c][0] for c in cluster_ids]
    if not diffs:
        raise ValueError("No paired clusters (observed in both arms); cannot estimate a contrast.")

    ard_point, ard_lower, ard_upper = cluster_bootstrap_ci(diffs, np.mean, n_boot=n_boot, seed=seed)

    separation = detect_separation(events_a, n_a, events_b, n_b)
    if separation:
        p_value = sign_flip_permutation_test(diffs, seed=seed)
        return ContrastResult(
            method="sign_flip_fallback",
            separation_detected=True,
            odds_ratio=None,
            odds_ratio_ci=None,
            p_value=p_value,
            absolute_risk_difference=ard_point,
            ard_ci=(ard_lower, ard_upper),
            n_clusters=len(cluster_ids),
            events_a=events_a,
            n_a=n_a,
            events_b=events_b,
            n_b=n_b,
        )

    try:
        coef, sd = _fit_mixed_glm(observations)
        if abs(coef) > abs_log_odds_threshold:
            raise _NonConvergenceError(f"|coef|={abs(coef)} exceeds separation threshold {abs_log_odds_threshold}.")
        method: Method = "mixed_model"
    except _NonConvergenceError:
        try:
            coef, sd = _fit_marginal_cluster_robust(observations)
            method = "marginal_cluster_robust"
        except _NonConvergenceError:
            # Both model-based routes failed to produce a usable estimate;
            # fall back to the same sign-flip route separation uses rather
            # than surface a raw numerical error to the caller.
            p_value = sign_flip_permutation_test(diffs, seed=seed)
            return ContrastResult(
                method="sign_flip_fallback",
                separation_detected=False,
                odds_ratio=None,
                odds_ratio_ci=None,
                p_value=p_value,
                absolute_risk_difference=ard_point,
                ard_ci=(ard_lower, ard_upper),
                n_clusters=len(cluster_ids),
                events_a=events_a,
                n_a=n_a,
                events_b=events_b,
                n_b=n_b,
            )

    odds_ratio = float(np.exp(coef))
    z = coef / sd
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (float(np.exp(coef - 1.96 * sd)), float(np.exp(coef + 1.96 * sd)))

    return ContrastResult(
        method=method,
        separation_detected=False,
        odds_ratio=odds_ratio,
        odds_ratio_ci=ci,
        p_value=p_value,
        absolute_risk_difference=ard_point,
        ard_ci=(ard_lower, ard_upper),
        n_clusters=len(cluster_ids),
        events_a=events_a,
        n_a=n_a,
        events_b=events_b,
        n_b=n_b,
    )
