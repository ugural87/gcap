"""Table 5's C-D1 replicate "budget row" selector (Section 9.6/9.7).

The C-D1 contrast (analysis.endpoints.PRIMARY_CONTRASTS: ("state_infeasible",
"C", "D1")) is pre-registered with a replicate-count floor of 3, a ceiling
of 5 (config.registry.ReplicateCounts.c_d1_floor/.c_d1_ceiling) and a fixed
cluster floor of 36 single-decision cases plus 36 episodes
(config.registry.ClusterFloors.c_d1_single/.c_d1_episodes). Table 5 is the
lookup that, given the pilot's own estimated intracluster correlation (ICC)
for that endpoint, picks which of the three replicate counts to actually
run the confirmatory C-D1 contrast at. This module computes that lookup
from first principles (a design-effect adjustment feeding the SAME
paired-binary power simulator gcap.pilot.report already uses for the
D1-D2 floor, analysis.power.simulate_paired_binary_power) rather than
hard-coding Table 5's rows as a literal table, so a revised ICC or a
revised registry value changes the selection the same way it would change
any other analysis in this codebase.

Design effect (Donner & Klar, cluster-randomised-trial planning):
DEFF(m, icc) = 1 + (m-1)*icc. With n_clusters independent clusters each
contributing m paired replicate observations, the design-effect-deflated
"effective" cluster count is

    effective_n_clusters(n_clusters, m, icc) = n_clusters * m / DEFF(m, icc)

which climbs with m but with strictly diminishing returns (it asymptotes
to n_clusters / icc as m grows without bound for icc > 0) -- exactly why
Table 5 only ever offers 3/4/5 rather than searching an unbounded m.

Planning discordance inputs (p10, p01): unlike the D1-D2 contrast, which
Section 9.7 pins to an explicit (121 clusters, p10=0.20, p01=0.05) floor
(see gcap.pilot.report's D1_D2_FLOOR_* constants), the proposal does not
give C-D1 its own literal override -- this falls back to Section 9.7's own
general planning target (quoted in analysis.power's module docstring:
"psi = p01+p10 near 0.15, 80% power, Holm level 0.05/4=0.0125"), split
symmetrically here as p10=0.10/p01=0.05 (psi=0.15). Like every other
placeholder in this build this is a versioned, overridable module
constant, never a silent literal buried in the selection logic.

"May raise this floor but may not lower it" (the same conservative-
revision spirit config.manifest.FreezeManifest.validate_conservative_
revision already enforces for c_d1_floor at freeze time): if even m=5
(the registry's ceiling) fails to reach target_power, select_c_d1_budget_
row still returns "5" -- the most conservative row available -- with
meets_target=False, rather than raising or returning nothing. The
confirmatory run still needs SOME row selected to proceed, and an
under-powered-but-documented outcome for a difficult endpoint is exactly
what Section 9.7 itself anticipates pilot data can reveal, not a bug to
hide (the same posture gcap.pilot.report takes toward its own
central_test_power finding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gcap.analysis.power import simulate_paired_binary_power
from gcap.config.manifest import BudgetRow
from gcap.config.registry import ClusterFloors, ReplicateCounts

# Section 9.7's general planning target (see module docstring) -- the
# fallback used for a primary contrast, like C-D1, that has no explicit
# literal override of its own.
C_D1_PLANNING_P10 = 0.10
C_D1_PLANNING_P01 = 0.05
C_D1_PLANNING_ALPHA = 0.05 / 4  # Holm level for 4 co-primary contrasts, 0.0125
C_D1_TARGET_POWER = 0.80


def design_effect(m: int, icc: float) -> float:
    """DEFF(m, icc) = 1 + (m-1)*icc (Donner & Klar)."""
    if m < 1:
        raise ValueError(f"Replicate count m must be >= 1, got {m}.")
    return 1.0 + (m - 1) * icc


def effective_n_clusters(n_clusters: int, m: int, icc: float) -> float:
    """n_clusters independent clusters, each contributing m paired
    replicate observations, deflated by the design effect."""
    return (n_clusters * m) / design_effect(m, icc)


@dataclass(frozen=True)
class BudgetRowCandidate:
    replicates: int
    effective_n_clusters: float
    power: float
    meets_target: bool


@dataclass(frozen=True)
class BudgetRowSelection:
    icc_used: float
    icc_source: Literal["pilot", "fallback_zero"]
    n_clusters: int
    candidates: tuple[BudgetRowCandidate, ...]
    selected_row: BudgetRow
    meets_target: bool


def select_c_d1_budget_row(
    *,
    icc: float | None,
    n_clusters: int | None = None,
    cluster_floors: ClusterFloors | None = None,
    replicate_counts: ReplicateCounts | None = None,
    p10: float = C_D1_PLANNING_P10,
    p01: float = C_D1_PLANNING_P01,
    alpha: float = C_D1_PLANNING_ALPHA,
    target_power: float = C_D1_TARGET_POWER,
    n_sims: int = 1000,
    n_perm: int = 500,
    seed: int | None = 0,
) -> BudgetRowSelection:
    """Picks the smallest replicate row (3, 4, or 5 -- `replicate_counts`'s
    [c_d1_floor, c_d1_ceiling]) whose design-effect-adjusted power on the
    C-D1 / state_infeasible contrast reaches `target_power`, given the
    pilot's own ICC estimate for that endpoint (e.g.
    PilotMetrics.icc["state_infeasible"]["D1"] or ["C"]).

    `icc=None` -- the pilot never observed enough state_infeasible
    variance to estimate one, a real possibility this build's own tiny
    pilot runs into (see gcap.analysis.icc.one_way_icc's own docstring:
    None for fewer than 2 groups or no within-group degrees of freedom) --
    falls back to icc=0.0, the most optimistic assumption (replicates
    behave as fully independent draws, i.e. no intracluster correlation at
    all), documented on the result via `icc_source` rather than silently
    treated the same as a pilot-confirmed zero.
    """
    cluster_floors = cluster_floors or ClusterFloors()
    replicate_counts = replicate_counts or ReplicateCounts()
    if replicate_counts.c_d1_floor < 3 or replicate_counts.c_d1_ceiling > 5:
        raise ValueError(
            f"Table 5's budget row is fixed to {{3, 4, 5}} replicates (config.manifest.BudgetRow); "
            f"got replicate_counts range [{replicate_counts.c_d1_floor}, {replicate_counts.c_d1_ceiling}]."
        )
    if n_clusters is None:
        n_clusters = cluster_floors.c_d1_single + cluster_floors.c_d1_episodes

    icc_source: Literal["pilot", "fallback_zero"] = "pilot" if icc is not None else "fallback_zero"
    # A small/noisy pilot sample can produce a slightly negative ICC
    # estimate (analysis.icc.one_way_icc's own ANOVA-estimator caveat); a
    # negative design effect makes no sense as a *conservative* planning
    # input, so this clamps to zero rather than letting DEFF < 1 overstate
    # the effective cluster count.
    icc_used = max(0.0, icc if icc is not None else 0.0)

    candidates: list[BudgetRowCandidate] = []
    for m in range(replicate_counts.c_d1_floor, replicate_counts.c_d1_ceiling + 1):
        eff_n = effective_n_clusters(n_clusters, m, icc_used)
        power = simulate_paired_binary_power(
            max(1, round(eff_n)), p10, p01, alpha=alpha, n_sims=n_sims, n_perm=n_perm, seed=seed
        )
        candidates.append(
            BudgetRowCandidate(
                replicates=m,
                effective_n_clusters=eff_n,
                power=power,
                meets_target=power >= target_power,
            )
        )

    qualifying = [c for c in candidates if c.meets_target]
    chosen = min(qualifying, key=lambda c: c.replicates) if qualifying else candidates[-1]
    selected_row: BudgetRow = (
        "3" if chosen.replicates == 3 else "4" if chosen.replicates == 4 else "5"
    )

    return BudgetRowSelection(
        icc_used=icc_used,
        icc_source=icc_source,
        n_clusters=n_clusters,
        candidates=tuple(candidates),
        selected_row=selected_row,
        meets_target=chosen.meets_target,
    )
