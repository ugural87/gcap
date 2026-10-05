"""Frozen-seed maximin Latin-hypercube candidate pool and the oracle-only
eligibility filter (Section 8.4 / 9.5).

Flow: frozen seed -> maximin LHC candidate pool -> oracle-only eligibility
filter (median aggregate add-on >= 25bp AND sign-flip count >= kmin) ->
select 30 qualifying draws by maximin distance -> expand the pool under the
same frozen algorithm if fewer than 30 qualify.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc


def sample_unit_cube(n_candidates: int, n_dims: int, seed: int) -> npt.NDArray[np.float64]:
    """Scrambled Latin-hypercube sample in [0,1]^n_dims, frozen by seed."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    result: npt.NDArray[np.float64] = sampler.random(n=n_candidates)
    return result


def scale_to_bounds(unit_cube: npt.NDArray[np.float64], bounds: list[tuple[float, float]]) -> npt.NDArray[np.float64]:
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    result: npt.NDArray[np.float64] = lows + unit_cube * (highs - lows)
    return result


def maximin_select(points: np.ndarray, n_select: int, start_index: int = 0) -> list[int]:
    """Greedy maximin selection: repeatedly picks the remaining point that
    maximises its minimum distance to the already-selected set."""
    n = points.shape[0]
    if n_select > n:
        raise ValueError(f"Cannot select {n_select} points from a pool of {n}.")
    selected = [start_index]
    remaining = set(range(n)) - {start_index}
    while len(selected) < n_select:
        best_idx: int | None = None
        best_mindist = -1.0
        selected_pts = points[selected]
        for idx in remaining:
            d = float(np.min(np.linalg.norm(selected_pts - points[idx], axis=1)))
            if d > best_mindist:
                best_mindist, best_idx = d, idx
        assert best_idx is not None  # `remaining` is non-empty here since len(selected) < n_select <= n
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


@dataclass(frozen=True)
class DrawEvaluation:
    draw_id: str
    params: dict[str, Decimal]
    median_addon_bps: Decimal
    sign_flip_count: int
    eligible: bool


def evaluate_eligibility(
    draw_id: str,
    params: dict[str, Decimal],
    median_addon_bps: Decimal,
    sign_flip_count: int,
    addon_floor_bps: Decimal,
    kmin: int,
) -> DrawEvaluation:
    eligible = median_addon_bps >= addon_floor_bps and sign_flip_count >= kmin
    return DrawEvaluation(
        draw_id=draw_id,
        params=params,
        median_addon_bps=median_addon_bps,
        sign_flip_count=sign_flip_count,
        eligible=eligible,
    )


def build_eligible_pool(
    seed: int,
    n_dims: int,
    bounds: list[tuple[float, float]],
    param_names: list[str],
    oracle_eval_fn: Callable[[dict[str, Decimal]], tuple[Decimal, int]],
    addon_floor_bps: Decimal,
    kmin: int,
    n_target: int = 30,
    initial_pool_size: int = 200,
    max_expansions: int = 5,
    expansion_factor: float = 2.0,
) -> tuple[list[DrawEvaluation], list[DrawEvaluation]]:
    """Returns (selected_30_draws, all_evaluated_draws). Expands the pool
    under the same frozen (seed, algorithm) if fewer than n_target qualify,
    per Section 8.4: "If fewer than 30 draws qualify, the pool is expanded
    under the same frozen seed and algorithm."
    """
    if len(param_names) != n_dims or len(bounds) != n_dims:
        raise ValueError("param_names and bounds must each have length n_dims.")

    pool_size = initial_pool_size
    all_evaluations: list[DrawEvaluation] = []
    for _attempt in range(max_expansions + 1):
        unit_cube = sample_unit_cube(pool_size, n_dims, seed)
        scaled = scale_to_bounds(unit_cube, bounds)
        all_evaluations = []
        for i in range(pool_size):
            params = {name: Decimal(str(round(float(scaled[i, j]), 8))) for j, name in enumerate(param_names)}
            median_addon_bps, sign_flip_count = oracle_eval_fn(params)
            all_evaluations.append(
                evaluate_eligibility(
                    draw_id=f"draw-{i:05d}",
                    params=params,
                    median_addon_bps=median_addon_bps,
                    sign_flip_count=sign_flip_count,
                    addon_floor_bps=addon_floor_bps,
                    kmin=kmin,
                )
            )
        eligible_indices = [i for i, e in enumerate(all_evaluations) if e.eligible]
        if len(eligible_indices) >= n_target:
            eligible = [all_evaluations[i] for i in eligible_indices]
            eligible_points = scaled[eligible_indices]
            selected_local = maximin_select(eligible_points, n_target)
            selected = [eligible[i] for i in selected_local]
            return selected, all_evaluations
        pool_size = int(pool_size * expansion_factor)

    eligible = [e for e in all_evaluations if e.eligible]
    return eligible, all_evaluations
