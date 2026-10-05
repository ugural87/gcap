from decimal import Decimal

import numpy as np

from gcap.analysis.lhc import build_eligible_pool, maximin_select, sample_unit_cube, scale_to_bounds


def test_sample_unit_cube_is_deterministic_given_seed():
    a = sample_unit_cube(20, 3, seed=42)
    b = sample_unit_cube(20, 3, seed=42)
    assert np.array_equal(a, b)


def test_sample_unit_cube_different_seed_differs():
    a = sample_unit_cube(20, 3, seed=1)
    b = sample_unit_cube(20, 3, seed=2)
    assert not np.array_equal(a, b)


def test_scale_to_bounds_respects_range():
    cube = sample_unit_cube(50, 2, seed=1)
    bounds = [(0.0, 0.01), (-0.005, 0.005)]
    scaled = scale_to_bounds(cube, bounds)
    assert scaled[:, 0].min() >= 0.0
    assert scaled[:, 0].max() <= 0.01
    assert scaled[:, 1].min() >= -0.005
    assert scaled[:, 1].max() <= 0.005


def test_maximin_select_spreads_points_more_than_first_n():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, size=(100, 2))
    selected_idx = maximin_select(points, 10)
    selected_pts = points[selected_idx]

    def min_pairwise_distance(pts):
        n = pts.shape[0]
        best = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                best = min(best, float(np.linalg.norm(pts[i] - pts[j])))
        return best

    naive_first_10 = points[:10]
    assert min_pairwise_distance(selected_pts) >= min_pairwise_distance(naive_first_10)


def test_build_eligible_pool_finds_30_draws_with_permissive_oracle():
    def oracle_eval_fn(params):
        # Every draw is eligible: constant median add-on above the floor and
        # a sign-flip count above kmin, regardless of parameters.
        return Decimal("40"), 15

    selected, all_evals = build_eligible_pool(
        seed=1,
        n_dims=2,
        bounds=[(0.0, 0.01), (0.0, 0.01)],
        param_names=["lp", "cp"],
        oracle_eval_fn=oracle_eval_fn,
        addon_floor_bps=Decimal("25"),
        kmin=10,
        n_target=30,
        initial_pool_size=50,
    )
    assert len(selected) == 30
    assert all(d.eligible for d in selected)


def test_build_eligible_pool_expands_when_initial_pool_too_restrictive():
    calls = {"n": 0}

    def oracle_eval_fn(params):
        calls["n"] += 1
        # Only every 5th evaluated draw is eligible -> initial pool of 20
        # cannot supply 30 eligible draws, forcing an expansion.
        eligible = calls["n"] % 5 == 0
        return (Decimal("40") if eligible else Decimal("10")), (15 if eligible else 2)

    selected, all_evals = build_eligible_pool(
        seed=2,
        n_dims=1,
        bounds=[(0.0, 0.01)],
        param_names=["lp"],
        oracle_eval_fn=oracle_eval_fn,
        addon_floor_bps=Decimal("25"),
        kmin=10,
        n_target=30,
        initial_pool_size=20,
        expansion_factor=3.0,
    )
    assert len(selected) == 30
    assert len(all_evals) > 20  # confirms an expansion actually happened
