import numpy as np

from gcap.analysis.mixed_model import ClusterObservation, fit_primary_contrast


def _make_observations(rng, n_clusters, n_reps, p_a, p_b, effect_scenario_sd=0.4):
    observations = []
    for s in range(n_clusters):
        scenario_effect = rng.normal(0, effect_scenario_sd)
        for arm, base_p in ((0, p_a), (1, p_b)):
            logit_base = np.log(base_p / (1 - base_p))
            p = 1 / (1 + np.exp(-(logit_base + scenario_effect)))
            p = min(max(p, 1e-6), 1 - 1e-6)
            for r in range(n_reps):
                y = int(rng.binomial(1, p))
                observations.append(
                    ClusterObservation(
                        cluster_id=f"s{s}", replicate=str(r), ordering=None, arm=arm, y=y
                    )
                )
    return observations


def test_separation_fallback_fires_on_zero_event_cell():
    # Arm A has zero events across the board; arm B has some -> classic
    # zero-cell separation (Section 9.6, and Section 11's explicit
    # acceptance criterion for analysis/).
    observations = []
    for s in range(10):
        for r in range(3):
            observations.append(ClusterObservation(cluster_id=f"s{s}", replicate=str(r), ordering=None, arm=0, y=0))
            observations.append(ClusterObservation(cluster_id=f"s{s}", replicate=str(r), ordering=None, arm=1, y=1))
    result = fit_primary_contrast(observations, seed=1)
    assert result.separation_detected is True
    assert result.method == "sign_flip_fallback"
    assert result.odds_ratio is None
    assert 0.0 <= result.p_value <= 1.0


def test_mixed_model_recovers_planted_negative_effect_direction():
    rng = np.random.default_rng(42)
    # Arm B (treatment) has a much lower event rate than arm A (baseline):
    # a strong, unambiguous planted effect with enough clusters/replicates
    # that both event cells are comfortably non-degenerate.
    observations = _make_observations(rng, n_clusters=40, n_reps=5, p_a=0.35, p_b=0.10)
    result = fit_primary_contrast(observations, seed=7)
    assert result.method in ("mixed_model", "marginal_cluster_robust")
    assert result.separation_detected is False
    assert result.odds_ratio is not None
    assert result.odds_ratio < 1.0  # treatment reduces the endpoint rate
    assert result.absolute_risk_difference < 0.0
    lower, upper = result.ard_ci
    assert lower <= result.absolute_risk_difference <= upper
    assert result.p_value < 0.05


def test_no_effect_does_not_falsely_reject():
    rng = np.random.default_rng(11)
    observations = _make_observations(rng, n_clusters=30, n_reps=4, p_a=0.25, p_b=0.25)
    result = fit_primary_contrast(observations, seed=3)
    assert result.separation_detected is False
    # Not a strict claim about p >= 0.05 every run (this is a single draw),
    # but the estimated odds ratio should be in a plausible neighbourhood
    # of 1 (no huge spurious effect) for a well-powered null design.
    assert result.odds_ratio is not None
    assert 0.3 < result.odds_ratio < 3.0


def test_paired_cluster_rates_excludes_clusters_missing_from_either_arm():
    from gcap.analysis.mixed_model import paired_cluster_rates

    observations = [
        ClusterObservation(cluster_id="only-a", replicate="0", ordering=None, arm=0, y=1),
        ClusterObservation(cluster_id="both", replicate="0", ordering=None, arm=0, y=0),
        ClusterObservation(cluster_id="both", replicate="0", ordering=None, arm=1, y=1),
    ]
    paired = paired_cluster_rates(observations)
    assert "only-a" not in paired
    assert paired["both"] == (0.0, 1.0)
