from gcap.analysis.power import simulate_paired_binary_power


def test_power_increases_with_more_clusters():
    small = simulate_paired_binary_power(n_clusters=20, p10=0.05, p01=0.20, n_sims=300, n_perm=300, seed=1)
    large = simulate_paired_binary_power(n_clusters=120, p10=0.05, p01=0.20, n_sims=300, n_perm=300, seed=1)
    assert large > small


def test_power_increases_with_larger_discordance_gap():
    weak = simulate_paired_binary_power(n_clusters=72, p10=0.05, p01=0.10, n_sims=300, n_perm=300, seed=2)
    strong = simulate_paired_binary_power(n_clusters=72, p10=0.05, p01=0.30, n_sims=300, n_perm=300, seed=2)
    assert strong > weak


def test_power_near_alpha_under_null_no_discordance_asymmetry():
    # p10 == p01: no directional effect, power should hover near alpha
    # (loosely — this is a Monte Carlo estimate, not an exact calibration
    # check).
    power = simulate_paired_binary_power(n_clusters=72, p10=0.10, p01=0.10, alpha=0.05, n_sims=500, n_perm=300, seed=3)
    assert power < 0.25


def test_invalid_discordance_probabilities_raise():
    import pytest

    with pytest.raises(ValueError):
        simulate_paired_binary_power(n_clusters=10, p10=0.6, p01=0.6)
