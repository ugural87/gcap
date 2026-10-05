from gcap.analysis.multiplicity import holm_correction


def test_holm_matches_hand_worked_example():
    # Classic textbook example: p = [0.01, 0.04, 0.03, 0.005], alpha=0.05, m=4.
    # Sorted: 0.005, 0.01, 0.03, 0.04 -> thresholds 0.05/4=0.0125, 0.05/3=0.01667,
    # 0.05/2=0.025, 0.05/1=0.05.
    # 0.005 <= 0.0125 reject; 0.01 <= 0.01667 reject; 0.03 <= 0.025? No -> stop.
    p_values = [0.01, 0.04, 0.03, 0.005]
    reject, adjusted = holm_correction(p_values, alpha=0.05)
    assert reject == [True, False, False, True]
    # adjusted p-values: sorted order gets running max of (m-rank)*p
    # 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06 -> running max 0.06, 0.04*1=0.04 -> running max stays 0.06
    assert adjusted[3] == 0.02  # original index 3 -> p=0.005, first in sorted order
    assert adjusted[0] == 0.03  # original index 0 -> p=0.01, second in sorted order
    assert adjusted[2] == 0.06  # original index 2 -> p=0.03, third in sorted order
    assert adjusted[1] == 0.06  # original index 1 -> p=0.04, running max stays at 0.06


def test_holm_all_reject_when_all_tiny():
    p_values = [1e-6, 1e-7, 1e-8, 1e-9]
    reject, _ = holm_correction(p_values, alpha=0.05)
    assert all(reject)


def test_holm_none_reject_when_all_large():
    p_values = [0.9, 0.8, 0.7, 0.6]
    reject, _ = holm_correction(p_values, alpha=0.05)
    assert not any(reject)


def test_holm_single_test_reduces_to_uncorrected():
    reject, adjusted = holm_correction([0.03], alpha=0.05)
    assert reject == [True]
    assert adjusted == [0.03]


def test_holm_empty_input():
    reject, adjusted = holm_correction([], alpha=0.05)
    assert reject == []
    assert adjusted == []
