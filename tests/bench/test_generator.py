from decimal import Decimal

from gcap.bench.generator import (
    PILOT_STRATUM_ALLOCATION,
    generate_lhc_candidate_pool,
    generate_pilot_slice,
    generate_slice,
)
from gcap.bench.labelling import label_case
from gcap.bench.schema import FAMILIES

TOLERANCES = {"portfolio_return_pct": Decimal("0.01")}
BAND_ANN = Decimal("0.0050")


def test_generate_slice_is_deterministic():
    a = generate_slice(seed=42, split="dev", n_per_family=2)
    b = generate_slice(seed=42, split="dev", n_per_family=2)
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]


def test_generate_slice_different_seed_differs():
    a = generate_slice(seed=1, split="dev", n_per_family=2)
    b = generate_slice(seed=2, split="dev", n_per_family=2)
    assert [c.model_dump_json() for c in a] != [c.model_dump_json() for c in b]


def test_generate_slice_covers_all_eight_families():
    cases = generate_slice(seed=7, split="dev", n_per_family=3)
    assert len(cases) == 24
    families_seen = {c.family for c in cases}
    assert families_seen == set(FAMILIES)


def test_generate_slice_mixes_equity_and_fixed_income():
    cases = generate_slice(seed=7, split="dev", n_per_family=4)
    asset_classes = {p.instrument.asset_class for c in cases for p in c.proposals}
    assert asset_classes == {"equity", "fixed_income"}


def test_all_generated_cases_produce_oracle_labels_without_error():
    cases = generate_slice(seed=11, split="dev", n_per_family=2)
    for case in cases:
        labels = label_case(case, TOLERANCES, BAND_ANN)
        assert labels.case_id == case.case_id


def test_family_endpoints_are_plausible_on_a_larger_sample():
    """Not a strict per-case oracle assertion (that's covered by the golden
    tests); a sanity check that each family's intended endpoint fires at a
    non-trivial rate over a larger seeded sample, i.e. the generator isn't
    silently producing only the trivial (always-clean) branch."""
    cases = generate_slice(seed=123, split="dev", n_per_family=20)
    by_family: dict[str, list] = {}
    for case in cases:
        by_family.setdefault(case.family, []).append(label_case(case, TOLERANCES, BAND_ANN))

    assert any(lbl.ep_governance_failure for lbl in by_family["stale_contradictory_evidence"])
    assert any(lbl.ep_numerical_error for lbl in by_family["numerical_reconciliation_error"])
    assert any(lbl.ep_pricing_inconsistent for lbl in by_family["alpha_eliminated_by_funding_cost"])
    assert any(lbl.ep_governance_failure for lbl in by_family["hard_mandate_breach"])
    assert any(lbl.expected_escalation_required for lbl in by_family["soft_corridor_escalation"])
    assert any(lbl.ep_state_infeasible for lbl in by_family["sequential_resource_exhaustion"])
    assert any(lbl.restructure_feasible for lbl in by_family["valid_restructure_alternative"])
    assert any(
        lbl.ep_governance_failure or lbl.ep_state_infeasible
        for lbl in by_family["injection_forged_stale_permission"]
    )


# ---------------------------------------------------------------------------
# generate_pilot_slice (M3, Section 9.7's 32-case development pilot).
# ---------------------------------------------------------------------------
def test_pilot_slice_allocation_sums_to_32():
    assert sum(PILOT_STRATUM_ALLOCATION.values()) == 32


def test_pilot_slice_is_deterministic():
    a = generate_pilot_slice(seed=42)
    b = generate_pilot_slice(seed=42)
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]


def test_pilot_slice_different_seed_differs():
    a = generate_pilot_slice(seed=1)
    b = generate_pilot_slice(seed=2)
    assert [c.model_dump_json() for c in a] != [c.model_dump_json() for c in b]


def test_pilot_slice_has_32_cases_all_split_pilot_with_unique_ids():
    cases = generate_pilot_slice(seed=7)
    assert len(cases) == 32
    assert {c.split for c in cases} == {"pilot"}
    assert len({c.case_id for c in cases}) == 32  # no case_id collisions across strata


def test_pilot_slice_matches_section_9_7_stratum_counts():
    cases = generate_pilot_slice(seed=7)
    counts = {"pricing": 0, "state": 0, "numerical": 0, "governance": 0}
    for case in cases:
        (stratum,) = case.strata  # every generator family declares exactly one stratum
        counts[stratum] += 1
    assert counts == PILOT_STRATUM_ALLOCATION


def test_pilot_slice_state_stratum_samples_both_state_families():
    cases = generate_pilot_slice(seed=7)
    state_families = {c.family for c in cases if "state" in c.strata}
    assert state_families == {"sequential_resource_exhaustion", "valid_restructure_alternative"}


def test_pilot_slice_governance_stratum_samples_all_four_governance_families():
    cases = generate_pilot_slice(seed=7)
    governance_families = {c.family for c in cases if "governance" in c.strata}
    assert governance_families == {
        "stale_contradictory_evidence",
        "hard_mandate_breach",
        "soft_corridor_escalation",
        "injection_forged_stale_permission",
    }


def test_all_pilot_cases_produce_oracle_labels_without_error():
    cases = generate_pilot_slice(seed=11)
    for case in cases:
        labels = label_case(case, TOLERANCES, BAND_ANN)
        assert labels.case_id == case.case_id


# ---------------------------------------------------------------------------
# generate_lhc_candidate_pool (M3, Section 9.5's LHC replay candidate pool).
# ---------------------------------------------------------------------------
def test_lhc_candidate_pool_is_deterministic():
    a = generate_lhc_candidate_pool(seed=3, n=40)
    b = generate_lhc_candidate_pool(seed=3, n=40)
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]


def test_lhc_candidate_pool_spans_all_three_tenor_strata_round_robin():
    cases = generate_lhc_candidate_pool(seed=3, n=60)
    assert len(cases) == 60
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.tenor_stratum] = counts.get(c.tenor_stratum, 0) + 1
    assert counts == {"<1m": 20, "1-12m": 20, ">12m": 20}


def test_lhc_candidate_pool_tenor_stratum_matches_its_own_tau_days():
    from gcap.core.ftp import tenor_stratum

    cases = generate_lhc_candidate_pool(seed=5, n=30)
    for case in cases:
        tau_days = case.proposals[0].oracle_proposal.tau_days
        assert case.tenor_stratum == tenor_stratum(tau_days, Decimal("365"))


def test_lhc_candidate_pool_produces_oracle_labels_without_error():
    cases = generate_lhc_candidate_pool(seed=9, n=20)
    for case in cases:
        labels = label_case(case, TOLERANCES, BAND_ANN)
        assert labels.case_id == case.case_id
