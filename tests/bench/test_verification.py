import pytest

from gcap.bench.generator import generate_slice
from gcap.bench.verification import CaseSplitRegistry, assert_case_verified, verify_case
from gcap.errors import UnverifiedCaseError


def _one_case():
    return generate_slice(seed=1, split="dev", n_per_family=1)[0]


def test_unverified_case_cannot_enter_registry():
    case = _one_case()
    registry = CaseSplitRegistry()
    with pytest.raises(UnverifiedCaseError):
        registry.add(case)


def test_verified_case_enters_registry():
    case = verify_case(_one_case(), reviewer_id="reviewer-1")
    assert_case_verified(case)  # does not raise
    registry = CaseSplitRegistry()
    registry.add(case)
    assert len(registry) == 1
    assert registry.get(case.case_id).case_id == case.case_id


def test_verify_case_does_not_mutate_original():
    original = _one_case()
    verified = verify_case(original, reviewer_id="reviewer-1")
    assert original.verified_by is None
    assert verified.verified_by == "reviewer-1"


def test_verify_case_requires_reviewer_id():
    with pytest.raises(UnverifiedCaseError):
        verify_case(_one_case(), reviewer_id="")


def test_registry_rejects_duplicate_case_id():
    case = verify_case(_one_case(), reviewer_id="reviewer-1")
    registry = CaseSplitRegistry()
    registry.add(case)
    with pytest.raises(UnverifiedCaseError):
        registry.add(case)


def test_by_split_and_by_family_filters():
    cases = generate_slice(seed=3, split="pilot", n_per_family=2)
    registry = CaseSplitRegistry()
    registry.add_many([verify_case(c, "reviewer-1") for c in cases])
    assert len(registry.by_split("pilot")) == len(cases)
    families = {c.family for c in cases}
    for family in families:
        expected = [c for c in cases if c.family == family]
        assert len(registry.by_family(family)) == len(expected)
