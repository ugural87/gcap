"""Second-reviewer verification gate (Section 7: "Every case is produced by
a seeded parametric generator, labelled by the oracle, and verified by a
second reviewer before it can enter any split." / Section 11: "Unverified
cases cannot enter a split.")
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from gcap.bench.schema import BenchCase
from gcap.errors import UnverifiedCaseError


class VerificationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    reviewer_id: str
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""


def verify_case(case: BenchCase, reviewer_id: str, notes: str = "") -> BenchCase:
    """Returns a new, verified BenchCase. Does not mutate the input (frozen
    models throughout)."""
    if not reviewer_id:
        raise UnverifiedCaseError("A verifying reviewer_id is required.")
    return case.model_copy(update={"verified_by": reviewer_id})


def assert_case_verified(case: BenchCase) -> None:
    if not case.is_verified():
        raise UnverifiedCaseError(
            f"Case {case.case_id!r} has no second-reviewer verification and cannot "
            "enter any split."
        )


class CaseSplitRegistry:
    """A minimal, in-memory split registry enforcing the verification gate
    at the point of entry. store/ (M2) wraps the same rule around the
    persistent benchmark-case table; the rule itself lives here so both the
    in-memory sandbox path and the persisted path share one gate.
    """

    def __init__(self) -> None:
        self._cases: dict[str, BenchCase] = {}

    def add(self, case: BenchCase) -> None:
        assert_case_verified(case)
        if case.case_id in self._cases:
            raise UnverifiedCaseError(f"Duplicate case_id {case.case_id!r}.")
        self._cases[case.case_id] = case

    def add_many(self, cases: list[BenchCase]) -> None:
        for case in cases:
            self.add(case)

    def get(self, case_id: str) -> BenchCase:
        return self._cases[case_id]

    def by_split(self, split: str) -> list[BenchCase]:
        return [c for c in self._cases.values() if c.split == split]

    def by_family(self, family: str) -> list[BenchCase]:
        return [c for c in self._cases.values() if c.family == family]

    def __len__(self) -> int:
        return len(self._cases)
