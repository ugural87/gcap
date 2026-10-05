"""Bridges a BenchCase to the oracle and back. This is the only place bench/
imports oracle/ (allowed: only oracle's *outbound* imports are restricted to
core, per the import-linter contract) — the oracle itself never imports
bench, so labels are always computed from the case definition alone
(invariant 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from gcap.bench.schema import BenchCase
from gcap.oracle.labels import compute_labels
from gcap.oracle.schema import OracleCaseInput, OracleLabels


def to_oracle_input(case: BenchCase) -> OracleCaseInput:
    all_claims = tuple(claim for proposal in case.proposals for claim in proposal.claims)
    return OracleCaseInput(
        case_id=case.case_id,
        initial_state=case.initial_state,
        proposals=tuple(p.oracle_proposal for p in case.proposals),
        restructure_proposal=case.restructure_proposal.oracle_proposal if case.restructure_proposal else None,
        claims=all_claims,
        policy_rules=case.policy_rules,
        policy_context_values=case.policy_context_values,
        policy_evaluated_at=case.as_of,
        policy_version=case.policy_version,
        governance_flags=case.governance_flags,
    )


def label_case(
    case: BenchCase,
    numerical_tolerances: Mapping[str, Decimal],
    marginal_band_ann: Decimal,
) -> OracleLabels:
    oracle_input = to_oracle_input(case)
    return compute_labels(
        oracle_input,
        numerical_tolerances,
        marginal_band_ann,
        generator_version=case.generator_version,
    )
