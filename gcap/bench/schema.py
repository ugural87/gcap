"""G-CAP Bench case schema (Section 7.1).

A BenchCase is a self-contained, typed scenario: enough to (a) feed an
OracleCaseInput and get ground-truth labels at authoring time, and (b) feed
the agent graph (built in M2) as point-in-time evidence + initial state.
Equity and fixed-income proposals both reduce to the same InstrumentSpec
union so FTP/ledger/Treasury/validator/policy stay asset-class agnostic.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gcap.core.ledger import LedgerState
from gcap.core.policy import PolicyRule
from gcap.oracle.schema import OracleClaim, OracleGovernanceFlags, OracleProposal

Split = Literal["dev", "pilot", "confirm", "audit", "roadmap"]
Stratum = Literal["numerical", "state", "pricing", "governance"]
TenorStratum = Literal["<1m", "1-12m", ">12m"]
AssetClass = Literal["equity", "fixed_income"]

FAMILIES: tuple[str, ...] = (
    "stale_contradictory_evidence",
    "numerical_reconciliation_error",
    "alpha_eliminated_by_funding_cost",
    "hard_mandate_breach",
    "soft_corridor_escalation",
    "sequential_resource_exhaustion",
    "valid_restructure_alternative",
    "injection_forged_stale_permission",
)


class EquityFundamentals(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: str
    revenue_ttm: Decimal | None = None
    ebitda_margin_pct: Decimal | None = None
    pe_ratio: Decimal | None = None
    expected_return_pct: Decimal | None = None
    downside_pct: Decimal | None = None
    transaction_cost_bps: Decimal | None = None
    concentration_key: str | None = None


class FixedIncomeAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    isin: str
    coupon_pct: Decimal
    maturity_date: str
    clean_price: Decimal
    yield_pct: Decimal
    spread_bps: Decimal
    modified_duration: Decimal
    dv01: Decimal
    carry_roll_down_bps: Decimal | None = None
    concentration_key: str | None = None


class InstrumentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_class: AssetClass
    equity: EquityFundamentals | None = None
    fixed_income: FixedIncomeAnalytics | None = None

    @model_validator(mode="after")
    def _check_payload_matches_asset_class(self) -> InstrumentSpec:
        if self.asset_class == "equity" and self.equity is None:
            raise ValueError("asset_class='equity' requires an EquityFundamentals payload.")
        if self.asset_class == "fixed_income" and self.fixed_income is None:
            raise ValueError("asset_class='fixed_income' requires a FixedIncomeAnalytics payload.")
        return self


class ProposalSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str
    instrument: InstrumentSpec
    oracle_proposal: OracleProposal
    claims: tuple[OracleClaim, ...] = ()


class PerturbationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["prompt_injection", "forged_evidence", "stale_state", "permission_escalation"]
    description: str


class BenchCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    split: Split
    family: str
    strata: frozenset[Stratum]
    materiality: str
    tenor_stratum: TenorStratum | None = None
    kind: Literal["single", "episode"]
    proposals: tuple[ProposalSpec, ...]
    restructure_proposal: ProposalSpec | None = None
    initial_state: LedgerState
    evidence_packet_ids: tuple[str, ...] = ()
    policy_rules: tuple[PolicyRule, ...] = ()
    policy_context_values: dict[str, object] = Field(default_factory=dict)
    policy_version: str
    governance_flags: OracleGovernanceFlags = OracleGovernanceFlags()
    as_of: datetime
    perturbation: PerturbationSpec | None = None
    generator_version: str
    generator_seed: int
    labels_ref: str = ""
    label_rationale: str = ""
    verified_by: str | None = None

    @model_validator(mode="after")
    def _check_proposal_count(self) -> BenchCase:
        n = len(self.proposals)
        if self.kind == "single" and n != 1:
            raise ValueError(f"kind='single' requires exactly 1 proposal, got {n}.")
        if self.kind == "episode" and n != 4:
            raise ValueError(f"kind='episode' requires exactly 4 proposals, got {n}.")
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown family {self.family!r}; must be one of {FAMILIES}.")
        return self

    def is_verified(self) -> bool:
        return self.verified_by is not None
