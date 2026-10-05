"""Deterministic FTP engine (Section 4.1 / proposal Section 5.2).

Invariant 2: the FTP engine never emits an action label. FTPOutput carries
only numerical fields, provenance and validation flags. Invariant 1: this
whole module is pure and typed; the oracle and the agents' tools call these
same functions, so a label and a tool result can never disagree because of
two implementations.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gcap.core.numerics import annual_from_horizon, horizon_from_annual
from gcap.errors import FTPValidationError

ENGINE_VERSION = "ftp-1.0.0"

# Field names an FTPOutput must never contain — invariant 2's schema guard.
_FORBIDDEN_OUTPUT_TOKENS = ("action", "approve", "reject", "defer", "decision", "recommend")


class FTPAddonVector(BaseModel):
    """LP, TP, CP, RP, CC — all decimals, all annualised rates, never basis
    points inside the engine (Section 4.1: "basis points appear only at the
    display layer")."""

    model_config = ConfigDict(frozen=True)

    lp: Decimal = Decimal("0")  # liquidity premium
    tp: Decimal = Decimal("0")  # tenor premium: residual effects only, never the observed term spread
    cp: Decimal = Decimal("0")  # capital-consumption charge
    rp: Decimal = Decimal("0")  # residual risk / balance-sheet charge
    cc: Decimal = Decimal("0")  # concentration charge

    @property
    def aggregate(self) -> Decimal:
        return self.lp + self.tp + self.cp + self.rp + self.cc


class FTPInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    tau_days: Decimal = Field(gt=0, description="Holding period, in day-count-basis days.")
    day_count_basis: Decimal = Field(gt=0, description="D, the documented day-count basis.")
    base_funding_curve_rate_ann: Decimal = Field(
        description="r_f^ann(tau): observable effective funding curve, interpolated at tau, annualised."
    )
    addons: FTPAddonVector = FTPAddonVector()
    trade_alpha_horizon: Decimal = Field(
        description="TradeAlpha_i over tau_i: the holding-period trade return from Trade Structuring."
    )
    tp_double_counts_term_spread: bool = Field(
        default=False,
        description="Set True by the caller when TP as specified duplicates the observed term "
        "spread already embedded in the base curve. Causes validation to reject.",
    )
    curve_source_id: str = ""
    curve_as_of: str = ""

    @model_validator(mode="after")
    def _validate(self) -> FTPInputs:
        if self.trade_alpha_horizon <= Decimal("-1"):
            raise FTPValidationError(
                f"TradeAlpha={self.trade_alpha_horizon} <= -1: a holding-period return cannot be "
                "-100% or worse (Section 4.1 validation rule)."
            )
        if self.tp_double_counts_term_spread:
            raise FTPValidationError(
                "TP specification double-counts the observed term spread already embedded in the "
                "base funding curve. TP must be restricted to residual tenor effects (behavioural "
                "maturity mismatch, rollover risk, term-funding-capacity reservation cost)."
            )
        return self


class FTPOutput(BaseModel):
    """Numerical fields, provenance and validation flags ONLY. No action
    field of any kind (invariant 2) — enforced by test_ftp_no_action_field
    below and re-checked at import time via `assert_no_action_field`."""

    model_config = ConfigDict(frozen=True)

    all_in_annualised_rate: Decimal  # f_i^ann
    ftp_horizon: Decimal  # FTP_i(tau_i)
    aggregate_addon_ann: Decimal  # a = LP+TP+CP+RP+CC
    addons: FTPAddonVector
    trade_alpha_horizon: Decimal
    trade_alpha_annualised: Decimal
    economic_alpha_horizon: Decimal  # TradeAlpha_i - FTP_i(tau_i)
    tau_days: Decimal
    day_count_basis: Decimal
    base_funding_curve_rate_ann: Decimal
    curve_source_id: str
    curve_as_of: str
    engine_version: str = ENGINE_VERSION
    validation_flags: tuple[str, ...] = ()

    def in_band(self, beta_ann: Decimal) -> bool:
        """Marginal band test on the trade horizon (Section 4.1):
        beta(tau) = (1+beta_ann)^(tau/D) - 1; in band iff
        |TradeAlpha - FTP(tau)| <= beta(tau), i.e. |EconomicAlpha| <= beta(tau).
        """
        beta_horizon = horizon_from_annual(beta_ann, self.tau_days, self.day_count_basis)
        return abs(self.economic_alpha_horizon) <= beta_horizon

    def distance_from_band_edge(self, beta_ann: Decimal) -> Decimal:
        """Signed distance: positive means inside the band, by this margin."""
        beta_horizon = horizon_from_annual(beta_ann, self.tau_days, self.day_count_basis)
        return beta_horizon - abs(self.economic_alpha_horizon)


def assert_no_action_field() -> None:
    """Schema guard for invariant 2: fails loudly (at import/test time, not
    silently at runtime) if an action-shaped field is ever added to
    FTPOutput."""
    for name in FTPOutput.model_fields:
        lowered = name.lower()
        for token in _FORBIDDEN_OUTPUT_TOKENS:
            if token in lowered:
                raise FTPValidationError(
                    f"FTPOutput gained a field {name!r} that looks like an action/decision "
                    f"field (matched forbidden token {token!r}). Invariant 2 requires the FTP "
                    "engine to never emit an action label."
                )


# Run the guard once at import time so a regression fails CI immediately,
# not only when the dedicated schema test happens to run.
assert_no_action_field()


def compute_ftp(inputs: FTPInputs) -> FTPOutput:
    """f_i^ann = r_f^ann(tau_i) + LP_i + TP_i + CP_i + RP_i + CC_i
    FTP_i(tau_i) = (1+f_i^ann)^(tau_i/D) - 1
    EconomicAlpha_i = TradeAlpha_i - FTP_i(tau_i)
    TradeAlpha_i^ann = (1+TradeAlpha_i)^(D/tau_i) - 1, TradeAlpha_i > -1
    """
    f_ann = inputs.base_funding_curve_rate_ann + inputs.addons.aggregate
    ftp_horizon = horizon_from_annual(f_ann, inputs.tau_days, inputs.day_count_basis)
    economic_alpha_horizon = inputs.trade_alpha_horizon - ftp_horizon
    trade_alpha_ann = annual_from_horizon(inputs.trade_alpha_horizon, inputs.tau_days, inputs.day_count_basis)

    flags: list[str] = []
    if inputs.addons.aggregate < 0:
        flags.append("negative_aggregate_addon")
    if f_ann < 0:
        flags.append("negative_all_in_rate")

    return FTPOutput(
        all_in_annualised_rate=f_ann,
        ftp_horizon=ftp_horizon,
        aggregate_addon_ann=inputs.addons.aggregate,
        addons=inputs.addons,
        trade_alpha_horizon=inputs.trade_alpha_horizon,
        trade_alpha_annualised=trade_alpha_ann,
        economic_alpha_horizon=economic_alpha_horizon,
        tau_days=inputs.tau_days,
        day_count_basis=inputs.day_count_basis,
        base_funding_curve_rate_ann=inputs.base_funding_curve_rate_ann,
        curve_source_id=inputs.curve_source_id,
        curve_as_of=inputs.curve_as_of,
        validation_flags=tuple(flags),
    )


TenorStratum = Literal["<1m", "1-12m", ">12m"]


def tenor_stratum(tau_days: Decimal, day_count_basis: Decimal) -> TenorStratum:
    """Buckets a holding period into the pre-specified tenor strata used for
    quota-stratifying the D1-D2 pricing cases (Section 9.5)."""
    months = (tau_days / day_count_basis) * Decimal(12)
    if months < 1:
        return "<1m"
    if months <= 12:
        return "1-12m"
    return ">12m"
