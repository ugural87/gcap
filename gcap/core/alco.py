"""ALCO packages (Section 4.4 / proposal Section 6.1).

Four fixed, versioned policy packages — NORMAL, CAUTION, STRESS, RECOVERY —
each holding FTP add-on ranges, liquidity-buffer targets, concentration
corridors and exception routes. Transitions are a deterministic function of
state observations with pre-registered thresholds. ALCO does not select
securities; it changes the environment future trades are judged in.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gcap.config.registry import AlcoTransitionThresholds

AlcoPackageName = Literal["NORMAL", "CAUTION", "STRESS", "RECOVERY"]


class AlcoPackage(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: AlcoPackageName
    ftp_addon_ranges: dict[str, tuple[Decimal, Decimal]]  # component -> (min, max) annualised
    liquidity_buffer_target_pct: Decimal
    concentration_corridor_pct: Decimal
    exception_route: str


# Placeholder ranges consistent with the AlcoTransitionThresholds placeholder
# defaults; both need joint pilot calibration before M2 sign-off (they are
# already flagged uncalibrated at the parameter-registry level).
PACKAGES: dict[AlcoPackageName, AlcoPackage] = {
    "NORMAL": AlcoPackage(
        name="NORMAL",
        ftp_addon_ranges={
            "lp": (Decimal("0.0000"), Decimal("0.0020")),
            "tp": (Decimal("0.0000"), Decimal("0.0010")),
            "cp": (Decimal("0.0000"), Decimal("0.0025")),
            "rp": (Decimal("0.0000"), Decimal("0.0010")),
            "cc": (Decimal("0.0000"), Decimal("0.0010")),
        },
        liquidity_buffer_target_pct=Decimal("1.00"),
        concentration_corridor_pct=Decimal("0.85"),
        exception_route="desk_head",
    ),
    "CAUTION": AlcoPackage(
        name="CAUTION",
        ftp_addon_ranges={
            "lp": (Decimal("0.0010"), Decimal("0.0035")),
            "tp": (Decimal("0.0005"), Decimal("0.0020")),
            "cp": (Decimal("0.0010"), Decimal("0.0040")),
            "rp": (Decimal("0.0005"), Decimal("0.0020")),
            "cc": (Decimal("0.0005"), Decimal("0.0020")),
        },
        liquidity_buffer_target_pct=Decimal("1.10"),
        concentration_corridor_pct=Decimal("0.75"),
        exception_route="alco_delegate",
    ),
    "STRESS": AlcoPackage(
        name="STRESS",
        ftp_addon_ranges={
            "lp": (Decimal("0.0025"), Decimal("0.0060")),
            "tp": (Decimal("0.0015"), Decimal("0.0035")),
            "cp": (Decimal("0.0025"), Decimal("0.0070")),
            "rp": (Decimal("0.0015"), Decimal("0.0040")),
            "cc": (Decimal("0.0015"), Decimal("0.0040")),
        },
        liquidity_buffer_target_pct=Decimal("1.25"),
        concentration_corridor_pct=Decimal("0.60"),
        exception_route="alco_committee",
    ),
    "RECOVERY": AlcoPackage(
        name="RECOVERY",
        ftp_addon_ranges={
            "lp": (Decimal("0.0015"), Decimal("0.0040")),
            "tp": (Decimal("0.0010"), Decimal("0.0025")),
            "cp": (Decimal("0.0015"), Decimal("0.0050")),
            "rp": (Decimal("0.0010"), Decimal("0.0025")),
            "cc": (Decimal("0.0010"), Decimal("0.0025")),
        },
        liquidity_buffer_target_pct=Decimal("1.15"),
        concentration_corridor_pct=Decimal("0.70"),
        exception_route="alco_delegate",
    ),
}


class AlcoObservations(BaseModel):
    model_config = ConfigDict(frozen=True)

    stress_liquidity_buffer_pct: Decimal
    funding_spread_percentile: Decimal
    maturity_gap_headroom_pct: Decimal
    collateral_usage_pct: Decimal


class AlcoTransitionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    package: AlcoPackageName
    triggering_observations: tuple[str, ...]


def determine_package(
    previous: AlcoPackageName,
    observations: AlcoObservations,
    thresholds: AlcoTransitionThresholds,
) -> AlcoTransitionResult:
    """Deterministic transition function. A package tightens when projected
    stress liquidity falls below target, funding-spread percentile exceeds
    a threshold, maturity-gap headroom is exhausted, or collateral usage
    breaches a corridor (Section 4.4 example). De-escalation passes through
    RECOVERY for one step (hysteresis) before returning to NORMAL, so a
    single noisy clean observation cannot flip STRESS straight to NORMAL.
    """
    triggers: list[str] = []
    if observations.stress_liquidity_buffer_pct < thresholds.stress_liquidity_buffer_floor_pct:
        triggers.append("stress_liquidity_buffer_below_floor")
    if observations.funding_spread_percentile > thresholds.funding_spread_percentile_ceiling:
        triggers.append("funding_spread_percentile_above_ceiling")
    if observations.maturity_gap_headroom_pct < thresholds.maturity_gap_headroom_floor_pct:
        triggers.append("maturity_gap_headroom_below_floor")
    if observations.collateral_usage_pct > thresholds.collateral_usage_corridor_ceiling_pct:
        triggers.append("collateral_usage_above_corridor")

    n = len(triggers)
    if n == 0:
        new_package: AlcoPackageName = "RECOVERY" if previous in ("STRESS", "CAUTION") else "NORMAL"
    elif n == 1:
        new_package = "CAUTION"
    else:
        new_package = "STRESS"

    return AlcoTransitionResult(package=new_package, triggering_observations=tuple(triggers))
