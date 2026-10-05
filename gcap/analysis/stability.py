"""The 29-of-30 directional-stability rule (Section 9.5 / 10): "Across 30
evaluable draws, this requires at least 29 sign-preserving draws: the lower
endpoint of the two-sided 95% Clopper-Pearson interval is approximately
0.828 for 29/30 and 0.780 for 28/30."
"""

from __future__ import annotations

from gcap.analysis.intervals import clopper_pearson


def stability_rule(
    sign_preserving: int,
    of: int = 30,
    min_sign_preserving: int = 29,
    lower_bound_floor: float = 0.80,
) -> tuple[bool, float]:
    """Returns (passes, clopper_pearson_lower_bound). Passes iff both the
    raw count meets the floor AND the two-sided 95% CI lower bound clears
    ``lower_bound_floor`` — the count condition and the CI condition are
    each necessary; Section 9.5 states both explicitly ("requires at least
    29... the lower endpoint... is approximately 0.828")."""
    lower, _upper = clopper_pearson(sign_preserving, of, confidence=0.95)
    passes = sign_preserving >= min_sign_preserving and lower >= lower_bound_floor
    return passes, lower
