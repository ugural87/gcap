"""Decimal-only numeric primitives shared by the deterministic engines.

Invariant 1 requires every number consumed downstream as truth to come from
Python arithmetic, never from an LLM. That only holds if the arithmetic
itself is exact enough not to introduce its own silent error, so every
financial computation in core/ uses ``decimal.Decimal`` with an explicit,
generous working precision — never ``float``.
"""

from __future__ import annotations

from decimal import Context, Decimal, InvalidOperation, localcontext

from gcap.errors import CoreValidationError

# 34 significant digits (roughly IEEE 754 decimal128) is ample headroom for
# chained FTP/portfolio arithmetic without accumulating visible rounding
# error at the basis-point level tested in property tests.
WORKING_PRECISION = 34


def working_context() -> Context:
    return Context(prec=WORKING_PRECISION)


def to_decimal(x: Decimal | int | str) -> Decimal:
    """Coerce to Decimal without ever passing through binary float.
    Floats are rejected outright: a caller with a float has already lost
    the precision guarantee this module exists to provide.
    """
    if isinstance(x, Decimal):
        return x
    if isinstance(x, float):
        raise CoreValidationError(
            f"Refusing to convert float {x!r} to Decimal implicitly; pass a "
            "Decimal or str to avoid silently importing binary-float error "
            "into a financial calculation."
        )
    with localcontext(working_context()):
        return Decimal(x)


def horizon_from_annual(rate_ann: Decimal, tau_days: Decimal, day_count_basis: Decimal) -> Decimal:
    """(1 + rate_ann) ** (tau/D) - 1 — converts an annualised rate to the
    holding-period rate over tau_days under day-count basis D."""
    if day_count_basis <= 0:
        raise CoreValidationError("day_count_basis must be positive.")
    if tau_days <= 0:
        raise CoreValidationError("tau_days must be positive.")
    base = Decimal(1) + rate_ann
    if base <= 0:
        raise CoreValidationError(
            f"Annualised rate {rate_ann} implies a non-positive base (1+r)={base}; "
            "cannot raise a non-positive base to a fractional power."
        )
    with localcontext(working_context()):
        exponent = tau_days / day_count_basis
        try:
            value = base**exponent
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise CoreValidationError(f"Fractional power failed for base={base}, exponent={exponent}") from exc
        return value - Decimal(1)


def annual_from_horizon(rate_horizon: Decimal, tau_days: Decimal, day_count_basis: Decimal) -> Decimal:
    """(1 + rate_horizon) ** (D/tau) - 1 — converts a holding-period rate to
    its annualised equivalent. Requires rate_horizon > -1 (Section 5.1)."""
    if rate_horizon <= Decimal("-1"):
        raise CoreValidationError("rate_horizon must be > -1 (a holding-period return of -100% or worse is not annualisable).")
    if day_count_basis <= 0:
        raise CoreValidationError("day_count_basis must be positive.")
    if tau_days <= 0:
        raise CoreValidationError("tau_days must be positive.")
    base = Decimal(1) + rate_horizon
    with localcontext(working_context()):
        exponent = day_count_basis / tau_days
        try:
            value = base**exponent
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise CoreValidationError(f"Fractional power failed for base={base}, exponent={exponent}") from exc
        return value - Decimal(1)


def quantize(value: Decimal, places: str = "0.00000001") -> Decimal:
    """Quantize a Decimal to a fixed number of places at the *display*
    layer only. Internal engine values are never quantized before being
    stored — basis points only appear at display (Section 4.1)."""
    with localcontext(working_context()):
        return value.quantize(Decimal(places))
