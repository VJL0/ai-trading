"""Decimal arithmetic helpers. No floats in accounting paths."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, getcontext

getcontext().prec = 34

ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10_000)
CENT = Decimal("0.01")
SUBPENNY = Decimal("0.0001")


def D(value: object) -> Decimal:
    """Convert to Decimal, refusing binary floats."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a decimal amount")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, float):
        raise TypeError("float rejected; use str(value) explicitly at an I/O boundary if unavoidable")
    raise TypeError(f"cannot convert {type(value).__name__} to Decimal")


def from_external_float(value: float | int | str | None) -> Decimal | None:
    """For third-party payloads that only expose floats (e.g. some SDK fields): go via repr string."""
    if value is None:
        return None
    if isinstance(value, float):
        return Decimal(repr(value))
    return D(value)


def price_tick(price: Decimal) -> Decimal:
    """U.S. equities: $0.01 increments at/above $1.00, $0.0001 below (Reg NMS Rule 612)."""
    return CENT if price >= ONE else SUBPENNY


def round_price(price: Decimal, *, side_buy: bool | None = None) -> Decimal:
    """Round to a valid tick. For limit prices, round *toward the passive side* when side is given."""
    tick = price_tick(price)
    if side_buy is None:
        return price.quantize(tick, rounding=ROUND_HALF_EVEN)
    return price.quantize(tick, rounding=ROUND_FLOOR if side_buy else ROUND_CEILING)


def floor_shares(qty: Decimal) -> Decimal:
    """Whole shares only (no fractional orders in the initial build)."""
    return qty.to_integral_value(rounding=ROUND_DOWN)


def bps(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        raise ZeroDivisionError("bps denominator is zero")
    return numerator / denominator * BPS


def apply_bps(value: Decimal, basis_points: Decimal) -> Decimal:
    return value * (ONE + basis_points / BPS)
