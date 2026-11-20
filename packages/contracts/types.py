"""Primitive validated types shared by every contract.

Rules enforced here (spec §5):
* Money, quantities and prices are ``Decimal``. Binary floats are *rejected* at the boundary so
  accounting never silently inherits float rounding error. Strings/ints/Decimals are accepted.
* Timestamps must be timezone-aware and are normalised to UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, PlainSerializer


def _reject_float(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid decimal amounts")
    if isinstance(value, float):
        raise ValueError(
            "binary float rejected for a money/quantity/price field; pass a string, int or Decimal"
        )
    if isinstance(value, str):
        try:
            d = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"invalid decimal string: {value!r}") from exc
        if not d.is_finite():
            raise ValueError("non-finite decimal rejected")
        return d
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("non-finite decimal rejected")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("naive datetime rejected; timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _decimal_to_str(value: Decimal) -> str:
    # Canonical, exponent-free string so JSON round-trips without float conversion.
    return format(value, "f")


def _dt_to_str(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


Dec = Annotated[
    Decimal,
    BeforeValidator(_reject_float),
    PlainSerializer(_decimal_to_str, return_type=str, when_used="json"),
]
"""Decimal amount: price, quantity, money, bps. Never a float."""

UtcDatetime = Annotated[
    datetime,
    AfterValidator(_require_utc),
    PlainSerializer(_dt_to_str, return_type=str, when_used="json"),
]
"""Timezone-aware UTC timestamp, serialised as ISO-8601 with a trailing Z."""


def utc_now() -> datetime:
    return datetime.now(UTC)
