"""Half-open Decimal intervals ``(a, b]`` over an order's cumulative filled quantity.

Each execution report with a cumulative quantity ``cum`` and fill quantity ``q`` covers ``(cum - q, cum]``.
Reconciliation books catch-up fills only for the *uncovered* part of ``(0, broker_cum]``; a late execution
report is then absorbed exactly by its overlap with those catch-up ranges, so nothing is double-booked and
nothing is dropped, whatever order the reports arrive in.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

Interval = tuple[Decimal, Decimal]

_ZERO = Decimal(0)


def normalize(intervals: Iterable[Interval]) -> list[Interval]:
    """Sorted, merged, non-empty intervals."""
    items = sorted((a, b) for a, b in intervals if b > a)
    merged: list[Interval] = []
    for a, b in items:
        if merged and a <= merged[-1][1]:
            last_a, last_b = merged[-1]
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def measure(intervals: Iterable[Interval]) -> Decimal:
    return sum((b - a for a, b in normalize(intervals)), _ZERO)


def overlap(interval: Interval, union: Iterable[Interval]) -> Decimal:
    """Length of ``interval`` intersected with the union of ``union``."""
    lo, hi = interval
    total = _ZERO
    for a, b in normalize(union):
        left, right = max(lo, a), min(hi, b)
        if right > left:
            total += right - left
    return total


def gaps(covered: Iterable[Interval], upto: Decimal) -> list[Interval]:
    """Parts of ``(0, upto]`` not covered by ``covered``."""
    out: list[Interval] = []
    cursor = _ZERO
    for a, b in normalize(covered):
        if a >= upto:
            break
        if a > cursor:
            out.append((cursor, min(a, upto)))
        cursor = max(cursor, b)
        if cursor >= upto:
            break
    if cursor < upto:
        out.append((cursor, upto))
    return [(a, b) for a, b in out if b > a]
