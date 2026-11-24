"""IEX-only versus consolidated (SIP) minute bars: evidence before any paid market-data decision (roadmap Phase 2).

Alpaca's free Basic plan streams *real-time* data from IEX only, but its historical endpoint serves full-market SIP
bars for anything older than the most recent 15 minutes. So the quality cost of trading on IEX can be measured for
free: fetch the same completed sessions from both feeds and compare them minute by minute.

Per symbol the report gives regular-session minute coverage of each feed, the IEX share of consolidated volume, the
distribution of close/VWAP differences on minutes both feeds have, 1-minute return correlation and the realized
volatility ratio. Nothing is stored, nothing is purchased, and no feed setting changes; the report is evidence for a
human decision. A model trained on SIP bars would also not see live what it was trained on (the runtime only has
IEX in real time), so the report states that parity consequence explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from statistics import median
from typing import Any

from .calendar import MarketCalendar, get_calendar
from .common import iso, to_utc


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        out = float(v if not isinstance(v, Decimal) else str(v))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return round(s[idx], 4)


def _corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb), 4)


def _index(records: Sequence[Mapping[str, Any]]) -> dict[datetime, Mapping[str, Any]]:
    out: dict[datetime, Mapping[str, Any]] = {}
    for r in records:
        ts = r.get("start_ts")
        if isinstance(ts, datetime) and r.get("close") is not None:
            out[to_utc(ts)] = r
    return out


def compare_bar_sets(iex: Sequence[Mapping[str, Any]], sip: Sequence[Mapping[str, Any]], *, start: datetime,
                     end: datetime, calendar: MarketCalendar | None = None) -> dict[str, Any]:
    """Minute-level comparison of one symbol's IEX and SIP regular-session bars inside ``[start, end)``."""
    cal = calendar or get_calendar()
    days = cal.sessions_in_range(cal.ny_date(start), cal.ny_date(end))
    expected = {t for d in days for t in cal.regular_bar_starts(d) if start <= t < end}
    ie = {t: r for t, r in _index(iex).items() if t in expected}
    si = {t: r for t, r in _index(sip).items() if t in expected}
    common = sorted(set(ie) & set(si))
    close_bps: list[float] = []
    vwap_bps: list[float] = []
    for t in common:
        a, b = _f(ie[t]["close"]), _f(si[t]["close"])
        if a and b:
            close_bps.append(abs(a - b) / b * 10_000)
        va, vb = _f(ie[t].get("vwap")), _f(si[t].get("vwap"))
        if va and vb:
            vwap_bps.append(abs(va - vb) / vb * 10_000)
    vol_iex = sum((_f(r.get("volume")) or 0.0) for r in ie.values())
    vol_sip = sum((_f(r.get("volume")) or 0.0) for r in si.values())
    # 1-minute log returns on consecutive minutes present in both feeds
    ra: list[float] = []
    rb: list[float] = []
    for p, q in pairwise(common):
        if (q - p).total_seconds() != 60:
            continue
        a0, a1, b0, b1 = (_f(ie[p]["close"]), _f(ie[q]["close"]), _f(si[p]["close"]), _f(si[q]["close"]))
        if a0 and a1 and b0 and b1:
            ra.append(math.log(a1 / a0))
            rb.append(math.log(b1 / b0))
    rv_iex = math.sqrt(sum(x * x for x in ra)) if ra else None
    rv_sip = math.sqrt(sum(x * x for x in rb)) if rb else None
    n = len(expected)
    return {
        "sessions": len(days), "expected_regular_minutes": n,
        "iex_minutes": len(ie), "sip_minutes": len(si), "common_minutes": len(common),
        "iex_minute_coverage": round(len(ie) / n, 4) if n else None,
        "sip_minute_coverage": round(len(si) / n, 4) if n else None,
        "minutes_missing_in_iex_present_in_sip": len(set(si) - set(ie)),
        "iex_volume_share": round(vol_iex / vol_sip, 4) if vol_sip else None,
        "close_diff_bps": {"median": round(median(close_bps), 4) if close_bps else None,
                           "p95": _pct(close_bps, 0.95), "max": round(max(close_bps), 4) if close_bps else None},
        "vwap_diff_bps": {"median": round(median(vwap_bps), 4) if vwap_bps else None, "p95": _pct(vwap_bps, 0.95)},
        "return_correlation_1m": _corr(ra, rb),
        "realized_vol_ratio_iex_over_sip": round(rv_iex / rv_sip, 4) if rv_iex and rv_sip else None,
    }


def summarize(per_symbol: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cov = [v["iex_minute_coverage"] for v in per_symbol.values() if v.get("iex_minute_coverage") is not None]
    share = [v["iex_volume_share"] for v in per_symbol.values() if v.get("iex_volume_share") is not None]
    p95 = [v["close_diff_bps"]["p95"] for v in per_symbol.values() if v.get("close_diff_bps", {}).get("p95") is not None]
    return {
        "median_iex_minute_coverage": round(median(cov), 4) if cov else None,
        "median_iex_volume_share": round(median(share), 4) if share else None,
        "median_close_diff_bps_p95": round(median(p95), 4) if p95 else None,
        "interpretation": [
            "Coverage below 1.0 means minutes with no IEX trade: IEX minute bars are missing there, never filled.",
            "Close differences are the price error of using IEX prints instead of consolidated prints for the same "
            "minute; compare them with the strategy's expected edge and cost estimates before deciding.",
            "Training on historical SIP while trading on real-time IEX breaks research/runtime parity (Phase 3); "
            "real-time SIP needs the paid Algo Trader Plus plan, which this system never purchases or enables.",
        ],
    }


async def compare_alpaca_feeds(secrets: Any, symbols: Sequence[str], start: datetime, end: datetime, *, clock: Any,
                               calendar: MarketCalendar | None = None) -> dict[str, Any]:
    """Fetch the same completed window from Alpaca IEX and historical SIP (read-only, nothing stored)."""
    from packages.contracts.enums import FeedMode

    from .alpaca_provider import AlpacaMarketDataProvider

    cal = calendar or get_calendar()
    iex = AlpacaMarketDataProvider(secrets, FeedMode.IEX_FREE, clock)
    sip = AlpacaMarketDataProvider(secrets, FeedMode.SIP_PAID, clock)  # historical SIP only; window ends >= 15 min ago
    out: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            a, _ = await iex.fetch_bar_records(sym, start, end, "1Min")
            b, _ = await sip.fetch_bar_records(sym, start, end, "1Min")
        except Exception as exc:  # report, never guess
            errors[sym] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        out[sym] = compare_bar_sets(a, b, start=start, end=end, calendar=cal)
    return {"window": {"start": iso(start), "end": iso(end)}, "symbols": out, "errors": errors,
            "summary": summarize(out), "source": "alpaca historical bars: feed=iex vs feed=sip (read-only)",
            "real_alpaca_data": True}
