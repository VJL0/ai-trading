"""Point-in-time corporate-action handling for research on RAW (unadjusted) bar data.

Historical datasets such as the LEAN sample are stored ``adjustment="raw"``: a 2-for-1 split shows up as a
fake -50 % overnight "crash" (a 1-for-20 reverse split as a +1900 % jump). Backtests, labels and features on
such prices are wrong, so research applies corporate actions explicitly:

* **Splits** (``split_factor`` = price multiplier, e.g. ``0.25`` for a 4-for-1 forward split): bars whose NYSE
  session date is *before* the ex-date get ``price x factor`` and ``volume / factor`` (cumulative over every
  later split). Every feature and strategy rule in this package is scale-free (log returns, ratios, z-scores,
  dollar ADV = volume x price), so backward adjustment equals "as-of" adjustment for them: it changes price
  *levels* only, never a return visible at decision time. ``asof`` additionally drops every record not yet
  published at ``asof`` (``available_at_ts > asof``), which is what the decision-time path uses.
* **Dividends** are returned as split-adjusted cash per share by ex-date; ``baselines.run_backtest`` credits
  them to cash for positions held into the ex-date (price-only labels/features do not use them).
* RENAME / DELISTING records are reported, not applied (bars are already keyed by the stored symbol).

Money stays ``Decimal``; adjusted prices are quantised to 1e-6 (the store's DECIMAL(18,6) scale) with
round-half-even, which preserves the OHLC ordering. Share counts in adjusted units differ from raw share
counts before a split, which only affects per-share fee terms (FINRA TAF; conservative) and whole-share
rounding. This module never imports ``services.market_data``: records are duck-typed (attributes or keys).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

import pandas as pd

from .frames import NY, OHLCV, BarDataError, to_decimal, utc_ts

PRICE_Q = Decimal("0.000001")
VOLUME_Q = Decimal("0.000001")
ADJUSTED_POLICIES = {"split", "all"}  # stored bars already split-adjusted -> never adjust twice


@dataclass(frozen=True)
class AdjustmentResult:
    bars: pd.DataFrame
    dividends: dict[str, dict[date, Decimal]] = field(default_factory=lambda: {})
    report: dict[str, Any] = field(default_factory=lambda: {})


def _get(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)  # pyright: ignore[reportUnknownMemberType]
    return getattr(record, name, None)


def _action_type(record: Any) -> str:
    value = _get(record, "action_type")
    return str(getattr(value, "value", value) or "").upper()


def _ex_date(record: Any) -> date:
    value = _get(record, "ex_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise BarDataError(f"corporate action without a valid ex_date: {value!r}")


def _positive_decimal(record: Any, name: str) -> Decimal:
    raw = _get(record, name)
    if raw is None:
        raise BarDataError(f"corporate action missing {name}")
    value = to_decimal(raw)
    if value <= 0:
        raise BarDataError(f"corporate action {name} must be positive, got {value}")
    return value


def apply_corporate_actions(
    bars_df: pd.DataFrame,
    actions: Iterable[Any],
    *,
    adjustment: str = "raw",
    asof: datetime | pd.Timestamp | None = None,
) -> AdjustmentResult:
    """Split-adjust a RAW bar frame and collect split-adjusted dividends (see module docstring).

    ``adjustment`` is the dataset's stored policy: ``split``/``all`` frames are returned unchanged (never
    adjusted twice; dividends are then not credited because their per-share basis is unknown).
    """
    policy = (adjustment or "raw").lower()
    report: dict[str, Any] = {"input_adjustment": policy, "asof": None if asof is None else utc_ts(asof)}
    if policy in ADJUSTED_POLICIES:
        report["applied"] = f"none (stored bars already adjusted: {policy})"
        return AdjustmentResult(bars_df, {}, report)
    if policy not in {"raw", "dividend"}:
        raise BarDataError(f"unknown adjustment policy {adjustment!r}")
    cutoff = None if asof is None else utc_ts(asof)
    symbols = {str(s) for s in bars_df["symbol"].unique()} if "symbol" in bars_df.columns else set()

    splits: dict[str, list[tuple[date, Decimal]]] = {}
    raw_dividends: dict[str, list[tuple[date, Decimal]]] = {}
    ignored_unpublished = 0
    not_applied: dict[str, int] = {}
    for rec in actions:
        sym = str(_get(rec, "symbol") or "")
        if sym not in symbols:
            continue
        avail = _get(rec, "available_at_ts")
        if cutoff is not None:
            if avail is None:
                ignored_unpublished += 1  # publication time unknown: not knowable at a decision time
                continue
            if utc_ts(avail) > cutoff:
                ignored_unpublished += 1
                continue
        kind = _action_type(rec)
        if kind == "SPLIT":
            splits.setdefault(sym, []).append((_ex_date(rec), _positive_decimal(rec, "split_factor")))
        elif kind == "DIVIDEND":
            raw_dividends.setdefault(sym, []).append((_ex_date(rec), _positive_decimal(rec, "cash_amount")))
        else:
            not_applied[kind or "UNKNOWN"] = not_applied.get(kind or "UNKNOWN", 0) + 1

    out = bars_df.copy()
    if splits and not out.empty:
        start = pd.to_datetime(out["start_ts"], utc=True)
        session = pd.Series(list(start.dt.tz_convert(NY).dt.date), index=out.index, dtype=object)
        sym_col = out["symbol"].astype(str)
        for col in OHLCV:
            out[col] = pd.Series([to_decimal(v) for v in out[col].tolist()], index=out.index, dtype=object)
        for sym, events in splits.items():
            ordered = sorted(events)
            idx = out.index[(sym_col == sym).to_numpy()]
            for i in idx:
                d = session.at[i]
                factor = Decimal(1)
                for ex, f in ordered:
                    if d < ex:
                        factor *= f
                if factor == 1:
                    continue
                for col in ("open", "high", "low", "close"):
                    out.at[i, col] = (out.at[i, col] * factor).quantize(PRICE_Q, rounding=ROUND_HALF_EVEN)
                out.at[i, "volume"] = (out.at[i, "volume"] / factor).quantize(VOLUME_Q, rounding=ROUND_HALF_EVEN)

    dividends: dict[str, dict[date, Decimal]] = {}
    if policy == "dividend":  # prices already reflect dividends: crediting cash too would double count
        raw_dividends = {}
        report["dividends_not_credited"] = "stored bars are dividend-adjusted"
    for sym, divs in raw_dividends.items():
        later = sorted(splits.get(sym, []))
        per: dict[date, Decimal] = {}
        for ex, cash in sorted(divs):
            factor = Decimal(1)
            for sx, f in later:
                if ex < sx:
                    factor *= f
            per[ex] = per.get(ex, Decimal(0)) + (cash * factor).quantize(PRICE_Q, rounding=ROUND_HALF_EVEN)
        dividends[sym] = per

    report.update(
        {
            "applied": "split (backward, cumulative) from point-in-time corporate-action records" if splits
            else "none (no split records for these symbols)",
            "splits": {s: [{"ex_date": ex.isoformat(), "split_factor": format(f, "f")} for ex, f in sorted(v)]
                       for s, v in sorted(splits.items())},
            "dividend_events": {s: len(v) for s, v in sorted(dividends.items())},
            "ignored_not_yet_published": ignored_unpublished,
            "reported_not_applied": not_applied,
            "dividend_treatment": "cash credit to positions held into the ex-date (backtests only); "
            "features/labels are price-only",
        }
    )
    return AdjustmentResult(out, dividends, report)
