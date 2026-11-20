"""Deterministic synthetic bar frames for research tests (explicitly synthetic: is_synthetic=True)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from services.research.synthetic import make_synthetic_bars


def make_bars(
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    *,
    timeframe: str = "1Day",
    n: int = 300,
    start: str = "2023-01-03",
    seed: int = 7,
    drift: float = 0.0003,
    vol: float = 0.012,
    latency_s: float = 0.0,
    dataset_version: str = "synthetic-test-v1",
    base_price: float = 100.0,
    base_volume: int = 2_000_000,
    is_synthetic: bool = True,
    price_type: str = "decimal",
    drifts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """``n`` = number of XNYS sessions (daily: one bar each; intraday: every bar of each session)."""
    return make_synthetic_bars(
        symbols,
        timeframe=timeframe,
        n_sessions=n,
        start=start,
        seed=seed,
        drift=drift,
        vol=vol,
        drifts=drifts,
        latency_s=latency_s,
        dataset_version=dataset_version,
        base_price=base_price,
        base_volume=base_volume,
        is_synthetic=is_synthetic,
        price_type=price_type,
    )


def ts(value: Any) -> pd.Timestamp:
    """Aware pandas Timestamp from a scalar/0-d value (typed helper; NaT rejected)."""
    out = pd.Timestamp(value)
    assert isinstance(out, pd.Timestamp) and not pd.isna(out), value
    return out


def to_dt(value: Any) -> datetime:
    """Python datetime from a timestamp-like value (typed helper; NaT rejected)."""
    out = ts(value).to_pydatetime()
    assert isinstance(out, datetime)
    return out


def bar_row(
    symbol: str,
    start: Any,
    *,
    minutes: int = 1,
    o: str = "100",
    h: str | None = None,
    lo: str | None = None,
    c: str = "100",
    v: str = "1000",
    latency_s: float = 0.0,
    timeframe: str = "1Min",
    dataset_version: str = "synthetic-test-v1",
    is_synthetic: bool = True,
) -> dict[str, Any]:
    """One explicit bar row (for hand-built edge-case frames)."""
    start = ts(start)
    end = start + pd.Timedelta(minutes=minutes)
    od, cd = Decimal(o), Decimal(c)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_ts": start,
        "end_ts": end,
        "open": od,
        "high": Decimal(h) if h is not None else max(od, cd),
        "low": Decimal(lo) if lo is not None else min(od, cd),
        "close": cd,
        "volume": Decimal(v),
        "available_at_ts": end + pd.Timedelta(seconds=latency_s),
        "dataset_version": dataset_version,
        "is_synthetic": is_synthetic,
    }


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("start_ts", "end_ts", "available_at_ts"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def shift_minutes(value: pd.Timestamp, minutes: float) -> pd.Timestamp:
    return ts(value + timedelta(minutes=minutes))
