"""Deterministic, explicitly SYNTHETIC bar frames for research tests and the CLI ``--source synthetic`` fallback.

Bars follow the real XNYS session calendar (early closes included) and a seeded geometric random walk.
Every row has ``is_synthetic=True`` (unless a test deliberately overrides it) and a ``dataset_version``
starting with ``research-synthetic`` so results can never be mistaken for market data. Prices are cents
(``Decimal``); nothing here is evidence about real markets.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .frames import timeframe_delta

DEFAULT_SYNTHETIC_VERSION = "research-synthetic-v1"


@lru_cache(maxsize=1)
def _xnys() -> Any:
    import exchange_calendars as xcals  # heavy import only when generating

    with warnings.catch_warnings():  # third-party pandas deprecation noise inside exchange_calendars
        warnings.simplefilter("ignore")
        return xcals.get_calendar("XNYS")


def xnys_sessions(start: str, n_sessions: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(open_utc, close_utc) of the first ``n_sessions`` XNYS sessions on/after ``start``."""
    cal = _xnys()
    first = pd.Timestamp(start)
    sessions = cal.sessions_in_range(first, first + pd.Timedelta(days=int(n_sessions * 1.6) + 15))
    if len(sessions) < n_sessions:
        raise ValueError("calendar range too short for requested sessions")
    return [(cal.session_open(s), cal.session_close(s)) for s in sessions[:n_sessions]]


def _spans(timeframe: str, sessions: Sequence[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    delta = timeframe_delta(timeframe)
    if delta >= pd.Timedelta(days=1):
        if delta != pd.Timedelta(days=1):
            raise ValueError("only 1Day daily bars are generated")
        return list(sessions)
    spans: list[tuple[Any, Any]] = []
    step = pd.Timedelta(delta)
    for o, c in sessions:
        t = o
        while t + step <= c:
            spans.append((t, t + step))
            t += step
    return spans


def _fmt_price(values: np.ndarray, price_type: str) -> list[Any]:
    cents = [f"{x:.2f}" for x in values.tolist()]
    if price_type == "decimal":
        return [Decimal(c) for c in cents]
    if price_type == "float":
        return [float(c) for c in cents]
    if price_type == "str":
        return cents
    raise ValueError(f"unknown price_type {price_type!r}")


def make_synthetic_bars(
    symbols: Sequence[str] = ("AAA", "BBB"),
    *,
    timeframe: str = "1Day",
    n_sessions: int = 300,
    start: str = "2023-01-03",
    seed: int = 7,
    drift: float = 0.0003,
    vol: float = 0.012,
    drifts: Mapping[str, float] | None = None,
    vols: Mapping[str, float] | None = None,
    latency_s: float = 0.0,
    dataset_version: str = DEFAULT_SYNTHETIC_VERSION,
    base_price: float = 100.0,
    base_volume: int = 2_000_000,
    is_synthetic: bool = True,
    price_type: str = "decimal",
) -> pd.DataFrame:
    """Seeded synthetic bars. ``drift``/``vol`` are per *daily session*; intraday bars are scaled by sqrt(time).

    Columns follow the research input contract: ``symbol, timeframe, start_ts, end_ts, open, high, low,
    close, volume, available_at_ts, dataset_version, is_synthetic``. ``available_at_ts = end_ts + latency_s``.
    """
    sessions = xnys_sessions(start, n_sessions)
    spans = _spans(timeframe, sessions)
    n = len(spans)
    starts = pd.DatetimeIndex([s for s, _ in spans]).tz_convert("UTC").as_unit("ns")
    ends = pd.DatetimeIndex([e for _, e in spans]).tz_convert("UTC").as_unit("ns")
    delta_s = timeframe_delta(timeframe).total_seconds()
    per_day = 1.0 if delta_s >= 86_400 else delta_s / (390 * 60)
    scale = float(np.sqrt(per_day))
    frames: list[pd.DataFrame] = []
    for k, sym in enumerate(symbols):
        rng = np.random.RandomState(seed + 1000 * k)
        mu = float((drifts or {}).get(sym, drift))
        sigma = float((vols or {}).get(sym, vol))
        gap = rng.normal(0.0, sigma * scale * 0.2, size=n)
        body = mu * per_day + rng.normal(0.0, sigma * scale, size=n)
        wick_hi = np.abs(rng.normal(0.0, sigma * scale * 0.3, size=n))
        wick_lo = np.abs(rng.normal(0.0, sigma * scale * 0.3, size=n))
        vol_noise = np.abs(rng.normal(0.0, 0.3, size=n))
        log_close = np.log(base_price * (1 + 0.1 * k)) + np.cumsum(gap + body)
        close = np.exp(log_close)
        open_ = np.exp(log_close - body)
        high = np.maximum(open_, close) * (1 + wick_hi)
        low = np.minimum(open_, close) * (1 - wick_lo)
        o, h, lo, c = (np.round(x, 2) for x in (open_, high, low, close))
        h = np.maximum.reduce([o, h, lo, c])
        lo = np.minimum.reduce([o, h, lo, c])
        lo = np.maximum(lo, 0.01)
        volume = np.floor(base_volume * per_day * (1 + vol_noise)).astype(np.int64)
        if price_type == "decimal":
            vcol: list[Any] = [Decimal(int(v)) for v in volume.tolist()]
        elif price_type == "float":
            vcol = [float(v) for v in volume.tolist()]
        else:
            vcol = [str(int(v)) for v in volume.tolist()]
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "timeframe": timeframe,
                    "start_ts": starts,
                    "end_ts": ends,
                    "open": _fmt_price(o, price_type),
                    "high": _fmt_price(h, price_type),
                    "low": _fmt_price(lo, price_type),
                    "close": _fmt_price(c, price_type),
                    "volume": vcol,
                    "available_at_ts": ends + pd.Timedelta(seconds=latency_s),
                    "dataset_version": dataset_version,
                    "is_synthetic": is_synthetic,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
