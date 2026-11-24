"""Deterministic synthetic daily bars for quantlab tests (explicitly SIMULATED; never evidence of anything)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.market_data.calendar import get_calendar


def sessions(start: str = "2018-01-02", n: int = 600) -> list[pd.Timestamp]:
    cal = get_calendar()
    end = min((pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6))).date(), cal.last_session)
    days = cal.sessions_in_range(pd.Timestamp(start).date(), end)
    return [pd.Timestamp(d) for d in days[:n]]


def bars(symbols: dict[str, dict[str, float]], n: int = 600, seed: int = 1, start: str = "2018-01-02",
         drop: dict[str, list[int]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``symbols``: name -> {drift, vol, price, volume}. Returns (raw, adjusted) frames in the fetch format."""
    rng = np.random.default_rng(seed)
    days = sessions(start, n)
    rows = []
    for sym, p in symbols.items():
        r = p.get("drift", 0.0) + p.get("vol", 0.01) * rng.standard_normal(n)
        close = p.get("price", 50.0) * np.exp(np.cumsum(r))
        gap = 0.3 * p.get("vol", 0.01) * rng.standard_normal(n)
        opn = close / np.exp(r) * np.exp(gap)
        for i, d in enumerate(days):
            if drop and i in drop.get(sym, []):
                continue
            t = pd.Timestamp(d).tz_localize("America/New_York").tz_convert("UTC")
            hi = max(opn[i], close[i]) * 1.005
            lo = min(opn[i], close[i]) * 0.995
            rows.append((sym, t, opn[i], hi, lo, close[i], p.get("volume", 2e6), 1000, (hi + lo) / 2))
    df = pd.DataFrame(rows, columns=["symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])
    return df, df.copy()


def assets(symbols: list[str], kind: str = "COMMON_STOCK", etfs: tuple[str, ...] = ("SPY",)) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "asset_class": "us_equity", "exchange": "NYSE",
                         "kind": ["ETF" if s in etfs else kind for s in symbols], "research_kind": True,
                         "tradable": True, "status": "active"})
