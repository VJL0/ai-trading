"""Strategy D — crypto trend / momentum on Alpaca-tradable USD pairs (24/7, UTC days or hours).

Literature: time-series momentum in crypto (Liu & Tsyvinski 2021 "Risks and Returns of Cryptocurrency"; Liu,
Tsyvinski & Wu 2022 cross-sectional crypto factors); volatility targeting (Moreira & Muir 2017). Alpaca tier-1
fees (25 bps taker) are large relative to daily edges, so turnover is the decisive variable.

The crypto panel re-uses :class:`EquityPanel` with UTC-day (or hour) rows; ``ret_co`` is the bar-open gap
(≈0 for a continuous market) so decisions at a bar close execute at the next bar open.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import apply_rebalance_schedule, hold_schedule
from ..panel import EquityPanel, compound, segment_count, shift, trailing_mean, trailing_std

FAMILY_TSMOM = "D1_crypto_tsmom"
FAMILY_XSMOM = "D2_crypto_xsmom"
FAMILY_INTRADAY = "D3_crypto_hourly_breakout"
CORE = ("BTC/USD", "ETH/USD", "SOL/USD")
MAX_GAP_DAYS = 3
"""A 24/7 pair missing more than this many days is treated as halted/delisted: new segment, forced exit."""


def build_crypto_panel(bars: pd.DataFrame, *, freq: str = "1D") -> EquityPanel:
    """``bars``: columns symbol,t,o,h,l,c,v,vw (UTC). ``freq`` '1D' resamples hourly bars to UTC days."""
    df = bars.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    if freq == "1D":
        df["k"] = df["t"].dt.floor("1D")
        g = df.sort_values("t").groupby(["symbol", "k"])
        agg = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        dv = (df["c"] * df["v"]).groupby([df["symbol"], df["k"]]).sum()
        agg["dv"] = dv
        agg = agg.reset_index()
    else:
        agg = df.rename(columns={"t": "k"})
        agg["dv"] = agg["c"] * agg["v"]
    idx = pd.DatetimeIndex(sorted(agg["k"].unique())).tz_convert(None)
    syms = np.array(sorted(agg["symbol"].unique()))
    T, N = len(idx), len(syms)
    di, si = pd.Index(idx), pd.Index(syms)
    r = di.get_indexer(pd.DatetimeIndex(agg["k"]).tz_convert(None))
    c = si.get_indexer(pd.Index(agg["symbol"].to_numpy()))

    def grid(col: str) -> np.ndarray:
        out = np.full((T, N), np.nan)
        out[r, c] = agg[col].to_numpy(dtype=float)
        return out

    f = {"open": grid("o"), "high": grid("h"), "low": grid("l"), "close": grid("c"), "volume": grid("v"),
         "dollar_vol": grid("dv")}
    close = f["close"]
    present = np.isfinite(close) & (close > 0)
    # a gap longer than MAX_GAP rows (a halt/delisting/relisting on a 24/7 market) starts a new segment: no return
    # spans it and the first bar's open is not trusted (relisting prints)
    max_gap = MAX_GAP_DAYS if freq == "1D" else MAX_GAP_DAYS * 24
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(present, rows, -1), axis=0)
    prev_obs = np.vstack([np.full((1, N), -1), last[:-1]])
    gap = np.where(present & (prev_obs >= 0), rows - prev_obs - 1, 0)
    brk = present & ((prev_obs < 0) | (gap > max_gap))
    prev = shift(close, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["ret_cc"] = np.where(present & ~brk, close / prev - 1, np.nan)
        f["ret_oc"] = np.where(present & ~brk, close / f["open"] - 1, np.nan)
        f["ret_co"] = np.where(present & ~brk, f["open"] / prev - 1, np.nan)
    f["segment"] = np.where(present, np.cumsum(brk, axis=0), np.nan).astype(float)
    f["adj_close"], f["adj_open"] = close.copy(), f["open"].copy()
    f["present"] = present.astype(float)
    win = 30 if freq == "1D" else 24 * 30
    f["adv20"] = shift(trailing_mean(np.nan_to_num(f["dollar_vol"]), win, win // 2), 1)
    f["obs_count"] = segment_count(f["segment"])
    return EquityPanel(idx, syms, f, np.array(["CRYPTO"] * N, dtype=object), {"freq": freq})


def vol_target_weights(panel: EquityPanel, signal: np.ndarray, *, target_vol: float, window: int,
                       periods: float) -> np.ndarray:
    vol = trailing_std(panel.f["ret_cc"], window, int(window * 0.8)) * np.sqrt(periods)
    raw = np.where(signal, target_vol / np.where(vol > 0, vol, np.nan), 0.0)
    raw = np.nan_to_num(raw)
    gross = raw.sum(axis=1, keepdims=True)
    return np.where(gross > 1.0, raw / np.where(gross > 0, gross, 1.0), raw)


def tsmom_variants() -> Iterator[dict[str, Any]]:
    for rule, lb, tv, reb, assets in itertools.product(("ret", "sma"), (20, 50, 100), (0.4, 0.8), (1, 7),
                                                       ("btc", "core3")):
        yield {"rule": rule, "lookback": lb, "target_vol": tv, "rebalance": reb, "assets": assets}


def tsmom_targets(panel: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    T, N = panel.shape
    names = ("BTC/USD",) if p["assets"] == "btc" else CORE
    sel = np.zeros(N, dtype=bool)
    for s in names:
        if s in set(panel.symbols):
            sel[panel.col(s)] = True
    lb = int(p["lookback"])
    if p["rule"] == "ret":
        sig = compound(panel.f["ret_cc"], lb, int(lb * 0.8)) > 0
    else:
        s = trailing_mean(panel.f["close"], lb, int(lb * 0.9))
        sig = np.isfinite(s) & (panel.f["close"] > s)
    sig &= sel[None, :] & (np.nan_to_num(panel.f["obs_count"]) >= lb + 30)
    w = vol_target_weights(panel, sig, target_vol=float(p["target_vol"]) / max(1, sel.sum()) ** 0.5,
                           window=30, periods=365.0)
    return apply_rebalance_schedule(w, hold_schedule(T, int(p["rebalance"])))


def xsmom_variants() -> Iterator[dict[str, Any]]:
    for lb, n, reb, regime in itertools.product((14, 30, 60), (2, 3, 5), (7, 14), ("none", "btc_sma50")):
        yield {"lookback": lb, "n": n, "rebalance": reb, "regime": regime}


def xsmom_targets(panel: EquityPanel, p: dict[str, Any], *, min_adv: float = 250_000.0) -> np.ndarray:
    from .momentum import top_n_weights

    T, _n = panel.shape
    lb = int(p["lookback"])
    sc = compound(panel.f["ret_cc"], lb, int(lb * 0.8))
    uni = (np.nan_to_num(panel.f["adv20"]) >= min_adv) & (np.nan_to_num(panel.f["obs_count"]) >= 90)
    w = top_n_weights(sc, uni, int(p["n"]), buffer=1.5)
    if p["regime"] == "btc_sma50":
        b = panel.col("BTC/USD")
        s = trailing_mean(panel.f["close"][:, [b]], 50, 45)[:, 0]
        on = np.isfinite(s) & (panel.f["close"][:, b] > s)
        w = w * on[:, None]
    return apply_rebalance_schedule(w, hold_schedule(T, int(p["rebalance"])))


def breakout_variants() -> Iterator[dict[str, Any]]:
    for lb, hold in itertools.product((24, 72, 168), (6, 24)):
        yield {"lookback_h": lb, "hold_h": hold}


def breakout_targets(panel: EquityPanel, p: dict[str, Any]) -> np.ndarray:
    """Hourly: long BTC/ETH after a close above the trailing ``lookback_h``-hour high; hold ``hold_h`` hours."""
    from .reversion import event_positions

    T, N = panel.shape
    hi = shift(pd.DataFrame(panel.f["high"]).rolling(int(p["lookback_h"]), min_periods=int(p["lookback_h"]) // 2)
               .max().to_numpy(), 1)
    sel = np.isin(panel.symbols, ("BTC/USD", "ETH/USD"))
    entry = sel[None, :] & np.isfinite(hi) & (panel.f["close"] > hi)
    never = np.zeros_like(entry)
    return event_positions(entry, never, np.zeros((T, N)), max_positions=2, max_hold=int(p["hold_h"]))


def buy_and_hold(panel: EquityPanel, symbols: tuple[str, ...]) -> np.ndarray:
    T, N = panel.shape
    w = np.zeros((T, N))
    for s in symbols:
        j = panel.col(s)
        w[:, j] = np.where(np.isfinite(panel.f["close"][:, j]), 1.0 / len(symbols), 0.0)
    return apply_rebalance_schedule(w, hold_schedule(T, 30))
