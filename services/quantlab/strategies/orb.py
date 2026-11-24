"""Strategy B — catalyst-confirmed opening-range breakout on stocks in play (intraday, same-session exits).

Reference: Zarattini, Barbon & Aziz (2024) "A Profitable Day Trading Strategy for the U.S. Equity Market" (5-minute
ORB on high relative-volume "stocks in play"). It is re-tested here on Alpaca SIP 5-minute bars after spread,
slippage and fees — not assumed to work.

Point-in-time selection (no look-ahead):
1. At the 09:30 open of session d: candidates are common stocks with prior close >= $5 and ADV20 (sessions < d)
   >= ``min_adv``, ranked by |opening gap| (open_d / close_{d-1} - 1; the open is known at 09:30). The top
   ``plan_top`` per day are fetched (5-minute bars) — this is the only use of the daily panel.
2. At the end of the opening range (09:30 + ``or_minutes``): relative volume = OR volume / (ADV20 shares x
   ``or_share``), where ``or_share`` is the median OR share of daily volume measured on *earlier* plan days.
   Stocks in play = top ``n`` by relative volume with relvol >= ``min_relvol``.
3. Long-only entry (executable form) if the OR candle closed up (and, optionally, above OR VWAP and SPY above its
   own open): buy at the next bar's open; stop at the OR low (or entry - ``atr_stop`` x ATR14); exit at the stop,
   at a ``target_r`` multiple, or at 15:55. A bar that opens through the stop exits at its open (gap risk).
   Short entries are simulated only when ``side='both'`` and labelled research-only (shorting is disabled).
Sizing: equal capital per position (1/n of equity, no leverage).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from ..backtest import BacktestResult
from ..costs import EquityCosts
from ..panel import EquityPanel, trailing_mean

FAMILY = "B_orb"


def _ts(x: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(x))


def build_plan(panel: EquityPanel, start: date, end: date, *, plan_top: int = 40, min_adv: float = 20e6,
               min_price: float = 5.0) -> dict[date, list[str]]:
    f = panel.f
    stocks = panel.kind == "COMMON_STOCK"
    plan: dict[date, list[str]] = {}
    prev_close = np.vstack([np.full((1, panel.shape[1]), np.nan), f["close"][:-1]])
    for t, d in enumerate(panel.dates):
        if not (pd.Timestamp(start) <= d <= pd.Timestamp(end)) or t == 0:
            continue
        gap = f["open"][t] / prev_close[t] - 1
        ok = stocks & (prev_close[t] >= min_price) & (np.nan_to_num(f["adv20"][t]) >= min_adv) & np.isfinite(gap)
        idx = np.where(ok)[0]
        order = idx[np.argsort(-np.abs(gap[idx]))][:plan_top]
        plan[d.date()] = sorted({*(str(panel.symbols[j]) for j in order), "SPY"})
    return plan


def variants() -> Iterator[dict[str, Any]]:
    for orm, n, rv, stop, filt, cat in itertools.product((5, 15), (5, 20), (1.0, 2.0), ("or_low", "atr10"),
                                                         ("candle", "candle_vwap_spy"), ("none", "news")):
        yield {"or_minutes": orm, "n": n, "min_relvol": rv, "stop": stop, "filter": filt, "target_r": 0.0,
               "side": "long", "catalyst": cat}


def _atr14(panel: EquityPanel) -> np.ndarray:
    f = panel.f
    prev = np.vstack([np.full((1, panel.shape[1]), np.nan), f["close"][:-1]])
    tr = np.nanmax(np.stack([f["high"] - f["low"], np.abs(f["high"] - prev), np.abs(f["low"] - prev)]), axis=0)
    atr = trailing_mean(tr, 14, 10)
    return np.vstack([np.full((1, panel.shape[1]), np.nan), atr[:-1]])  # known before the open


@dataclass(frozen=True)
class SymbolDay:
    """Regular-session bars of one symbol on one day as numpy arrays (minutes since 09:30)."""

    mins: np.ndarray
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    vw: np.ndarray


class OrbSimulator:
    """Holds the intraday bars (5-minute, regular session) and daily context; simulates variants."""

    def __init__(self, bars: pd.DataFrame, panel: EquityPanel, costs: EquityCosts | None = None,
                 bar_minutes: int = 5, news: pd.DataFrame | None = None) -> None:
        self.news: set[tuple[str, str]] = set()
        if news is not None and not news.empty:
            hit = news[news["news"].to_numpy(dtype=float) > 0]
            self.news = {(str(a), str(b)) for a, b in zip(hit["session"], hit["symbol"], strict=True)}
        df = bars.copy()
        ts = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).tz_convert("America/New_York")
        df["d"] = ts.tz_localize(None).normalize()
        df["mins"] = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
        df = df.sort_values(["d", "symbol", "mins"])
        self.panel = panel
        self.costs = costs or EquityCosts()
        self.bar_minutes = bar_minutes
        self.atr = _atr14(panel)
        self.row: dict[pd.Timestamp, int] = {_ts(d): i for i, d in enumerate(panel.dates)}
        self.adv_shares = np.vstack([np.full((1, panel.shape[1]), np.nan),
                                     trailing_mean(panel.f["volume"], 20, 15)[:-1]])
        self.data: dict[pd.Timestamp, dict[str, SymbolDay]] = {}
        d_arr = df["d"].to_numpy()
        sym_arr = df["symbol"].to_numpy().astype(str)
        cols = {k: df[k].to_numpy(dtype=float) for k in ("mins", "o", "h", "l", "c", "v")}
        vw = df["vw"].to_numpy(dtype=float)
        vw = np.where(np.isfinite(vw), vw, cols["c"])
        keys = pd.MultiIndex.from_arrays([d_arr, sym_arr])
        starts = np.r_[0, np.where(keys[1:] != keys[:-1])[0] + 1, len(df)] if len(df) else np.array([0])
        for i in range(len(starts) - 1):
            lo_i, hi_i = int(starts[i]), int(starts[i + 1])
            day = _ts(d_arr[lo_i])
            self.data.setdefault(day, {})[str(sym_arr[lo_i])] = SymbolDay(
                cols["mins"][lo_i:hi_i].astype(int), cols["o"][lo_i:hi_i], cols["h"][lo_i:hi_i],
                cols["l"][lo_i:hi_i], cols["c"][lo_i:hi_i], cols["v"][lo_i:hi_i], vw[lo_i:hi_i])
        self.days: list[pd.Timestamp] = sorted(self.data)

    @staticmethod
    def _or_share(upto: pd.Timestamp, history: dict[pd.Timestamp, float], default: float) -> float:
        """Median opening-range share of daily volume over plan days strictly before ``upto`` (PIT)."""
        past = [v for d, v in history.items() if d < upto]
        return float(np.median(past[-60:])) if len(past) >= 10 else default

    def run(self, p: dict[str, Any], cost_multiplier: float = 1.0, *, capital: float = 100_000.0,
            trade_days: set[pd.Timestamp] | None = None) -> BacktestResult:
        """Simulate every day in order. With ``trade_days`` only those days trade (others still feed the PIT
        opening-range volume normalizer) and only those days are returned."""
        orm = int(p["or_minutes"])
        n = int(p["n"])
        costs = self.costs if cost_multiplier == 1.0 else EquityCosts(cost_multiplier=cost_multiplier)
        share_hist: dict[pd.Timestamp, float] = {}
        daily_net: list[float] = []
        daily_gross: list[float] = []
        daily_cost: list[float] = []
        turn: list[float] = []
        expo: list[float] = []
        npos: list[float] = []
        idx: list[pd.Timestamp] = []
        trades = 0
        for d in self.days:
            t = self.row.get(d)
            if t is None:
                continue
            day = self.data[d]
            share = self._or_share(d, share_hist, 0.06 * orm / 5)
            spy_open = spy_now = None
            spy = day.get("SPY")
            if spy is not None and len(spy.o):
                spy_open = float(spy.o[0])
                in_or = spy.mins < orm
                spy_now = float(spy.c[in_or][-1]) if in_or.any() else None
            cands: list[tuple[float, str, int]] = []
            ratios: list[float] = []
            for sym, b in day.items():
                in_or = b.mins < orm
                v_or = float(b.v[in_or].sum())
                v_day = float(b.v.sum())
                if v_day > 0:
                    ratios.append(v_or / v_day)
                if sym == "SPY" or v_or <= 0:
                    continue
                try:
                    j = self.panel.col(sym)
                except KeyError:
                    continue
                if self.panel.kind[j] != "COMMON_STOCK":  # plan may predate a reclassification: stocks only
                    continue
                base = float(self.adv_shares[t, j]) * share
                if np.isfinite(base) and base > 0:
                    cands.append((v_or / base, sym, j))
            if ratios:  # today's realized OR share enters the estimate only for LATER days (PIT)
                share_hist[d] = float(np.median(ratios))
            if trade_days is not None and d not in trade_days:
                continue
            if p.get("catalyst") == "news":
                key = str(d)[:10]
                cands = [c for c in cands if (key, c[1]) in self.news]
            cands = sorted((c for c in cands if c[0] >= float(p["min_relvol"])), reverse=True)
            pnl = gross_pnl = cost_usd = traded = 0.0
            open_pos = 0
            for _rv, sym, j in cands[:n]:
                b = day[sym]
                in_or = b.mins < orm
                rest = ~in_or
                if not in_or.any() or not rest.any():
                    continue
                o, c = float(b.o[in_or][0]), float(b.c[in_or][-1])
                hi, lo = float(b.h[in_or].max()), float(b.lo[in_or].min())
                vol_or = float(b.v[in_or].sum())
                vwap = float((b.vw[in_or] * b.v[in_or]).sum() / vol_or) if vol_or > 0 else c
                long_ok = c > o
                if p["filter"] == "candle_vwap_spy":
                    long_ok = long_ok and c > vwap and spy_open is not None and spy_now is not None \
                        and spy_now > spy_open
                short_ok = (c < o) and p.get("side") == "both"
                if not (long_ok or short_ok):
                    continue
                side = 1 if long_ok else -1
                ro, rh, rl, rc, rm = b.o[rest], b.h[rest], b.lo[rest], b.c[rest], b.mins[rest]
                entry = float(ro[0])
                atr = float(self.atr[t, j])
                if p["stop"] == "or_low":
                    stop = lo if side > 0 else hi
                else:
                    if not np.isfinite(atr):
                        continue
                    stop = entry - side * 0.10 * atr
                risk = (entry - stop) * side
                if risk <= 0:
                    continue
                target_r = float(p.get("target_r") or 0.0)
                target = entry + side * target_r * risk if target_r > 0 else None
                exit_px = _exit_price(ro[1:], rh[1:], rl[1:], rm[1:], side, stop, target)
                if exit_px is None:
                    exit_px = float(rc[-1])
                alloc = capital / n
                adv_usd = float(self.panel.f["adv20"][t, j])
                c_in = float(costs.one_way_bps(np.array([adv_usd]), np.array([alloc]), np.array([entry]),
                                               sell=np.array([side < 0]), timing="open")[0])
                c_out = float(costs.one_way_bps(np.array([adv_usd]), np.array([alloc]), np.array([exit_px]),
                                                sell=np.array([side > 0]), timing="close")[0])
                g_ret = side * (exit_px / entry - 1)
                c_ret = (c_in + c_out) / 1e4
                gross_pnl += alloc * g_ret
                cost_usd += alloc * c_ret
                pnl += alloc * (g_ret - c_ret)
                traded += 2 * alloc
                open_pos += 1
                trades += 2
            idx.append(d)
            daily_net.append(pnl / capital)
            daily_gross.append(gross_pnl / capital)
            daily_cost.append(cost_usd / capital)
            turn.append(traded / capital)
            expo.append(open_pos / n)
            npos.append(float(open_pos))
        return BacktestResult(pd.DatetimeIndex(idx), np.array(daily_net), np.array(daily_gross), np.array(daily_cost),
                              np.array(turn), np.array(expo), np.array(npos), trades, 0, 0,
                              {"capital": capital, "execution": "intraday_5min", "cost_multiplier": cost_multiplier})


def _exit_price(o: np.ndarray, h: np.ndarray, lo: np.ndarray, mins: np.ndarray, side: int, stop: float,
                target: float | None) -> float | None:
    """First exit after entry: gap through the stop (exit at that bar's open), stop touch (at the stop), target
    touch (at the target), or the 15:55 bar (at its open). Same-bar stop and target -> stop first (conservative)."""
    if len(o) == 0:
        return None
    eod = mins >= 385
    gap = (o <= stop) if side > 0 else (o >= stop)
    touch = (lo <= stop) if side > 0 else (h >= stop)
    tgt = np.zeros_like(touch) if target is None else ((h >= target) if side > 0 else (lo <= target))
    events = eod | gap | touch | tgt
    if not events.any():
        return None
    i = int(np.argmax(events))
    if eod[i] or gap[i]:
        return float(o[i])
    if touch[i]:
        return float(stop)
    return float(target) if target is not None else float(o[i])


def spy_intraday_benchmark(sim: OrbSimulator) -> BacktestResult:
    """Benchmark for a flat-overnight strategy: SPY bought at the open and sold at the close each day."""
    out: list[float] = []
    days: list[pd.Timestamp] = []
    for d in sim.days:
        if d not in sim.row:
            continue
        spy = sim.data[d].get("SPY")
        days.append(d)
        out.append(float(spy.c[-1] / spy.o[0] - 1) if spy is not None and len(spy.o) else 0.0)
    r = np.array(out)
    z = np.zeros_like(r)
    return BacktestResult(pd.DatetimeIndex(days), r, r, z, z, np.ones_like(r), np.ones_like(r), len(r) * 2, 0, 0,
                          {"benchmark": "SPY open->close"})
