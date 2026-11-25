"""Point-in-time intraday (5-minute, regular-session) panels and a lifecycle-faithful trade simulator.

Used by the session-timing families (``strategies/session.py``): intraday momentum, VWAP reversion, opening-gap
behaviour, overnight holds and turn-of-month holds on allowlisted ETFs and large caps.

Grid: ``f[name][d, s, j]`` for session ``d``, slot ``s`` (bar starting 09:30 + 5*s minutes, New York) and symbol
``j``. Only regular-session bars are kept; ``last_slot[d]`` is 77 on a normal day (15:55 bar) and 41 on a 13:00
early close. A bar's close is known at the END of its slot, so a decision on the close of slot ``s`` can trade
from slot ``s + 1`` (its open) at the earliest.

The simulator applies the runtime lifecycle constraints instead of idealised fills:

* entries are marketable LIMIT BUYs at ``decision close x (1 + limit_bps)``; the order fills at the entry bar's open
  when that open is at or below the limit, otherwise it is NOT filled (the runtime cancels an unfilled entry after
  15 s; a 5-minute bar cannot resolve that, so a gap past the limit counts as a non-fill);
* no entry in the last 30 minutes of the session (``ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS``: entry slot <=
  ``last_slot - 6``); intraday positions are sold at the open of the flatten slot (10 minutes before the close,
  ``PositionPolicy.flatten_before_close_s``: slot ``last_slot - 1``), early closes included;
* every position carries a protective stop ``stop_bps`` below the entry (the runtime always rests one). A bar that
  opens through the stop exits at that open (gap risk, including the next session's open for overnight holds),
  otherwise a bar whose low reaches it exits at the stop. A stop is a trigger, not a guaranteed price;
* costs: :class:`services.quantlab.costs.EquityCosts` per leg (the runtime model's spread buckets, fixed slippage,
  square-root impact on the traded notional and sell-side SEC/FINRA fees), charged exactly once per leg, with the
  opening spread multiplier for fills in the first 15 minutes; ADV and price levels come from the daily RAW panel
  as of the previous close (point in time).

A position held across sessions is marked to market at every held session's close (entry cost on the entry
session, exit cost on the exit session), so daily returns, drawdowns and betas see the overnight risk as it occurs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.market_data.calendar import get_calendar
from services.positions.policy import PositionPolicy

from .backtest import BacktestResult
from .costs import EquityCosts
from .panel import EquityPanel

BAR_MINUTES = 5
N_SLOTS = 78  # 09:30 ... 15:55 bar starts on a full session
FIELDS = ("o", "h", "l", "c", "v", "vw")
ENTRY_CUTOFF_SLOTS = 6  # no entry bar starting within the last 30 minutes (6 x 5 min)
FLATTEN_SLOTS_BEFORE_CLOSE = 1  # flatten at the open of the bar starting 10 minutes before the close (default)
OPEN_TIMING_SLOTS = 3  # fills in the first 15 minutes pay the opening spread multiplier


def slot_of(clock: str) -> int:
    """Slot index of the bar that STARTS at ``clock`` ('HH:MM', New York)."""
    h, m = (int(x) for x in clock.split(":"))
    minutes = h * 60 + m - (9 * 60 + 30)
    if minutes < 0 or minutes % BAR_MINUTES:
        raise ValueError(f"{clock} is not a regular-session {BAR_MINUTES}-minute bar start")
    return minutes // BAR_MINUTES


def decision_slot(clock: str) -> int:
    """Slot whose bar CLOSES at ``clock``: a decision at 10:00 uses the 09:55 bar (slot 5) and trades at slot 6."""
    return slot_of(clock) - 1


@dataclass
class IntradayPanel:
    days: pd.DatetimeIndex
    symbols: np.ndarray
    f: dict[str, np.ndarray]  # (D, S, N) float32
    last_slot: np.ndarray  # (D,) int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.days), N_SLOTS, len(self.symbols)

    def col(self, symbol: str) -> int:
        idx = np.where(self.symbols == symbol)[0]
        if not len(idx):
            raise KeyError(symbol)
        return int(idx[0])

    def cols(self, symbols: Sequence[str]) -> np.ndarray:
        have = set(self.symbols.tolist())
        return np.array([self.col(s) for s in symbols if s in have], dtype=int)

    def slice_days(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> IntradayPanel:
        m = np.ones(len(self.days), dtype=bool)
        if start is not None:
            m &= self.days >= start
        if end is not None:
            m &= self.days <= end
        return IntradayPanel(self.days[m], self.symbols, {k: v[m] for k, v in self.f.items()}, self.last_slot[m],
                             {**self.meta, "sliced": [str(start), str(end)]})

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, bars: pd.DataFrame, symbols: Sequence[str] | None = None) -> IntradayPanel:
        cal = get_calendar()
        df = bars.copy()
        ts = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).tz_convert("America/New_York")
        df["d"] = ts.tz_localize(None).normalize()
        df["slot"] = ((ts.hour * 60 + ts.minute) - (9 * 60 + 30)) // BAR_MINUTES
        on_grid = ((ts.minute % BAR_MINUTES) == 0) & (ts.second == 0)
        df = df.loc[np.asarray(on_grid) & (df["slot"].to_numpy() >= 0) & (df["slot"].to_numpy() < N_SLOTS)]
        if df.empty:
            raise ValueError("no regular-session bars")
        start, end = df["d"].min().date(), df["d"].max().date()
        days = pd.DatetimeIndex(pd.to_datetime(cal.sessions_in_range(start, end)))
        last = np.array([(int((cal.session_close(d.date()) - cal.session_open(d.date())).total_seconds() // 60)
                          // BAR_MINUTES) - 1 for d in days], dtype=int)
        syms = np.array(sorted(set(symbols) if symbols is not None else set(df["symbol"].astype(str))), dtype=object)
        di, si = pd.Index(days), pd.Index(syms)
        r = di.get_indexer(pd.DatetimeIndex(df["d"]))
        c = si.get_indexer(pd.Index(df["symbol"].astype(str).to_numpy()))
        s = df["slot"].to_numpy(dtype=int)
        ok = (r >= 0) & (c >= 0)
        ok &= s <= last[np.clip(r, 0, len(last) - 1)]  # drop post-close bars of early-close sessions
        D, N = len(days), len(syms)
        f: dict[str, np.ndarray] = {}
        for name in FIELDS:
            arr = np.full((D, N_SLOTS, N), np.nan, dtype=np.float32)
            vals = df[name].to_numpy(dtype=float)
            arr[r[ok], s[ok], c[ok]] = vals[ok]
            f[name] = arr
        f["vw"] = np.where(np.isfinite(f["vw"]), f["vw"], f["c"]).astype(np.float32)
        f["v"] = np.nan_to_num(f["v"]).astype(np.float32)
        panel = cls(days, syms, f, last, {"built_at": pd.Timestamp.now("UTC").isoformat(), "bar_minutes": BAR_MINUTES,
                                          "rows": int(ok.sum())})
        return panel

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {  # fixed-width strings: the loader never unpickles (allow_pickle=False)
            "days": np.array(self.days.strftime("%Y-%m-%d").tolist(), dtype="U10"),
            "symbols": np.array([str(x) for x in self.symbols.tolist()], dtype="U16"),
            "last_slot": np.asarray(self.last_slot, dtype=np.int64)}
        arrays.update({f"f_{k}": v for k, v in self.f.items()})
        np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        self.meta["sha256"] = h
        path.with_suffix(".json").write_text(json.dumps({**self.meta, "sha256": h, "days": len(self.days),
                                                         "symbols": self.symbols.astype(str).tolist()}, indent=1,
                                                        default=str))

    @classmethod
    def load(cls, path: Path) -> IntradayPanel:
        z = np.load(path, allow_pickle=False)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        f = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        return cls(pd.DatetimeIndex(pd.to_datetime(z["days"])), z["symbols"].astype(object), f,
                   z["last_slot"].astype(int), meta)

    # ------------------------------------------------------------------ derived features (point in time)
    def ffill_close(self) -> np.ndarray:
        """Close carried forward within each session (never across sessions)."""
        c = self.f["c"].astype(float)
        out = c.copy()
        for s in range(1, N_SLOTS):
            out[:, s] = np.where(np.isfinite(out[:, s]), out[:, s], out[:, s - 1])
        return out

    def session_vwap(self) -> np.ndarray:
        """Cumulative session VWAP through the END of each slot."""
        v = self.f["v"].astype(float)
        pv = np.nan_to_num(self.f["vw"].astype(float)) * v
        cv = np.cumsum(v, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cv > 0, np.cumsum(pv, axis=1) / cv, np.nan)

    def prev_close(self, cf: np.ndarray | None = None) -> np.ndarray:
        """(D, N): last regular-session close of the PREVIOUS session (NaN on the first day)."""
        cf = self.ffill_close() if cf is None else cf
        last = cf[np.arange(len(self.days)), self.last_slot]  # (D, N)
        return np.vstack([np.full((1, len(self.symbols)), np.nan), last[:-1]])

    def first_open(self) -> np.ndarray:
        """(D, N): opening price (first regular-session bar's open, NaN if the 09:30 bar is missing)."""
        return self.f["o"][:, 0].astype(float)


def trailing_std_rows(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """(D, N) std over the previous ``window`` rows, EXCLUDING the current row (point in time)."""
    df = pd.DataFrame(x)
    sd = df.rolling(window, min_periods=min_periods).std().shift(1)
    return sd.to_numpy()


@dataclass(frozen=True)
class DailyContext:
    """Daily-panel facts aligned to intraday days, all as of the PREVIOUS close (point in time)."""

    adv_usd: np.ndarray  # (D, N) 20-session ADV in USD (sessions strictly before d)
    raw_prev_close: np.ndarray  # (D, N) as-traded close of d-1 (price level for share rounding and fees)
    adj_prev_close: np.ndarray  # (D, N) adjusted close of d-1
    above_sma50: np.ndarray  # (D, N) bool, adjusted close of d-1 above its 50-session mean
    above_sma200: np.ndarray  # (D, N) bool
    spy_ret_cc: np.ndarray  # (D,) SPY adjusted close-to-close return OF day d (benchmark only, never a signal)

    @classmethod
    def align(cls, ipanel: IntradayPanel, daily: EquityPanel) -> DailyContext:
        from .panel import trailing_mean

        D, N = len(ipanel.days), len(ipanel.symbols)
        rows = pd.Index(daily.dates).get_indexer(ipanel.days)
        cols = np.array([daily.col(s) if s in set(daily.symbols.tolist()) else -1 for s in ipanel.symbols])

        def take(a: np.ndarray, lag: int) -> np.ndarray:
            out = np.full((D, N), np.nan)
            src = rows - lag
            ok_r = (rows >= 0) & (src >= 0)
            for k, j in enumerate(cols):
                if j >= 0:
                    out[ok_r, k] = a[src[ok_r], j]
            return out

        adj = daily.f["adj_close"]
        sma50 = trailing_mean(adj, 50, 45)
        sma200 = trailing_mean(adj, 200, 180)
        prev_adj = take(adj, 1)
        spy = np.full(D, np.nan)
        if "SPY" in set(daily.symbols.tolist()):
            j = daily.col("SPY")
            ok = rows >= 0
            spy[ok] = daily.f["ret_cc"][rows[ok], j]
        return cls(adv_usd=take(daily.f["adv20"], 0), raw_prev_close=take(daily.f["close"], 1),
                   adj_prev_close=prev_adj, above_sma50=np.nan_to_num(prev_adj > take(sma50, 1)).astype(bool),
                   above_sma200=np.nan_to_num(prev_adj > take(sma200, 1)).astype(bool), spy_ret_cc=spy)


@dataclass(frozen=True)
class Leg:
    """One requested round trip. ``exit_day``/``exit_slot`` give the planned sale at that bar's open."""

    d: int
    j: int
    decision_slot: int
    exit_day: int
    exit_slot: int
    score: float = 0.0
    stop_bps: float = 150.0


@dataclass(frozen=True)
class SimConfig:
    capital: float = 100_000.0
    weight: float = 0.25  # fraction of capital per position
    max_concurrent: int = 4
    limit_bps: float = 10.0  # marketable entry limit above the decision close
    costs: EquityCosts = field(default_factory=EquityCosts)
    whole_shares: bool = False
    stop_limit_gap_bps: float = float(PositionPolicy().stop_limit_gap_bps)
    """Protective orders are STOP_LIMIT SELLs with limit = stop - gap. A bar trading below that limit cannot fill the
    broker order; the runtime escalates to a marketable exit (STOP_BREACH), booked at min(limit, bar close)."""


@dataclass
class LegOutcome:
    leg: Leg
    filled: bool
    entry_slot: int = -1
    entry_px: float = float("nan")
    exit_px: float = float("nan")
    exit_reason: str = ""
    weight: float = 0.0
    gross: float = 0.0
    cost: float = 0.0
    cost_in: float = 0.0
    cost_out: float = 0.0
    held_slots: int = 0
    exit_day_actual: int = -1
    exit_slot_actual: int = -1
    marks: list[tuple[int, float]] = field(default_factory=list)
    """Session closes (day index, price) between entry and exit for multi-session legs: P&L is marked to market
    every session instead of being booked in one lump on the exit day."""


def flatten_slot(last_slot: int) -> int:
    return int(last_slot) - FLATTEN_SLOTS_BEFORE_CLOSE


def latest_entry_slot(last_slot: int) -> int:
    return int(last_slot) - ENTRY_CUTOFF_SLOTS


def _first_finite(a: np.ndarray, start: int, stop: int) -> int:
    """First index in [start, stop] with a finite value, or -1."""
    for s in range(max(0, start), min(stop, len(a) - 1) + 1):
        if np.isfinite(a[s]):
            return s
    return -1


def simulate_leg(p: IntradayPanel, ctx: DailyContext, leg: Leg, cfg: SimConfig, weight: float) -> LegOutcome:
    f = p.f
    d, j = leg.d, leg.j
    out = LegOutcome(leg, filled=False, weight=weight)
    entry_slot = leg.decision_slot + 1
    if entry_slot > latest_entry_slot(p.last_slot[d]) or leg.exit_day < d or leg.exit_day >= len(p.days):
        out.exit_reason = "ENTRY_WINDOW"
        return out
    ref = float(f["c"][d, leg.decision_slot, j])
    o_entry = float(f["o"][d, entry_slot, j])
    if not (np.isfinite(ref) and np.isfinite(o_entry)) or o_entry > ref * (1 + cfg.limit_bps / 1e4):
        out.exit_reason = "NON_FILL"
        return out
    entry = o_entry
    stop = entry * (1 - leg.stop_bps / 1e4)
    # scan bars from the entry bar to the bar before the planned exit (across sessions for multi-day legs)
    exit_px = float("nan")
    reason = ""
    held = 0
    exit_d, exit_s = leg.exit_day, leg.exit_slot
    for dd in range(d, leg.exit_day + 1):
        s0 = entry_slot if dd == d else 0
        s_last = (min(leg.exit_slot, flatten_slot(p.last_slot[dd])) if dd == leg.exit_day
                  else int(p.last_slot[dd]) + 1)  # an overnight hold scans the whole session
        for s in range(s0, s_last):
            o, lo = float(f["o"][dd, s, j]), float(f["l"][dd, s, j])
            if not np.isfinite(o):
                continue
            held += 1
            if not (dd == d and s == entry_slot) and o <= stop:
                exit_px, reason, exit_d, exit_s = o, "STOP_GAP", dd, s
                break
            if np.isfinite(lo) and lo <= stop:
                limit = stop * (1 - cfg.stop_limit_gap_bps / 1e4)
                if lo >= limit:
                    exit_px, reason = stop, "STOP"
                else:  # traded through the stop-limit's limit: unfilled broker order -> marketable escalation
                    c_bar = float(f["c"][dd, s, j])
                    exit_px, reason = min(limit, c_bar if np.isfinite(c_bar) else limit), "STOP_BREACH"
                exit_d, exit_s = dd, s
                break
        if reason:
            break
    if not reason:
        last_ok = flatten_slot(p.last_slot[leg.exit_day])
        s_exit = min(leg.exit_slot, last_ok)
        k = _first_finite(f["o"][leg.exit_day, :, j].astype(float), s_exit, int(p.last_slot[leg.exit_day]))
        if k >= 0:
            exit_px, reason, exit_s = float(f["o"][leg.exit_day, k, j]), "TIME" if k < last_ok else "EOD", k
        else:  # no bar left: forced at the last available close at or before the exit session (never dropped)
            exit_px = float("nan")
            for dd in range(leg.exit_day, d - 1, -1):
                cf = f["c"][dd, :, j].astype(float)
                fin = np.where(np.isfinite(cf))[0]
                if dd == d:
                    fin = fin[fin >= entry_slot]
                if len(fin):
                    exit_px, reason, exit_d, exit_s = float(cf[fin[-1]]), "FORCED", dd, int(fin[-1])
                    break
            if not np.isfinite(exit_px):  # cannot happen after a fill (the entry bar exists); kept fail-safe
                exit_px, reason, exit_d, exit_s = entry, "FORCED", d, entry_slot
    out.filled = True
    out.entry_slot, out.entry_px, out.exit_px, out.exit_reason = entry_slot, entry, exit_px, reason
    out.held_slots = max(1, held)
    # costs: one entry leg + one exit leg, each once
    adv = np.array([ctx.adv_usd[d, j]])
    px_raw = float(ctx.raw_prev_close[d, j])
    notional = np.array([cfg.capital * weight])
    if cfg.whole_shares:
        adj_prev = float(ctx.adj_prev_close[d, j])
        raw_entry = entry * (px_raw / adj_prev) if np.isfinite(px_raw) and np.isfinite(adj_prev) and adj_prev > 0 \
            else float("nan")
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            out.filled, out.exit_reason = False, "NO_PRICE"
            return out
        shares = np.floor(cfg.capital * weight / raw_entry)
        if shares < 1:
            out.filled, out.exit_reason = False, "LESS_THAN_ONE_SHARE"
            return out
        out.weight = weight = float(shares * raw_entry / cfg.capital)
        notional = np.array([cfg.capital * weight])
    price = np.array([px_raw if np.isfinite(px_raw) else entry])
    t_in = "open" if entry_slot < OPEN_TIMING_SLOTS else "close"
    t_out = "open" if exit_s < OPEN_TIMING_SLOTS else "close"
    c_in = float(cfg.costs.one_way_bps(adv, notional, price, sell=np.array([False]), timing=t_in)[0])
    c_out = float(cfg.costs.one_way_bps(adv, notional, price, sell=np.array([True]), timing=t_out)[0])
    out.gross = exit_px / entry - 1.0
    out.cost_in, out.cost_out = c_in / 1e4, c_out / 1e4
    out.cost = out.cost_in + out.cost_out
    out.exit_day_actual, out.exit_slot_actual = (exit_d, exit_s) if reason.startswith(("STOP", "FORCED")) \
        else (leg.exit_day, exit_s)
    for dd in range(d, out.exit_day_actual):  # mark every completed session of a multi-session hold
        closes = f["c"][dd, : int(p.last_slot[dd]) + 1, j].astype(float)
        fin = np.where(np.isfinite(closes))[0]
        if len(fin):
            out.marks.append((dd, float(closes[fin[-1]])))
    return out


def simulate(p: IntradayPanel, ctx: DailyContext, legs: Sequence[Leg], cfg: SimConfig, *,
             day_mask: np.ndarray | None = None) -> tuple[BacktestResult, list[LegOutcome]]:
    """Run legs in time order with at most ``cfg.max_concurrent`` open positions (higher score first among legs
    deciding at the same bar). Returns the daily after-cost series over ``p.days`` (or ``day_mask`` days)."""
    D = len(p.days)
    net = np.zeros(D)
    gross = np.zeros(D)
    cost = np.zeros(D)
    turn = np.zeros(D)
    expo = np.zeros(D)
    npos = np.zeros(D)
    outcomes: list[LegOutcome] = []
    trades = non_fills = 0
    open_until: dict[int, tuple[int, int]] = {}  # symbol column -> (exit day, exit slot) of its open position
    order = sorted(legs, key=lambda x: (x.d, x.decision_slot, -x.score, x.j))
    for leg in order:
        if day_mask is not None and not day_mask[leg.d]:
            continue
        now = (leg.d, leg.decision_slot + 1)
        open_until = {j: e for j, e in open_until.items() if e >= now}  # an exit bar frees nothing earlier than itself
        if leg.j in open_until or len(open_until) >= cfg.max_concurrent:
            continue  # one position per symbol (one open plan per symbol) and the concurrency cap
        res = simulate_leg(p, ctx, leg, cfg, cfg.weight)
        outcomes.append(res)
        if not res.filled:
            non_fills += res.exit_reason == "NON_FILL"
            continue
        open_until[leg.j] = (res.exit_day_actual, res.exit_slot_actual)
        book = res.exit_day_actual
        w = res.weight
        # mark to market: entry -> each held session's close -> exit (segments sum to the round-trip gross return,
        # weights fixed at entry); the entry cost is booked on the entry session, the exit cost on the exit session
        prev = res.entry_px
        for dd, px in res.marks:
            seg = w * (px - prev) / res.entry_px
            gross[dd] += seg
            net[dd] += seg
            prev = px
        seg = w * (res.exit_px - prev) / res.entry_px
        gross[book] += seg
        net[book] += seg
        net[leg.d] -= w * res.cost_in
        net[book] -= w * res.cost_out
        cost[leg.d] += w * res.cost_in
        cost[book] += w * res.cost_out
        turn[leg.d] += w
        turn[book] += w
        for dd in range(leg.d, book + 1):  # exposure: held fraction of every session in the position
            first = res.entry_slot if dd == leg.d else 0
            last = res.exit_slot_actual if dd == book else int(p.last_slot[dd]) + 1
            frac = max(0, last - first) / (int(p.last_slot[dd]) + 1)
            expo[dd] += w * min(1.0, frac)
            npos[dd] += 1
        trades += 2
    keep = np.ones(D, dtype=bool) if day_mask is None else day_mask
    result = BacktestResult(p.days[keep], net[keep], gross[keep], cost[keep], turn[keep], expo[keep], npos[keep],
                            trades, 0, non_fills, {"capital": cfg.capital, "execution": "intraday_5min_limit",
                                                   "weight": cfg.weight, "max_concurrent": cfg.max_concurrent,
                                                   "limit_bps": cfg.limit_bps, "whole_shares": cfg.whole_shares,
                                                   "cost_multiplier": cfg.costs.cost_multiplier})
    return result, outcomes


def benchmark_spy_buy_and_hold(p: IntradayPanel, ctx: DailyContext) -> BacktestResult:
    """Passive alternative for every session family: SPY held close-to-close (no costs), on the same sessions."""
    r = np.nan_to_num(ctx.spy_ret_cc)
    z = np.zeros_like(r)
    return BacktestResult(p.days, r, r, z, z, np.ones_like(r), np.ones_like(r), 2, 0, 0, {"benchmark": "SPY B&H"})


def session_profile(p: IntradayPanel, symbols: Sequence[str], *, start: date | None = None) -> dict[str, Any]:
    """Descriptive (not a strategy, nothing selected): mean and t-stat of the return of each half-hour window and of
    the overnight gap per symbol. Answers *when* returns accrue; trading rules are tested only in the families."""
    cf = p.ffill_close()
    pc = p.prev_close(cf)
    op = p.first_open()
    mask = np.ones(len(p.days), dtype=bool) if start is None else (p.days >= pd.Timestamp(start))
    full = p.last_slot == N_SLOTS - 1
    out: dict[str, Any] = {"days": int((mask & full).sum()), "note": "full sessions only; descriptive statistics"}
    for sym in symbols:
        try:
            j = p.col(sym)
        except KeyError:
            continue
        rows = mask & full
        windows: dict[str, Any] = {}
        on = op[rows, j] / pc[rows, j] - 1
        windows["overnight"] = _tstat(on)
        prev = op[rows, j]
        for s in range(5, N_SLOTS, 6):  # closes at 10:00, 10:30, ..., 16:00
            cur = cf[rows, s, j]
            h, m = divmod(9 * 60 + 30 + (s + 1) * BAR_MINUTES, 60)
            windows[f"to_{h:02d}:{m:02d}"] = _tstat(cur / prev - 1)
            prev = cur
        out[sym] = windows
    return out


def _tstat(x: np.ndarray) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return {"n": float(len(x)), "mean_bps": None, "t": None}
    sd = float(x.std(ddof=1))
    return {"n": float(len(x)), "mean_bps": float(x.mean() * 1e4),
            "t": float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else None, "std_bps": sd * 1e4}


def leg_diagnostics(outcomes: Sequence[LegOutcome]) -> dict[str, Any]:
    """Trade-level economics of one simulated variant: how large the gross edge per round trip is relative to the
    modelled cost (break-even cost), how often entries do not fill and how positions end."""
    filled = [o for o in outcomes if o.filled]
    attempted = len(outcomes)
    out: dict[str, Any] = {"attempted": attempted, "filled": len(filled),
                           "fill_rate": (len(filled) / attempted) if attempted else None}
    if not filled:
        return out
    g = np.array([o.gross for o in filled]) * 1e4
    c = np.array([o.cost for o in filled]) * 1e4
    reasons: dict[str, int] = {}
    for o in filled:
        reasons[o.exit_reason] = reasons.get(o.exit_reason, 0) + 1
    sd = float(g.std(ddof=1)) if len(g) > 1 else float("nan")
    out.update({
        "mean_gross_bps": float(g.mean()), "median_gross_bps": float(np.median(g)),
        "t_gross": float(g.mean() / (sd / np.sqrt(len(g)))) if len(g) > 1 and sd > 0 else None,
        "mean_cost_bps": float(c.mean()), "mean_net_bps": float((g - c).mean()),
        "break_even_round_trip_cost_bps": float(g.mean()), "hit_rate_gross": float((g > 0).mean()),
        "hit_rate_net": float((g - c > 0).mean()), "exit_reasons": reasons,
        "avg_held_bars": float(np.mean([o.held_slots for o in filled])),
    })
    return out
