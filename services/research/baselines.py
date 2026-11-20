"""No-LLM quantitative baselines + a strictly point-in-time, event-driven backtester (spec §3).

Strategies (long-only, no leverage, one position at a time, whole shares):

* ``buy_and_hold`` — enter at the first executable open, hold to the end.
* ``momentum`` — long while the trailing ``lookback``-bar return is positive (N-bar return sign).
* ``intraday_mean_reversion`` — long when price is ``entry_z`` standard deviations below the session
  VWAP; exit on reversion (``z >= exit_z``) and **always flat before the scheduled session close**
  (intraday timeframes only; skipped with a recorded reason on daily bars).
* ``filtered_momentum`` — momentum, but only when trailing ADV (USD) clears a liquidity floor and
  annualised realised volatility is below a cap.

Point-in-time discipline (tested):

* the target for bar ``t`` is computed from bars ``<= t`` only (rolling/causal ops; prefix-invariant);
* the decision time is ``available_at_ts[t]``; the order executes at the **open of the first bar whose
  ``start_ts >= decision time``** plus modelled costs (half-spread, slippage, SELL-side fees);
* liquidity inputs to the cost model (ADV) are themselves point-in-time.

Results are written as JSON under ``<out_dir>/<run_id>/``: ``results.json`` (everything),
``manifest.json`` (provenance) and one ``runs/<strategy>__<symbol>/summary.json`` per backtest in the
shape served by ``GET /api/v1/backtests`` (docs/architecture/api-contract.md).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from packages.contracts.enums import Side
from packages.core.money import ZERO, floor_shares

from .costs import CostEstimate, CostModel
from .frames import (
    TRADING_DAYS_PER_YEAR,
    SegmentPolicy,
    dataset_version_of,
    is_intraday,
    is_simulated_of,
    iter_symbol_frames,
    last_clean_segment,
    periods_per_year,
    prepare_bars,
    session_close_utc,
    timeframe_delta,
    trailing_adv_shares,
    ts_ns,
)
from .provenance import (
    NO_EDGE_NOTE,
    code_version_hash,
    config_hash,
    library_versions,
    new_run_id,
    relpath,
    utc_now_iso,
    write_json,
)

DEFAULT_INITIAL_CASH = Decimal(100_000)
EQUITY_CURVE_MAX_POINTS = 1500
TRADES_MAX_RECORDED = 500
ENGINE_LABEL = "PYTHON_BASELINE"


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def params(self) -> dict[str, Any]: ...

    def applicable(self, timeframe: str) -> tuple[bool, str]: ...

    def targets(self, sym_bars: pd.DataFrame) -> np.ndarray: ...


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    assert isinstance(s, pd.Series)
    return s


@dataclass(frozen=True)
class BuyAndHold:
    name: str = "buy_and_hold"

    def params(self) -> dict[str, Any]:
        return {}

    def applicable(self, timeframe: str) -> tuple[bool, str]:
        return True, ""

    def targets(self, sym_bars: pd.DataFrame) -> np.ndarray:
        return np.ones(len(sym_bars), dtype=np.int8)


@dataclass(frozen=True)
class Momentum:
    lookback: int = 20
    threshold: float = 0.0
    name: str = "momentum"

    def params(self) -> dict[str, Any]:
        return {"lookback_bars": self.lookback, "threshold": self.threshold, "rule": "long iff C_t/C_{t-L}-1 > threshold"}

    def applicable(self, timeframe: str) -> tuple[bool, str]:
        return True, ""

    def targets(self, sym_bars: pd.DataFrame) -> np.ndarray:
        close = _col(sym_bars, "close_f").to_numpy(dtype=np.float64)
        out = np.zeros(len(close), dtype=np.int8)
        if len(close) > self.lookback:
            ret = close[self.lookback :] / close[: -self.lookback] - 1.0
            out[self.lookback :] = (ret > self.threshold).astype(np.int8)
        return out


@dataclass(frozen=True)
class IntradayMeanReversion:
    window: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.0
    flat_before_close_minutes: int = 5
    name: str = "intraday_mean_reversion"
    entries_expire_at_session_end: bool = True
    """An entry decided in one session never fills in a later one (e.g. after a data gap before the close)."""

    def params(self) -> dict[str, Any]:
        return {
            "window_bars": self.window,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "flat_before_close_minutes": self.flat_before_close_minutes,
            "entries_expire_at_session_end": self.entries_expire_at_session_end,
            "reference": "session VWAP (typical price), z = (close/vwap - 1) / rolling_std within the session",
        }

    def applicable(self, timeframe: str) -> tuple[bool, str]:
        if not is_intraday(timeframe):
            return False, "intraday mean reversion requires intraday bars"
        return True, ""

    def zscores(self, sym_bars: pd.DataFrame) -> np.ndarray:
        high = _col(sym_bars, "high_f").to_numpy(dtype=np.float64)
        low = _col(sym_bars, "low_f").to_numpy(dtype=np.float64)
        close = _col(sym_bars, "close_f").to_numpy(dtype=np.float64)
        vol = _col(sym_bars, "volume_f").to_numpy(dtype=np.float64)
        days: list[Any] = list(sym_bars["session_date"])
        z = np.full(len(close), np.nan)
        start = 0
        while start < len(close):
            end = start
            while end < len(close) and days[end] == days[start]:
                end += 1
            tp = (high[start:end] + low[start:end] + close[start:end]) / 3.0
            v = vol[start:end]
            cum_v = np.cumsum(v)
            cum_pv = np.cumsum(tp * v)
            vwap = np.where(cum_v > 0, cum_pv / np.where(cum_v > 0, cum_v, 1.0), close[start:end])
            dev = pd.Series(close[start:end] / vwap - 1.0)
            sd = dev.rolling(self.window, min_periods=self.window).std().to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                z[start:end] = np.where(sd > 0, dev.to_numpy(dtype=np.float64) / sd, np.nan)
            start = end
        return z

    def targets(self, sym_bars: pd.DataFrame) -> np.ndarray:
        z = self.zscores(sym_bars)
        tf = str(sym_bars["timeframe"].iloc[0])
        buffer = timedelta(minutes=self.flat_before_close_minutes) + timeframe_delta(tf)
        days: list[Any] = list(sym_bars["session_date"])
        cutoffs = {d: int(pd.Timestamp(session_close_utc(d) - buffer).as_unit("ns").value) for d in sorted(set(days))}
        decision = ts_ns(sym_bars["available_at_ts"])
        out = np.zeros(len(sym_bars), dtype=np.int8)
        pos = 0
        prev_day = None
        for i in range(len(sym_bars)):
            if days[i] != prev_day:
                pos = 0  # a fresh session never inherits a position decision
                prev_day = days[i]
            if decision[i] >= cutoffs[days[i]]:
                pos = 0
            elif math.isnan(z[i]):
                pass
            elif pos == 0 and z[i] <= -self.entry_z:
                pos = 1
            elif pos == 1 and z[i] >= self.exit_z:
                pos = 0
            out[i] = pos
        return out


@dataclass(frozen=True)
class FilteredMomentum:
    lookback: int = 20
    vol_window: int = 20
    min_adv_usd: float = 10_000_000.0
    max_annual_vol: float = 0.60
    adv_sessions: int = 20
    name: str = "filtered_momentum"

    def params(self) -> dict[str, Any]:
        return {
            "lookback_bars": self.lookback,
            "vol_window_bars": self.vol_window,
            "min_adv_usd": self.min_adv_usd,
            "max_annual_vol": self.max_annual_vol,
            "adv_sessions": self.adv_sessions,
            "rule": "momentum AND trailing ADV(USD) >= floor AND annualised realised vol <= cap",
        }

    def applicable(self, timeframe: str) -> tuple[bool, str]:
        return True, ""

    def targets(self, sym_bars: pd.DataFrame) -> np.ndarray:
        close = _col(sym_bars, "close_f").to_numpy(dtype=np.float64)
        n = len(close)
        mom = np.full(n, np.nan)
        if n > self.lookback:
            mom[self.lookback :] = close[self.lookback :] / close[: -self.lookback] - 1.0
        logret = np.full(n, np.nan)
        logret[1:] = np.log(close[1:] / close[:-1])
        tf = str(sym_bars["timeframe"].iloc[0])
        ann_vol = pd.Series(logret).rolling(self.vol_window, min_periods=self.vol_window).std().to_numpy(
            dtype=np.float64
        ) * math.sqrt(periods_per_year(tf))
        adv_usd = trailing_adv_shares(sym_bars, sessions=self.adv_sessions) * close
        with np.errstate(invalid="ignore"):
            ok = (mom > 0) & (adv_usd >= self.min_adv_usd) & (ann_vol <= self.max_annual_vol)
        return np.where(np.isnan(mom) | np.isnan(ann_vol) | np.isnan(adv_usd), 0, ok).astype(np.int8)


def default_strategies(timeframe: str) -> list[Strategy]:
    if is_intraday(timeframe):
        return [
            BuyAndHold(),
            Momentum(lookback=30),
            IntradayMeanReversion(),
            FilteredMomentum(lookback=30, vol_window=60),
        ]
    return [BuyAndHold(), Momentum(lookback=20), IntradayMeanReversion(), FilteredMomentum()]


# --------------------------------------------------------------------------------------------
# Backtester
# --------------------------------------------------------------------------------------------


@dataclass
class TradeRecord:
    bar_index: int
    decision_index: int
    ts: pd.Timestamp
    decision_ts: pd.Timestamp
    side: str
    quantity: Decimal
    price: Decimal
    notional: Decimal
    cost_usd: Decimal
    fees_usd: Decimal
    spread_usd: Decimal
    slippage_usd: Decimal
    reason: str
    realized_pl: Decimal | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts
        d["decision_ts"] = self.decision_ts
        return d


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    params: dict[str, Any]
    timeframe: str
    status: str
    skip_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=lambda: {})
    trades: list[TradeRecord] = field(default_factory=lambda: [])
    equity: list[Decimal] = field(default_factory=lambda: [])
    equity_ts: list[pd.Timestamp] = field(default_factory=lambda: [])
    diagnostics: dict[str, Any] = field(default_factory=lambda: {})

    def equity_curve_points(self, max_points: int = EQUITY_CURVE_MAX_POINTS) -> list[dict[str, Any]]:
        n = len(self.equity)
        if n == 0:
            return []
        idx = np.unique(np.linspace(0, n - 1, num=min(n, max_points)).round().astype(int))
        return [{"time": int(self.equity_ts[i].timestamp()), "equity": format(self.equity[i], "f")} for i in idx]

    def to_json(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "params": self.params,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "trades_total": len(self.trades),
            "trades_recorded": len(self.trades[:TRADES_MAX_RECORDED]),
            "trades": [t.to_json() for t in self.trades[:TRADES_MAX_RECORDED]],
            "equity_curve": self.equity_curve_points(),
        }


def _adv_dec(value: float) -> Decimal | None:
    if not math.isfinite(value) or value <= 0:
        return None
    return Decimal(f"{value:.0f}")


@dataclass
class _Pending:
    target: int
    decision_ns: int
    decision_index: int


def run_backtest(
    sym_bars: pd.DataFrame,
    targets: np.ndarray,
    cost_model: CostModel,
    *,
    strategy: str,
    params: dict[str, Any] | None = None,
    initial_cash: Decimal = DEFAULT_INITIAL_CASH,
    liquidate_at_end: bool = True,
    entries_expire_at_session_end: bool = False,
    dividends: Mapping[date, Decimal] | None = None,
) -> BacktestResult:
    """Event-driven single-symbol backtest of a 0/1 long target series (see module docstring).

    ``sym_bars`` must be a *prepared* frame (``prepare_bars``) for one symbol. With
    ``entries_expire_at_session_end`` a pending BUY decided in one session is cancelled instead of filling in a
    later session (DAY-order semantics for intraday strategies); pending exits always carry over (exit ASAP).
    ``dividends`` maps ex-date -> cash per share (in the same share units as the bars, i.e. split-adjusted when
    the bars are): a position held at the close before the ex-date is credited ``qty x cash`` at the first bar
    of the ex-date session (before any order executes there - selling at the ex-date open keeps the dividend,
    buying there does not get it).
    """
    if len(targets) != len(sym_bars):
        raise ValueError("targets length must match bars")
    if {int(x) for x in np.unique(targets).tolist()} - {0, 1}:
        raise ValueError("targets must be 0/1 (long-only, no leverage)")
    if len(set(sym_bars["symbol"].astype(str))) != 1:
        raise ValueError("run_backtest expects a single symbol")
    if not isinstance(initial_cash, Decimal) or initial_cash <= 0:
        raise ValueError("initial_cash must be a positive Decimal")
    symbol = str(sym_bars["symbol"].iloc[0])
    tf = str(sym_bars["timeframe"].iloc[0])
    n = len(sym_bars)
    start_ns = ts_ns(sym_bars["start_ts"])
    avail_ns = ts_ns(sym_bars["available_at_ts"])
    opens: list[Decimal] = list(sym_bars["open"])
    closes: list[Decimal] = list(sym_bars["close"])
    adv = trailing_adv_shares(sym_bars)
    end_ts: list[pd.Timestamp] = list(sym_bars["end_ts"])
    start_ts: list[pd.Timestamp] = list(sym_bars["start_ts"])
    avail_ts: list[pd.Timestamp] = list(sym_bars["available_at_ts"])
    days: list[Any] = list(sym_bars["session_date"])

    cash = initial_cash
    qty = ZERO
    entry_cost_basis = ZERO
    pending: _Pending | None = None
    trades: list[TradeRecord] = []
    equity: list[Decimal] = []
    bars_in_market = 0
    skipped_buys = 0
    overnight_holds = 0
    expired_entries = 0
    totals = {"cost": ZERO, "fees": ZERO, "spread": ZERO, "slippage": ZERO, "notional": ZERO, "dividends": ZERO}
    div_dates = sorted((dividends or {}).items())
    for ex, cash_ps in div_dates:
        if not isinstance(cash_ps, Decimal) or not cash_ps.is_finite() or cash_ps <= 0:
            raise ValueError(f"dividend on {ex} must be a positive finite Decimal")
    div_ptr = 0
    while div_ptr < len(div_dates) and div_dates[div_ptr][0] <= days[0]:
        div_ptr += 1  # ex-dates on/before the first bar cannot be earned inside this backtest
    dividend_events = 0

    def book(est: CostEstimate) -> None:
        totals["cost"] += est.total_usd
        totals["fees"] += est.fees_usd
        totals["spread"] += est.spread_cost_usd
        totals["slippage"] += est.slippage_usd
        totals["notional"] += est.notional

    def execute(i: int, p: _Pending, price: Decimal, ts: pd.Timestamp, reason: str) -> None:
        nonlocal cash, qty, entry_cost_basis, skipped_buys
        adv_shares = _adv_dec(float(adv[p.decision_index]))
        if p.target == 1 and qty == 0:
            q = floor_shares(cash / price)
            est: CostEstimate | None = None
            while q > 0:
                est = cost_model.estimate(symbol, Side.BUY, q, price, adv_shares=adv_shares)
                if q * price + est.total_usd <= cash:
                    break
                shrink = floor_shares((cash - est.total_usd) / price)
                q = shrink if shrink < q else q - 1
            if q <= 0 or est is None:
                skipped_buys += 1
                return
            cash -= q * price + est.total_usd
            qty = q
            entry_cost_basis = q * price + est.total_usd
            book(est)
            trades.append(
                TradeRecord(i, p.decision_index, ts, avail_ts[p.decision_index], "BUY", q, price, q * price,
                            est.total_usd, est.fees_usd, est.spread_cost_usd, est.slippage_usd, reason)
            )
        elif p.target == 0 and qty > 0:
            est = cost_model.estimate(symbol, Side.SELL, qty, price, adv_shares=adv_shares)
            proceeds = qty * price - est.total_usd
            cash += proceeds
            book(est)
            trades.append(
                TradeRecord(i, p.decision_index, ts, avail_ts[p.decision_index], "SELL", qty, price, qty * price,
                            est.total_usd, est.fees_usd, est.spread_cost_usd, est.slippage_usd, reason,
                            realized_pl=proceeds - entry_cost_basis)
            )
            qty = ZERO
            entry_cost_basis = ZERO

    for i in range(n):
        # 0a) dividends with ex-date in (previous session, this session] go to the holder at the previous close
        if i > 0 and days[i] != days[i - 1]:
            while div_ptr < len(div_dates) and div_dates[div_ptr][0] <= days[i]:
                if qty > 0:
                    income = qty * div_dates[div_ptr][1]
                    cash += income
                    totals["dividends"] += income
                    dividend_events += 1
                div_ptr += 1
        # 0b) DAY semantics for entries (optional): an unfilled entry never crosses into another session
        if (
            entries_expire_at_session_end
            and pending is not None
            and pending.target == 1
            and days[i] != days[pending.decision_index]
        ):
            pending = None
            expired_entries += 1
        # 1) execute a pending decision at this bar's open iff the bar opened at/after the decision time
        if pending is not None and start_ns[i] >= pending.decision_ns:
            execute(i, pending, opens[i], start_ts[i], "signal")
            pending = None
        if qty > 0:
            bars_in_market += 1
            if i + 1 < n and days[i + 1] != days[i]:
                overnight_holds += 1
        equity.append(cash + qty * closes[i])
        if i == n - 1:
            break
        # 2) decide using bar i (knowable at available_at_ts[i]); executes at a later bar's open
        desired = int(targets[i])
        actual = 1 if qty > 0 else 0
        if desired == actual:
            pending = None
        elif pending is None or pending.target != desired:
            pending = _Pending(desired, int(avail_ns[i]), i)

    if liquidate_at_end and qty > 0:
        execute(n - 1, _Pending(0, int(avail_ns[n - 1]), n - 1), closes[n - 1], end_ts[n - 1], "terminal_liquidation")
        equity[-1] = cash

    result = BacktestResult(
        strategy=strategy,
        symbol=symbol,
        params=params or {},
        timeframe=tf,
        status="OK",
        trades=trades,
        equity=equity,
        equity_ts=end_ts,
    )
    result.metrics = compute_metrics(
        equity=equity,
        end_ts=end_ts,
        session_dates=days,
        initial_cash=initial_cash,
        timeframe=tf,
        trades=trades,
        totals=totals,
        bars_in_market=bars_in_market,
        first_start=start_ts[0],
    )
    result.diagnostics = {
        "skipped_buys_insufficient_cash": skipped_buys,
        "overnight_holds": overnight_holds,
        "expired_entry_orders": expired_entries,
        "dividends_credited": dividend_events,
        "pending_unexecuted_at_end": pending is not None,
        "bars": n,
    }
    return result


def _fmt(value: float | None, places: int = 6) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return f"{value:.{places}f}"


def compute_metrics(
    *,
    equity: Sequence[Decimal],
    end_ts: Sequence[pd.Timestamp],
    session_dates: Sequence[Any],
    initial_cash: Decimal,
    timeframe: str,
    trades: Sequence[TradeRecord],
    totals: dict[str, Decimal],
    bars_in_market: int,
    first_start: pd.Timestamp,
) -> dict[str, Any]:
    """Performance metrics. Money/returns are Decimal strings; ratio statistics are float strings."""
    final = equity[-1]
    total_return = final / initial_cash - 1
    peak = initial_cash
    mdd = ZERO
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > mdd:
            mdd = dd
    # daily returns from end-of-session equity (first return vs initial cash)
    day_end: dict[Any, Decimal] = {}
    for d, e in zip(session_dates, equity, strict=True):
        day_end[d] = e
    series = [initial_cash, *day_end.values()]
    daily = [float(series[k] / series[k - 1] - 1) for k in range(1, len(series))]
    sharpe: float | None = None
    vol: float | None = None
    if len(daily) >= 2:
        sd = statistics.stdev(daily)
        vol = sd * math.sqrt(TRADING_DAYS_PER_YEAR)
        if sd > 0:
            sharpe = statistics.fmean(daily) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)
    cagr: float | None = None
    span_days = (end_ts[-1] - first_start).total_seconds() / 86400.0
    if not is_intraday(timeframe) and span_days > 0 and final > 0:
        cagr = float(final / initial_cash) ** (365.25 / span_days) - 1.0
    mean_equity = sum(equity, ZERO) / len(equity)
    turnover = totals["notional"] / mean_equity if mean_equity > 0 else ZERO
    sells = [t for t in trades if t.side == "SELL" and t.realized_pl is not None]
    wins = [t for t in sells if t.realized_pl is not None and t.realized_pl > 0]
    q6 = Decimal("0.000001")
    return {
        "initial_cash": format(initial_cash, "f"),
        "final_equity": format(final.quantize(Decimal("0.01")), "f"),
        "total_return": format(total_return.quantize(q6), "f"),
        "cagr": _fmt(cagr),
        "cagr_note": "daily+ bars only: (final/initial)^(365.25/calendar_span_days)-1; null for intraday runs",
        "sharpe": _fmt(sharpe, 4),
        "annual_volatility": _fmt(vol),
        "sharpe_annualization": "mean/stdev of end-of-session (America/New_York) equity returns x sqrt(252); rf=0",
        "daily_return_observations": len(daily),
        "max_drawdown": format(mdd.quantize(q6), "f"),
        "turnover": format(turnover.quantize(q6), "f"),
        "turnover_definition": "sum of traded notional (both legs) / mean equity",
        "trades": len(trades),
        "round_trips": len(sells),
        "win_rate": _fmt(len(wins) / len(sells), 4) if sells else None,
        "total_costs": format(totals["cost"].quantize(Decimal("0.01")), "f"),
        "fees": format(totals["fees"].quantize(Decimal("0.01")), "f"),
        "spread_costs": format(totals["spread"].quantize(Decimal("0.01")), "f"),
        "slippage_costs": format(totals["slippage"].quantize(Decimal("0.01")), "f"),
        "dividend_income": format(totals.get("dividends", ZERO).quantize(Decimal("0.01")), "f"),
        "exposure": _fmt(bars_in_market / len(equity), 4),
        "exposure_definition": "fraction of bars ending with an open position",
        "periods_per_year_bars": periods_per_year(timeframe),
    }


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def _summary_for(
    r: BacktestResult, *, run_id: str, dataset_version: str, simulated: bool, started: str, results_path: Path
) -> dict[str, Any]:
    """One run in the ``GET /api/v1/backtests`` shape."""
    return {
        "run_id": f"{run_id}:{r.strategy}:{r.symbol}",
        "parent_run_id": run_id,
        "engine": ENGINE_LABEL,
        "strategy": r.strategy,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "dataset": dataset_version,
        "is_simulated": simulated,
        "started_at": started,
        "metrics": {
            "total_return": r.metrics.get("total_return"),
            "sharpe": r.metrics.get("sharpe"),
            "max_drawdown": r.metrics.get("max_drawdown"),
            "trades": int(r.metrics.get("trades", 0)),
            "fees": r.metrics.get("total_costs"),
            "cagr": r.metrics.get("cagr"),
            "turnover": r.metrics.get("turnover"),
            "exposure": r.metrics.get("exposure"),
        },
        "equity_curve": r.equity_curve_points(),
        "source_path": relpath(results_path),
        "notes": ("SIMULATED DATA. " if simulated else "") + NO_EDGE_NOTE,
    }


def run_all_baselines(
    bars_df: pd.DataFrame,
    cost_model: CostModel,
    out_dir: Path | str,
    *,
    strategies: Sequence[Strategy] | None = None,
    initial_cash: Decimal = DEFAULT_INITIAL_CASH,
    symbols: Sequence[str] | None = None,
    run_id: str | None = None,
    dividends: Mapping[str, Mapping[date, Decimal]] | None = None,
    data_adjustments: dict[str, Any] | None = None,
    segment_policy: SegmentPolicy | None = SegmentPolicy(),  # frozen dataclass: safe default
) -> dict[str, Any]:
    """Run every baseline on every symbol and write ``<out_dir>/<run_id>/...`` (see module docstring).

    ``out_dir`` is the *base* directory (normally ``experiments/baselines``). Returns the full results
    payload (``run_dir`` is relative to the project root when possible). ``dividends`` (per symbol, ex-date ->
    cash per share) and ``data_adjustments`` come from ``adjustments.apply_corporate_actions`` for RAW data.
    ``segment_policy`` (default on; ``None`` disables) keeps only the last clean contiguous segment per symbol
    (``frames.last_clean_segment``) so no backtest trades across a multi-session hole or a ticker change.
    """
    bars = prepare_bars(bars_df)
    data_quality: dict[str, Any] = {"applied": "none (segment_policy=None)"}
    if segment_policy is not None:
        bars, data_quality = last_clean_segment(bars, segment_policy)
    tf = str(bars["timeframe"].iloc[0])
    strats = list(strategies) if strategies is not None else default_strategies(tf)
    run_id = run_id or new_run_id("bl")
    run_dir = Path(out_dir) / run_id
    dataset_version = dataset_version_of(bars)
    simulated = is_simulated_of(bars)
    started = utc_now_iso()

    results: list[BacktestResult] = []
    for sym, sb in iter_symbol_frames(bars, symbols):
        for strat in strats:
            ok, why = strat.applicable(tf)
            if not ok:
                results.append(BacktestResult(strat.name, sym, strat.params(), tf, "SKIPPED", skip_reason=why))
                continue
            if len(sb) < 3:
                results.append(
                    BacktestResult(strat.name, sym, strat.params(), tf, "SKIPPED", skip_reason="fewer than 3 bars")
                )
                continue
            tgt = strat.targets(sb)
            results.append(
                run_backtest(sb, tgt, cost_model, strategy=strat.name, params=strat.params(), initial_cash=initial_cash,
                             entries_expire_at_session_end=bool(getattr(strat, "entries_expire_at_session_end", False)),
                             dividends=(dividends or {}).get(sym))
            )

    aggregate: dict[str, Any] = {}
    for strat in strats:
        rows = [r for r in results if r.strategy == strat.name and r.status == "OK"]
        if not rows:
            skipped = [r.skip_reason for r in results if r.strategy == strat.name]
            aggregate[strat.name] = {"symbols": 0, "skipped": sorted(set(skipped))}
            continue
        rets = [float(r.metrics["total_return"]) for r in rows]
        sharpes = [float(r.metrics["sharpe"]) for r in rows if r.metrics["sharpe"] is not None]
        aggregate[strat.name] = {
            "symbols": len(rows),
            "mean_total_return": _fmt(statistics.fmean(rets)),
            "median_total_return": _fmt(statistics.median(rets)),
            "mean_sharpe": _fmt(statistics.fmean(sharpes), 4) if sharpes else None,
            "total_trades": sum(int(r.metrics["trades"]) for r in rows),
            "total_costs": format(sum((Decimal(r.metrics["total_costs"]) for r in rows), ZERO), "f"),
        }

    parameters = {
        "strategies": {s.name: s.params() for s in strats},
        "initial_cash": format(initial_cash, "f"),
        "symbols": sorted({r.symbol for r in results}),
        "execution": "next bar open with start_ts >= decision time (available_at_ts) + modelled costs",
        "position_rules": "long-only, no leverage, one position at a time per backtest, whole shares",
        "terminal": "open positions liquidated at the final bar close with SELL costs",
        "dividends": "credited as cash to positions held into the ex-date" if dividends else "not credited (none supplied)",
    }
    cost_desc = cost_model.describe()
    cfg_hash = config_hash({"parameters": parameters, "cost_model": cost_desc, "dataset_version": dataset_version})
    results_path = run_dir / "results.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "kind": "baseline_backtest",
        "engine": ENGINE_LABEL,
        "created_at": started,
        "dataset_version": dataset_version,
        "is_simulated": simulated,
        "data_label": "SIMULATED (synthetic fixture)" if simulated else "HISTORICAL (non-synthetic bars)",
        "timeframe": tf,
        "bar_count": len(bars),
        "period": {"start": bars["start_ts"].min(), "end": bars["end_ts"].max()},
        "code_version": code_version_hash(),
        "config_hash": cfg_hash,
        "parameters": parameters,
        "cost_model": cost_desc,
        "assumptions": [
            "Signals use only bars with available_at_ts <= decision time.",
            "Fills at the next executable bar open; no partial fills, no queue modelling.",
            "Spread/slippage/fees from the stated cost model; bars carry no quotes so spreads are bucket estimates.",
            "Sharpe uses daily end-of-session equity returns, annualised by sqrt(252), risk-free rate 0.",
            "Splits: bars must be split-adjusted (the CLI applies point-in-time split records to RAW datasets); "
            "dividends are cash credits only when supplied; other corporate actions are not modelled.",
        ],
        "data_adjustments": data_adjustments or {"applied": "none (caller supplied no corporate-action handling)"},
        "data_quality": data_quality,
        "no_edge_note": NO_EDGE_NOTE,
        "aggregate": aggregate,
        "runs": [r.to_json() for r in results],
        "run_dir": relpath(run_dir),
    }
    write_json(results_path, payload)
    summary_files: list[str] = []
    for r in results:
        if r.status != "OK":
            continue
        path = run_dir / "runs" / f"{r.strategy}__{r.symbol}" / "summary.json"
        write_json(
            path,
            _summary_for(r, run_id=run_id, dataset_version=dataset_version, simulated=simulated, started=started,
                         results_path=results_path),
        )
        summary_files.append(path.relative_to(run_dir).as_posix())
    payload["summary_files"] = summary_files
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "kind": "baseline_backtest",
            "created_at": started,
            "dataset_version": dataset_version,
            "is_simulated": simulated,
            "code_version": code_version_hash(),
            "config_hash": cfg_hash,
            "files": ["results.json", *summary_files],
            "libraries": library_versions("pandas", "numpy", "exchange-calendars"),
            "no_edge_note": NO_EDGE_NOTE,
        },
    )
    return payload
