"""Vectorized daily portfolio backtester with point-in-time execution and after-cost accounting.

Convention: ``target[t]`` is decided from information available at the close of session ``t`` (plus settle time)
and is executed at session ``t+1`` — at the open (``execution="open"``: overnight return on the old holdings,
trade at the open, intraday return on the new holdings) or at the close (``execution="close"``). A name with no
bar at the execution time cannot be traded that day (non-fill: the old holding stays). A held name that stops
trading for ``delist_after`` sessions is force-exited at its last price less ``delist_haircut``.

Costs per leg come from :class:`services.quantlab.costs.EquityCosts` (half spread by liquidity bucket, fixed
slippage, square-root impact on actual traded dollars, SEC/FINRA fees on sells). Long-short is supported for
*research only* (shorting is disabled by the execution policy) with an annual borrow charge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from .costs import EquityCosts


@dataclass(frozen=True)
class BacktestConfig:
    capital: float = 100_000.0
    execution: str = "open"
    long_only: bool = True
    max_gross: float = 1.0
    borrow_bps_annual: float = 50.0
    costs: EquityCosts = field(default_factory=EquityCosts)
    rebalance_band: float = 0.0
    delist_after: int = 5
    delist_haircut: float = 0.0
    flat_cost_bps: tuple[float, ...] | None = None
    """Per-column one-way cost in bps (crypto/other non-equity panels); overrides ``costs`` when given."""
    periods_per_year: float = 252.0
    whole_shares: bool = False
    """Round every traded position to whole shares at the execution price (execution-tier realism)."""


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    net: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    turnover: np.ndarray
    exposure: np.ndarray
    n_positions: np.ndarray
    trades: int
    forced_exits: int
    non_fills: int
    config: dict[str, Any]

    def series(self) -> pd.Series:
        return pd.Series(self.net, index=self.dates)

    def metrics(self, periods_per_year: float = 252.0) -> dict[str, Any]:
        return performance_metrics(self.net, self.dates, periods_per_year=periods_per_year, gross=self.gross,
                                   cost=self.cost, turnover=self.turnover, exposure=self.exposure,
                                   trades=self.trades, n_positions=self.n_positions)


def _day(ts: object) -> str:
    return str(pd.Timestamp(str(ts)).date())


def performance_metrics(net: np.ndarray, dates: pd.DatetimeIndex | None = None, *, periods_per_year: float = 252.0,
                        gross: np.ndarray | None = None, cost: np.ndarray | None = None,
                        turnover: np.ndarray | None = None, exposure: np.ndarray | None = None,
                        trades: int | None = None, n_positions: np.ndarray | None = None) -> dict[str, Any]:
    r = np.nan_to_num(np.asarray(net, dtype=float))
    n = len(r)
    if n < 2:
        return {"days": n}
    eq = np.cumprod(1 + r)
    years = n / periods_per_year
    total = float(eq[-1] - 1)
    cagr = float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 and years > 0 else -1.0
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    downside = r[r < 0]
    dd_sd = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2))) if len(downside) else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    mdd = float(dd.min())
    q05 = float(np.quantile(r, 0.05))
    cvar95 = float(r[r <= q05].mean()) if (r <= q05).any() else q05
    q01 = float(np.quantile(r, 0.01))
    cvar99 = float(r[r <= q01].mean()) if (r <= q01).any() else q01
    out: dict[str, Any] = {
        "days": n, "years": round(years, 3), "total_return": total, "cagr": cagr,
        "ann_vol": sd * math.sqrt(periods_per_year),
        "sharpe": (mu / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "sortino": (mu / dd_sd * math.sqrt(periods_per_year)) if dd_sd > 0 else 0.0,
        "max_drawdown": mdd, "calmar": (cagr / abs(mdd)) if mdd < 0 else 0.0,
        "cvar95_daily": cvar95, "cvar99_daily": cvar99, "worst_day": float(r.min()), "best_day": float(r.max()),
        "skew": float(sps.skew(r)), "kurtosis": float(sps.kurtosis(r)),
        "hit_rate": float((r > 0).sum() / max(1, (r != 0).sum())),
    }
    if dates is not None and len(dates) == n:
        years_arr = np.asarray(dates.year)
        yearly = {int(y): float(np.prod(1 + r[years_arr == y]) - 1) for y in np.unique(years_arr)}
        out["yearly"] = yearly
        out["pct_positive_years"] = float(np.mean([v > 0 for v in yearly.values()]))
        out["start"], out["end"] = _day(dates[0]), _day(dates[-1])
    if gross is not None:
        g = np.nan_to_num(gross)
        out["gross_cagr"] = float(np.prod(1 + g) ** (1 / years) - 1) if years > 0 else 0.0
        out["gross_sharpe"] = float(g.mean() / g.std(ddof=1) * math.sqrt(periods_per_year)) if g.std() > 0 else 0.0
    if cost is not None:
        out["cost_drag_annual"] = float(np.nansum(cost) / years) if years > 0 else 0.0
    if turnover is not None:
        out["turnover_annual"] = float(np.nansum(turnover) / years) if years > 0 else 0.0
    if exposure is not None:
        out["avg_exposure"] = float(np.nanmean(exposure))
    if n_positions is not None:
        out["avg_positions"] = float(np.nanmean(n_positions))
    if trades is not None:
        out["trades"] = int(trades)
    return out


def run_weights(f: dict[str, np.ndarray], dates: pd.DatetimeIndex, target: np.ndarray,
                cfg: BacktestConfig | None = None) -> BacktestResult:
    """Simulate ``target`` (T, N) weights. A row that is entirely NaN means "no rebalance decision" (hold, drift);
    otherwise NaN entries mean 0. Returns daily net/gross returns for rows 1..T-1."""
    cfg = cfg or BacktestConfig()
    T, _n = target.shape
    hold_rows = np.asarray(np.isnan(target).all(axis=1), dtype=bool)
    tgt = np.nan_to_num(target.astype(float))
    if cfg.long_only:
        tgt = np.clip(tgt, 0.0, None)
    gross_w = np.abs(tgt).sum(axis=1, keepdims=True)
    scale = np.where(gross_w > cfg.max_gross, cfg.max_gross / np.where(gross_w > 0, gross_w, 1), 1.0)
    tgt = tgt * scale
    used = np.where(np.abs(tgt).sum(axis=0) > 0)[0]
    tg = tgt[:, used]
    ret_co = f["ret_co"][:, used]
    ret_oc = f["ret_oc"][:, used]
    ret_cc = f["ret_cc"][:, used]
    opn = f["open"][:, used]
    cls = f["close"][:, used]
    adv = f["adv20"][:, used]
    M = len(used)
    h = np.zeros(M)
    missing = np.zeros(M, dtype=int)
    V = cfg.capital
    net = np.zeros(T - 1)
    gross = np.zeros(T - 1)
    cost_arr = np.zeros(T - 1)
    turn = np.zeros(T - 1)
    expo = np.zeros(T - 1)
    npos = np.zeros(T - 1)
    trades = forced = nonfill = 0
    borrow_daily = cfg.borrow_bps_annual / 1e4 / cfg.periods_per_year
    costs = cfg.costs
    flat = None if cfg.flat_cost_bps is None else np.asarray(cfg.flat_cost_bps, dtype=float)[used]
    for t in range(T - 1):
        d = t + 1
        if cfg.execution == "open":
            r1 = ret_co[d]
            px = opn[d]
        else:
            r1 = ret_cc[d]
            px = cls[d]
        tradable = np.isfinite(px) & (px > 0)
        r1z = np.where(np.isfinite(r1), r1, 0.0)
        # delisting / prolonged absence: force exit of held names
        missing = np.where(tradable, 0, missing + 1)
        stuck = (missing >= cfg.delist_after) & (h != 0)
        hair = 0.0
        if stuck.any():
            hair = float(np.sum(h[stuck]) * cfg.delist_haircut)
            forced += int(stuck.sum())
            h = np.where(stuck, 0.0, h)
        p1 = float(np.dot(h, r1z)) - hair
        denom = 1.0 + p1
        h_mid = h * (1.0 + r1z) / denom if denom > 0 else np.zeros(M)
        w = h_mid.copy() if bool(hold_rows[t]) else tg[t].copy()
        blocked = ~tradable & (np.abs(w - h_mid) > 1e-12)
        nonfill += int(blocked.sum())
        w = np.where(tradable, w, h_mid)
        if cfg.whole_shares:
            V_pre = V * denom
            pxs = np.where(tradable, px, np.nan)
            shares = np.floor(np.abs(w) * V_pre / np.where(pxs > 0, pxs, np.inf))
            w_round = np.sign(w) * shares * np.nan_to_num(pxs) / V_pre if V_pre > 0 else w * 0
            w = np.where(tradable & (np.abs(w - h_mid) > 1e-12), w_round, w)
        delta = w - h_mid
        if cfg.rebalance_band > 0:
            small = (np.abs(delta) < cfg.rebalance_band) & (w != 0) & (h_mid != 0)
            w = np.where(small, h_mid, w)
            delta = w - h_mid
        traded = np.abs(delta) > 1e-9
        trades += int(traded.sum())
        V_mid = V * denom
        if traded.any():
            if flat is not None:
                bps = flat[traded]
            else:
                bps = costs.one_way_bps(adv[d][traded], np.abs(delta[traded]) * V_mid, px[traded],
                                        sell=delta[traded] < 0, timing=cfg.execution)
            c = float(np.sum(np.abs(delta[traded]) * bps) / 1e4)
        else:
            c = 0.0
        short_cost = float(np.sum(np.abs(np.minimum(w, 0.0)))) * borrow_daily
        if cfg.execution == "open":
            r2 = np.where(np.isfinite(ret_oc[d]), ret_oc[d], 0.0)
            p2 = float(np.dot(w, r2))
            h_new = w * (1.0 + r2) / (1.0 + p2) if (1.0 + p2) > 0 else np.zeros(M)
        else:
            p2 = 0.0
            h_new = w
        g = denom * (1.0 + p2) - 1.0
        nr = denom * (1.0 - c - short_cost) * (1.0 + p2) - 1.0
        V = V * (1.0 + nr)
        h = h_new
        net[t], gross[t], cost_arr[t] = nr, g, c + short_cost
        turn[t] = float(np.sum(np.abs(delta)))
        expo[t] = float(np.sum(np.abs(w)))
        npos[t] = float(np.sum(np.abs(w) > 1e-9))
        if V <= 0:
            net[t + 1:] = 0.0
            break
    return BacktestResult(dates[1:], net, gross, cost_arr, turn, expo, npos, trades, forced, nonfill,
                          {"capital": cfg.capital, "execution": cfg.execution, "long_only": cfg.long_only,
                           "max_gross": cfg.max_gross, "rebalance_band": cfg.rebalance_band,
                           "cost_multiplier": cfg.costs.cost_multiplier, "cost_version": cfg.costs.version})


def hold_schedule(T: int, every: int, offset: int = 0) -> np.ndarray:
    """Boolean rebalance mask: True every ``every`` rows starting at ``offset``."""
    m = np.zeros(T, dtype=bool)
    m[offset::max(1, every)] = True
    return m


def apply_rebalance_schedule(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Decisions only on scheduled rows: other rows become all-NaN (hold and let weights drift)."""
    out = np.where(mask[:, None], np.nan_to_num(target.astype(float)), np.nan)
    return out
