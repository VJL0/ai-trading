"""Point-in-time features for the model suite (float64 statistics; never accounting).

Every value at bar ``t`` of a symbol uses only bars ``<= t`` of that symbol, all of which have
``available_at_ts <= available_at_ts[t]`` (``prepare_bars`` enforces monotone availability), so a row is
knowable at its decision time ``available_at_ts[t]``. Cross-symbol information is used only by the regime
model's market proxy, which aggregates bars of the SAME timestamp (also knowable at that time).

Columns (``model_feature_columns``):

* research ``FeatureSpec`` columns (reused from ``services.research.features``): trailing log returns
  ``ret_1/5/15/60``, realised vol ``rv_30``, volume z-score, bar range, distance from the 60-bar mean;
* ``tod``: minutes from the scheduled 09:30 America/New_York open to the bar end, / 390 (known in advance);
* ``ret_sod``: ``ln(C_t / O_first)`` where ``O_first`` is the open of the session's first bar seen so far.

Volatility inputs (``vol_inputs``): the intra-session one-bar log return ``r1`` (close-to-close inside a
session, open-to-close for a session's first bar, so overnight gaps never enter realised variance), the HAR-RV
components (trailing sums of ``r1**2``) and the RiskMetrics EWMA variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import numpy as np
import pandas as pd

from ..features import FeatureSpec, symbol_features
from ..frames import NY, REGULAR_SESSION_MINUTES, bar_trading_minutes, ts_ns

EXTRA_COLUMNS: tuple[str, ...] = ("tod", "ret_sod")


def model_feature_columns(spec: FeatureSpec) -> tuple[str, ...]:
    return (*spec.columns, *EXTRA_COLUMNS)


def session_open_utc_ns(day: date) -> int:
    """Scheduled 09:30 New York open of a session date (calendar knowledge, not data)."""
    dt = datetime.combine(day, time(9, 30), tzinfo=NY).astimezone(UTC)
    return int(pd.Timestamp(dt).as_unit("ns").value)


def _session_open_ns(days: list[date]) -> np.ndarray:
    cache: dict[date, int] = {}
    out = np.empty(len(days), dtype=np.int64)
    for i, d in enumerate(days):
        v = cache.get(d)
        if v is None:
            v = session_open_utc_ns(d)
            cache[d] = v
        out[i] = v
    return out


def same_session_prev(days: list[date]) -> np.ndarray:
    """``True`` where bar ``i`` and bar ``i-1`` belong to the same session (index 0 -> False)."""
    ords = np.asarray([d.toordinal() for d in days], dtype=np.int64)
    out = np.zeros(len(ords), dtype=bool)
    if len(ords) > 1:
        out[1:] = ords[1:] == ords[:-1]
    return out


def intra_session_r1(sym_bars: pd.DataFrame) -> np.ndarray:
    """One-bar log return excluding overnight gaps (first bar of a session: ``ln(C/O)``)."""
    close = sym_bars["close_f"].to_numpy(dtype=np.float64)
    open_ = sym_bars["open_f"].to_numpy(dtype=np.float64)
    days = list(sym_bars["session_date"])
    same = same_session_prev(days)
    r = np.log(close / open_)
    if len(close) > 1:
        cc = np.log(close[1:] / close[:-1])
        r[1:] = np.where(same[1:], cc, r[1:])
    return r


def symbol_model_features(sym_bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Feature frame aligned with a *prepared* single-symbol frame (NaN during warm-up)."""
    base = symbol_features(sym_bars, spec)
    days = list(sym_bars["session_date"])
    end_ns = ts_ns(sym_bars["end_ts"])
    open_ns = _session_open_ns(days)
    base["tod"] = (end_ns - open_ns) / 60e9 / REGULAR_SESSION_MINUTES
    opens = sym_bars["open_f"].to_numpy(dtype=np.float64)
    close = sym_bars["close_f"].to_numpy(dtype=np.float64)
    first_open = pd.Series(opens).groupby(pd.Series([d.toordinal() for d in days])).transform("first")
    base["ret_sod"] = np.log(close / first_open.to_numpy(dtype=np.float64))
    return base


def momentum_score(X: np.ndarray, columns: tuple[str, ...], spec: FeatureSpec) -> np.ndarray:
    """Deterministic momentum t-stat ``ret_L / (rv * sqrt(L))`` with ``L`` = longest return lookback."""
    lag = max(spec.return_horizons)
    ret = X[:, columns.index(f"ret_{lag}")]
    rv = X[:, columns.index(f"rv_{spec.vol_window}")]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(rv > 0, ret / (rv * np.sqrt(lag)), 0.0)
    return np.asarray(s, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# volatility inputs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSpec:
    horizon_bars: int
    base_bars: int
    lags: tuple[int, int, int]
    ewma_lambda: float
    ewma_init_bars: int = 30

    @property
    def warmup_bars(self) -> int:
        return max(self.base_bars * self.lags[2], self.ewma_init_bars)


def vol_spec_for(timeframe: str, horizon_minutes: int, base_minutes: int | None, lags: tuple[int, int, int],
                 ewma_lambda: float) -> VolSpec:
    m = bar_trading_minutes(timeframe)
    hb = max(1, round(horizon_minutes / m))
    bb = max(1, round((base_minutes or horizon_minutes) / m))
    return VolSpec(horizon_bars=hb, base_bars=bb, lags=lags, ewma_lambda=ewma_lambda)


def ewma_variance(r1: np.ndarray, lam: float, init_bars: int) -> np.ndarray:
    """RiskMetrics ``s2_t = lam * s2_{t-1} + (1 - lam) * r_t**2`` (includes bar t). NaN before ``init_bars``."""
    n = len(r1)
    out = np.full(n, np.nan)
    if n < init_bars:
        return out
    s2 = float(np.mean(r1[:init_bars] ** 2))
    out[init_bars - 1] = s2
    for i in range(init_bars, n):
        s2 = lam * s2 + (1.0 - lam) * float(r1[i]) ** 2
        out[i] = s2
    return out


def _trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    c = np.concatenate([[0.0], np.cumsum(x)])
    out[w - 1:] = c[w:] - c[:-w]
    return out


def vol_inputs(sym_bars: pd.DataFrame, vs: VolSpec) -> pd.DataFrame:
    """HAR components (variance per base period) and the EWMA per-bar variance at each bar (PIT)."""
    r1 = intra_session_r1(sym_bars)
    sq = r1 ** 2
    d, w, m = vs.lags
    har_d = _trailing_sum(sq, vs.base_bars * d) / d
    har_w = _trailing_sum(sq, vs.base_bars * w) / w
    har_m = _trailing_sum(sq, vs.base_bars * m) / m
    ew = ewma_variance(r1, vs.ewma_lambda, vs.ewma_init_bars)
    return pd.DataFrame({"r1": r1, "har_d": har_d, "har_w": har_w, "har_m": har_m, "ewma_var": ew}, index=sym_bars.index)


# --------------------------------------------------------------------------------------------
# market proxy for the regime model
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBlocks:
    """Complete, session-anchored blocks of the equal-weighted market proxy (log return, bps)."""

    avail_ns: np.ndarray  # availability of the block (max available_at of its bars)
    end_ns: np.ndarray  # end_ts of the block's last bar
    ret_bps: np.ndarray

    def __len__(self) -> int:
        return len(self.ret_bps)


def market_blocks(bars: pd.DataFrame, block_minutes: int) -> RegimeBlocks:
    """Aggregate a *prepared* multi-symbol frame into complete session-anchored blocks.

    Per timestamp the proxy is the mean intra-session one-bar log return across symbols with a bar at that
    timestamp. Block ``k`` of a session covers bars whose start lies in ``[open + k*B, open + (k+1)*B)``; only
    blocks with every scheduled bar present are kept (a partial / still-open block is never used). A block is
    knowable at the latest ``available_at_ts`` of its bars.
    """
    tf = str(bars["timeframe"].iloc[0])
    bar_min = bar_trading_minutes(tf)
    per_block = max(1, block_minutes // bar_min)
    parts: list[pd.DataFrame] = []
    for _, g in bars.groupby("symbol", sort=True):
        g = g.reset_index(drop=True)
        parts.append(
            pd.DataFrame(
                {
                    "start_ns": ts_ns(g["start_ts"]),
                    "end_ns": ts_ns(g["end_ts"]),
                    "avail_ns": ts_ns(g["available_at_ts"]),
                    "day": [d.toordinal() for d in g["session_date"]],
                    "r1": intra_session_r1(g),
                }
            )
        )
    if not parts:
        return RegimeBlocks(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0))
    allp = pd.concat(parts, ignore_index=True)
    per_ts = allp.groupby("start_ns", sort=True).agg(
        end_ns=("end_ns", "max"), avail_ns=("avail_ns", "max"), day=("day", "first"), r1=("r1", "mean")
    ).reset_index()
    days = [date.fromordinal(int(d)) for d in per_ts["day"]]
    open_ns = _session_open_ns(days)
    slot = ((per_ts["start_ns"].to_numpy(dtype=np.int64) - open_ns) // int(bar_min * 60e9)).astype(np.int64)
    per_ts["block"] = slot // per_block
    agg = per_ts.groupby(["day", "block"], sort=True).agg(
        n=("r1", "size"), ret=("r1", "sum"), end_ns=("end_ns", "max"), avail_ns=("avail_ns", "max")
    ).reset_index()
    agg = agg.loc[agg["n"] == per_block]
    return RegimeBlocks(
        avail_ns=agg["avail_ns"].to_numpy(dtype=np.int64),
        end_ns=agg["end_ns"].to_numpy(dtype=np.int64),
        ret_bps=agg["ret"].to_numpy(dtype=np.float64) * 10_000.0,
    )
