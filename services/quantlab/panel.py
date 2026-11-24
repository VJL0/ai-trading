"""Point-in-time daily panels (dates x instruments) for vectorized research.

``EquityPanel.build(raw, adj, assets)`` aligns raw and split/dividend-adjusted daily bars on the XNYS session grid:

* ``close/open/high/low/volume/vwap`` — RAW levels (as traded that day; used for price/liquidity filters and costs)
* ``ret_cc``  — adjusted close-to-close total return (dividends reinvested on the ex-date)
* ``ret_oc``/``ret_co`` — adjusted open->close and close->next-open returns (gap), consistent with ``ret_cc``
* ``dollar_vol`` (raw close x volume), ``adv20`` (trailing mean of completed sessions, excludes today)
* segment breaks: a gap of more than ``max_gap_sessions`` missing sessions, or an unexplained |log adjusted
  return| > ``max_abs_log_ret``, starts a new segment — features never span a break (ticker reuse, bad prints)

Knowledge time: a daily bar is available ``settle`` after the session close; every signal built from day ``t``
can only trade at ``t+1`` (open or close). Universe masks are computed from data up to and including ``t``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.market_data.calendar import get_calendar

FIELDS = ("open", "high", "low", "close", "volume", "vwap", "adj_close", "adj_open")


@dataclass
class EquityPanel:
    dates: pd.DatetimeIndex  # session dates (tz-naive, NY calendar date)
    symbols: np.ndarray
    f: dict[str, np.ndarray]  # name -> (T, N) float64
    kind: np.ndarray  # instrument kind per symbol
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.symbols)

    def col(self, symbol: str) -> int:
        idx = np.where(self.symbols == symbol)[0]
        if not len(idx):
            raise KeyError(symbol)
        return int(idx[0])

    def slice_dates(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> EquityPanel:
        m = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            m &= self.dates >= start
        if end is not None:
            m &= self.dates <= end
        return EquityPanel(self.dates[m], self.symbols, {k: v[m] for k, v in self.f.items()}, self.kind,
                           {**self.meta, "sliced": [str(start), str(end)]})

    def subset(self, symbols: list[str] | tuple[str, ...]) -> tuple[EquityPanel, np.ndarray]:
        """Panel restricted to ``symbols`` (those present), plus their column indices in this panel."""
        have = set(self.symbols.tolist())
        cols = np.array([self.col(s) for s in symbols if s in have], dtype=int)
        return EquityPanel(self.dates, self.symbols[cols], {k: v[:, cols] for k, v in self.f.items()},
                           self.kind[cols], dict(self.meta)), cols

    def expand(self, sub_w: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Embed a (T, len(cols)) weight matrix into full width (NaN hold-rows preserved)."""
        out = np.zeros(self.shape)
        out[:, cols] = sub_w
        hold = np.isnan(sub_w).all(axis=1)
        out[hold] = np.nan
        return out

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, raw: pd.DataFrame, adj: pd.DataFrame, assets: pd.DataFrame, *, max_gap_sessions: int = 10,
              max_abs_log_ret: float = 1.2, min_obs: int = 60) -> EquityPanel:
        cal = get_calendar()
        raw = raw.copy()
        adj = adj.copy()
        for df in (raw, adj):
            df["d"] = df["t"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        start, end = raw["d"].min().date(), raw["d"].max().date()
        dates = pd.DatetimeIndex(pd.to_datetime(cal.sessions_in_range(start, end)))
        counts = raw["symbol"].value_counts()
        syms = np.array(sorted(str(k) for k, v in counts.items() if int(v) >= min_obs), dtype=object)
        keep = set(syms.tolist())
        raw = raw.loc[raw["symbol"].isin(sorted(keep))]
        adj = adj.loc[adj["symbol"].isin(sorted(keep))]
        T, N = len(dates), len(syms)
        di = pd.Index(dates)
        si = pd.Index(syms)

        def grid(df: pd.DataFrame, col: str) -> np.ndarray:
            out = np.full((T, N), np.nan)
            r = di.get_indexer(pd.DatetimeIndex(df["d"]))
            c = si.get_indexer(pd.Index(df["symbol"].to_numpy()))
            ok = (r >= 0) & (c >= 0)
            out[r[ok], c[ok]] = df[col].to_numpy(dtype=float)[ok]
            return out

        f = {"open": grid(raw, "o"), "high": grid(raw, "h"), "low": grid(raw, "l"), "close": grid(raw, "c"),
             "volume": grid(raw, "v"), "vwap": grid(raw, "vw"), "trades": grid(raw, "n"),
             "adj_close": grid(adj, "c"), "adj_open": grid(adj, "o")}
        known = pd.Series(assets.drop_duplicates("symbol")["kind"].to_numpy(),
                          index=assets.drop_duplicates("symbol")["symbol"].to_numpy())
        kinds = infer_missing_kinds(syms, known)
        panel = cls(dates, syms, f, kinds, {"built_at": pd.Timestamp.now('UTC').isoformat()})
        panel._derive(max_gap_sessions=max_gap_sessions, max_abs_log_ret=max_abs_log_ret)
        return panel

    def _derive(self, *, max_gap_sessions: int, max_abs_log_ret: float) -> None:
        f = self.f
        T, N = self.shape
        present = np.isfinite(f["close"]) & np.isfinite(f["adj_close"]) & (f["close"] > 0) & (f["adj_close"] > 0)
        # last observed index per column (for gap length) and segment id
        idx = np.where(present, np.arange(T)[:, None], -1)
        last_obs = np.maximum.accumulate(idx, axis=0)
        prev_obs = np.vstack([np.full((1, N), -1), last_obs[:-1]])
        gap = np.where(present & (prev_obs >= 0), np.arange(T)[:, None] - prev_obs - 1, 0)
        adj_c = np.where(present, f["adj_close"], np.nan)
        prev_adj = _ffill_prev(adj_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(adj_c / prev_adj)
        brk = present & ((prev_obs < 0) | (gap > max_gap_sessions) | (np.abs(np.nan_to_num(lr)) > max_abs_log_ret))
        seg = np.cumsum(brk, axis=0)
        f["segment"] = np.where(present, seg, np.nan).astype(float)
        # returns only within a segment and between consecutive observations (gap <= max_gap_sessions)
        valid = present & ~brk
        f["ret_cc"] = np.where(valid, np.exp(lr) - 1.0, np.nan)
        adj_o = np.where(present & (f["adj_open"] > 0), f["adj_open"], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["ret_oc"] = np.where(valid, adj_c / adj_o - 1.0, np.nan)  # first bar of a segment: open untrusted
            f["ret_co"] = np.where(valid, adj_o / prev_adj - 1.0, np.nan)  # overnight gap into today's open
        f["dollar_vol"] = np.where(present, f["close"] * f["volume"], np.nan)
        f["present"] = present.astype(float)
        f["adv20"] = trailing_mean(f["dollar_vol"], 20, min_periods=15)  # excludes today: shifted below
        f["adv20"] = shift(f["adv20"], 1)
        f["obs_count"] = segment_count(f["segment"])
        self.meta["max_gap_sessions"] = max_gap_sessions
        self.meta["max_abs_log_ret"] = max_abs_log_ret

    # ------------------------------------------------------------------ universe
    def universe(self, *, min_price: float = 5.0, min_adv: float = 20e6, min_history: int = 252,
                 kinds: tuple[str, ...] = ("COMMON_STOCK", "ETF"), top_n: int | None = None) -> np.ndarray:
        """Boolean (T, N): eligible at the *close* of day t using only data <= t (ADV uses days < t)."""
        f = self.f
        kind_ok = np.isin(self.kind, kinds)[None, :]
        u = (f["present"] > 0) & (f["close"] >= min_price) & (np.nan_to_num(f["adv20"]) >= min_adv)
        u &= np.nan_to_num(f["obs_count"]) >= min_history
        u &= kind_ok
        if top_n:
            adv = np.where(u, f["adv20"], -np.inf)
            rank = np.argsort(np.argsort(-adv, axis=1), axis=1)
            u &= rank < top_n
        return u

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"f_{k}": v.astype(np.float32 if k not in ("close", "open", "adj_close", "adj_open") else np.float64)
                  for k, v in self.f.items()}
        payload: dict[str, np.ndarray] = {
            "dates": np.asarray(self.dates.values).astype("datetime64[D]").astype("int64"),
            "symbols": self.symbols.astype(str), "kind": self.kind.astype(str), **arrays}
        np.savez_compressed(path, **payload)  # pyright: ignore[reportArgumentType]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = {**self.meta, "sha256": digest, "shape": list(self.shape),
                "first": str(self.dates[0])[:10], "last": str(self.dates[-1])[:10]}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        return digest

    @classmethod
    def load(cls, path: Path) -> EquityPanel:
        z = np.load(path, allow_pickle=False)
        dates = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
        f = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("f_")}
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return cls(dates, z["symbols"].astype(object), f, z["kind"].astype(object), meta)


def infer_missing_kinds(symbols: np.ndarray, known: pd.Series) -> np.ndarray:
    """Delisted symbols recovered from corporate actions are absent from the asset list: plain 1-4 letter tickers
    (and class shares like BRK.B) are treated as common stock; SPAC unit/warrant/right suffixes are excluded."""
    out = known.reindex(symbols).to_numpy(dtype=object)
    for i, s in enumerate(symbols):
        if isinstance(out[i], str):
            continue
        sym = str(s)
        if "." in sym:
            _base, _, suf = sym.partition(".")
            out[i] = "COMMON_STOCK" if len(suf) == 1 and suf in "ABC" else "OTHER"
        elif len(sym) <= 4:
            out[i] = "COMMON_STOCK"
        else:
            out[i] = "OTHER" if sym[-1] in "WRUZ" else "COMMON_STOCK"
    return out


# ------------------------------------------------------------------------------------------- array helpers
def shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _ffill_prev(a: np.ndarray) -> np.ndarray:
    """Value of the previous finite observation (strictly before t) per column."""
    T, N = a.shape
    idx = np.where(np.isfinite(a), np.arange(T)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)
    prev = np.vstack([np.full((1, N), -1), last[:-1]])
    out = np.full_like(a, np.nan, dtype=float)
    ok = prev >= 0
    cols = np.broadcast_to(np.arange(N), (T, N))
    out[ok] = a[prev[ok], cols[ok]]
    return out


def trailing_mean(a: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    min_periods = window if min_periods is None else min_periods
    return np.asarray(pd.DataFrame(a).rolling(window, min_periods=min_periods).mean(), dtype=float)


def trailing_std(a: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    min_periods = window if min_periods is None else min_periods
    return np.asarray(pd.DataFrame(a).rolling(window, min_periods=min_periods).std(), dtype=float)


def trailing_sum(a: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    min_periods = window if min_periods is None else min_periods
    return np.asarray(pd.DataFrame(a).rolling(window, min_periods=min_periods).sum(), dtype=float)


def segment_count(segment: np.ndarray) -> np.ndarray:
    """Observations so far within the current segment (1 on the first bar of a segment)."""
    T, N = segment.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        s = segment[:, j]
        ok = np.isfinite(s)
        if not ok.any():
            continue
        vals = pd.Series(s[ok])
        out[np.where(ok)[0], j] = np.asarray(vals.groupby(vals).cumcount(), dtype=float) + 1
    return out


def compound(ret: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """Trailing compounded return over ``window`` rows (NaN returns break nothing: treated as missing)."""
    lr = np.log1p(np.where(np.isfinite(ret), ret, np.nan))
    s = trailing_sum(lr, window, min_periods=min_periods if min_periods is not None else int(window * 0.8))
    return np.expm1(s)
