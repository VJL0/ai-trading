"""Causal feature and forward-label engineering for supervised research (spec §3).

Features at bar ``t`` use only bars ``<= t`` of the same symbol (trailing windows), so they are
knowable at the decision time ``available_at_ts[t]``. They are float64 statistics, not accounting.

Labels follow the execution convention of the baselines: a decision at ``available_at_ts[t]`` enters at
the open of the first bar with ``start_ts >= decision`` (index ``e``) and exits ``H`` bars later at the
open of bar ``e + H``. ``fwd_ret = open[e+H] / open[e] - 1``. The label *interval* used for purging is
``[decision_ts, label_end_ts]`` with ``label_end_ts = available_at_ts[e + H]`` (conservative: the time the
exit bar's data is published). For intraday timeframes labels may be restricted to one session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .frames import is_intraday, iter_symbol_frames, trailing_adv_shares, ts_ns


@dataclass(frozen=True)
class FeatureSpec:
    return_horizons: tuple[int, ...] = (1, 5, 20)
    vol_window: int = 20
    volume_window: int = 20
    ma_window: int = 20
    extra: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.return_horizons or min(self.return_horizons) < 1:
            raise ValueError("return horizons must be positive")
        for w in (self.vol_window, self.volume_window, self.ma_window):
            if w < 2:
                raise ValueError("windows must be >= 2")

    @property
    def columns(self) -> tuple[str, ...]:
        return (
            *(f"ret_{h}" for h in self.return_horizons),
            f"rv_{self.vol_window}",
            f"vol_z_{self.volume_window}",
            "range_1",
            f"dist_ma_{self.ma_window}",
        )

    @property
    def warmup_bars(self) -> int:
        return max(max(self.return_horizons), self.vol_window + 1, self.volume_window, self.ma_window)

    def describe(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "definitions": {
                "ret_h": "ln(C_t / C_{t-h})",
                f"rv_{self.vol_window}": f"stdev(ddof=1) of the last {self.vol_window} one-bar log returns",
                f"vol_z_{self.volume_window}": f"z-score of ln(1+V_t) vs trailing {self.volume_window} bars (incl. t)",
                "range_1": "(H_t - L_t) / C_t",
                f"dist_ma_{self.ma_window}": f"C_t / SMA_{self.ma_window}(C)_t - 1",
            },
            "causality": "trailing windows only; value at t uses bars <= t",
            "warmup_bars": self.warmup_bars,
        }


def symbol_features(sym_bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Feature matrix aligned with ``sym_bars`` (a prepared single-symbol frame). NaN during warm-up."""
    close = sym_bars["close_f"].to_numpy(dtype=np.float64)
    high = sym_bars["high_f"].to_numpy(dtype=np.float64)
    low = sym_bars["low_f"].to_numpy(dtype=np.float64)
    volume = sym_bars["volume_f"].to_numpy(dtype=np.float64)
    n = len(close)
    logc = np.log(close)
    out: dict[str, np.ndarray] = {}
    for h in spec.return_horizons:
        r = np.full(n, np.nan)
        if n > h:
            r[h:] = logc[h:] - logc[:-h]
        out[f"ret_{h}"] = r
    lr = np.full(n, np.nan)
    if n > 1:
        lr[1:] = logc[1:] - logc[:-1]
    out[f"rv_{spec.vol_window}"] = np.asarray(
        pd.Series(lr).rolling(spec.vol_window, min_periods=spec.vol_window).std(), dtype=np.float64
    )
    lv = pd.Series(np.log1p(volume))
    mu = np.asarray(lv.rolling(spec.volume_window, min_periods=spec.volume_window).mean(), dtype=np.float64)
    sd = np.asarray(lv.rolling(spec.volume_window, min_periods=spec.volume_window).std(), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"vol_z_{spec.volume_window}"] = np.where(sd > 0, (lv.to_numpy(dtype=np.float64) - mu) / sd, 0.0)
    out[f"vol_z_{spec.volume_window}"][np.isnan(mu)] = np.nan
    out["range_1"] = (high - low) / close
    ma = np.asarray(pd.Series(close).rolling(spec.ma_window, min_periods=spec.ma_window).mean(), dtype=np.float64)
    out[f"dist_ma_{spec.ma_window}"] = close / ma - 1.0
    return pd.DataFrame({c: out[c] for c in spec.columns}, index=sym_bars.index)


def symbol_labels(sym_bars: pd.DataFrame, horizon_bars: int, *, same_session: bool) -> pd.DataFrame:
    """Forward-return labels aligned with ``sym_bars`` (see module docstring). ``valid`` marks usable rows."""
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    start_ns = ts_ns(sym_bars["start_ts"])
    avail_ns = ts_ns(sym_bars["available_at_ts"])
    opens = sym_bars["open_f"].to_numpy(dtype=np.float64)
    n = len(opens)
    entry = np.searchsorted(start_ns, avail_ns, side="left")
    exit_ = entry + horizon_bars
    valid = exit_ < n
    days = np.asarray([d.toordinal() for d in sym_bars["session_date"]], dtype=np.int64)
    safe_entry = np.minimum(entry, n - 1)
    safe_exit = np.minimum(exit_, n - 1)
    if same_session:
        valid &= (days[safe_entry] == days) & (days[safe_exit] == days)
    fwd = np.where(valid, opens[safe_exit] / opens[safe_entry] - 1.0, np.nan)
    return pd.DataFrame(
        {
            "decision_ns": avail_ns,
            "entry_idx": entry,
            "exit_idx": exit_,
            "label_start_ns": np.where(valid, start_ns[safe_entry], -1),
            "label_end_ns": np.where(valid, avail_ns[safe_exit], -1),
            "fwd_ret": fwd,
            "valid": valid,
        },
        index=sym_bars.index,
    )


def build_panel(
    bars: pd.DataFrame,
    spec: FeatureSpec,
    horizon_bars: int,
    *,
    same_session: bool | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Multi-symbol supervised panel from a *prepared* frame, sorted by ``(decision_ns, symbol)``.

    Columns: ``symbol, bar_index, session_date, decision_ns, label_start_ns, label_end_ns, close, adv_shares,
    <features>, fwd_ret, y_up``. Rows with incomplete features or labels are dropped (never filled).
    """
    tf = str(bars["timeframe"].iloc[0])
    if same_session is None:
        same_session = is_intraday(tf)
    parts: list[pd.DataFrame] = []
    for sym, sb in iter_symbol_frames(bars, symbols):
        feats = symbol_features(sb, spec)
        labels = symbol_labels(sb, horizon_bars, same_session=same_session)
        part = pd.DataFrame(
            {
                "symbol": sym,
                "bar_index": np.arange(len(sb)),
                "session_date": list(sb["session_date"]),
                "decision_ns": labels["decision_ns"].to_numpy(),
                "label_start_ns": labels["label_start_ns"].to_numpy(),
                "label_end_ns": labels["label_end_ns"].to_numpy(),
                "close": sb["close_f"].to_numpy(dtype=np.float64),
                "adv_shares": trailing_adv_shares(sb),
            }
        )
        for c in spec.columns:
            part[c] = feats[c].to_numpy(dtype=np.float64)
        part["fwd_ret"] = labels["fwd_ret"].to_numpy(dtype=np.float64)
        ok = labels["valid"].to_numpy() & np.isfinite(part[list(spec.columns)].to_numpy()).all(axis=1)
        parts.append(part.loc[ok])
    if not parts:
        raise ValueError("no symbols in bars")
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["decision_ns", "symbol"], kind="mergesort").reset_index(drop=True)
    panel["y_up"] = (panel["fwd_ret"] > 0).astype(np.int64)
    return panel


def latest_feature_row(sym_bars: pd.DataFrame, spec: FeatureSpec) -> np.ndarray | None:
    """Features of the last (most recent) bar of a prepared single-symbol frame, or None if incomplete."""
    if len(sym_bars) < spec.warmup_bars + 1:
        return None
    tail = sym_bars.iloc[-(spec.warmup_bars + 1) :]
    row = symbol_features(tail, spec).iloc[-1].to_numpy(dtype=np.float64)
    if not np.isfinite(row).all():
        return None
    return row
