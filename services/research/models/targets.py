"""Forward NET return targets, success labels, cross-sectional rank targets, vol and execution-cost labels.

Decision time (cross-sectional, ``aligned_decisions``): the bars of all symbols that share a bar slot (``start_ts``)
form one cross-section; it is complete - and the decision is taken - at the LATEST ``available_at_ts`` among them
(forced strictly increasing across slots). With equal publication latencies this is every bar's own
``available_at_ts``; with per-symbol latencies (live-recorded data) no row is decided before all of its
cross-section is knowable, so rank targets and the one-position strategy compare symbols at one common time.

Execution convention (identical to the research backtester / ``features.symbol_labels``, see ``forward_labels``):
a decision enters at the open of the first bar whose ``start_ts >= decision`` (the *next* bar that opens after the
decision; never a bar already open) and exits ``H`` bars later at that bar's open, inside the same regular
session. For each horizon ``h`` in minutes (``H = h / bar minutes``):

* ``gross_bps[h] = (open[e+H] / open[e] - 1) * 1e4``;
* ``net_bps[h] = gross_bps[h] - cost_rt_bps`` where ``cost_rt_bps`` is the ``CostModel`` round trip (BUY + SELL)
  for ``reference_notional`` at the decision-time close, the trailing completed-session ADV and - when quotes are
  supplied - the last quote spread published by the decision time (else the liquidity-bucket estimate);
* ``success = net_bps[success_h] > 0``;
* ``rank_pct[h]`` = cross-sectional percentile of ``net_bps[rank_h]`` among the symbols at the same decision time
  (0 = worst, 1 = best; NaN when fewer than two symbols), ``rank_rel`` = integer relevance for lambdarank.

Vol label: realised variance of the holding window ``sum(r1[e .. e+Hv-1] ** 2)`` (intra-session returns).

Cost label (``realized_cost_bps``, one way, BUY): an arrival-price shortfall PROXY = half the quoted spread at the
execution time (quote when available, else the bucket estimate) + entry fees + ``(open[e] / close[t] - 1) * 1e4``.
There are no broker fills in this project's data: on synthetic data this "realised" slippage is SIMULATED.

Rows whose features or labels are incomplete, or whose liquidity is unknown (no completed-session ADV, which the
signal engine also refuses), are dropped - never filled. ``label_end_ns`` (used for purging) is the latest label
end over every target of the row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from packages.core.money import floor_shares

from ..costs import CostModel
from ..features import FeatureSpec
from ..frames import (
    REGULAR_SESSION_MINUTES,
    bar_trading_minutes,
    dataset_version_of,
    is_intraday,
    is_simulated_of,
    iter_symbol_frames,
    trailing_adv_shares,
    ts_ns,
)
from .config import SuiteConfig
from .features import (
    RegimeBlocks,
    VolSpec,
    market_blocks,
    model_feature_columns,
    symbol_model_features,
    vol_inputs,
    vol_spec_for,
)

VOL_COLUMNS: tuple[str, ...] = ("har_d", "har_w", "har_m", "ewma_var")


def cost_feature_columns(spec: FeatureSpec) -> tuple[str, ...]:
    """Features of the execution-cost residual model (all knowable at the decision time)."""
    return ("quoted_spread_bps", f"rv_{spec.vol_window}", "participation", "log_adv_usd", "tod",
            f"ret_{min(spec.return_horizons)}")


@dataclass
class QuoteSpreads:
    """Per-symbol quote spreads (bps of mid) sorted by availability (int64 ns)."""

    by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=lambda: {})
    source: str = "none"

    @classmethod
    def from_frame(cls, df: pd.DataFrame | None, source: str = "quotes") -> QuoteSpreads:
        """``df`` columns: ``symbol, available_at_ns (int64), spread_bps (float)``."""
        if df is None or df.empty:
            return cls()
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sym, g in df.groupby("symbol", sort=True):
            g = g.sort_values("available_at_ns", kind="mergesort")
            sp = g["spread_bps"].to_numpy(dtype=np.float64)
            ok = np.isfinite(sp) & (sp >= 0)
            out[str(sym)] = (g["available_at_ns"].to_numpy(dtype=np.int64)[ok], sp[ok])
        return cls(out, source)

    def lookup(self, symbol: str, query_ns: np.ndarray, max_age_ns: int) -> np.ndarray:
        """Last spread published at or before each query time and no older than ``max_age_ns`` (else NaN)."""
        q = np.asarray(query_ns, dtype=np.int64)
        res = np.full(len(q), np.nan)
        got = self.by_symbol.get(symbol)
        if got is None or len(got[0]) == 0:
            return res
        avail, sp = got
        idx = np.searchsorted(avail, q, side="right") - 1
        ok = idx >= 0
        safe = np.where(ok, idx, 0)
        fresh = ok & (q - avail[safe] <= max_age_ns)
        res[fresh] = sp[safe[fresh]]
        return res


@dataclass
class SuiteDataset:
    panel: pd.DataFrame
    X: np.ndarray
    feature_columns: tuple[str, ...]
    decision_ns: np.ndarray
    label_end_ns: np.ndarray
    keys: list[str]
    horizons_bars: dict[int, int]
    gross_bps: dict[int, np.ndarray]
    net_bps: dict[int, np.ndarray]
    exit_ns: dict[int, np.ndarray]
    cost_rt_bps: np.ndarray
    success: np.ndarray
    rank_pct: np.ndarray
    rank_rel: np.ndarray
    vol_X: np.ndarray
    vol_valid: np.ndarray
    rv_target: np.ndarray
    cost_X: np.ndarray
    baseline_cost_bps: np.ndarray
    cost_floor_bps: np.ndarray
    realized_cost_bps: np.ndarray
    blocks: RegimeBlocks
    regime_symbols: list[str]
    vol_spec: VolSpec
    har_base_reason: str
    timeframe: str
    bar_minutes: int
    dataset_version: str
    is_simulated: bool
    spread_source: str
    symbols: list[str]

    def __len__(self) -> int:
        return len(self.decision_ns)


@dataclass
class _CostCache:
    model: CostModel
    notional: Decimal
    cache: dict[tuple[str, str, str, str], tuple[float, float, float, float, float, float]] = field(
        default_factory=lambda: {}
    )

    def get(self, sym: str, price: Decimal, adv_f: float, spread_f: float) -> tuple[float, float, float, float, float, float]:
        """(round-trip bps, entry total bps, entry half-spread bps, entry fees bps, quoted spread bps, participation)."""
        price_q = price.quantize(Decimal("0.01"))
        adv_key = f"{adv_f:.3g}" if math.isfinite(adv_f) and adv_f > 0 else "none"
        sp_key = f"{spread_f:.1f}" if math.isfinite(spread_f) else "none"
        key = (sym, format(price_q, "f"), adv_key, sp_key)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        adv = None if adv_key == "none" else Decimal(f"{float(adv_key):.0f}")
        spread = None if sp_key == "none" else Decimal(sp_key)
        qty = max(Decimal(1), floor_shares(self.notional / price_q))
        rt = self.model.estimate_round_trip(sym, qty, price_q, spread_bps=spread, adv_shares=adv)
        part = float(rt.entry.participation) if rt.entry.participation is not None else float("nan")
        val = (float(rt.total_bps), float(rt.entry.total_bps), float(rt.entry.spread_cost_bps),
               float(rt.entry.fees_bps), float(rt.entry.quoted_spread_bps), part)
        self.cache[key] = val
        return val


def round_trip_cost_bps(cost_model: CostModel, symbol: str, price: Decimal, adv_shares: float, spread_bps: float | None,
                        notional: Decimal) -> float:
    """Round-trip cost (bps) for one decision, using exactly the training-time convention."""
    cc = _CostCache(cost_model, notional)
    return cc.get(symbol, price, adv_shares, float("nan") if spread_bps is None else spread_bps)[0]


def aligned_decisions(bar_start_ns: np.ndarray, avail_ns: np.ndarray) -> np.ndarray:
    """Cross-sectional decision time of every bar (see module docstring).

    ``decision = max(available_at)`` over all bars sharing the bar's ``start_ts`` (all symbols), then made strictly
    increasing across slots (``d_k >= d_{k-1} + 1 ns``) so each slot has its own decision time and every symbol's
    decision sequence stays monotone. Never earlier than the bar's own availability (features stay knowable).
    """
    starts = np.asarray(bar_start_ns, dtype=np.int64)
    avail = np.asarray(avail_ns, dtype=np.int64)
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64)
    slots, inv = np.unique(starts, return_inverse=True)
    smax = np.full(len(slots), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(smax, inv, avail)
    k = np.arange(len(slots), dtype=np.int64)
    smax = np.maximum.accumulate(smax - k) + k
    out = smax[inv]
    assert (out >= avail).all()
    return out


def forward_labels(sb: pd.DataFrame, decision_ns: np.ndarray, horizon_bars: int, *, same_session: bool
                   ) -> dict[str, np.ndarray]:
    """``features.symbol_labels`` with an explicit decision time per bar (identical when decision = own availability).

    Entry = first bar with ``start_ts >= decision``; exit = entry + ``horizon_bars``; with ``same_session`` the entry
    and exit bars must belong to the decision bar's session. ``label_end_ns`` = availability of the exit bar.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    start_ns = ts_ns(sb["start_ts"])
    avail_ns = ts_ns(sb["available_at_ts"])
    opens = sb["open_f"].to_numpy(dtype=np.float64)
    n = len(opens)
    d = np.asarray(decision_ns, dtype=np.int64)
    entry = np.searchsorted(start_ns, d, side="left")
    exit_ = entry + horizon_bars
    valid = exit_ < n
    days = np.asarray([x.toordinal() for x in sb["session_date"]], dtype=np.int64)
    safe_entry = np.minimum(entry, n - 1)
    safe_exit = np.minimum(exit_, n - 1)
    if same_session:
        valid &= (days[safe_entry] == days) & (days[safe_exit] == days)
    return {
        "entry_idx": entry,
        "exit_idx": exit_,
        "label_start_ns": np.where(valid, start_ns[safe_entry], -1),
        "label_end_ns": np.where(valid, avail_ns[safe_exit], -1),
        "fwd_ret": np.where(valid, opens[safe_exit] / opens[safe_entry] - 1.0, np.nan),
        "valid": valid,
    }


def choose_har_base_minutes(n_sessions: int, cfg: SuiteConfig) -> tuple[int, str]:
    """HAR base period: one regular session (literal daily/weekly/monthly) when the history supports it."""
    if cfg.har_base_minutes is not None:
        return cfg.har_base_minutes, f"configured: base period {cfg.har_base_minutes} trading minutes"
    need = cfg.har_lags[2] + cfg.har_min_fit_sessions
    if n_sessions >= need:
        return REGULAR_SESSION_MINUTES, (
            f"literal HAR: base period = one regular session ({REGULAR_SESSION_MINUTES} trading minutes), "
            f"daily/weekly/monthly = {cfg.har_lags[0]}/{cfg.har_lags[1]}/{cfg.har_lags[2]} sessions")
    return cfg.vol_horizon_minutes, (
        f"FALLBACK: only {n_sessions} sessions of history (< {need} = {cfg.har_lags[2]}-session monthly window + "
        f"{cfg.har_min_fit_sessions} sessions to fit); the HAR cascade uses {cfg.vol_horizon_minutes}-minute base "
        f"periods ({cfg.har_lags[0]}/{cfg.har_lags[1]}/{cfg.har_lags[2]} x {cfg.vol_horizon_minutes} min)")


def _per_symbol(sb: pd.DataFrame, decision_ns: np.ndarray, cfg: SuiteConfig, vs: VolSpec, hb: dict[int, int],
                same_session: bool, quotes: QuoteSpreads, cost: _CostCache) -> pd.DataFrame:
    sym = str(sb["symbol"].iloc[0])
    feats = symbol_model_features(sb, cfg.feature_spec)
    vi = vol_inputs(sb, vs)
    n = len(sb)
    start_ns = ts_ns(sb["start_ts"])
    dec = np.asarray(decision_ns, dtype=np.int64)
    opens = sb["open_f"].to_numpy(dtype=np.float64)
    close_f = sb["close_f"].to_numpy(dtype=np.float64)
    closes: list[Decimal] = list(sb["close"])
    adv = trailing_adv_shares(sb)
    out = pd.DataFrame({"symbol": sym, "decision_ns": dec, "bar_index": np.arange(n), "bar_start_ns": start_ns,
                        "own_available_ns": ts_ns(sb["available_at_ts"])})
    for c in feats.columns:
        out[c] = feats[c].to_numpy(dtype=np.float64)
    valid = np.isfinite(feats.to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(adv) & (adv > 0)
    label_end = np.full(n, -1, dtype=np.int64)
    entry = None
    for h_min, H in sorted(hb.items()):
        lab = forward_labels(sb, dec, H, same_session=same_session)
        out[f"gross_{h_min}"] = lab["fwd_ret"] * 10_000.0
        le = lab["label_end_ns"].astype(np.int64)
        out[f"exit_ns_{h_min}"] = le
        valid &= lab["valid"]
        label_end = np.maximum(label_end, le)
        entry = lab["entry_idx"].astype(np.int64)
    assert entry is not None
    # vol label: realised variance of bars e .. e+Hv-1 (same session)
    vlab = forward_labels(sb, dec, vs.horizon_bars, same_session=same_session)
    vvalid = vlab["valid"]
    sq = vi["r1"].to_numpy(dtype=np.float64) ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    e = np.minimum(entry, n)
    e_end = np.minimum(e + vs.horizon_bars, n)
    rv = np.where(vvalid, csum[e_end] - csum[e], np.nan)
    out["rv_target"] = rv
    label_end = np.maximum(label_end, np.where(vvalid, vlab["label_end_ns"].astype(np.int64), -1))
    for c in VOL_COLUMNS:
        out[c] = vi[c].to_numpy(dtype=np.float64)
    out["vol_valid"] = vvalid & np.isfinite(vi[list(VOL_COLUMNS)].to_numpy(dtype=np.float64)).all(axis=1) & (rv > 0)
    out["label_end_ns"] = label_end
    # costs (Decimal CostModel; PIT inputs only: the last quote published by the decision time)
    max_age = int(cfg.max_quote_age_s * 1e9)
    dec_spread = quotes.lookup(sym, dec, max_age)
    safe_e = np.minimum(entry, n - 1)
    exec_spread = quotes.lookup(sym, start_ns[safe_e], max_age)
    rt = np.full(n, np.nan)
    base1 = np.full(n, np.nan)
    qspread = np.full(n, np.nan)
    part = np.full(n, np.nan)
    real_half = np.full(n, np.nan)
    half1 = np.full(n, np.nan)
    fees1 = np.full(n, np.nan)
    for i in np.flatnonzero(valid):
        r, b, half, fee, qs, pa = cost.get(sym, closes[i], float(adv[i]), float(dec_spread[i]))
        rt[i], base1[i], half1[i], fees1[i], qspread[i], part[i] = r, b, half, fee, qs, pa
        if math.isfinite(exec_spread[i]):
            real_half[i] = exec_spread[i] / 2.0
        else:
            real_half[i] = half  # no execution-time quote: bucket estimate (same as the baseline)
    delay = (opens[safe_e] / close_f - 1.0) * 10_000.0
    out["cost_rt_bps"] = rt
    out["baseline_cost_bps"] = base1
    out["cost_floor_bps"] = half1 + fees1
    out["realized_cost_bps"] = real_half + fees1 + delay
    out["quoted_spread_bps"] = qspread
    out["participation"] = np.where(np.isfinite(part), part, 0.0)
    out["log_adv_usd"] = np.log10(np.maximum(adv * close_f, 1.0))
    out["valid"] = valid
    return out


def build_suite_dataset(
    bars: pd.DataFrame,
    cost_model: CostModel,
    cfg: SuiteConfig,
    *,
    quotes: QuoteSpreads | None = None,
) -> SuiteDataset:
    """Panel over a *prepared* (and data-quality-cut) bar frame, sorted by ``(decision_ns, symbol)``."""
    tf = str(bars["timeframe"].iloc[0])
    bar_min = bar_trading_minutes(tf)
    hb = {h: max(1, round(h / bar_min)) for h in cfg.return_horizons_minutes}
    if len(set(hb.values())) != len(hb):
        raise ValueError(f"timeframe {tf} maps distinct return horizons onto the same bar count: {hb}")
    n_sessions = len(set(bars["session_date"]))
    har_base, har_reason = choose_har_base_minutes(n_sessions, cfg)
    vs = vol_spec_for(tf, cfg.vol_horizon_minutes, har_base, cfg.har_lags, cfg.ewma_lambda)
    same_session = is_intraday(tf)
    q = quotes or QuoteSpreads()
    cost = _CostCache(cost_model, cfg.reference_notional)
    frames = list(iter_symbol_frames(bars))
    if not frames:
        raise ValueError("no symbols in bars")
    all_start = np.concatenate([ts_ns(sb["start_ts"]) for _, sb in frames])
    all_avail = np.concatenate([ts_ns(sb["available_at_ts"]) for _, sb in frames])
    all_dec = aligned_decisions(all_start, all_avail)
    parts: list[pd.DataFrame] = []
    pos = 0
    for _, sb in frames:
        parts.append(_per_symbol(sb, all_dec[pos:pos + len(sb)], cfg, vs, hb, same_session, q, cost))
        pos += len(sb)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.loc[panel["valid"].to_numpy(dtype=bool)]
    if panel.empty:
        raise ValueError("no complete feature/label rows (history too short or liquidity unknown everywhere)")
    panel = panel.sort_values(["decision_ns", "symbol"], kind="mergesort").reset_index(drop=True)
    cols = model_feature_columns(cfg.feature_spec)
    X = panel[list(cols)].to_numpy(dtype=np.float64)
    gross = {h: panel[f"gross_{h}"].to_numpy(dtype=np.float64) for h in hb}
    cost_rt = panel["cost_rt_bps"].to_numpy(dtype=np.float64)
    net = {h: gross[h] - cost_rt for h in hb}
    decision_ns = panel["decision_ns"].to_numpy(dtype=np.int64)
    rank_pct, rank_rel = cross_sectional_rank(decision_ns, net[cfg.rank_horizon_minutes], cfg.rank_levels)
    panel["rank_pct"] = rank_pct
    cost_X = panel[list(cost_feature_columns(cfg.feature_spec))].to_numpy(dtype=np.float64)
    cost_X[:, 0] = np.nan_to_num(cost_X[:, 0], nan=0.0)
    return SuiteDataset(
        panel=panel,
        X=X,
        feature_columns=cols,
        decision_ns=decision_ns,
        label_end_ns=panel["label_end_ns"].to_numpy(dtype=np.int64),
        keys=[f"{s}|{d}" for s, d in zip(panel["symbol"], decision_ns, strict=True)],
        horizons_bars=hb,
        gross_bps=gross,
        net_bps=net,
        exit_ns={h: panel[f"exit_ns_{h}"].to_numpy(dtype=np.int64) for h in hb},
        cost_rt_bps=cost_rt,
        success=(net[cfg.success_horizon_minutes] > 0).astype(np.int64),
        rank_pct=rank_pct,
        rank_rel=rank_rel,
        vol_X=panel[list(VOL_COLUMNS)].to_numpy(dtype=np.float64),
        vol_valid=panel["vol_valid"].to_numpy(dtype=bool),
        rv_target=panel["rv_target"].to_numpy(dtype=np.float64),
        cost_X=cost_X,
        baseline_cost_bps=panel["baseline_cost_bps"].to_numpy(dtype=np.float64),
        cost_floor_bps=panel["cost_floor_bps"].to_numpy(dtype=np.float64),
        realized_cost_bps=panel["realized_cost_bps"].to_numpy(dtype=np.float64),
        blocks=market_blocks(bars, cfg.regime_block_minutes),
        regime_symbols=sorted({str(s) for s, _ in frames}),
        vol_spec=vs,
        har_base_reason=har_reason,
        timeframe=tf,
        bar_minutes=bar_min,
        dataset_version=dataset_version_of(bars),
        is_simulated=is_simulated_of(bars),
        spread_source=q.source if q.by_symbol else "liquidity-bucket estimate (no quotes supplied)",
        symbols=sorted(set(panel["symbol"].astype(str))),
    )


def cross_sectional_rank(decision_ns: np.ndarray, values: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Percentile rank of ``values`` within each decision time (average ties) and integer relevance labels.

    ``pct = (rank - 1) / (n - 1)`` in [0, 1]; NaN (relevance -1) for groups with fewer than two members.
    ``relevance = min(levels - 1, floor(pct * levels))``.
    """
    df = pd.DataFrame({"t": decision_ns, "v": values})
    n = df.groupby("t")["v"].transform("size").to_numpy(dtype=np.float64)
    r = df.groupby("t")["v"].rank(method="average").to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(n >= 2, (r - 1.0) / (n - 1.0), np.nan)
    rel = np.where(np.isfinite(pct), np.minimum(levels - 1, np.floor(pct * levels)), -1).astype(np.int64)
    return pct, rel


def describe_targets(cfg: SuiteConfig, ds: SuiteDataset) -> dict[str, Any]:
    return {
        "decision_time": "latest available_at_ts among the bars of the same bar slot (cross-section complete), "
                         "strictly increasing across slots",
        "execution": "enter at the open of the first bar starting at/after the decision; exit H bars later at the open; "
                     "same regular session",
        "return_horizons": {f"{h}m": {"bars": b} for h, b in ds.horizons_bars.items()},
        "net_return": "gross open-to-open return (bps) minus the CostModel round trip for reference_notional "
                      f"{cfg.reference_notional} USD at decision-time close / trailing ADV / PIT quote spread",
        "success": f"net return over {cfg.success_horizon_minutes} minutes > 0",
        "rank": f"cross-sectional percentile of the {cfg.rank_horizon_minutes}-minute net return per decision time; "
                f"lambdarank relevance = floor(pct * {cfg.rank_levels})",
        "vol": f"realised variance of the {cfg.vol_horizon_minutes}-minute holding window (intra-session 1-bar log "
               "returns)",
        "cost": "one-way BUY arrival-price shortfall proxy: half quoted spread at execution + fees + (next open / "
                "decision close - 1); no broker fills exist - SIMULATED on synthetic data",
        "spread_source": ds.spread_source,
        "har_base": ds.har_base_reason,
        "decisions_delayed_by_cross_section": int(np.count_nonzero(
            ds.panel["decision_ns"].to_numpy(dtype=np.int64) > ds.panel["own_available_ns"].to_numpy(dtype=np.int64))),
        "rows": len(ds),
        "vol_rows": int(ds.vol_valid.sum()),
        "regime_blocks": len(ds.blocks),
    }


def quote_spreads_from_mapping(m: Mapping[str, Decimal] | None) -> dict[str, float]:
    return {} if not m else {str(k): float(v) for k, v in m.items()}
