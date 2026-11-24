"""Point-in-time opportunity scanner over the daily equity panel (and the crypto panel).

At a given row ``t`` it only uses data through the close of ``t``. It reports *conditions*, not trades: unusual
volume, opening gaps (stocks in play), relative strength vs the market and the data-inferred sector, momentum
quality (return per unit of volatility — the most volatile name is never ranked first merely for moving most),
volatility expansion/compression, 52-week-high proximity, oversold-in-uptrend reversals, changing correlations,
sector leadership and the market regime. Each list names the strategy family that would evaluate it; actual
trade candidates come only from validated strategies through the opportunity engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .panel import EquityPanel, compound, trailing_mean, trailing_std

SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH")


def _corr_last(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Correlation of each column of ``a`` (W, N) with vector ``b`` (W,) over the last ``window`` rows."""
    A = a[-window:]
    B = b[-window:, None]
    ok = np.isfinite(A) & np.isfinite(B)
    n = ok.sum(axis=0)
    Az = np.where(ok, A, 0.0)
    Bz = np.where(ok, B, 0.0)
    ma = Az.sum(0) / np.maximum(n, 1)
    mb = Bz.sum(0) / np.maximum(n, 1)
    cov = (np.where(ok, (A - ma) * (B - mb), 0.0)).sum(0)
    va = (np.where(ok, (A - ma) ** 2, 0.0)).sum(0)
    vb = (np.where(ok, (B - mb) ** 2, 0.0)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = cov / np.sqrt(va * vb)
    return np.where(n >= window * 0.8, c, np.nan)


def assign_sectors(panel: EquityPanel, t: int, window: int = 126) -> tuple[np.ndarray, np.ndarray]:
    """Data-inferred sector per symbol: the sector ETF with the highest trailing return correlation (PIT)."""
    r = panel.f["ret_cc"][: t + 1]
    names = [s for s in SECTOR_ETFS if s in set(panel.symbols)]
    if not names:
        return np.full(len(panel.symbols), None, dtype=object), np.full(len(panel.symbols), np.nan)
    cors = np.stack([_corr_last(r, r[:, panel.col(s)], window) for s in names])
    best = np.nanargmax(np.where(np.isfinite(cors), cors, -9), axis=0)
    sector = np.array([names[i] for i in best], dtype=object)
    bestc = cors[best, np.arange(cors.shape[1])]
    sector = np.array([s if (np.isfinite(c) and c > 0.3) else None for s, c in zip(sector, bestc, strict=True)],
                      dtype=object)
    return sector, bestc


def scan(panel: EquityPanel, t: int | None = None, *, min_price: float = 5.0, min_adv: float = 20e6,
         top: int = 15) -> dict[str, Any]:
    f = panel.f
    t = len(panel.dates) - 1 if t is None else t
    asof = panel.dates[t]
    hist = slice(max(0, t - 300), t + 1)
    sub = {k: v[hist] for k, v in f.items()}
    k = sub["close"].shape[0] - 1
    present = sub["present"][k] > 0
    kind_ok = np.isin(panel.kind, ("COMMON_STOCK", "ETF"))
    liquid = present & kind_ok & (sub["close"][k] >= min_price) & (np.nan_to_num(sub["adv20"][k]) >= min_adv)
    ret = sub["ret_cc"]
    vol63 = trailing_std(ret, 63, 50)[k] * np.sqrt(252)
    vol10 = trailing_std(ret, 10, 8)[k] * np.sqrt(252)
    m21, m63 = compound(ret, 21, 17)[k], compound(ret, 63, 50)[k]
    m126, m252 = compound(ret, 126, 100)[k], compound(ret, 252, 200)[k]
    spy = panel.col("SPY")
    rs63 = m63 - m63[spy]
    adv_shares = trailing_mean(sub["volume"], 20, 15)
    relvol = sub["volume"][k] / np.where(adv_shares[k - 1] > 0, adv_shares[k - 1], np.nan)
    gap = sub["open"][k] / np.where(sub["close"][k - 1] > 0, sub["close"][k - 1], np.nan) - 1
    sma50 = trailing_mean(sub["adj_close"], 50, 45)[k]
    sma200 = trailing_mean(sub["adj_close"], 200, 180)[k]
    px = sub["adj_close"][k]
    hi252 = pd.DataFrame(sub["adj_close"]).rolling(252, min_periods=200).max().to_numpy()[k]
    bb = trailing_std(sub["adj_close"], 20, 18)[k] / np.where(px > 0, px, np.nan)
    bb_hist = trailing_std(sub["adj_close"], 20, 18) / np.where(sub["adj_close"] > 0, sub["adj_close"], np.nan)
    bb_pct = (bb_hist[-252:] <= bb).mean(axis=0) if bb_hist.shape[0] >= 60 else np.full_like(bb, np.nan)
    corr_s = _corr_last(ret, ret[:, spy], 21)
    corr_l = _corr_last(ret, ret[:, spy], 126)
    sector, _sector_corr = assign_sectors(panel, t)
    quality63 = m63 / np.where(vol63 > 0, vol63 * np.sqrt(63 / 252), np.nan)

    def rows(mask: np.ndarray, key: np.ndarray, why: str, family: str, desc: bool = True) -> list[dict[str, Any]]:
        idx = np.where(mask & np.isfinite(key))[0]
        order = idx[np.argsort(-key[idx] if desc else key[idx])][:top]
        return [{"symbol": str(panel.symbols[j]), "kind": str(panel.kind[j]), "score": float(key[j]),
                 "close": float(sub["close"][k][j]), "adv20_usd": float(np.nan_to_num(sub["adv20"][k][j])),
                 "relvol": _f(relvol[j]), "gap": _f(gap[j]), "mom_63d": _f(m63[j]), "rs_vs_spy_63d": _f(rs63[j]),
                 "vol_63d": _f(vol63[j]), "sector": sector[j], "why": why, "family": family} for j in order]

    stocks = liquid & (panel.kind == "COMMON_STOCK")
    breadth = float(np.nanmean((px > sma50)[liquid])) if liquid.any() else float("nan")
    spy_vol = float(vol10[spy])
    spy_vol_hist = trailing_std(ret[:, [spy]], 21, 18)[:, 0] * np.sqrt(252)
    regime = {
        "spy_above_sma200": bool(px[spy] > sma200[spy]) if np.isfinite(sma200[spy]) else None,
        "spy_vol_10d": spy_vol, "spy_vol_pctile_1y": float((spy_vol_hist[-252:] <= spy_vol_hist[-1]).mean()),
        "breadth_above_sma50": breadth, "spy_mom_63d": _f(m63[spy]),
    }
    regime["label"] = ("RISK_ON" if regime["spy_above_sma200"] and breadth > 0.5 else
                       "RISK_OFF" if regime["spy_above_sma200"] is False else "MIXED")
    sector_rows = []
    for s in SECTOR_ETFS:
        if s in set(panel.symbols):
            j = panel.col(s)
            members = stocks & (sector == s)
            sector_rows.append({"sector_etf": s, "mom_63d": _f(m63[j]), "mom_126d": _f(m126[j]),
                                "rs_vs_spy_63d": _f(rs63[j]), "members": int(members.sum()),
                                "member_breadth": float(np.nanmean((px > sma50)[members])) if members.any() else None})
    sector_rows.sort(key=lambda x: -(x["rs_vs_spy_63d"] or -9))
    sec_mom = {r["sector_etf"]: r["mom_63d"] or 0.0 for r in sector_rows}
    rel_sector = m63 - np.array([sec_mom.get(s, np.nan) if s else np.nan for s in sector], dtype=float)
    return {
        "asof": str(asof)[:10], "row": int(t), "universe_size": int(liquid.sum()), "regime": regime,
        "sector_leadership": sector_rows,
        "relative_strength": rows(stocks & (px > sma50), quality63, "63d return per unit vol, above 50d SMA",
                                  "A_momentum"),
        "sector_relative_strength": rows(stocks & np.isfinite(rel_sector), rel_sector,
                                         "63d return minus its inferred sector ETF", "A_momentum"),
        "unusual_volume": rows(liquid & (relvol >= 2.0), relvol, "volume >= 2x 20d average", "B_orb"),
        "gap_movers": rows(liquid & (np.abs(gap) >= 0.03), np.abs(gap), "|open/prev close - 1| >= 3%", "B_orb"),
        "near_52w_high": rows(stocks & (px / hi252 >= 0.98), m252, "within 2% of 52-week high", "A_momentum"),
        "vol_expansion": rows(liquid & (vol10 / vol63 >= 1.8), vol10 / vol63, "10d vol >= 1.8x 63d vol",
                              "risk/E_options"),
        "vol_compression": rows(liquid & (bb_pct <= 0.05), -bb_pct, "20d band width in lowest 5% of 1y",
                                "breakout watch"),
        "oversold_in_uptrend": rows(liquid & (px > sma200) & (m21 < -0.08), -m21,
                                    "21d return < -8% while above 200d SMA", "C2_pullback"),
        "correlation_breaks": rows(stocks & np.isfinite(corr_s) & np.isfinite(corr_l),
                                   np.abs(corr_l - corr_s), "|corr21 - corr126| with SPY", "cross-asset"),
        "notes": ["Conditions only; trade candidates require a validated strategy (see opportunity engine).",
                  "Sector = ETF with highest trailing 126d return correlation (data-inferred, PIT)."],
    }


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
