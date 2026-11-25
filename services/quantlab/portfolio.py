"""Cost-aware portfolio construction: trade a name only when its calibrated edge beats what trading it costs.

At each rebalance row the selector compares, per eligible name, a conservative edge estimate (the lower bound of
the calibrated expected *excess* return over the holding horizon, see :mod:`services.quantlab.calibration`) with
the round-trip cost of entering and later leaving that position at the intended size (the same spread / slippage /
square-root impact / fee model the backtester charges, :class:`services.quantlab.costs.EquityCosts`):

    utility_i = edge_lcb_i - (0 if i is already held else round_trip_cost_i)

Names with ``utility > min_edge`` compete for at most ``n`` equal slots. A held name pays no new entry cost, so it is
kept until a candidate's net edge beats it (a no-trade region instead of chasing small rank changes). Slots without
a qualifying name go to ``park_col`` (the market ETF) when given: the edge is measured *relative to the market*, so
"no stock beats its costs" means "hold the market", not "hold cash". Rows that are not rebalance rows are NaN
(hold and drift), matching :func:`services.quantlab.backtest.run_weights`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .costs import EquityCosts


def round_trip_cost_bps(f: dict[str, np.ndarray], costs: EquityCosts, notional: float,
                        timing: str = "open") -> np.ndarray:
    """Entry (buy) + exit (sell, incl. SEC/FINRA fees) cost in bps of ``notional`` per name, from data known at the
    close of each row (ADV of completed sessions, the row's raw close)."""
    adv = f["adv20"]
    px = f["close"]
    size = np.full(adv.shape, float(notional))
    buy = costs.one_way_bps(adv, size, px, sell=np.zeros(adv.shape, dtype=bool), timing=timing)
    sell = costs.one_way_bps(adv, size, px, sell=np.ones(adv.shape, dtype=bool), timing=timing)
    return np.where(np.isfinite(px) & (px > 0), buy + sell, np.nan)


def select_cost_aware(edge_lcb_bps: np.ndarray, cost_rt_bps: np.ndarray, eligible: np.ndarray, n: int,
                      rebalance: np.ndarray, *, min_edge_bps: float = 0.0, park_col: int | None = None,
                      active: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Target weights (T, N) and selection statistics. ``active`` (T,) limits trading (and parking) to rows where
    the strategy is live; inactive rebalance rows are flat."""
    if n < 1:
        raise ValueError("n must be >= 1")
    T, N = edge_lcb_bps.shape
    out = np.full((T, N), np.nan)
    held = np.zeros(N, dtype=bool)
    live = np.ones(T, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    picked: list[int] = []
    kept = entered = 0
    for t in np.where(rebalance)[0]:
        row = np.zeros(N)
        if not live[t]:
            held[:] = False
            out[t] = row
            continue
        valid = eligible[t] & np.isfinite(edge_lcb_bps[t]) & np.isfinite(cost_rt_bps[t])
        if park_col is not None:
            valid[park_col] = False
        util = edge_lcb_bps[t] - np.where(held, 0.0, cost_rt_bps[t])
        cand = np.where(valid & (util > min_edge_bps))[0]
        if len(cand):
            order = cand[np.argsort(-util[cand], kind="stable")][:n]
            row[order] = 1.0 / n
            kept += int(held[order].sum())
            entered += int((~held[order]).sum())
            new_held = np.zeros(N, dtype=bool)
            new_held[order] = True
            held = new_held
            picked.append(len(order))
        else:
            held[:] = False
            picked.append(0)
        if park_col is not None:
            row[park_col] += 1.0 - row.sum()
        out[t] = row
    stats = {"rebalances": len(picked), "avg_names": float(np.mean(picked)) if picked else 0.0,
             "share_rebalances_with_no_name": float(np.mean([p == 0 for p in picked])) if picked else None,
             "kept_slots": kept, "new_slots": entered}
    return out, stats
