"""Parity: the same snapshot and strategy through the research simulator (``services.quantlab.intraday``) and the
replay of the real stack, trade by trade.

The research side runs unchanged ``intraday.simulate`` on an ``IntradayPanel`` built from the snapshot's bars and a
``DailyContext`` built from the snapshot's own point-in-time daily aggregates (an adapter: QuantLab normally aligns
a daily equity panel; here ADV and previous closes come from the same bars, as-traded, no adjustments). Sizing is
the runtime's (``max_order_notional x 0.5``, whole shares), the concurrency cap is ``policy.max_positions`` and the
entry limit is the decision close (the runtime's limit is the ask at the decision, and the replay fills at the ask
of the next open, so "next open <= decision close" is the shared fill rule).

Reported differences:

* ``entries_missed_in_replay`` - research filled an entry the replay did not trade (risk rejection, lock, fill
  model, sizing);
* ``entries_missed_in_research`` - the replay traded an entry research did not fill or had no leg for;
* per matched trade: entry and exit fill price deltas (bps; the replay pays the synthesized half spread in the
  price, research charges it as cost), exit reason categories, exit time deltas, gross and net bps.

Expected, documented differences: replay fills at bid/ask (research: trade price + modelled half spread), the
position manager's exit order lands a few seconds after the trigger (research: the trigger bar's open), the
replay stop is relative to the entry LIMIT (research: the fill price), research's opening-spread multiplier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np

from packages.contracts.policy import RiskPolicy
from services.order_ledger.timeutil import parse_iso
from services.positions.policy import PositionPolicy

from .snapshot import BAR, BarSnapshot, session_open_close
from .strategies import ReplayStrategy

STOP_REASONS = ("STOP", "STOP_GAP", "STOP_BREACH")


def exit_category(reason: str | None) -> str:
    r = (reason or "").upper()
    if r.startswith("STOP"):
        return "STOP"
    if r in ("EOD", "HOLDING_LIMIT"):
        return "EOD"
    if r == "TIME":
        return "TIME"
    if r == "FORCED":
        return "FORCED"
    return r or "UNKNOWN"


@dataclass(frozen=True)
class Tolerance:
    entry_bps: float = 5.0
    exit_bps: float = 10.0
    exit_minutes: float = 5.0
    net_bps: float = 15.0


def daily_context(snapshot: BarSnapshot, panel: Any) -> Any:
    from services.quantlab.intraday import DailyContext

    D, N = len(panel.days), len(panel.symbols)
    adv = np.full((D, N), np.nan)
    prev = np.full((D, N), np.nan)
    for d, day in enumerate(panel.days):
        for j, sym in enumerate(panel.symbols.tolist()):
            a = snapshot.adv_shares(str(sym), day.date())
            p = snapshot.prev_close(str(sym), day.date())
            if p is not None:
                prev[d, j] = p
                if a is not None:
                    adv[d, j] = a * p
    false = np.zeros((D, N), dtype=bool)
    return DailyContext(adv_usd=adv, raw_prev_close=prev, adj_prev_close=prev, above_sma50=false,
                        above_sma200=false.copy(), spy_ret_cc=np.full(D, np.nan))


def research_outcomes(snapshot: BarSnapshot, strategy: ReplayStrategy, days: Sequence[date], policy: RiskPolicy,
                      equity: Decimal) -> tuple[Any, list[Any], Any]:
    from services.quantlab.intraday import IntradayPanel, SimConfig, simulate

    if strategy.research is None:
        raise ValueError(f"strategy {strategy.strategy_id} has no research counterpart (parity unavailable)")
    panel = IntradayPanel.build(snapshot.research_frame(), symbols=list(snapshot.symbols))
    ctx = daily_context(snapshot, panel)
    legs = strategy.research(panel, list(snapshot.symbols), policy)
    mask = np.isin(np.array([d.date() for d in panel.days]), np.array(list(days)))
    notional = float(policy.max_order_notional) * 0.5
    cfg = SimConfig(capital=float(equity), weight=notional / float(equity), max_concurrent=max(1, policy.max_positions),
                    limit_bps=0.0, whole_shares=True, stop_limit_gap_bps=float(PositionPolicy().stop_limit_gap_bps))
    result, outcomes = simulate(panel, ctx, legs, cfg, day_mask=mask)
    return result, outcomes, panel


def _slot_of(at: datetime) -> tuple[date, int]:
    from services.order_ledger.timeutil import trading_day

    day = trading_day(at)
    b = session_open_close(day)
    assert b is not None
    return day, int((at - b[0]).total_seconds() // BAR.total_seconds())


def compare(replay_summary: dict[str, Any], snapshot: BarSnapshot, strategy: ReplayStrategy, days: Sequence[date],
            policy: RiskPolicy, equity: Decimal, tolerance: Tolerance | None = None) -> dict[str, Any]:
    tol = tolerance or Tolerance()
    _, outcomes, panel = research_outcomes(snapshot, strategy, days, policy, equity)
    research: dict[tuple[str, date, int], Any] = {}
    research_nonfills: dict[tuple[str, date, int], str] = {}
    for o in outcomes:
        sym = str(panel.symbols[o.leg.j])
        day = panel.days[o.leg.d].date()
        key = (sym, day, o.entry_slot if o.filled else o.leg.decision_slot + 1)
        (research if o.filled else research_nonfills).__setitem__(key, o if o.filled else o.exit_reason)
    replay: dict[tuple[str, date, int], dict[str, Any]] = {}
    for t in replay_summary.get("trades", []):
        day, slot = _slot_of(parse_iso(t["entry_at"]))
        replay[(t["symbol"], day, slot)] = t
    matched: list[dict[str, Any]] = []
    for key in sorted(set(research) & set(replay)):
        o, t = research[key], replay[key]
        entry_r, exit_r = float(o.entry_px), float(o.exit_px)
        entry_p, exit_p = float(t["entry_avg_price"]), float(t["exit_avg_price"])
        exit_day = panel.days[o.exit_day_actual].date()
        b = session_open_close(exit_day)
        assert b is not None
        exit_r_at = b[0] + o.exit_slot_actual * BAR
        exit_p_at = parse_iso(t["exit_at"]) if t.get("exit_at") else exit_r_at
        net_r = (o.gross - o.cost) * 1e4
        matched.append({
            "symbol": key[0], "session": key[1].isoformat(), "entry_slot": key[2],
            "entry_px_research": round(entry_r, 4), "entry_px_replay": entry_p,
            "entry_delta_bps": round((entry_p / entry_r - 1) * 1e4, 3),
            "exit_px_research": round(exit_r, 4), "exit_px_replay": exit_p,
            "exit_delta_bps": round((exit_p / exit_r - 1) * 1e4, 3),
            "exit_reason_research": o.exit_reason, "exit_reason_replay": t.get("close_reason") or t.get("exit_reason"),
            "exit_category_research": exit_category(o.exit_reason),
            "exit_category_replay": exit_category(t.get("exit_reason") or t.get("close_reason")),
            "exit_time_delta_min": round((exit_p_at - exit_r_at).total_seconds() / 60, 2),
            "gross_bps_research": round(o.gross * 1e4, 3), "gross_bps_replay": t.get("gross_bps"),
            "net_bps_research": round(net_r, 3), "net_bps_replay": t.get("net_bps"),
            "net_delta_bps": None if t.get("net_bps") is None else round(float(t["net_bps"]) - net_r, 3),
        })
    missed_replay = [{"symbol": k[0], "session": k[1].isoformat(), "entry_slot": k[2],
                      "research_entry_px": round(float(research[k].entry_px), 4)} for k in sorted(set(research) - set(replay))]
    missed_research = [{"symbol": k[0], "session": k[1].isoformat(), "entry_slot": k[2],
                        "research_outcome": research_nonfills.get(k, "NO_LEG")} for k in sorted(set(replay) - set(research))]

    def _stat(name: str) -> dict[str, float | None]:
        xs = [abs(float(m[name])) for m in matched if m.get(name) is not None]
        return {"mean_abs": round(float(np.mean(xs)), 3) if xs else None, "max_abs": round(max(xs), 3) if xs else None}

    agree = [m["exit_category_research"] == m["exit_category_replay"] for m in matched]
    within = [abs(m["entry_delta_bps"]) <= tol.entry_bps and abs(m["exit_delta_bps"]) <= tol.exit_bps
              and abs(m["exit_time_delta_min"]) <= tol.exit_minutes and m["exit_category_research"] ==
              m["exit_category_replay"] and (m["net_delta_bps"] is None or abs(m["net_delta_bps"]) <= tol.net_bps)
              for m in matched]
    n_r, n_p = len(research), len(replay)
    return {
        "strategy_id": strategy.strategy_id,
        "research_simulator": "services.quantlab.intraday.simulate",
        "research_trades": n_r, "replay_trades": n_p, "matched": len(matched),
        "entries_missed_in_replay": missed_replay, "entries_missed_in_research": missed_research,
        "research_nonfills": [{"symbol": k[0], "session": k[1].isoformat(), "entry_slot": k[2], "reason": v}
                              for k, v in sorted(research_nonfills.items())],
        "entry_delta_bps": _stat("entry_delta_bps"), "exit_delta_bps": _stat("exit_delta_bps"),
        "exit_time_delta_min": _stat("exit_time_delta_min"), "net_delta_bps": _stat("net_delta_bps"),
        "exit_reason_agreement": round(sum(agree) / len(agree), 4) if agree else None,
        "tolerance": tol.__dict__, "all_within_tolerance": bool(matched) and all(within) and not missed_replay
        and not missed_research,
        "trades": matched,
        "research_total_net_bps": round(float(sum((research[k].gross - research[k].cost) * 1e4 for k in research)), 3),
        "replay_total_net_bps": round(float(sum(float(t["net_bps"] or 0.0) for t in replay.values())), 3),
        "notes": ["replay fills at the synthesized bid/ask; research fills at the bar's trade price and charges the "
                  "half spread as cost", "replay exits land after the position manager's cancel-then-sell sequence "
                  "(seconds after the trigger)", "research applies an opening-spread multiplier in the first 15 min"],
    }

