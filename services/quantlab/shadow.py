"""Forward-only portfolio shadow journal, modelled after costs (never broker fills).

A completed-data decision must be committed before its future execution open. The existing backtester replays
only those committed decisions from cash, including drift, gaps, non-fills and costs. Every intervening session
is marked, even when monitoring missed a day; a missed decision is HOLD, never retroactively reconstructed.
Legacy gross-only rows remain readable but cannot qualify a strategy for PAPER.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from services.market_data.calendar import get_calendar

from .backtest import BacktestConfig, run_weights
from .gates import HOLDOUT_GATE, SHADOW_GATE
from .ledger import code_hash, spec_hash
from .panel import EquityPanel
from .pipeline import Lab, latest_capabilities
from .registry import State
from .targets import latest_decision

ASSET_CAPABILITY = {"us_equity": "us_equity_long", "etf": "us_equity_long", "crypto": "crypto_spot"}
MODEL_VERSION = "shadow-after-cost-v2"


def lifecycle_supported(lab: Lab, strategy: dict[str, Any]) -> tuple[bool, str]:
    """Require an implemented lifecycle and the active policy, independently of research eligibility."""
    from services.capabilities.policy import load_active_policy

    if strategy.get("asset_class") == "crypto":
        return False, "crypto broker lifecycle is not implemented; research only"
    if strategy.get("asset_class") not in ("us_equity", "etf"):
        return False, "asset lifecycle is not implemented"
    policy = load_active_policy(lab.root)
    if policy is None:
        return False, "active risk policy unreadable"
    if strategy.get("holding") == "intraday":
        return True, "intraday lifecycle"
    if strategy.get("holding") != "multi_day":
        return False, "unknown holding period"
    sid = str(strategy.get("strategy_id"))
    ok = policy.allow_overnight_positions and sid in policy.multi_day_strategy_allowlist
    return ok, ("multi-day holding allowed for this strategy" if ok else
                "allow_overnight_positions is off or the strategy is not on multi_day_strategy_allowlist")


def _family_specs(lab: Lab, panel: EquityPanel, asset_class: str) -> dict[str, Any]:
    if asset_class == "crypto":
        return {f.family: f for f in lab.crypto_families(panel)}
    return {f.family: f for f in lab.equity_families(panel)}


def _bounds(session: str, crypto: bool) -> tuple[datetime, datetime]:
    day = date.fromisoformat(session)
    if crypto:
        op = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        return op, op + timedelta(days=1)
    cal = get_calendar()
    return cal.session_open(day), cal.session_close(day)


def _targets_dict(panel: EquityPanel, row: np.ndarray) -> dict[str, float]:
    return {str(panel.symbols[j]): float(row[j]) for j in np.flatnonzero(row > 1e-9)}


def _valid_decision(row: dict[str, Any], *, code: str, config: str) -> bool:
    m = row["metadata"]
    if (m.get("model_version"), m.get("code_hash"), m.get("config_hash")) != (MODEL_VERSION, code, config):
        return False
    try:
        return datetime.fromisoformat(m["data_complete_at"]) <= datetime.fromisoformat(row["decided_at"]) \
            < datetime.fromisoformat(m["execution_open"])
    except (KeyError, TypeError, ValueError):
        return False


def update(lab: Lab, panel: EquityPanel, *, asset_class: str = "us_equity",
           now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("shadow clock must be timezone-aware")
    now = now.astimezone(UTC)
    out: dict[str, Any] = {"recorded": [], "settled": [], "promotion_checks": [], "skipped": [],
                           "note": "Prospective modelled returns after costs; these are not broker fills or realized P&L."}
    strategies = [s for s in lab.registry.in_state(State.SHADOW, State.PAPER)
                  if (s["asset_class"] == "crypto") == (asset_class == "crypto")]
    if not strategies or not len(panel.dates):
        return out
    crypto = asset_class == "crypto"
    if crypto and panel.meta.get("freq", "1D") != "1D":
        out["skipped"].append("daily shadow requires a daily crypto panel")
        return out
    # Never feed an incomplete/future bar into a prospective decision or a settlement.
    complete = [i for i, d in enumerate(panel.dates) if _bounds(str(d)[:10], crypto)[1] <= now]
    if not complete:
        out["skipped"].append("no completed session")
        return out
    end = pd.Timestamp(str(panel.dates[complete[-1]]))
    if not isinstance(end, pd.Timestamp):
        raise ValueError("invalid completed panel date")
    panel = panel.slice_dates(None, end)
    dates = [str(d)[:10] for d in panel.dates]
    indices = {day: i for i, day in enumerate(dates)}
    specs = _family_specs(lab, panel, asset_class)
    caps = latest_capabilities(lab.root).get("asset_classes", {})
    code = code_hash()
    for strategy in strategies:
        sid = strategy["strategy_id"]
        spec = specs.get(strategy["family"])
        if spec is None or spec.runner is not None:
            out["skipped"].append({"strategy_id": sid, "reason": "daily target simulator unavailable"})
            continue
        cfg = spec.config_for(strategy["spec"]) if spec.config_for else spec.config
        if cfg.execution != "open":
            out["skipped"].append({"strategy_id": sid, "reason": "shadow requires next-open execution"})
            continue
        config = spec_hash(asdict(cfg))
        raw = spec.make_targets(strategy["spec"])
        prior = lab.registry.shadow_rows(sid, 100_000)
        decisions = [r for r in prior if _valid_decision(r, code=code, config=config)]
        t = len(dates) - 1
        source = dates[t]
        source_close = _bounds(source, crypto)[1]
        if crypto:
            # Daily crypto bars finish exactly at the next open. That price has already occurred by the time
            # the bar is processed, so the earliest honest daily-open fill is the NEXT future UTC midnight.
            execute_day = now.date() + timedelta(days=1)
            fresh = date.fromisoformat(source) == now.date() - timedelta(days=1)
        else:
            execute_day = get_calendar().next_session(date.fromisoformat(source))
            fresh = True
        execution_open = _bounds(execute_day.isoformat(), crypto)[0]
        if fresh and source_close <= now < execution_open and date.fromisoformat(source) > lab.holdout[1]:
            decision_i, desired = latest_decision(raw, t)
            last = max(decisions, key=lambda r: r["session"], default=None)
            decision_source = dates[decision_i] if decision_i is not None else None
            kind = "REBALANCE" if last is None or last["metadata"].get("decision_source") != decision_source else "HOLD"
            metadata = {"model_version": MODEL_VERSION, "code_hash": code, "config_hash": config,
                        "decision_kind": kind, "decision_source": decision_source, "signal_session": source,
                        "execution_session": execute_day.isoformat(), "data_complete_at": source_close.isoformat(),
                        "execution_open": execution_open.isoformat(),
                        "latency_note": "next future UTC open (one extra bar)" if crypto else "next XNYS open"}
            if lab.registry.record_shadow(sid, source, _targets_dict(panel, desired), metadata=metadata, now=now):
                out["recorded"].append({"strategy_id": sid, "session": source, "targets": _targets_dict(panel, desired),
                                        "decision_kind": kind, "execution_session": execute_day.isoformat()})
                decisions = [r for r in lab.registry.shadow_rows(sid, 100_000)
                             if _valid_decision(r, code=code, config=config)]
        else:
            out["skipped"].append({"strategy_id": sid,
                                   "reason": "stale data, historical/holdout session, or execution open passed"})
        replay_ok = _settle(lab, panel, strategy, raw, cfg, decisions, indices, crypto, now, out)
        out["promotion_checks"].append(promotion_check(lab, strategy, caps, code=code, config=config,
                                                       replay_ok=replay_ok))
    return out


def _settle(lab: Lab, panel: EquityPanel, strategy: dict[str, Any], raw: np.ndarray, cfg: BacktestConfig,
            decisions: list[dict[str, Any]], indices: dict[str, int], crypto: bool, now: datetime,
            out: dict[str, Any]) -> bool:
    sid = strategy["strategy_id"]
    # Replay only committed decisions. Missing observation days cannot invent rebalances, exits or entries.
    committed = np.full(panel.shape, np.nan)
    replay = np.full(panel.shape, np.nan)
    executable = sorted([r for r in decisions if r["metadata"]["execution_session"] in indices],
                        key=lambda r: r["metadata"]["execution_session"])
    if not executable:
        return True
    for r in executable:
        i = indices[r["metadata"]["execution_session"]] - 1
        source = indices.get(r["session"])
        if i < 0 or source is None:
            return False  # required panel history is missing; never assume a cash position
        _, desired = latest_decision(raw, source)
        if r["metadata"]["decision_kind"] == "REBALANCE":
            committed[i] = 0.0
            for symbol, weight in r["targets"].items():
                committed[i, panel.col(symbol)] = weight
            replay[i] = desired
        elif _targets_dict(panel, desired) != r["targets"]:
            return False  # revisions changed a held target even if there was no new rebalance
    actual = run_weights(panel.f, panel.dates, committed, cfg)
    replayed = run_weights(panel.f, panel.dates, replay, cfg)
    old_marks = {r["session"]: r for r in lab.registry.shadow_marks(sid)}
    first = executable[0]["metadata"]["execution_session"]
    ok = True
    cursor = 0
    for i, day in enumerate(actual.dates):
        session = str(day)[:10]
        if session < first:
            continue
        while cursor + 1 < len(executable) and executable[cursor + 1]["metadata"]["execution_session"] <= session:
            cursor += 1
        decision = executable[cursor]
        op, cl = _bounds(session, crypto)
        provenance = {k: decision["metadata"][k] for k in ("model_version", "code_hash", "config_hash")}
        provenance.update({"decision_id": decision["id"], "execution_open": op.isoformat(),
                           "data_complete_at": cl.isoformat(), "observed_at": now.isoformat(),
                           "basis": "modelled_after_cost_not_broker_fills"})
        values = {"net": float(actual.net[i]), "gross": float(actual.gross[i]), "cost": float(actual.cost[i]),
                  "replayed": float(replayed.net[i])}
        if session in old_marks:
            old = old_marks[session]
            if any(not math.isclose(values[k], old[k], rel_tol=1e-10, abs_tol=1e-12) for k in values):
                ok = False  # append-only evidence cannot be silently rewritten by a market-data revision
            continue
        if lab.registry.record_shadow_mark(sid, session, **values, provenance=provenance):
            out["settled"].append({"strategy_id": sid, "session": session, **values})
        if session == decision["metadata"]["execution_session"]:
            lab.registry.settle_shadow(sid, decision["session"], values["net"], values["replayed"])
    return ok


def promotion_check(lab: Lab, strategy: dict[str, Any], caps: dict[str, Any], *, code: str | None = None,
                    config: str | None = None, replay_ok: bool = True) -> dict[str, Any]:
    code = code or code_hash()
    all_rows = lab.registry.shadow_marks(strategy["strategy_id"])
    rows = [r for r in all_rows if r["provenance"].get("code_hash") == code
            and (config is None or r["provenance"].get("config_hash") == config)]
    finite = all(all(math.isfinite(r[k]) for k in ("net", "replayed", "cost")) for r in rows)
    gaps = [abs(r["net"] - r["replayed"]) * 1e4 for r in rows] if finite else []
    curve = np.r_[1.0, np.cumprod([1 + r["net"] for r in rows])] if finite else np.array([1.0])
    drawdown = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    cap_key = ASSET_CAPABILITY.get(strategy["asset_class"], "")
    lifecycle, lifecycle_detail = lifecycle_supported(lab, strategy)
    latest_path = lab.out / "latest_research.json"
    latest = json.loads(latest_path.read_text()) if latest_path.is_file() else {}
    research = latest.get("families", {}).get(strategy["family"], {})
    dev_dd = research.get("wf_oos", {}).get("max_drawdown")
    dev_valid = isinstance(dev_dd, (int, float)) and math.isfinite(dev_dd) and dev_dd <= 0 \
        and research.get("selected_spec_hash") == strategy["spec_hash"]
    floor = max(HOLDOUT_GATE["max_drawdown_floor"], 1.5 * min(dev_dd, -0.01)) if dev_valid else None
    evidence = {
        "shadow_sessions": len(rows), "tracking_gap_bps": float(np.mean(gaps)) if gaps else None,
        "execution_eligible": bool(caps.get(cap_key, {}).get("execution_eligible")),
        "lifecycle_supported": lifecycle, "lifecycle_detail": lifecycle_detail,
        "prospective_return": float(curve[-1] - 1), "prospective_drawdown": drawdown,
        "drawdown_floor": floor, "code_hash": code, "config_hash": config,
        "basis": "modelled_after_cost_not_broker_fills", "replay_valid": replay_ok,
    }
    blockers = []
    if len(rows) < SHADOW_GATE["min_sessions"]:
        blockers.append(f"{len(rows)}/{int(SHADOW_GATE['min_sessions'])} provenance-checked after-cost shadow sessions")
    if not evidence["execution_eligible"]:
        blockers.append(f"{cap_key or strategy['asset_class']} not execution-eligible")
    if not lifecycle:
        blockers.append(f"{strategy['holding']} holding not enabled: {lifecycle_detail}")
    gap = evidence["tracking_gap_bps"]
    if not finite or gap is None or not math.isfinite(gap) or gap > SHADOW_GATE["max_tracking_gap_bps"]:
        blockers.append("shadow tracking not yet within tolerance")
    if not replay_ok or len(rows) != len(all_rows):
        blockers.append("prospective evidence changed on replay or uses a different code/cost model")
    if floor is None:
        blockers.append("matching development drawdown evidence unavailable")
    elif drawdown < floor:
        blockers.append("prospective drawdown breach")
    ready = not blockers and strategy["state"] == State.SHADOW
    if ready:
        lab.registry.transition(strategy["strategy_id"], State.PAPER, reason="after-cost shadow evidence and execution path OK",
                                evidence=evidence)
    elif strategy["state"] == State.PAPER and blockers:
        if floor is not None and drawdown < floor:
            suspend_on_breach(lab, strategy["strategy_id"], drawdown, float(dev_dd))
        else:
            lab.registry.transition(strategy["strategy_id"], State.SUSPENDED,
                                    reason="prospective evidence or execution eligibility degraded", evidence=evidence)
    return {"strategy_id": strategy["strategy_id"], "state": str(lab.registry.state(strategy["strategy_id"])),
            "promoted": ready, "blockers": blockers, "evidence": evidence}


def suspend_on_breach(lab: Lab, strategy_id: str, prospective_drawdown: float, dev_drawdown: float) -> bool:
    """Reversible demotion on non-finite risk evidence or a breach of the development/absolute drawdown floor."""
    if lab.registry.state(strategy_id) != State.PAPER:
        return False
    valid = math.isfinite(prospective_drawdown) and math.isfinite(dev_drawdown)
    floor = max(HOLDOUT_GATE["max_drawdown_floor"], 1.5 * min(dev_drawdown, -0.01)) if valid else 0.0
    if not valid or prospective_drawdown < floor:
        lab.registry.transition(strategy_id, State.SUSPENDED, reason="prospective drawdown breach",
                                evidence={"prospective_drawdown": prospective_drawdown, "dev_drawdown": dev_drawdown})
        return True
    return False
