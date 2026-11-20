"""``dispatch``: validate input -> call the runtime facade -> validate output. The only tool executor.

Used in-process by the API route ``POST /api/v1/agent/tools/{name}`` (which the MCP server reaches over
HTTP) and by in-app AI runtimes through ``InProcessToolBackend``. Guarantees:

* unknown tool -> ``ToolError(404)``; invalid input (unknown fields, floats in money, bad symbols, injection-
  shaped text) -> ``ToolError(422)``; outside PAPER/REPLAY or with a LIVE execution label -> ``ToolError(403)``;
* outputs are validated against the strict output models (``ToolError(502)`` if the runtime returned
  something that violates the contract — fail closed, nothing unvalidated reaches a model);
* the four mutating tools reach the runtime only through ``submit_agent_proposal`` / ``halt`` / ``cancel`` /
  ``run_cycle`` (risk engine -> signed intent -> final gate -> paper gateway). Nothing is retried: a failed
  mutating call reports its outcome as UNKNOWN and tells the agent to inspect orders first;
* ``propose_paper_trade`` never trusts agent arithmetic: quantity and limit are requests, the limit is rounded
  to a passive tick, ``max_planned_loss`` is computed here deterministically (conservatively), ``data_asof``
  comes from a fresh snapshot, and every cited evidence id must have been handed to this agent session by a
  tool output (or be a persisted signal id).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, cast

from pydantic import BaseModel, ValidationError

from packages.contracts import MarketSnapshot, PredictionSet, QuantSignal, RiskDecision, TradeProposal
from packages.contracts.enums import (
    TERMINAL_ORDER_STATES,
    DecisionAction,
    ExecutionLabel,
    OrderType,
    Proposer,
    Side,
    TradingMode,
)
from packages.core.ids import new_causal_id, new_id
from packages.core.logging import log_event
from packages.core.money import BPS, round_price
from services.sessions.errors import SessionError

from . import schemas as s
from .protocol import ToolRuntime
from .registry import TOOL_NAMES, get_spec
from .sanitize import sanitize_actor, truncate
from .session import IssuedId, state_for

log = logging.getLogger("lat.tools")

AGENT_MODES: Final[frozenset[TradingMode]] = frozenset({TradingMode.PAPER, TradingMode.REPLAY})
_TERMINAL: Final[frozenset[str]] = frozenset(st.value for st in TERMINAL_ORDER_STATES)
_PROPOSAL_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"SUBMITTED", "REJECTED", "BLOCKED", "AWAITING_HUMAN_APPROVAL", "RATE_LIMITED", "NO_TRADE", "ERROR"}
)


class ToolError(Exception):
    """A refused or failed tool call. ``status`` is an HTTP status; ``detail`` is JSON-serialisable."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail

    def message(self) -> str:
        if isinstance(self.detail, str):
            return self.detail
        if isinstance(self.detail, list):
            parts = []
            for item in cast(list[Any], self.detail)[:8]:
                if isinstance(item, dict):
                    d = cast(dict[str, Any], item)
                    loc = ".".join(str(p) for p in d.get("loc", ()))
                    parts.append(f"{loc}: {d.get('msg', '')}" if loc else str(d.get("msg", "")))
                else:
                    parts.append(str(item))
            return "; ".join(parts)
        return str(self.detail)


Handler = Callable[[ToolRuntime, Any, str], Awaitable[dict[str, Any]]]


# ------------------------------------------------------------------------------------------------ dispatch
async def dispatch(runtime: ToolRuntime, name: str, args: Any, *, actor: str) -> dict[str, Any]:
    spec = get_spec(name)
    if spec is None:
        raise ToolError(404, f"unknown tool {truncate(name, 64)!r}; available: {', '.join(TOOL_NAMES)}")
    who = sanitize_actor(actor)
    _require_agent_mode(runtime)
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ToolError(422, "tool arguments must be a JSON object")
    try:
        inp = spec.input_model.model_validate(dict(cast(Mapping[str, Any], args)))
    except ValidationError as exc:
        raise ToolError(422, _validation_detail(exc)) from None
    handler = _HANDLERS[name]
    try:
        raw = await handler(runtime, inp, who)
    except ToolError as exc:
        log_event(log, "agent tool refused", logging.INFO, tool=name, actor=who, status=exc.status)
        raise
    except Exception as exc:
        log_event(log, "agent tool failed", logging.WARNING, tool=name, actor=who, error=type(exc).__name__)
        if spec.mutating:
            raise ToolError(
                500, f"{name} failed ({type(exc).__name__}); its outcome is UNKNOWN. Inspect get_orders/get_risk_status"
                     " before trying again; do not retry blindly.",
            ) from None
        raise ToolError(503, f"{name} is temporarily unavailable ({type(exc).__name__})") from None
    try:
        out = spec.output_model.model_validate(raw)
    except ValidationError as exc:
        log_event(log, "agent tool output invalid", logging.ERROR, tool=name, actor=who, errors=exc.error_count())
        raise ToolError(502, f"{name} output failed contract validation and was withheld (fail closed)") from None
    _register_issued_ids(runtime, who, out)
    log_event(log, "agent tool call", logging.INFO, tool=name, actor=who, mutating=spec.mutating,
              outcome=getattr(out, "outcome", "OK"))
    return out.model_dump(mode="json")


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    # Never echo the rejected input back (it may be large or adversarial); location + message only.
    return [{"loc": list(e.get("loc", ())), "msg": truncate(e.get("msg", ""), 240), "type": e.get("type")}
            for e in exc.errors(include_url=False, include_input=False, include_context=False)[:20]]


def _require_agent_mode(runtime: ToolRuntime) -> None:
    mode = runtime.settings.mode
    if runtime.execution_label == ExecutionLabel.LIVE or mode not in AGENT_MODES:
        raise ToolError(403, "the agent tool interface is available only in PAPER/REPLAY mode with a non-live gateway")


def _base(runtime: ToolRuntime, data_simulated: bool = False) -> dict[str, Any]:
    label = runtime.execution_label
    return {"execution_label": label, "is_simulated": bool(data_simulated) or label == ExecutionLabel.SIMULATED,
            "asof": runtime.now()}


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)


def _str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list | tuple) else []


# ------------------------------------------------------------------------------------------------ market
async def _get_market_snapshot(rt: ToolRuntime, inp: s.GetMarketSnapshotIn, actor: str) -> dict[str, Any]:
    snap = _as_snapshot(await rt.market_snapshot(inp.symbol))
    return _snapshot_out(rt, snap)


def _as_snapshot(obj: Any) -> MarketSnapshot:
    return obj if isinstance(obj, MarketSnapshot) else MarketSnapshot.model_validate(obj)


def _snapshot_out(rt: ToolRuntime, snap: MarketSnapshot) -> dict[str, Any]:
    q = snap.quote
    quote = None
    if q is not None:
        quote = {"bid": q.bid, "ask": q.ask, "bid_size": q.bid_size, "ask_size": q.ask_size, "mid": q.mid,
                 "spread_bps": q.spread_bps.quantize(Decimal("0.01")), "exchange_ts": q.exchange_ts,
                 "available_at_ts": q.available_at_ts}
    b = snap.last_bar
    bar = None if b is None else {"t": b.start_ts, "o": b.open, "h": b.high, "l": b.low, "c": b.close, "v": b.volume}
    return {
        **_base(rt, snap.is_simulated), "snapshot_id": snap.snapshot_id, "symbol": snap.symbol,
        "feed": snap.feed.value, "feed_mode": snap.feed_mode.value, "session": snap.session.value,
        "data_asof": snap.data_asof, "dataset_version": snap.dataset_version, "source_id": snap.source_id,
        "quote": quote, "last_trade_price": snap.last_trade_price, "last_bar": bar,
        "limitations": [truncate(x, 200) for x in snap.limitations][:8],
    }


async def _get_bars(rt: ToolRuntime, inp: s.GetBarsIn, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.market_bars, inp.symbol, inp.timeframe, inp.limit))
    rows = _list(data.get("bars"))[-inp.limit:]
    bars: list[dict[str, Any]] = []
    last_avail: Any = None
    for r in rows:
        d = _dict(r)
        bars.append({"t": datetime.fromtimestamp(int(d["time"]), UTC), "o": d["open"], "h": d["high"], "l": d["low"],
                     "c": d["close"], "v": d["volume"]})
        last_avail = d.get("available_at") or last_avail
    return {
        **_base(rt, bool(data.get("is_simulated"))), "symbol": inp.symbol, "timeframe": inp.timeframe,
        "feed": str(data.get("feed") or ""), "feed_mode": str(data.get("feed_mode") or ""),
        "dataset_version": _str(data.get("dataset_version")), "source_id": _str(data.get("source_id")),
        "last_available_at": last_avail,
        "limitations": [truncate(x, 200) for x in _list(data.get("limitations"))][:8], "bars": bars,
    }


async def _get_feed_quality(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.market_quality))
    symbols = [
        {"symbol": d.get("symbol"), "last_quote_ts": d.get("last_quote_ts"), "staleness_s": d.get("staleness_s"),
         "spread_bps": d.get("spread_bps"), "gaps_today": int(d.get("gaps_today") or 0),
         "rejected_today": int(d.get("rejected_today") or 0)}
        for d in map(_dict, _list(data.get("symbols"))[:50])
    ]
    events = [
        {"ts": d.get("ts"), "symbol": _str(d.get("symbol")), "kind": _str(d.get("kind")), "severity": _str(d.get("severity"))}
        for d in map(_dict, _list(data.get("events"))[:10])
    ]
    return {
        **_base(rt, bool(data.get("is_simulated"))), "feed_mode": str(data.get("feed_mode") or ""),
        "provider": _str(data.get("provider")), "feed": _str(data.get("feed")),
        "status": str(data.get("status") or "UNKNOWN"), "detail": truncate(data.get("detail"), 240),
        "staleness_s": data.get("staleness_s"), "max_staleness_s": data.get("max_staleness_s"),
        "last_bar_ts": data.get("last_bar_ts"), "last_quote_ts": data.get("last_quote_ts"),
        "limitations": [truncate(x, 200) for x in _list(data.get("limitations"))][:8],
        "symbols": symbols, "recent_events": events,
    }


# ------------------------------------------------------------------------------------------------ models
async def _get_predictions(rt: ToolRuntime, inp: s.GetPredictionsIn, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.predictions, inp.symbol))
    preds: list[dict[str, Any]] = []
    for raw in _list(data.get("predictions"))[:25]:
        try:
            p = raw if isinstance(raw, PredictionSet) else PredictionSet.model_validate(raw)
        except ValidationError:
            continue
        preds.append({
            "prediction_id": p.prediction_id, "symbol": p.symbol, "asof": p.asof, "features_asof": p.features_asof,
            "forecasts": [{"horizon_minutes": f.horizon_minutes, "expected_return_bps": f.expected_return_bps,
                           "model_id": f.model_id} for f in p.return_forecasts[:6]],
            "success_probability": p.success_probability, "success_horizon_minutes": p.success_horizon_minutes,
            "probability_calibrated": p.probability_calibrated, "calibration_method": p.calibration_method,
            "rank": p.rank, "universe_size": p.universe_size, "vol_forecast_bps": p.vol_forecast_bps,
            "regime": p.regime.value, "est_cost_bps": p.est_cost_bps, "is_simulated": p.is_simulated,
        })
    return {
        **_base(rt, bool(data.get("is_simulated"))), "registry_version": _str(data.get("registry_version")),
        "detail": truncate(data.get("detail"), 240), "predictions": preds,
    }


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value  # research metrics (not money) may be floats
    if isinstance(value, Decimal):
        return format(value, "f")
    return truncate(value, 120)


def _scalars(d: Mapping[str, Any], limit: int = 16) -> dict[str, Any]:
    return {str(k): _scalar(v) for k, v in list(d.items())[:limit] if not isinstance(v, dict | list)}


async def _get_model_report(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.model_report))
    targets: list[dict[str, Any]] = []
    for t in map(_dict, _list(data.get("targets"))[:12]):
        cands = [_dict(c) for c in _list(t.get("candidates"))]
        sel_name = _str(t.get("selected_model"))
        sel = next((c for c in cands if c.get("selected") or (sel_name and c.get("model_id") == sel_name)), None)
        selected = None
        if sel is not None:
            metrics = {k: v for k, v in sel.items() if k not in ("model_id", "selected", "rejected_reason")}
            selected = {"model_id": str(sel.get("model_id")), "metrics": _scalars(metrics)}
        rejected = [{"model_id": str(c.get("model_id")), "reason": truncate(c.get("rejected_reason"), 160)}
                    for c in cands if c.get("rejected_reason")][:3]
        calib = t.get("calibration")
        targets.append({
            "target": str(t.get("target")), "selected_model": sel_name, "selected": selected, "n_candidates": len(cands),
            "rejected": rejected, "calibration": _scalars(_dict(calib)) if isinstance(calib, dict) else None,
        })
    wf = data.get("walk_forward")
    walk_forward = None
    if isinstance(wf, dict):
        walk_forward = {str(k): (_scalars(_dict(v)) if isinstance(v, dict) else _scalar(v))
                        for k, v in list(cast(dict[str, Any], wf).items())[:12] if not isinstance(v, list)}
    mt = data.get("multiple_testing")
    multiple = None
    if isinstance(mt, dict):
        mtd = cast(dict[str, Any], mt)
        survivors = [str(_dict(x).get("model_id", x)) if isinstance(x, dict) else str(x)
                     for x in _list(mtd.get("survivors"))][:10]
        multiple = {**_scalars(mtd), "survivors": survivors}
    return {
        **_base(rt, bool(data.get("is_simulated"))), "registry_version": _str(data.get("registry_version")),
        "trained_at": data.get("trained_at"), "dataset_version": _str(data.get("dataset_version")),
        "selection_metric": _str(data.get("selection_metric")), "targets": targets, "walk_forward": walk_forward,
        "multiple_testing": multiple, "notes": [cast(str, truncate(n, 200)) for n in _list(data.get("notes"))][:8],
    }


async def _get_signals(rt: ToolRuntime, inp: s.GetSignalsIn, actor: str) -> dict[str, Any]:
    filtered = inp.symbol is not None or inp.candidates_only
    data = _dict(await _call(rt.signals, 200 if filtered else inp.limit))
    out: list[dict[str, Any]] = []
    skipped = 0
    any_sim = False
    for raw in _list(data.get("signals")):
        try:
            sig = raw if isinstance(raw, QuantSignal) else QuantSignal.model_validate(raw)
        except ValidationError:
            skipped += 1
            continue
        if (inp.symbol and sig.symbol != inp.symbol) or (inp.candidates_only and not sig.is_candidate):
            continue
        any_sim = any_sim or sig.is_simulated
        out.append({
            "signal_id": sig.signal_id, "symbol": sig.symbol, "strategy_id": sig.strategy_id,
            "model_version": sig.model_version, "side": sig.side.value, "horizon_minutes": sig.horizon_minutes,
            "score": sig.score, "rank": sig.rank, "expected_return_bps": sig.expected_return_bps,
            "est_total_cost_bps": sig.est_total_cost_bps, "net_edge_bps": sig.net_edge_bps,
            "probability_up": sig.probability_up, "probability_calibrated": sig.probability_calibrated,
            "is_candidate": sig.is_candidate, "reason": truncate(sig.reason, 200), "features_asof": sig.features_asof,
            "generated_at": sig.generated_at, "is_simulated": sig.is_simulated,
        })
        if len(out) >= inp.limit:
            break
    lc = data.get("latest_cycle")
    latest = None
    if isinstance(lc, dict):
        d = cast(dict[str, Any], lc)
        latest = {"causal_id": _str(d.get("causal_id")), "ran_at": d.get("ran_at"),
                  "candidate_symbol": _str(d.get("candidate_symbol")), "outcome": _str(d.get("outcome")),
                  "reason": truncate(d.get("reason"), 240)}
    return {**_base(rt, any_sim), "signals": out, "latest_cycle": latest, "skipped_invalid": skipped}


# ------------------------------------------------------------------------------------------------ portfolio
async def _get_portfolio(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.portfolio))
    acct = data.get("account")
    account = None
    if isinstance(acct, dict):
        a = cast(dict[str, Any], acct)
        account = {k: a.get(k) for k in ("status", "is_paper", "equity", "cash", "buying_power", "trading_blocked",
                                         "account_blocked", "asof", "source")}
    positions = [
        {k: p.get(k) for k in ("symbol", "quantity", "avg_entry_price", "market_value", "unrealized_pl", "asof", "source")}
        for p in map(_dict, _list(data.get("broker_positions"))[:50])
    ]
    ledger = [{k: p.get(k) for k in ("symbol", "quantity", "avg_cost", "realized_pl")}
              for p in map(_dict, _list(data.get("ledger_positions"))[:50])]
    pnl = _dict(data.get("pnl"))
    return {
        **_base(rt), "account": account, "broker_positions": positions, "ledger_positions": ledger,
        "pnl": {"daily_realized": pnl.get("daily_realized", "0"), "daily_unrealized": pnl.get("daily_unrealized", "0"),
                "start_of_day_equity": pnl.get("start_of_day_equity")},
    }


_ORDER_KEYS: Final = ("client_order_id", "symbol", "side", "quantity", "filled_quantity", "avg_fill_price", "limit_price",
                      "state", "gateway", "execution_label", "created_at", "updated_at")


def _order_out(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k) for k in _ORDER_KEYS}
    for k in ("side", "state", "gateway", "execution_label"):
        out[k] = _str(out[k])
    return out


async def _get_orders(rt: ToolRuntime, inp: s.GetOrdersIn, actor: str) -> dict[str, Any]:
    data = _dict(await _call(rt.orders, 200 if inp.open_only else inp.limit))
    rows = [_dict(r) for r in _list(data.get("orders"))]
    if inp.open_only:
        rows = [r for r in rows if _str(r.get("state")) not in _TERMINAL]
    return {**_base(rt), "orders": [_order_out(r) for r in rows[: inp.limit]]}


async def _get_order(rt: ToolRuntime, inp: s.GetOrderIn, actor: str) -> dict[str, Any]:
    detail = await _call(rt.order_detail, inp.client_order_id)
    if not isinstance(detail, dict):
        raise ToolError(404, "unknown client_order_id")
    d = cast(dict[str, Any], detail)
    intent = d.get("intent")
    brief = None
    if isinstance(intent, dict):
        i = cast(dict[str, Any], intent)  # signature/key material deliberately omitted
        brief = {k: i.get(k) for k in ("intent_id", "proposal_id", "policy_version", "expires_at",
                                       "requires_human_approval")}
    transitions = [{"from_state": _str(t.get("from_state")), "to_state": _str(t.get("to_state")), "at": t.get("at"),
                    "reason": truncate(t.get("reason"), 200)} for t in map(_dict, _list(d.get("transitions")))][-20:]
    fills = [{k: f.get(k) for k in ("fill_id", "quantity", "price", "ts")} for f in map(_dict, _list(d.get("fills")))][:50]
    return {**_base(rt), "order": _order_out(_dict(d.get("order"))), "intent": brief, "transitions": transitions,
            "fills": fills, "n_events": len(_list(d.get("events")))}


async def _get_risk_status(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    status = _dict(await _call(rt.status))
    view = _dict(await _call(rt.risk_view))
    pol = rt.policy
    limits = {
        "long_only": pol.long_only, "regular_session_only": pol.regular_session_only, "max_positions": pol.max_positions,
        "max_pending_orders": pol.max_pending_orders, "max_orders_per_minute": pol.max_orders_per_minute,
        "max_orders_per_day": pol.max_orders_per_day, "max_order_notional": pol.max_order_notional,
        "max_position_notional": pol.max_position_notional, "max_concentration_pct": pol.max_concentration_pct,
        "max_daily_loss": pol.max_daily_loss, "max_planned_loss_per_trade": pol.max_planned_loss_per_trade,
        "max_spread_bps": pol.max_spread_bps, "max_quote_age_seconds": pol.max_quote_age_seconds,
        "max_limit_deviation_bps": pol.max_limit_deviation_bps, "default_stop_loss_bps": pol.default_stop_loss_bps,
        "min_price": pol.min_price, "min_account_equity": pol.min_account_equity,
        "symbol_allowlist": list(pol.symbol_allowlist)[:50],
    }
    halts = [{"halt_id": str(h.get("halt_id")), "reason": truncate(h.get("reason"), 200) or "",
              "source": str(h.get("source") or ""), "created_at": h.get("created_at")}
             for h in map(_dict, _list(status.get("halts")))][:10]
    util = [{k: u.get(k) for k in ("name", "used", "limit", "pct")} for u in map(_dict, _list(view.get("utilization")))]
    decisions = []
    for raw in _list(view.get("recent_decisions"))[:5]:
        d = _dict(raw)
        decisions.append({"decision_id": str(d.get("decision_id")), "proposal_id": str(d.get("proposal_id")),
                          "verdict": str(d.get("verdict")),
                          "reasons": [cast(str, truncate(r, 160)) for r in _list(d.get("reasons"))][:4],
                          "evaluated_at": d.get("evaluated_at")})
    locked = status.get("entries_locked")
    return {
        **_base(rt), "mode": rt.settings.mode.value, "live_execution_compiled": bool(status.get("live_execution_compiled")),
        "policy_version": pol.version, "entries_locked": True if locked is None else bool(locked),
        "lock_reasons": [cast(str, truncate(r, 200)) for r in _list(status.get("lock_reasons"))][:10],
        "halts": halts, "limits": limits, "utilization": util, "recent_decisions": decisions,
        "agent_max_proposals_per_hour": int(rt.settings.agent_max_proposals_per_hour),
    }


# ------------------------------------------------------------------------------------------------ paper trading
async def _propose_paper_trade(rt: ToolRuntime, inp: s.ProposePaperTradeIn, actor: str) -> dict[str, Any]:
    side = Side(inp.side)
    cited = list(dict.fromkeys(inp.evidence_ids))
    issued = await _resolve_evidence(rt, actor, cited, inp.symbol)

    buy = side == Side.BUY
    qty: Decimal = inp.quantity
    limit = round_price(inp.limit_price, side_buy=buy)
    if limit <= 0:
        raise ToolError(422, "limit_price rounds to zero")
    stop: Decimal | None = None
    if not buy and inp.stop_price is not None:
        raise ToolError(422, "stop_price applies only to BUY proposals (a SELL reduces an existing long)")
    if buy:
        if inp.stop_price is not None:
            stop = round_price(inp.stop_price, side_buy=False)
            if not (Decimal(0) < stop < limit):
                raise ToolError(422, f"stop_price {stop} must be positive and below the BUY limit {limit}")
        default_stop = round_price(limit * (1 - rt.policy.default_stop_loss_bps / BPS), side_buy=True)
        worst_stop = min(stop, default_stop) if stop is not None else default_stop
        # Deterministic and conservative (never understates): the risk engine recomputes it independently anyway.
        max_loss = qty * (limit - worst_stop)
    else:
        max_loss = Decimal(0)

    snap = _as_snapshot(await rt.market_snapshot(inp.symbol))  # fresh point-in-time snapshot -> data_asof
    signal_id = next((eid for eid in cited if issued[eid].kind == "signal" and issued[eid].symbol == inp.symbol), None)
    rationale = f"[agent:{actor}] thesis: {inp.thesis} | counterarguments: " + " ; ".join(inp.counterarguments)
    proposal = TradeProposal(
        proposal_id=new_id("prop"), causal_id=new_causal_id(), created_at=rt.now(), data_asof=snap.data_asof,
        action=DecisionAction.TRADE, proposer=Proposer.AGENT_TOOL, strategy_id=f"agent:{actor}", symbol=inp.symbol,
        side=side, quantity=qty, order_type=OrderType.LIMIT, limit_price=limit, stop_price=stop,
        max_planned_loss=max_loss, evidence_ids=tuple(dict.fromkeys([snap.snapshot_id, *cited])), signal_id=signal_id,
        policy_version_seen=rt.policy.version, dataset_version=snap.dataset_version, rationale=rationale[:4000],
    )
    result = _dict(await rt.submit_agent_proposal(proposal, actor))
    outcome = str(result.get("outcome") or "UNKNOWN")
    if outcome == "REFUSED":
        raise ToolError(403, truncate(result.get("reason"), 300) or "agent proposals refused")
    decision = None
    raw_dec = result.get("decision")
    if raw_dec is not None:
        dec = raw_dec if isinstance(raw_dec, RiskDecision) else RiskDecision.model_validate(raw_dec)
        decision = {
            "decision_id": dec.decision_id, "verdict": dec.verdict.value,
            "failed_checks": [c.name for c in dec.checks if not c.passed][:20],
            "reasons": [cast(str, truncate(r, 200)) for r in dec.reasons][:8],
            "recomputed_quantity": dec.recomputed_quantity, "recomputed_notional": dec.recomputed_notional,
            "recomputed_max_loss": dec.recomputed_max_loss, "reference_price": dec.reference_price,
            "policy_version": dec.policy_version,
        }
    order = None
    raw_order = result.get("order")
    if isinstance(raw_order, dict):
        o = cast(dict[str, Any], raw_order)
        order = {"client_order_id": str(o.get("client_order_id")), "state": _str(o.get("state")) or "UNKNOWN",
                 "filled_quantity": o.get("filled_quantity"), "avg_fill_price": o.get("avg_fill_price")}
    return {
        **_base(rt, snap.is_simulated), "outcome": outcome if outcome in _PROPOSAL_OUTCOMES else "UNKNOWN",
        "reason": truncate(result.get("reason"), 400), "proposal_id": proposal.proposal_id,
        "causal_id": proposal.causal_id, "data_asof": proposal.data_asof,
        "requested": {"symbol": inp.symbol, "side": side.value, "quantity": qty, "limit_price": limit, "stop_price": stop},
        "decision": decision, "client_order_id": _str(result.get("client_order_id")), "order": order,
    }


async def _resolve_evidence(rt: ToolRuntime, actor: str, cited: list[str], symbol: str) -> dict[str, IssuedId]:
    ids = state_for(rt).ids_for(actor)
    resolved: dict[str, IssuedId] = {}
    missing: list[str] = []
    for eid in cited:
        rec = ids.get(eid)
        if rec is None:
            missing.append(eid)
        else:
            resolved[eid] = rec
    if missing:  # persisted quant signals are citable even if this session did not list them
        data = _dict(await _call(rt.signals, 500))
        known = {str(_dict(x).get("signal_id")): _str(_dict(x).get("symbol")) for x in _list(data.get("signals"))}
        for eid in list(missing):
            if eid in known:
                resolved[eid] = IssuedId("signal", known[eid], 0.0)
                missing.remove(eid)
    if missing:
        raise ToolError(422, "evidence_ids not obtained from a tool output in this agent session: "
                             + ", ".join(missing[:6]) + ". Call get_signals / get_predictions / get_market_snapshot "
                             "first and cite the ids they return.")
    if not any(r.symbol == symbol for r in resolved.values()):
        raise ToolError(422, f"at least one evidence id must belong to {symbol} "
                             "(e.g. its snapshot_id, signal_id or prediction_id)")
    return resolved


async def _request_halt(rt: ToolRuntime, inp: s.RequestHaltIn, actor: str) -> dict[str, Any]:
    halt_id = await _call(rt.halt, inp.reason, f"agent:{actor}")
    return {**_base(rt), "halt_id": str(halt_id), "entries_locked": True,
            "detail": "new entries are stopped; only the human operator can resume after a CLEAN reconciliation"}


async def _cancel_paper_order(rt: ToolRuntime, inp: s.CancelPaperOrderIn, actor: str) -> dict[str, Any]:
    detail = await _call(rt.order_detail, inp.client_order_id)
    if not isinstance(detail, dict):
        raise ToolError(404, "unknown client_order_id")
    order = _dict(cast(dict[str, Any], detail).get("order"))
    if _str(order.get("execution_label")) == ExecutionLabel.LIVE.value:
        raise ToolError(403, "live orders are not accessible through the agent tool interface")
    state = _str(order.get("state"))
    if state in _TERMINAL:
        return {**_base(rt), "client_order_id": inp.client_order_id, "outcome": "ALREADY_TERMINAL", "state": state}
    res = _dict(await rt.cancel(inp.client_order_id))
    return {**_base(rt), "client_order_id": inp.client_order_id, "outcome": "CANCEL_REQUESTED",
            "state": _str(res.get("state")) or state}


async def _run_quant_cycle(rt: ToolRuntime, inp: s.RunQuantCycleIn, actor: str) -> dict[str, Any]:
    universe = tuple(rt.settings.symbols)
    symbols = list(dict.fromkeys(inp.symbols)) if inp.symbols else None
    if symbols is not None:
        unknown = [x for x in symbols if x not in universe]
        if unknown:
            raise ToolError(422, f"symbols outside the configured universe {list(universe)}: {unknown}")
    empty: dict[str, Any] = {"causal_id": None, "reason": None, "candidate_symbol": None, "signals": None,
                             "client_order_id": None, "entries_locked": None, "lock_reasons": [], "ran_at": None}
    limit = int(rt.settings.agent_max_proposals_per_hour)
    if not state_for(rt).cycle_limiter.try_acquire(limit):
        return {**_base(rt), **empty, "outcome": "RATE_LIMITED",
                "reason": f"agent-initiated quant cycles are limited to {limit}/hour"}
    # use_ai=False: the quant engine never depends on AI, and an agent must not trigger nested paid model calls.
    res = _dict(await rt.run_cycle(symbols=symbols, use_ai=False, actor=f"agent:{actor}"))
    return {
        **_base(rt), "outcome": str(res.get("outcome") or "UNKNOWN"), "causal_id": _str(res.get("causal_id")),
        "reason": truncate(res.get("reason"), 400), "candidate_symbol": _str(res.get("candidate_symbol")),
        "signals": res.get("signals") if isinstance(res.get("signals"), int) else None,
        "client_order_id": _str(res.get("client_order_id")),
        "entries_locked": res.get("entries_locked") if isinstance(res.get("entries_locked"), bool) else None,
        "lock_reasons": [cast(str, truncate(r, 200)) for r in _list(res.get("lock_reasons"))][:10],
        "ran_at": res.get("ran_at"),
    }


# ------------------------------------------------------------------------------------------------ sessions & ops
_SESSION_KEYS: Final = tuple(s.SessionOut.model_fields)
_SCHEDULE_KEYS: Final = tuple(s.ScheduleOut.model_fields)
_RUN_KEYS: Final = tuple(s.ScheduleRunOut.model_fields)
_SESSION_DETAIL: Final = (
    "The session runs inside the local trading service and keeps running if this chat ends or hits a usage limit."
    " Decision cycles run only during XNYS regular hours and every order still needs independent risk-engine"
    " approval (PAPER only). Do not poll: check back later with get_session_report or explain_decisions."
)


def _manager(rt: ToolRuntime) -> Any:
    mgr = rt.sessions
    if mgr is None:
        raise ToolError(503, "the session manager is not running in the local API; restart it (make dev)")
    return mgr


async def _session_call(name: str, fn: Callable[..., Any], *args: Any, mutating: bool = False, **kwargs: Any) -> Any:
    try:
        return await _call(fn, *args, **kwargs)
    except SessionError as exc:
        raise ToolError(exc.status, truncate(exc.message, 400) or "refused") from None
    except Exception as exc:
        log_event(log, "session tool failed", logging.WARNING, tool=name, error=type(exc).__name__)
        if mutating:
            raise ToolError(500, f"{name} failed ({type(exc).__name__}); its outcome is UNKNOWN. Check"
                                 " get_session_status once before trying again; do not retry blindly.") from None
        raise ToolError(503, f"{name} is temporarily unavailable ({type(exc).__name__})") from None


def _session_out(d: Any) -> dict[str, Any] | None:
    if not isinstance(d, dict):
        return None
    row = cast(dict[str, Any], d)
    out = {k: row.get(k) for k in _SESSION_KEYS}
    out["stop_reason"] = truncate(out.get("stop_reason"), 300)
    out["symbols"] = [str(x) for x in _list(row.get("symbols"))][:10]
    return out


def _schedule_out(d: Any) -> dict[str, Any] | None:
    if not isinstance(d, dict):
        return None
    row = cast(dict[str, Any], d)
    out = {k: row.get(k) for k in _SCHEDULE_KEYS}
    last = row.get("last_run")
    out["last_run"] = {k: _dict(last).get(k) for k in _RUN_KEYS} if isinstance(last, dict) else None
    return out


async def _start_paper_session(rt: ToolRuntime, inp: s.StartPaperSessionIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    session = await _session_call("start_paper_session", mgr.start, inp.duration_minutes,
                                  symbols=list(inp.symbols) if inp.symbols else None,
                                  cadence_minutes=inp.cadence_minutes, use_ai=inp.use_ai,
                                  requested_by=f"agent:{actor}", mutating=True)
    return {**_base(rt), "outcome": "STARTED", "session": _session_out(session), "detail": _SESSION_DETAIL}


_STOP_DETAIL: Final[dict[str, str]] = {
    "STOPPED": "The session is stopped; no further decision cycles. Open orders and positions are unchanged.",
    "STOPPING": "Stop accepted: the in-flight decision cycle finishes first (never cancelled mid-order), then the"
                " session stops. Check get_session_status once later if needed.",
    "ALREADY_ENDED": "That session had already ended.",
    "NO_ACTIVE_SESSION": "No paper session is running.",
}


async def _stop_session(rt: ToolRuntime, inp: s.StopSessionIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    res = _dict(await _session_call("stop_session", mgr.stop, inp.session_id, reason=inp.reason,
                                    requested_by=f"agent:{actor}", mutating=True))
    outcome = str(res.get("outcome") or "")
    if outcome not in _STOP_DETAIL:
        raise ToolError(500, "stop_session returned an unexpected outcome; check get_session_status once")
    return {**_base(rt), "outcome": outcome, "session": _session_out(res.get("session")), "detail": _STOP_DETAIL[outcome]}


async def _get_session_status(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    active = await _session_call("get_session_status", mgr.active)
    recent = [x for x in _list(await _session_call("get_session_status", mgr.list, 4))
              if not (isinstance(active, dict) and _dict(x).get("session_id") == _dict(active).get("session_id"))][:3]
    schedules = [x for x in _list(await _session_call("get_session_status", mgr.list_schedules)) if _dict(x).get("enabled")]
    sched = _dict(await _session_call("get_session_status", mgr.scheduler_status))
    status = _dict(await _call(rt.status))
    locked = status.get("entries_locked")
    running = sched.get("loop_running")
    if active is not None:
        detail = "A paper session is running in the local service; it continues without this chat."
    else:
        detail = "No paper session is running."
    if running is False:
        detail += " Warning: the session scheduler loop is not running in the API process."
    return {
        **_base(rt), "active": _session_out(active), "recent": [_session_out(x) for x in recent],
        "schedules": [_schedule_out(x) for x in schedules][:10], "entries_locked": True if locked is None else bool(locked),
        "lock_reasons": [cast(str, truncate(r, 200)) for r in _list(status.get("lock_reasons"))][:10],
        "market_open": sched.get("market_open") if isinstance(sched.get("market_open"), bool) else None,
        "next_open": sched.get("next_open"), "scheduler_running": running if isinstance(running, bool) else None,
        "detail": detail,
    }


async def _schedule_session(rt: ToolRuntime, inp: s.ScheduleSessionIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    sch = await _session_call("schedule_session", mgr.add_schedule, list(inp.weekdays), inp.start_time_et,
                              inp.duration_minutes, list(inp.symbols) if inp.symbols else None, inp.cadence_minutes,
                              inp.use_ai, f"agent:{actor}", mutating=True)
    return {**_base(rt), "outcome": "SCHEDULED", "schedule": _schedule_out(sch),
            "detail": "Scheduled sessions are started by the local service (no chat needed), at most once per day, only"
                      " inside XNYS regular hours; a missed window is recorded MISSED and never run late."}


async def _list_schedules(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    rows = _list(await _session_call("list_schedules", mgr.list_schedules))
    return {**_base(rt), "schedules": [_schedule_out(x) for x in rows][:20]}


async def _cancel_schedule(rt: ToolRuntime, inp: s.CancelScheduleIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    res = _dict(await _session_call("cancel_schedule", mgr.cancel_schedule, inp.schedule_id,
                                    requested_by=f"agent:{actor}", mutating=True))
    outcome = str(res.get("outcome") or "")
    if outcome not in ("CANCELLED", "ALREADY_CANCELLED"):
        raise ToolError(500, "cancel_schedule returned an unexpected outcome; check list_schedules once")
    return {**_base(rt), "outcome": outcome, "schedule": _schedule_out(res.get("schedule")),
            "detail": "No new sessions will start from this schedule; a session it already started keeps running."}


def _perf_summary(p: Mapping[str, Any]) -> dict[str, Any]:
    return {k: p.get(k) for k in s.PerformanceSummaryOut.model_fields}


async def _get_session_report(rt: ToolRuntime, inp: s.GetSessionReportIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    rep = await _session_call("get_session_report", mgr.report, inp.session_id)
    if not isinstance(rep, dict):
        raise ToolError(404, "unknown session" if inp.session_id else "no paper session has run yet")
    r = cast(dict[str, Any], rep)
    perf = _dict(r.get("performance"))
    trades = [{k: _dict(t).get(k) for k in s.TradeOut.model_fields} for t in _list(r.get("trades"))][:20]
    return {
        **_base(rt, bool(perf.get("is_simulated"))), "session": _session_out(r.get("session")),
        "performance": _perf_summary(perf), "cycles": int(r.get("cycles") or 0),
        "decisions": {str(k): int(v) for k, v in _dict(r.get("decisions")).items()},
        "top_no_trade_reasons": [{"reason": truncate(_dict(x).get("reason"), 300), "count": _dict(x).get("count")}
                                 for x in _list(r.get("top_no_trade_reasons"))][:5],
        "top_risk_rejections": [{"check": str(_dict(x).get("check")), "count": _dict(x).get("count")}
                                for x in _list(r.get("top_risk_rejections"))][:5],
        "trades": trades, "api_cost_usd": r.get("api_cost_usd"), "narrative": truncate(r.get("narrative"), 2000),
    }


async def _explain_decisions(rt: ToolRuntime, inp: s.ExplainDecisionsIn, actor: str) -> dict[str, Any]:
    mgr = _manager(rt)
    data = _dict(await _session_call("explain_decisions", mgr.explain, inp.limit))
    cycles = []
    for c in map(_dict, _list(data.get("cycles"))[: inp.limit]):
        cycles.append({k: c.get(k) for k in s.ExplainCycleOut.model_fields} | {
            "no_trade_reason": truncate(c.get("no_trade_reason"), 300), "ai_review": truncate(c.get("ai_review"), 200),
            "failed_checks": [cast(str, truncate(x, 240)) for x in _list(c.get("failed_checks"))][:12],
            "lock_reasons": [cast(str, truncate(x, 200)) for x in _list(c.get("lock_reasons"))][:8],
        })
    agg = _dict(data.get("aggregate"))
    aggregate = {k: agg.get(k) for k in s.ExplainAggregateOut.model_fields}
    return {**_base(rt), "answer": truncate(data.get("answer"), 1200) or "", "aggregate": aggregate, "cycles": cycles}


async def _get_performance(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    p = _dict(await _call(rt.performance))
    curve = _list(p.get("equity_curve"))
    fields = [k for k in s.PerformanceOut.model_fields if k not in s.ToolOutput.model_fields
              and k not in ("equity_points", "equity_curve")]
    return {**_base(rt, bool(p.get("is_simulated"))), **{k: p.get(k) for k in fields}, "equity_points": len(curve),
            "equity_curve": [{"time": _dict(x).get("time"), "equity": _dict(x).get("equity")} for x in curve[-20:]]}


_ACTIVITY_FETCH: Final = 500


def _activity_message(ev: Mapping[str, Any]) -> str:
    event = str(ev.get("event") or "activity")
    if event == "stage":
        msg = f"{ev.get('stage')} {ev.get('status')}"
        if ev.get("item"):
            msg += f" [{ev.get('item')}]"
        return msg + (f": {ev.get('detail')}" if ev.get("detail") else "")
    if event == "decision":
        return f"decision {ev.get('action')} {ev.get('symbol') or ''} {ev.get('status')}: {ev.get('reason') or ''}".strip()
    return str(ev.get("message") or "")


async def _get_activity(rt: ToolRuntime, inp: s.GetActivityIn, actor: str) -> dict[str, Any]:
    rows = _list(await _call(rt.activity.recent, _ACTIVITY_FETCH if inp.kind else inp.limit, None))
    out: list[dict[str, Any]] = []
    for ev in map(_dict, rows):
        event = str(ev.get("event") or "activity")
        kind = ev.get("kind") if event == "activity" else event
        if inp.kind and inp.kind not in (kind, event):
            continue
        out.append({"seq": int(ev.get("seq") or 0), "ts": ev.get("ts"), "event": event, "kind": _str(kind),
                    "message": truncate(_activity_message(ev), 300) or "", "session_id": _str(ev.get("session_id")),
                    "causal_id": _str(ev.get("causal_id"))})
        if len(out) >= inp.limit:
            break
    return {**_base(rt), "events": out}


async def _get_system_health(rt: ToolRuntime, inp: s.NoInput, actor: str) -> dict[str, Any]:
    h = _dict(await _call(rt.health_report))
    comps = [{"name": str(c.get("name")), "status": str(c.get("status")), "detail": truncate(c.get("detail"), 240) or "",
              "last_ok": _str(c.get("last_ok"))} for c in map(_dict, _list(h.get("components")))][:20]
    return {**_base(rt), "overall": str(h.get("overall") or "DOWN"), "components": comps}


# ------------------------------------------------------------------------------------------------ id issuing
def _register_issued_ids(rt: ToolRuntime, actor: str, out: BaseModel) -> None:
    ids = state_for(rt).ids_for(actor)
    if isinstance(out, s.MarketSnapshotOut):
        ids.add(out.snapshot_id, "snapshot", out.symbol)
    elif isinstance(out, s.SignalsOut):
        for sig in out.signals:
            ids.add(sig.signal_id, "signal", sig.symbol)
    elif isinstance(out, s.PredictionsOut):
        for p in out.predictions:
            ids.add(p.prediction_id, "prediction", p.symbol)
    elif isinstance(out, s.ModelReportOut):
        ids.add(out.registry_version, "model_registry", None)


async def _search_memory(rt: ToolRuntime, inp: s.SearchMemoryIn, actor: str) -> dict[str, Any]:
    return {**_base(rt), "memory": await asyncio.to_thread(
        rt.search_memory, inp.query, symbol=inp.symbol or None, kind=inp.kind or None, limit=inp.limit)}


_HANDLERS: Final[dict[str, Handler]] = {
    "search_memory": _search_memory,
    "get_market_snapshot": _get_market_snapshot,
    "get_bars": _get_bars,
    "get_feed_quality": _get_feed_quality,
    "get_predictions": _get_predictions,
    "get_model_report": _get_model_report,
    "get_signals": _get_signals,
    "get_portfolio": _get_portfolio,
    "get_orders": _get_orders,
    "get_order": _get_order,
    "get_risk_status": _get_risk_status,
    "propose_paper_trade": _propose_paper_trade,
    "request_halt": _request_halt,
    "cancel_paper_order": _cancel_paper_order,
    "run_quant_cycle": _run_quant_cycle,
    "start_paper_session": _start_paper_session,
    "stop_session": _stop_session,
    "schedule_session": _schedule_session,
    "cancel_schedule": _cancel_schedule,
    "get_session_status": _get_session_status,
    "list_schedules": _list_schedules,
    "get_session_report": _get_session_report,
    "explain_decisions": _explain_decisions,
    "get_performance": _get_performance,
    "get_activity": _get_activity,
    "get_system_health": _get_system_health,
}
