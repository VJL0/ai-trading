"""Autonomous position manager: the entry-fill -> protection -> exit -> flat lifecycle (PAPER/REPLAY only).

The manager never sends an order itself. Every protective stop, exit and re-price is a ``TradeProposal`` from
``Proposer.POSITION_MANAGER`` that goes through the unchanged single order path: independent risk evaluation ->
ECDSA-signed intent -> durable ledger -> final gate (fresh context) -> gateway. Cancels go through
``ExecutionService.cancel``. Everything it decides is derived from durable state (booked fills, ledger order states,
the position plan) plus a fresh quote, so a restart, disconnect or crash simply resumes on the next tick after a
CLEAN broker reconciliation.

Lifecycle per symbol (one tick, at most one order action per symbol):

* **flat**: stale protective/exit SELLs are canceled; a working entry is canceled after ``entry_fill_timeout_s``;
  an open plan is closed with realized P&L from booked fills once nothing is working.
* **open, holding**: the whole position (net of other working SELLs) is covered by broker-held STOP_LIMIT SELLs at
  the plan's stop. Missing coverage is added (a top-up for the delta), over-coverage is canceled and re-placed.
  A partial entry is protected for its filled quantity; the rest of the entry is canceled after the timeout.
* **exit triggers** (sticky once chosen, persisted on the plan): EOD (``flatten_before_close_s`` before the XNYS
  close, early closes included), OVERNIGHT (opened on an earlier trading day), STOP (bid at the stop and no
  acknowledged broker stop), STOP_BREACH (the broker stop triggered but the bid gapped below its limit, or it is
  still unfilled after ``breach_grace_s``), TIME (the plan's horizon elapsed). Exit sequence: cancel protection ->
  wait until the broker confirms it terminal (a racing stop fill simply shrinks the position) -> marketable LIMIT
  SELL for the remaining quantity at/below the bid within the risk band -> cancel and re-price after
  ``exit_reprice_s`` (a tick lower, never duplicating a recently authorized order).
* **authorized multi-day equities**: the first entry's approved holding request fixes a deadline before the final
  XNYS session close. DAY stops expire overnight and are re-armed from confirmed broker state at the next open;
  gaps through the stop exit immediately. Policy revocation exits at the next permitted session. Existing plans
  and entries without an approved holding request retain the intraday lifecycle.

Safety state (``position_safety``, locks NEW ENTRIES, never exits): UNPROTECTED (no confirmed protection after
``protect_grace_s``), EXITING, BREACHED, OVERNIGHT (held outside regular hours: broker DAY stops expired), and
UNVERIFIED (state cannot be verified). It clears only when ledger state shows the position protected or flat.

A protective stop is a trigger, not guaranteed protection: gaps, halts and thin books can fill far below it or
not at all, and nothing protects a position while the broker order is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Protocol

from packages.contracts.enums import (
    DecisionAction,
    OrderState,
    OrderType,
    Proposer,
    Side,
    TradingMode,
)
from packages.contracts.models import MarketSnapshot, Quote, RiskDecision, TradeProposal
from packages.contracts.policy import RiskPolicy
from packages.core.audit import record_failure
from packages.core.ids import new_causal_id, new_id
from packages.core.logging import log_event
from packages.core.money import BPS, ZERO, floor_shares, price_tick
from services.order_ledger.entries import EntriesLock
from services.order_ledger.ledger import OrderLedger, OrderRecord
from services.order_ledger.sessions import session_bounds, session_offset
from services.order_ledger.state_machine import IN_DOUBT, OPEN, PRE_SUBMISSION, TERMINAL, WORKING
from services.order_ledger.timeutil import NY_TZ, iso, parse_iso, trading_day
from services.risk import limits
from services.risk.context import ANY_STRATEGY, order_fingerprint
from services.risk.policy_store import PolicyReader

from .policy import PositionPolicy
from .store import PositionPlan, PositionStore

log = logging.getLogger("lat.positions")

S = OrderState
ACKED: frozenset[OrderState] = frozenset({S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED})
STRATEGY_PREFIX = "position-manager"
PROTECTION_NOTE = "a protective stop is a trigger, not a guaranteed exit price (gaps can fill far below it or not at all)"
EXIT_REASONS = ("POLICY_REVOKED", "HOLDING_LIMIT", "EOD", "OVERNIGHT", "STOP_BREACH", "STOP", "TIME")
SESSION_MODES = frozenset({TradingMode.PAPER, TradingMode.REPLAY})


class PositionPort(Protocol):
    """What the manager needs from the runtime (tests provide a rig). No method may bypass the risk engine."""

    @property
    def mode(self) -> TradingMode: ...

    @property
    def ledger(self) -> OrderLedger: ...

    @property
    def entries(self) -> EntriesLock: ...

    @property
    def dataset_version(self) -> str: ...

    def now(self) -> datetime: ...

    def risk_policy(self) -> RiskPolicy: ...

    async def snapshot(self, symbol: str) -> MarketSnapshot: ...

    async def submit_proposal(self, proposal: TradeProposal, snapshot: MarketSnapshot) -> dict[str, Any]:
        """Risk-evaluate, authorize, record and submit through the single order path. Returns the outcome dict."""
        ...

    async def cancel_order(self, client_order_id: str) -> OrderState | None: ...

    async def request_reconcile(self) -> Any: ...


@dataclass
class SymbolView:
    """Everything one decision for one symbol reads (built fresh every tick from durable state)."""

    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    plan: PositionPlan | None
    working_buys: list[OrderRecord] = field(default_factory=list)
    protective: list[OrderRecord] = field(default_factory=list)
    exits: list[OrderRecord] = field(default_factory=list)
    in_doubt: list[OrderRecord] = field(default_factory=list)

    @property
    def protected_qty(self) -> Decimal:
        """Quantity covered by broker-acknowledged protective stops (CANCEL_PENDING / SUBMITTING do not count)."""
        return sum((r.quantity - r.filled_quantity for r in self.protective if r.state in ACKED), ZERO)

    @property
    def other_sell_qty(self) -> Decimal:
        return sum((r.quantity - r.filled_quantity for r in self.exits if r.state in WORKING | IN_DOUBT), ZERO)

    @property
    def protection_target(self) -> Decimal:
        return max(self.quantity - self.other_sell_qty, ZERO)


def _age(now: datetime, then: datetime | None) -> float:
    return 0.0 if then is None else (now - then).total_seconds()


def _floor_tick(price: Decimal) -> Decimal:
    tick = price_tick(price)
    return (price / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def _ceil_tick(price: Decimal) -> Decimal:
    tick = price_tick(price)
    return (price / tick).to_integral_value(rounding=ROUND_CEILING) * tick


class PositionManager:
    def __init__(self, port: PositionPort, store: PositionStore, policy: PositionPolicy | None = None) -> None:
        self.port = port
        self.store = store
        self.policy = policy or PositionPolicy()
        self._lock = asyncio.Lock()
        self._unprotected_since: dict[str, datetime] = {}
        self._unprotected_ticks: dict[str, int] = {}
        self._breach_since: dict[str, datetime] = {}
        self._cooldown_until: dict[str, datetime] = {}
        self._last_reconcile_request: datetime | None = None
        self._last_quote: dict[str, Quote] = {}
        self.ticks = 0
        self.last_tick_at: datetime | None = None
        self.last_error: str | None = None

    # ================================================================== loop
    async def run(self, stop_event: asyncio.Event, *, interval_s: float = 1.0) -> None:
        while not stop_event.is_set():
            try:
                await self.tick()
            except Exception as exc:  # the loop survives everything; the failure is recorded
                self._failure("tick", exc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)

    async def tick(self) -> list[dict[str, Any]]:
        """One pass over every symbol with a position, a plan or a working order. Returns the actions taken."""
        async with self._lock:
            return await self._tick()

    async def _tick(self) -> list[dict[str, Any]]:
        if not self.policy.enabled or self.port.mode not in SESSION_MODES:
            return []
        now = self.port.now()
        self.ticks += 1
        self.last_tick_at = now
        ledger = self.port.ledger
        positions = {p.symbol: p for p in ledger.positions()}
        records = ledger.records(sorted(OPEN, key=lambda st: st.value))
        plans = {p.symbol: p for p in self.store.open_plans()}
        symbols = sorted({s for s, p in positions.items() if p.quantity != 0} | set(plans) | {r.symbol for r in records})
        exits_locked, exit_reasons = self.port.entries.exits_locked()
        out: list[dict[str, Any]] = []
        for sym in symbols:
            pos = positions.get(sym)
            view = SymbolView(
                symbol=sym, quantity=pos.quantity if pos else ZERO, avg_cost=pos.avg_entry_price if pos else ZERO,
                plan=plans.get(sym),
            )
            for r in records:
                if r.symbol != sym or r.state in PRE_SUBMISSION:
                    continue
                if r.state in IN_DOUBT:
                    view.in_doubt.append(r)
                if r.side == Side.BUY:
                    if r.state in WORKING:
                        view.working_buys.append(r)
                elif r.is_protective:
                    view.protective.append(r)
                else:
                    view.exits.append(r)
            try:
                out.extend(await self._manage(view, now, exits_locked, exit_reasons))
            except Exception as exc:
                self._failure(f"manage:{sym}", exc)
                if view.quantity > 0:
                    self._safety(sym, "UNVERIFIED", f"position manager error: {type(exc).__name__}: {exc}"[:300])
        for sym in [s for s in self._unprotected_since if s not in symbols]:
            self._unprotected_since.pop(sym, None)
            self._unprotected_ticks.pop(sym, None)
        return out

    # ================================================================== per symbol
    async def _manage(self, v: SymbolView, now: datetime, exits_locked: bool,
                      exit_reasons: Sequence[str]) -> list[dict[str, Any]]:
        sym = v.symbol
        if v.quantity < 0:
            self._safety(sym, "UNVERIFIED", f"negative ledger position {v.quantity} in a long-only system")
            return []
        bounds = session_bounds(now)
        session_open = bounds is not None and bounds[0] <= now < bounds[1]
        if v.quantity > 0 and not session_open:
            # Whatever the locks say: a position outside regular hours is unprotected (broker DAY stops expired).
            self._safety(sym, "OVERNIGHT", "position held outside XNYS regular hours; broker DAY protective orders "
                                           "expire at the close, so it is unprotected until the next open")
            return []
        if exits_locked:
            # Broker/ledger state is uncertain: no order actions. Broker-held stops keep working at the broker.
            if v.quantity > 0:
                self._evaluate_protection(v, now, exiting=False, detail="exits locked: " + "; ".join(exit_reasons)[:200])
            await self._maybe_reconcile(now, exit_reasons)
            return []
        if v.in_doubt:
            return []  # a submission/lookup is in flight (UNKNOWN orders also lock exits): never act on doubt
        if v.quantity == 0:
            return await self._flat(v, now)
        plan = self._ensure_plan(v, now)
        v.plan = plan
        if plan.max_quantity < v.quantity:
            self.store.update_plan(plan.plan_id, now, max_quantity=v.quantity)
        assert bounds is not None  # session_open
        # DAY orders from a previous session cannot establish today's protection. An expiry update may have
        # been lost during a restart; resolve it from broker truth before replacing or canceling anything.
        if any(trading_day(r.submitted_at or r.created_at) < trading_day(now) for r in v.protective):
            self._safety(sym, "UNVERIFIED", "prior-session DAY protection awaits broker reconciliation")
            await self._maybe_reconcile(now, (), force=True)
            return []
        snapshot = await self.port.snapshot(sym)
        quote = snapshot.quote
        if quote is not None:
            self._last_quote[sym] = quote
        actions: list[dict[str, Any]] = []
        # ---- entries: cancel a stale working entry (partial or unfilled); EOD cancels any entry
        flatten_at = bounds[1] - timedelta(seconds=self.policy.flatten_before_close_s)
        for b in v.working_buys:
            if b.state == S.CANCEL_PENDING:
                continue
            if _age(now, b.submitted_at or b.created_at) >= self.policy.entry_fill_timeout_s or now >= flatten_at:
                actions.append(await self._cancel(plan, b, "ENTRY_EOD" if now >= flatten_at else "ENTRY_TIMEOUT", now))
                return actions
        # ---- exit trigger (sticky)
        reason = plan.exit_reason or self._exit_trigger(v, plan, now, flatten_at, quote)
        if reason is not None:
            if plan.exit_reason is None:
                self.store.update_plan(plan.plan_id, now, exit_reason=reason, exit_started_at=now)
                self.store.record_action(plan_id=plan.plan_id, symbol=sym, at=now, action="EXIT_TRIGGERED", purpose=reason,
                                         coid=None, outcome="OK", detail=self._trigger_detail(reason, plan, quote))
            state = "BREACHED" if reason == "STOP_BREACH" else "OVERNIGHT" if reason == "OVERNIGHT" else "EXITING"
            self._safety(sym, state, f"exit in progress ({reason}): {v.quantity} shares; protection canceled for the exit")
            actions.extend(await self._exit(v, plan, reason, now, snapshot))
            return actions
        actions.extend(await self._protect(v, plan, now, snapshot))
        self._evaluate_protection(self._refresh(v) if actions else v, now, exiting=False)
        return actions

    def _refresh(self, v: SymbolView) -> SymbolView:
        """Re-read the symbol's working orders after this tick acted (broker responses may already be applied)."""
        fresh = SymbolView(symbol=v.symbol, quantity=self.port.ledger.position_quantities().get(v.symbol, ZERO),
                           avg_cost=v.avg_cost, plan=v.plan)
        for r in self.port.ledger.records(sorted(OPEN, key=lambda st: st.value)):
            if r.symbol != v.symbol or r.state in PRE_SUBMISSION:
                continue
            if r.state in IN_DOUBT:
                fresh.in_doubt.append(r)
            if r.side == Side.BUY:
                if r.state in WORKING:
                    fresh.working_buys.append(r)
            elif r.is_protective:
                fresh.protective.append(r)
            else:
                fresh.exits.append(r)
        return fresh

    # ------------------------------------------------------------------ flat
    async def _flat(self, v: SymbolView, now: datetime) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        bounds = session_bounds(now)
        flatten = bounds is not None and now >= bounds[1] - timedelta(seconds=self.policy.flatten_before_close_s)
        for r in [*v.protective, *v.exits]:
            if r.state in ACKED:
                actions.append(await self._cancel(v.plan, r, "FLAT_CLEANUP", now))
                return actions
        for b in v.working_buys:
            if b.state != S.CANCEL_PENDING and (
                flatten or _age(now, b.submitted_at or b.created_at) >= self.policy.entry_fill_timeout_s
            ):
                actions.append(await self._cancel(v.plan, b, "ENTRY_EOD" if flatten else "ENTRY_TIMEOUT", now))
                return actions
        working = [r for r in [*v.protective, *v.exits, *v.working_buys] if r.state not in TERMINAL]
        await self._stuck_cancel([r for r in working if r.state == S.CANCEL_PENDING], now)
        if v.plan is not None and not working:
            self._close_plan(v.plan, now)
        self._clear_safety(v.symbol, "position flat")
        self._unprotected_since.pop(v.symbol, None)
        self._unprotected_ticks.pop(v.symbol, None)
        self._breach_since.pop(v.symbol, None)
        return actions

    # ------------------------------------------------------------------ protection
    async def _protect(self, v: SymbolView, plan: PositionPlan, now: datetime,
                       snapshot: MarketSnapshot) -> list[dict[str, Any]]:
        if not self.policy.protective_stops:
            return []
        pending_cancel = [r for r in v.protective if r.state == S.CANCEL_PENDING]
        if pending_cancel:
            await self._stuck_cancel(pending_cancel, now)
            return []
        target = v.protection_target
        covered = v.protected_qty
        if covered > target:
            # Over-covered (position shrank, e.g. an external SELL filled): cancel and re-place exactly.
            largest = max(v.protective, key=lambda r: r.quantity - r.filled_quantity)
            return [await self._cancel(plan, largest, "RESIZE_PROTECTION", now)]
        if covered == target or target <= 0:
            return []
        if self._cooling(v.symbol, now):
            return []
        open_sells = len([r for r in [*v.protective, *v.exits] if r.state not in TERMINAL])
        if open_sells >= limits.MAX_OPEN_SELLS_PER_SYMBOL:
            # No slot for a top-up: replace the smallest protective order with one for the full quantity.
            smallest = min(v.protective, key=lambda r: r.quantity - r.filled_quantity)
            return [await self._cancel(plan, smallest, "CONSOLIDATE_PROTECTION", now)]
        quote = snapshot.quote
        if quote is None:
            return [self._note(plan, v.symbol, now, "PROTECT", "SKIPPED", "no fresh quote; cannot price a protective stop")]
        delta = floor_shares(target - covered)
        if delta <= 0:
            return []
        stop = plan.stop_price
        gap = self.policy.stop_limit_gap_bps / BPS
        limit = self._unique_price(v.symbol, delta, _floor_tick(stop * (1 - gap)), floor=_ceil_tick(
            stop * (1 - limits.MAX_STOP_LIMIT_GAP_BPS / BPS)), now=now)
        if limit is None:
            return [self._note(plan, v.symbol, now, "PROTECT", "SKIPPED",
                               "every protective limit price inside the gap was authorized recently (duplicate window)")]
        purpose = "TOP_UP" if covered > 0 else "PROTECT"
        proposal = self._proposal(
            plan, snapshot, now, purpose=purpose.lower(), qty=delta, limit=limit, stop=stop,
            rationale=(f"protective STOP_LIMIT SELL {delta} {v.symbol}: stop {stop}, limit {limit} "
                       f"({plan.stop_source}); {PROTECTION_NOTE}"))
        return [await self._submit(plan, v.symbol, purpose, proposal, snapshot, now)]

    def _evaluate_protection(self, v: SymbolView, now: datetime, *, exiting: bool, detail: str | None = None) -> None:
        """Raise UNPROTECTED once a position lacks confirmed coverage for longer than the grace period."""
        sym = v.symbol
        if exiting:
            return
        target = v.protection_target
        ok = self.policy.protective_stops and target > 0 and v.protected_qty >= target
        if target == 0 and v.other_sell_qty >= v.quantity > 0:
            ok = True  # the whole position is already being sold by working SELLs
        if ok:
            self._unprotected_since.pop(sym, None)
            self._unprotected_ticks.pop(sym, None)
            self._clear_safety(sym, f"protected: {v.protected_qty} shares covered by broker-held stops")
            return
        since = self._unprotected_since.setdefault(sym, now)
        ticks = self._unprotected_ticks[sym] = self._unprotected_ticks.get(sym, 0) + 1
        if _age(now, since) >= self.policy.protect_grace_s and ticks >= 2:
            why = (f"{v.quantity} shares, {v.protected_qty} covered by acknowledged broker stops for "
                   f"{_age(now, since):.0f}s" + (f"; {detail}" if detail else ""))
            if not self.policy.protective_stops:
                why = "broker-held protective stops are disabled; " + why
            self._safety(sym, "UNPROTECTED", why)

    # ------------------------------------------------------------------ exits
    def _exit_trigger(self, v: SymbolView, plan: PositionPlan, now: datetime, flatten_at: datetime,
                      quote: Quote | None) -> str | None:
        if plan.holding_sessions > 1:
            policy = self.port.risk_policy()
            if (not policy.allow_overnight_positions or plan.strategy_id not in policy.multi_day_strategy_allowlist
                    or plan.holding_sessions > policy.max_holding_sessions):
                return "POLICY_REVOKED"
            # Never extend a durable deadline when the process restarts or its position policy changes.
            if now >= plan.exit_deadline or (
                trading_day(now) >= trading_day(plan.exit_deadline) and now >= flatten_at
            ):
                return "HOLDING_LIMIT"
        else:
            if now >= flatten_at:
                return "EOD"
            if plan.trading_day < trading_day(now).isoformat():
                return "OVERNIGHT"
        if quote is not None:
            acked = [r for r in v.protective if r.state in ACKED]
            if acked and quote.bid <= plan.stop_price:
                limit_floor = min((r.limit_price for r in acked if r.limit_price is not None), default=plan.stop_price)
                if quote.bid < limit_floor:
                    return "STOP_BREACH"  # gapped through the stop-limit: the broker order will not fill
                since = self._breach_since.setdefault(v.symbol, now)
                if _age(now, since) >= self.policy.breach_grace_s:
                    return "STOP_BREACH"
            else:
                self._breach_since.pop(v.symbol, None)
            zone = plan.stop_price * (1 + limits.MIN_STOP_DISTANCE_BPS / BPS)
            if not acked and quote.bid <= zone:
                return "STOP"  # no broker stop resting and the bid is at the stop: exit now
        if plan.holding_sessions == 1 and now >= plan.exit_deadline:
            return "TIME"
        return None

    async def _exit(self, v: SymbolView, plan: PositionPlan, reason: str, now: datetime,
                    snapshot: MarketSnapshot) -> list[dict[str, Any]]:
        # 1. protection must be gone first (shares held by a resting stop cannot be sold twice)
        live_prot = [r for r in v.protective if r.state not in TERMINAL]
        to_cancel = [r for r in live_prot if r.state != S.CANCEL_PENDING]
        if to_cancel:
            return [await self._cancel(plan, to_cancel[0], f"EXIT_{reason}", now)]
        if live_prot:
            await self._stuck_cancel(live_prot, now)
            return []
        # 2. one exit at a time; re-price a stale one
        live_exits = [r for r in v.exits if r.state not in TERMINAL]
        if live_exits:
            e = live_exits[0]
            if e.state == S.CANCEL_PENDING:
                await self._stuck_cancel([e], now)
                return []
            if _age(now, e.submitted_at or e.created_at) >= self.policy.exit_reprice_s:
                return [await self._cancel(plan, e, "EXIT_REPRICE", now)]
            return []
        if self._cooling(v.symbol, now):
            return []
        quote = snapshot.quote
        if quote is None:
            return [self._note(plan, v.symbol, now, "EXIT", "SKIPPED", "no fresh quote; cannot price the exit")]
        qty = floor_shares(v.quantity)
        if qty <= 0:
            return []
        attempt = self._exit_attempts(plan)
        limit = self._exit_limit(v.symbol, qty, quote, attempt, now)
        if limit is None:
            return [self._note(plan, v.symbol, now, "EXIT", "SKIPPED",
                               "every exit price inside the risk band was authorized recently (duplicate window)")]
        proposal = self._proposal(
            plan, snapshot, now, purpose=f"exit-{reason.lower()}", qty=qty, limit=limit, stop=None,
            rationale=(f"{reason} exit: LIMIT SELL {qty} {v.symbol} @ {limit} (bid {quote.bid}, attempt {attempt + 1}); "
                       f"plan {plan.plan_id}"))
        return [await self._submit(plan, v.symbol, "EXIT", proposal, snapshot, now)]

    def _exit_limit(self, symbol: str, qty: Decimal, quote: Quote, attempt: int, now: datetime) -> Decimal | None:
        """Marketable SELL limit: at the bid, one tick lower per attempt, inside both risk bands."""
        bid, mid = quote.bid, quote.mid
        tick = price_tick(bid)
        band = self.port.risk_policy().max_limit_deviation_bps
        floor = max(_ceil_tick(bid * (1 - limits.MAX_THROUGH_TOUCH_BPS / BPS)), _ceil_tick(mid * (1 - band / BPS)))
        start = max(_floor_tick(bid) - tick * attempt, floor)
        return self._unique_price(symbol, qty, start, floor=floor, now=now)

    def _unique_price(self, symbol: str, qty: Decimal, start: Decimal, *, floor: Decimal, now: datetime) -> Decimal | None:
        """First price at or below ``start`` (down to ``floor``) whose SELL fingerprint is not in the risk engine's
        duplicate window, so a re-price is never refused as a duplicate of a canceled order."""
        window = self.port.risk_policy().duplicate_window_seconds
        recent = set(self.port.ledger.recent_fingerprints(now - timedelta(seconds=window)))
        twins = {o.limit_price for o in self.port.ledger.open_orders()
                 if o.symbol == symbol and o.side == Side.SELL and o.quantity == qty}
        price = start
        while price >= floor and price > 0:
            if price not in twins and order_fingerprint(symbol, Side.SELL, qty, price, ANY_STRATEGY) not in recent:
                return price
            price -= price_tick(price)
        return None

    def _exit_attempts(self, plan: PositionPlan) -> int:
        row = self.port.ledger.db.query_one(
            "SELECT COUNT(*) AS n FROM position_actions WHERE plan_id=? AND action='EXIT' AND outcome='SUBMITTED'",
            (plan.plan_id,))
        return int(row["n"]) if row else 0

    @staticmethod
    def _trigger_detail(reason: str, plan: PositionPlan, quote: Quote | None) -> str:
        q = f"bid {quote.bid} ask {quote.ask}" if quote is not None else "no quote"
        return f"{reason}: stop {plan.stop_price}, deadline {iso(plan.exit_deadline)}, {q}"

    # ------------------------------------------------------------------ plans
    def _ensure_plan(self, v: SymbolView, now: datetime) -> PositionPlan:
        if v.plan is not None:
            return v.plan
        db = self.port.ledger.db
        since = self.store.last_closed_at(v.symbol)
        fill = db.query_one(
            "SELECT client_order_id, ts FROM fills WHERE symbol=? AND side='BUY' AND ts >= ? ORDER BY ts, fill_id LIMIT 1",
            (v.symbol, "" if since is None else iso(since)))
        opened_at = parse_iso(str(fill["ts"])) if fill else now
        entry_coid = str(fill["client_order_id"]) if fill else None
        proposal: dict[str, Any] = {}
        horizon = self.policy.default_horizon_minutes
        holding_sessions = 1
        entry_risk_policy_version: str | None = None
        if entry_coid is not None:
            row = db.query_one("SELECT p.payload_json FROM orders o JOIN proposals p ON p.proposal_id=o.proposal_id "
                               "WHERE o.client_order_id=?", (entry_coid,))
            proposal = json.loads(row["payload_json"]) if row else {}
            holding_sessions, entry_risk_policy_version = self._entry_holding(entry_coid, proposal)
            sid = proposal.get("signal_id")
            if sid:
                sig = db.query_one("SELECT payload_json FROM quant_signals WHERE signal_id=?", (sid,))
                if sig:
                    with contextlib.suppress(Exception):
                        horizon = max(1, min(int(json.loads(sig["payload_json"]).get("horizon_minutes") or horizon), 390))
        avg = v.avg_cost if v.avg_cost > 0 else None
        raw_stop = proposal.get("stop_price")
        stop: Decimal | None = Decimal(str(raw_stop)) if raw_stop is not None else None
        source = "entry_proposal_stop"
        reference = avg or (Decimal(str(proposal["limit_price"])) if proposal.get("limit_price") else None)
        if stop is None or reference is None or not (ZERO < stop < reference):
            if reference is None:
                raise RuntimeError(f"cannot plan {v.symbol}: no average cost or entry price")
            default_bps = self.port.risk_policy().default_stop_loss_bps
            stop = _floor_tick(reference * (1 - default_bps / BPS))
            source = "policy_default_stop" if raw_stop is None else "policy_default_stop(entry stop invalid)"
        deadline = opened_at + timedelta(minutes=horizon)
        if holding_sessions > 1:
            last_day = session_offset(trading_day(opened_at), holding_sessions)
            last_bounds = session_bounds(datetime.combine(last_day, time(12), tzinfo=NY_TZ))
            if last_bounds is None:
                raise RuntimeError("cannot establish final holding session; no multi-day plan created")
            deadline = last_bounds[1] - timedelta(seconds=self.policy.flatten_before_close_s)
        plan = self.store.create_plan(
            symbol=v.symbol, opened_at=opened_at, strategy_id=proposal.get("strategy_id"), entry_coid=entry_coid,
            stop_price=stop, stop_source=source, horizon_minutes=horizon,
            exit_deadline=deadline, quantity=v.quantity, holding_sessions=holding_sessions,
            entry_risk_policy_version=entry_risk_policy_version, policy_version=self.policy.version, at=now)
        log_event(log, "position plan opened", symbol=v.symbol, plan_id=plan.plan_id, stop=str(stop))
        return plan

    def _entry_holding(self, entry_coid: str, proposal: dict[str, Any]) -> tuple[int, str | None]:
        """Only a filled entry with an independently approved holding request creates a multi-day plan.

        Older approved orders lack the holding-period check and stay intraday even if the active policy later
        enables overnight holding. Read the entry's policy, not today's policy, so revocation before first manager
        tick still reconstructs its original plan and immediately takes the POLICY_REVOKED exit.
        """
        order = self.port.ledger.get_record(entry_coid)
        if order is None or order.side != Side.BUY or order.filled_quantity <= 0:
            return 1, None
        intent = order.intent
        version = intent.policy_version
        requested = proposal.get("holding_sessions")
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 1:
            return 1, version
        row = self.port.ledger.db.query_one("SELECT payload_json FROM risk_decisions WHERE decision_id=?",
                                             (order.decision_id,))
        if row is None:
            return 1, version
        decision = RiskDecision.model_validate_json(row["payload_json"])
        approved = decision.approved and decision.proposal_id == order.proposal_id and decision.policy_version == version
        approved = approved and any(c.name == "holding_period_authorized" and c.passed for c in decision.checks)
        if not approved:
            return 1, version
        policy = PolicyReader(self.port.ledger.db).get(version)
        if (limits.validate_policy(policy) or not policy.allow_overnight_positions
                or proposal.get("strategy_id") not in policy.multi_day_strategy_allowlist
                or requested > policy.max_holding_sessions):
            return 1, version
        return requested, version

    def _close_plan(self, plan: PositionPlan, now: datetime) -> None:
        db = self.port.ledger.db
        rows = db.query(
            "SELECT f.side, f.quantity, f.price, f.realized_pl, o.order_type FROM fills f JOIN orders o ON "
            "o.client_order_id=f.client_order_id WHERE f.symbol=? AND f.ts >= ? ORDER BY f.ts, f.fill_id",
            (plan.symbol, iso(plan.opened_at)))
        bq = bn = sq = sn = realized = ZERO
        last_sell_type: str | None = None
        for r in rows:
            q, px = Decimal(r["quantity"]), Decimal(r["price"])
            if r["side"] == Side.BUY.value:
                bq, bn = bq + q, bn + q * px
            else:
                sq, sn = sq + q, sn + q * px
                realized += Decimal(r["realized_pl"])
                last_sell_type = r["order_type"]
        reason = plan.exit_reason or ("STOP" if last_sell_type == OrderType.STOP_LIMIT.value else "EXTERNAL")
        if plan.exit_reason and last_sell_type == OrderType.STOP_LIMIT.value and plan.exit_reason not in ("STOP",):
            reason = f"STOP (during {plan.exit_reason} exit)"
        self.store.close_plan(plan, at=now, reason=reason, entry_qty=bq, entry_avg=(bn / bq).quantize(Decimal("0.0001"))
                              if bq else None, exit_qty=sq, exit_avg=(sn / sq).quantize(Decimal("0.0001")) if sq else None,
                              realized=realized)
        log_event(log, "position plan closed", symbol=plan.symbol, plan_id=plan.plan_id, reason=reason,
                  realized=str(realized))

    # ------------------------------------------------------------------ order actions
    def _proposal(self, plan: PositionPlan, snapshot: MarketSnapshot, now: datetime, *, purpose: str, qty: Decimal,
                  limit: Decimal, stop: Decimal | None, rationale: str) -> TradeProposal:
        return TradeProposal(
            proposal_id=new_id("prop"), causal_id=new_causal_id(), created_at=now, data_asof=snapshot.data_asof,
            action=DecisionAction.TRADE, proposer=Proposer.POSITION_MANAGER, strategy_id=f"{STRATEGY_PREFIX}/{purpose}",
            symbol=plan.symbol, side=Side.SELL, quantity=qty,
            order_type=OrderType.STOP_LIMIT if stop is not None else OrderType.LIMIT, limit_price=limit,
            stop_price=stop, max_planned_loss=ZERO, evidence_ids=(f"position:{plan.plan_id}",),
            dataset_version=self.port.dataset_version, rationale=rationale[:4000],
        )

    async def _submit(self, plan: PositionPlan, symbol: str, action: str, proposal: TradeProposal,
                      snapshot: MarketSnapshot, now: datetime) -> dict[str, Any]:
        try:
            result = await self.port.submit_proposal(proposal, snapshot)
        except Exception as exc:
            result = {"outcome": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}
        outcome = str(result.get("outcome") or "UNKNOWN")
        coid = result.get("client_order_id")
        reason = result.get("reason")
        if outcome != "SUBMITTED":
            self._cooldown_until[symbol] = now + timedelta(seconds=self.policy.exit_reprice_s)
        detail = proposal.rationale if outcome == "SUBMITTED" else f"{outcome}: {reason}"[:1000]
        self.store.record_action(plan_id=plan.plan_id, symbol=symbol, at=now, action=action,
                                 purpose=proposal.strategy_id, coid=coid, outcome=outcome, detail=detail)
        return {"action": action, "symbol": symbol, "outcome": outcome, "client_order_id": coid, "reason": reason,
                "order_type": proposal.order_type.value, "quantity": format(proposal.quantity or ZERO, "f"),
                "limit_price": format(proposal.limit_price or ZERO, "f"),
                "stop_price": None if proposal.stop_price is None else format(proposal.stop_price, "f")}

    async def _cancel(self, plan: PositionPlan | None, rec: OrderRecord, purpose: str, now: datetime) -> dict[str, Any]:
        try:
            state = await self.port.cancel_order(rec.client_order_id)
            outcome = "CANCEL_REQUESTED" if state is None else state.value
        except Exception as exc:
            outcome = f"ERROR {type(exc).__name__}"
            state = None
        self.store.record_action(plan_id=plan.plan_id if plan else None, symbol=rec.symbol, at=now, action="CANCEL",
                                 purpose=purpose, coid=rec.client_order_id, outcome=outcome,
                                 detail=f"{rec.side.value} {rec.order_type.value} {rec.quantity} (filled "
                                        f"{rec.filled_quantity}) state {rec.state.value}")
        return {"action": "CANCEL", "symbol": rec.symbol, "purpose": purpose, "client_order_id": rec.client_order_id,
                "outcome": outcome}

    async def _stuck_cancel(self, recs: Sequence[OrderRecord], now: datetime) -> None:
        stuck = [r for r in recs if _age(now, r.cancel_requested_at or r.updated_at) >= self.policy.cancel_stuck_s]
        if stuck:
            await self._maybe_reconcile(now, [f"cancel pending for {r.client_order_id}" for r in stuck], force=True)

    def _note(self, plan: PositionPlan | None, symbol: str, now: datetime, action: str, outcome: str,
              detail: str) -> dict[str, Any]:
        last = self.store.actions(symbol, limit=1)
        if not (last and last[0]["action"] == action and last[0]["outcome"] == outcome and last[0]["detail"] == detail):
            self.store.record_action(plan_id=plan.plan_id if plan else None, symbol=symbol, at=now, action=action,
                                     purpose=None, coid=None, outcome=outcome, detail=detail)
        return {"action": action, "symbol": symbol, "outcome": outcome, "reason": detail}

    def _cooling(self, symbol: str, now: datetime) -> bool:
        until = self._cooldown_until.get(symbol)
        return until is not None and now < until

    async def _maybe_reconcile(self, now: datetime, reasons: Sequence[str], *, force: bool = False) -> None:
        """Ask for a broker-authoritative reconciliation when exits are blocked by state uncertainty (rate-limited).

        Never when the block is an operator kill switch or a mode that cannot trade.
        """
        text = " ".join(reasons)
        if not force and ("scope=ALL" in text or "mode " in text):
            return
        last = self._last_reconcile_request
        if last is not None and _age(now, last) < self.policy.reconcile_request_interval_s:
            return
        self._last_reconcile_request = now
        try:
            await self.port.request_reconcile()
        except Exception as exc:
            self._failure("reconcile_request", exc)

    # ------------------------------------------------------------------ safety
    def _safety(self, symbol: str, state: str, reason: str) -> None:
        try:
            if self.port.entries.set_position_safety(symbol, state, reason):
                log_event(log, "position safety state", logging.WARNING, symbol=symbol, state=state, reason=reason)
        except Exception as exc:
            self._failure("safety", exc)

    def _clear_safety(self, symbol: str, reason: str) -> None:
        try:
            self.port.entries.clear_position_safety(symbol, reason)
        except Exception as exc:
            self._failure("safety_clear", exc)

    def _failure(self, kind: str, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"[:1000]
        self.last_error = f"{kind}: {detail}"
        log_event(log, "position manager failure", logging.ERROR, kind=kind, error=detail)
        with contextlib.suppress(Exception):
            self.port.ledger.db.write(lambda c: record_failure(c, component="position_manager", kind=kind, detail=detail))

    # ================================================================== read-only view
    def view(self) -> dict[str, Any]:
        """Position-level state and risk for the API and agent tools (read-only, JSON-safe)."""
        ledger = self.port.ledger
        positions = {p.symbol: p for p in ledger.positions()}
        records = ledger.records(sorted(OPEN, key=lambda st: st.value))
        safety = {s["symbol"]: s for s in self.port.entries.position_safety()}
        open_rows: list[dict[str, Any]] = []
        total_risk = ZERO
        for plan in self.store.open_plans():
            pos = positions.get(plan.symbol)
            qty = pos.quantity if pos else ZERO
            avg = pos.avg_entry_price if pos else ZERO
            quote = self._last_quote.get(plan.symbol)
            mark = quote.bid if quote is not None else None
            prot = [r for r in records if r.symbol == plan.symbol and r.is_protective]
            protected = sum((r.quantity - r.filled_quantity for r in prot if r.state in ACKED), ZERO)
            risk_to_stop = qty * max((mark if mark is not None else avg) - plan.stop_price, ZERO)
            total_risk += risk_to_stop
            open_rows.append({
                **plan.to_json(), "quantity": format(qty, "f"), "avg_cost": format(avg, "f"),
                "mark_bid": None if mark is None else format(mark, "f"),
                "unrealized_pl": None if mark is None else format(qty * (mark - avg), "f"),
                "risk_to_stop": format(risk_to_stop, "f"),
                "risk_to_stop_note": "loss if exited exactly at the stop; gaps can lose more (not guaranteed)",
                "protected_quantity": format(protected, "f"),
                "protective_orders": [{"client_order_id": r.client_order_id, "state": r.state.value,
                                       "quantity": format(r.quantity, "f"), "stop_price": None if r.stop_price is None
                                       else format(r.stop_price, "f"),
                                       "limit_price": None if r.limit_price is None else format(r.limit_price, "f")}
                                      for r in prot],
                "safety": safety.get(plan.symbol),
            })
        closed = self.store.closed_plans(limit=20)
        return {
            "open": open_rows, "closed": [p.to_json() for p in closed],
            "realized_pl_closed_recent": format(sum((p.realized_pl or ZERO for p in closed), ZERO), "f"),
            "open_risk_to_stop": format(total_risk, "f"), "safety": list(safety.values()),
            "policy": self.policy.model_dump(mode="json"), "policy_version": self.policy.version,
            "manager": {"ticks": self.ticks, "last_tick_at": None if self.last_tick_at is None else iso(self.last_tick_at),
                        "last_error": self.last_error, "enabled": self.policy.enabled},
            "note": PROTECTION_NOTE,
        }
