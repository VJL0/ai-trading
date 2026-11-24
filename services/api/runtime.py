"""Runtime composition root: wires every service behind the API (integration owner).

Order path (the only one): DecisionCycle -> RiskEngine.evaluate/authorize (signed intent) -> OrderLedger
-> ExecutionService.submit (final_gate at the boundary) -> ExecutionGateway (FakeBroker | LEAN bridge).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from packages.contracts import (
    ExecutionStatus,
    Quote,
    ReconciliationReport,
    RiskContext,
    RiskPolicy,
    SystemHealth,
    TradeProposal,
)
from packages.contracts.enums import DataFeed, ExecutionLabel, FeedMode, GatewayKind, ReconciliationStatus, Side
from packages.contracts.models import PositionSnapshot
from packages.core import modes
from packages.core.audit import active_halts, add_halt, append_audit, clear_halt, get_state, record_failure, set_state
from packages.core.clock import Clock, SystemClock
from packages.core.config import Settings
from packages.core.db import Database, utc_iso
from packages.core.logging import configure_logging, log_event
from packages.core.secrets import SecretName, SecretStore
from packages.core.signing import IntentSigner
from services.order_ledger.timeutil import trading_day

from .events import DbTailer, EventBus
from .security import AGENT_TOKEN_FILE, ENGINE_TOKEN_FILE, OPERATOR_TOKEN_FILE, ensure_token
from .simclock import SimulationClock

log = logging.getLogger("lat.runtime")


class Runtime:
    """Holds long-lived service objects. Construct with ``build_runtime``."""

    def __init__(self, settings: Settings) -> None:
        from services.market_data.quality import FeedQualityMonitor
        from services.market_data.store import MarketDataStore
        from services.order_ledger.entries import EntriesLock
        from services.order_ledger.ledger import OrderLedger
        from services.risk.policy_store import PolicyStore

        self.settings = settings
        settings.ensure_dirs()
        configure_logging(settings.log_level, settings.log_dir / f"api-{settings.namespace}.log")
        self.secrets = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
        ensure_token(settings.keys_dir, OPERATOR_TOKEN_FILE)
        ensure_token(settings.keys_dir, ENGINE_TOKEN_FILE)
        ensure_token(settings.keys_dir, AGENT_TOKEN_FILE)
        self.db = Database(settings.db_path)
        self.bus = EventBus()
        self.tailer = DbTailer(self.db, self.bus)
        self.started_at = datetime.now(UTC)
        self.startup_phase = "STARTING"
        self.startup_detail = ""

        # ---- mode / gateway decision (fail closed, audited)
        modes.assert_gateway_allowed(
            settings.mode, settings.gateway
        ) if settings.mode != modes.TradingMode.RESEARCH_ONLY else None
        self.execution_label: ExecutionLabel = modes.execution_label(settings.mode, settings.gateway)
        self.feed_mode: FeedMode = settings.feed_mode or (
            FeedMode.IEX_FREE if self.secrets.alpaca_paper_available() else FeedMode.SIMULATED
        )

        # ---- market data
        self.store = MarketDataStore(settings.datasets_dir)
        self.clock: Clock
        self.sim_clock: SimulationClock | None = None
        self.dataset_version: str = "none"
        self.provider: Any
        self._build_market_data()
        from .activity import ActivityLog

        self.activity = ActivityLog(self.db, self.bus, self.clock)
        self._cycle_lock = asyncio.Lock()
        # Serializes "capture the single-use final-gate context + submit" across decision cycles, agent proposals
        # and the position manager (each holds it only around prepare_gate + ExecutionService.submit).
        self.submit_lock = asyncio.Lock()
        self.feed_monitor = FeedQualityMonitor(self.feed_mode, self.provider_name, self.clock,
                                               symbols=tuple(settings.symbols))

        # ---- risk + ledger + execution
        self.signer = IntentSigner.load_or_create(settings.keys_dir)
        self.verifier = self.signer.verifier
        self.policy_store = PolicyStore(self.db, settings.mode)
        self.policy_store.ensure_default()
        self.ledger = OrderLedger(self.db, self.clock, settings.mode)
        self.entries = EntriesLock(self.db, settings.mode)
        self._build_risk()
        self._build_execution()
        self._build_positions()
        self._build_research()
        self._build_sessions()
        self._record_mode_decision()

    # ------------------------------------------------------------------ builders
    def _build_market_data(self) -> None:
        from services.market_data.replay import ReplayMarketDataProvider

        if self.feed_mode == FeedMode.SIMULATED:
            from services.market_data.fixtures import ensure_synthetic_fixture

            manifest = ensure_synthetic_fixture(self.settings.datasets_dir, symbols=self.settings.symbols,
                                                store=self.store)
            self.dataset_version = str(manifest["dataset_version"])
            self.data_feed: DataFeed | None = DataFeed.SYNTHETIC
            sessions = _sessions_from_manifest(manifest)
            self.sim_clock = SimulationClock(sessions, start_session=max(len(sessions) - 3, 0), speed=30.0)
            self.clock = self.sim_clock
            self.provider = ReplayMarketDataProvider(self.store, self.clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED,
                                                     dataset_version=self.dataset_version)
            self.provider_name = f"replay:{self.dataset_version}"
        else:
            from services.market_data.alpaca_provider import AlpacaMarketDataProvider

            self.clock = SystemClock()
            self.data_feed = DataFeed.IEX if self.feed_mode == FeedMode.IEX_FREE else DataFeed.SIP
            self.provider = AlpacaMarketDataProvider(self.secrets, self.feed_mode, self.clock)
            self.provider_name = f"alpaca:{'iex' if self.feed_mode == FeedMode.IEX_FREE else 'sip'}"
            self.dataset_version = f"live-{self.feed_mode.value.lower()}"

    def _build_risk(self) -> None:
        from services.risk.engine import RiskEngine

        self.policy: RiskPolicy = self.policy_store.active()
        self.risk = RiskEngine(self.policy, self.signer, self.clock)

    def _quote_source(self, symbol: str) -> Any:
        """Latest point-in-time quote (available_at <= now) for the fake broker's simulated fills."""
        sync = getattr(self.provider, "latest_quote_asof", None)
        if sync is not None:
            return sync(symbol, self.clock.now())
        return _run_sync(self.provider.get_latest_quote(symbol))

    def _build_execution(self) -> None:
        from services.order_ledger.execution import ExecutionService
        from services.order_ledger.gateway_factory import build_gateway
        from services.order_ledger.reconcile import Reconciler

        self.gateway = build_gateway(
            self.settings, self.secrets, ledger=self.ledger, clock=self.clock, quote_source=self._quote_source
        )
        self.execution = ExecutionService(
            self.ledger, self.gateway, self._final_gate, self.clock, submit_timeout_s=10.0, entries_lock=self.entries
        )
        self.reconciler = Reconciler(self.db, self.ledger, self.gateway, self.entries, self.clock)
        self.execution.on_reconnect = self.reconcile

    def _build_positions(self) -> None:
        from services.positions.manager import PositionManager
        from services.positions.policy import PositionPolicy
        from services.positions.store import PositionStore

        self.position_policy = PositionPolicy()
        self.position_store = PositionStore(self.db, self.settings.mode.value)
        self.positions = PositionManager(RuntimePositionPort(self), self.position_store, self.position_policy)

    def _build_research(self) -> None:
        from services.ai.pipeline import ResearchPipeline
        from services.research.costs import CostModel
        from services.research.signals import QuantSignalEngine

        self.cost_model = CostModel()
        self.signal_engine = QuantSignalEngine(self.cost_model, clock=self.clock)
        self.ai = ResearchPipeline(
            self.settings, self.secrets, transport=self.settings.ai_transport, db=self.db, clock=self.clock
        )

    def _record_mode_decision(self) -> None:
        detail = (
            f"mode={self.settings.mode} gateway={self.settings.gateway} label={self.execution_label} "
            f"feed={self.feed_mode} live_compiled={modes.LIVE_EXECUTION_COMPILED}"
        )

        def _w(conn: Any) -> None:
            conn.execute(
                "INSERT INTO mode_transitions(at, from_mode, to_mode, actor, result, detail) VALUES (?,?,?,?,?,?)",
                (utc_iso(), None, self.settings.mode.value, "process-start", "ACCEPTED", detail),
            )
            append_audit(conn, event_type="RUNTIME_STARTED", source="api", mode=self.settings.mode.value,
                         payload={"detail": detail, "dataset_version": self.dataset_version,
                                  "position_policy": self.position_policy.model_dump(mode="json"),
                                  "position_policy_version": self.position_policy.version},
                         policy_version=self.policy.version, dataset_version=self.dataset_version)

        self.db.write(_w)

    # ------------------------------------------------------------------ time helpers
    def now(self) -> datetime:
        return self.clock.now()

    def now_iso(self) -> str:
        return utc_iso(self.now())

    # ------------------------------------------------------------------ health / context
    def entries_status(self) -> tuple[bool, list[str]]:
        locked, reasons = self.entries.is_locked()
        reasons = list(reasons)
        pnl = get_state(self.db, "daily_pnl") or {}
        if pnl.get("trading_day") == trading_day(self.now()).isoformat() and pnl.get("available") is False:
            reasons.append("daily P&L unavailable: " + "; ".join(pnl.get("errors") or ["unverified opening marks"]))
        return bool(locked or reasons), reasons

    def engine_health(self) -> tuple[bool, str, str | None]:
        if self.settings.gateway == GatewayKind.FAKE:
            return True, "in-process deterministic fake broker (SIMULATED)", None
        if self.settings.gateway == GatewayKind.ALPACA_PAPER:
            return True, "Alpaca SDK adapter; broker connectivity checked by reconciliation", None
        hb = get_state(self.db, "engine_heartbeat")
        if not hb:
            return False, "no LEAN heartbeat received", None
        ts = hb.get("ts") or hb.get("received_at")
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))).total_seconds()
        except ValueError:
            return False, "unparseable LEAN heartbeat", None
        return age < 15, f"LEAN heartbeat {age:.0f}s ago ({hb.get('status')})", str(ts)

    def health(self) -> SystemHealth:
        locked, reasons = self.entries_status()
        exits_locked, exit_reasons = self.entries.exits_locked()
        feed = self.feed_monitor.status()
        feed_ok = feed.get("status") == "OK"
        engine_ok, engine_detail, _ = self.engine_health()
        last = get_state(self.db, "last_reconciliation") or {}
        reconciled = last.get("status") == ReconciliationStatus.CLEAN.value
        broker_ok = bool(get_state(self.db, "broker_connected", False))
        return SystemHealth(
            feed_ok=feed_ok, feed_detail=str(feed.get("status")),
            broker_connected=broker_ok, broker_detail=str(get_state(self.db, "broker_detail", "")),
            reconciled=reconciled, reconcile_detail=str(last.get("status")),
            engine_ok=engine_ok, engine_detail=engine_detail,
            entries_locked=locked, lock_reasons=tuple(reasons),
            exits_locked=exits_locked, exit_lock_reasons=tuple(exit_reasons),
        )

    async def risk_context(self, market: Any) -> RiskContext:
        now = self.now()
        try:
            account = await self.gateway.get_account()
            positions = await self.gateway.get_positions()
            self._set_broker(True, "ok")
        except Exception as exc:  # fail closed: unknown account state blocks entries
            account, positions = None, []
            self._set_broker(False, f"{type(exc).__name__}: {exc}")
        day = trading_day(now)
        sod = get_state(self.db, f"sod_equity:{day.isoformat()}")
        if sod is None and account is not None:
            baseline = account.last_equity if account.last_equity is not None else account.equity
            sod = format(baseline, "f")
            self.db.write(lambda c: set_state(c, f"sod_equity:{day.isoformat()}", sod))
        realized, unreal, pnl_errors = self._daily_position_pnl(day, positions)
        self.db.write(lambda c: set_state(c, "daily_pnl", {
            "daily_realized": format(realized, "f"), "daily_unrealized": format(unreal, "f"),
            "start_of_day_equity": sod, "trading_day": day.isoformat(), "asof": self.now_iso(),
            "available": not pnl_errors and account is not None, "errors": pnl_errors,
            "basis": "previous-close carried market value plus booked intraday cash flows"}))
        return RiskContext(
            now=now, mode=self.settings.mode, gateway=self.settings.gateway, account=account,
            positions=tuple(positions), open_orders=tuple(self.ledger.open_orders()),
            recent_submission_times=tuple(self.ledger.recent_submission_times(now - timedelta(days=1))),
            recent_intent_fingerprints=tuple(
                self.ledger.recent_fingerprints(now - timedelta(seconds=self.policy.duplicate_window_seconds))
            ),
            start_of_day_equity=Decimal(sod) if sod else None,
            daily_realized_pl=realized, daily_unrealized_pl=unreal, market=market, health=self.health(),
        )

    def _daily_position_pnl(self, day: date, positions: list[PositionSnapshot]) -> tuple[Decimal, Decimal, list[str]]:
        """Rebase overnight holdings at the previous close, including shares sold during this session.

        Lifetime cost-basis gains on an overnight SELL must not mask today's losses. The broker's intraday P&L
        seeds a durable opening market value; cash flows then preserve that reference after partial/full exits
        and process restarts. A missing or inconsistent reference blocks entries through ``entries_status``.
        Protective exits stay available. No broker or ledger history is rewritten.
        """
        zero = Decimal(0)
        fills = self.db.query("SELECT symbol, side, quantity, price FROM fills WHERE trading_day=?", (day.isoformat(),))
        flow: dict[str, dict[str, Decimal]] = {}
        for fill in fills:
            item = flow.setdefault(fill["symbol"], {"buy_qty": zero, "sell_qty": zero, "buys": zero, "sells": zero})
            qty, price = Decimal(fill["quantity"]), Decimal(fill["price"])
            if fill["side"] == Side.BUY.value:
                item["buy_qty"] += qty
                item["buys"] += qty * price
            else:
                item["sell_qty"] += qty
                item["sells"] += qty * price
        key = f"daily_position_marks:{day.isoformat()}"
        marks = get_state(self.db, key) or {}
        held = {p.symbol: p for p in positions if p.quantity != zero}
        ledger_qty = self.ledger.position_quantities()
        errors: list[str] = []
        total, unreal = zero, zero
        for symbol in sorted(set(held) | set(flow) | set(marks) | set(ledger_qty)):
            p = held.get(symbol)
            qty = p.quantity if p is not None else zero
            if qty != ledger_qty.get(symbol, zero):
                errors.append(f"{symbol}: broker and ledger quantities differ")
                continue
            f = flow.get(symbol, {"buy_qty": zero, "sell_qty": zero, "buys": zero, "sells": zero})
            carried = qty - f["buy_qty"] + f["sell_qty"]
            if carried < zero or qty < zero:
                errors.append(f"{symbol}: unsupported negative opening/current quantity")
                continue
            value = p.market_value if p is not None else zero
            if value is None:
                errors.append(f"{symbol}: current market value unavailable")
                continue
            opening_value = zero
            if carried > zero:
                if p is not None and p.unrealized_intraday_pl is None:
                    errors.append(f"{symbol}: overnight position has no broker intraday P&L")
                    continue
                mark = marks.get(symbol)
                if mark is None and p is not None and p.unrealized_intraday_pl is not None:
                    if f["sell_qty"] == zero:
                        opening_value = value - p.unrealized_intraday_pl - f["buys"]
                    elif f["buy_qty"] == zero:
                        opening_value = (value - p.unrealized_intraday_pl) / qty * carried
                    else:
                        errors.append(f"{symbol}: opening mark unavailable after mixed intraday buys/sells")
                        continue
                    if opening_value <= zero:
                        errors.append(f"{symbol}: invalid previous-close market value")
                        continue
                    mark = {"quantity": format(carried, "f"), "market_value": format(opening_value, "f"),
                            "source": "broker_unrealized_intraday_pl", "captured_at": self.now_iso()}
                    marks[symbol] = mark
                if mark is None:
                    errors.append(f"{symbol}: opening mark unavailable after full overnight exit")
                    continue
                if Decimal(mark["quantity"]) != carried:
                    errors.append(f"{symbol}: opening quantity changed; mark cannot be trusted")
                    continue
                opening_value = Decimal(mark["market_value"])
            total += value + f["sells"] - f["buys"] - opening_value
            if p is not None:
                intraday = p.unrealized_intraday_pl if carried > zero else p.unrealized_pl
                if intraday is None:
                    errors.append(f"{symbol}: unrealized P&L unavailable")
                else:
                    unreal += intraday
        self.db.write(lambda c: set_state(c, key, marks))
        # Unknown contributions never become an authoritative zero: the availability flag locks new entries.
        return total - unreal, unreal, errors

    def _set_broker(self, ok: bool, detail: str) -> None:
        def _w(c: Any) -> None:
            set_state(c, "broker_connected", ok)
            set_state(c, "broker_detail", detail)

        self.db.write(_w)

    def _final_gate(self, intent: Any) -> list[str]:
        """Final order-boundary check (signature, expiry, mode, active policy, health, halts, duplicates).

        Uses the risk context captured immediately before submission (``prepare_gate``); without a fresh
        context the gate fails closed.
        """
        ctx = self._gate_ctx
        self._gate_ctx = None  # single use
        if ctx is None:
            return ["no fresh risk context captured for the final gate"]
        if abs((self.now() - ctx.now).total_seconds()) > 5:
            return ["risk context for the final gate is stale"]
        active = self.policy_store.active().version
        row = self.db.query_one("SELECT payload_json FROM proposals WHERE proposal_id=?", (intent.proposal_id,))
        proposal = None
        if row is not None:
            with contextlib.suppress(ValueError):
                proposal = TradeProposal.model_validate_json(row["payload_json"])
        if proposal is None and intent.side == Side.BUY:
            return ["persisted BUY proposal missing or invalid at the final gate"]
        return list(self.risk.final_gate(intent, ctx, self.verifier, active, proposal=proposal,
                                        strategy_id=proposal.strategy_id if proposal is not None else None))

    async def prepare_gate(self, market: Any) -> RiskContext:
        self._gate_ctx = await self.risk_context(market)
        return self._gate_ctx

    _gate_ctx: RiskContext | None = None

    # ------------------------------------------------------------------ views
    def status(self) -> dict[str, Any]:
        locked, reasons = self.entries_status()
        exits_locked, exit_reasons = self.entries.exits_locked()
        feed = self.feed_monitor.status()
        engine_ok, engine_detail, hb_ts = self.engine_health()
        last = get_state(self.db, "last_reconciliation") or {}
        status = ExecutionStatus(
            mode=self.settings.mode, label=self.execution_label, gateway=self.settings.gateway,
            feed_mode=self.feed_mode, live_execution_compiled=modes.LIVE_EXECUTION_COMPILED,
        )
        return {
            "mode": status.mode.value,
            "execution_label": status.label.value,
            "gateway": status.gateway.value,
            "live_execution_compiled": status.live_execution_compiled,
            "entries_locked": locked,
            "lock_reasons": reasons,
            "exits_locked": exits_locked,
            "exit_lock_reasons": exit_reasons,
            "position_safety": self.entries.position_safety(),
            "halts": active_halts(self.db),
            "startup": {"phase": self.startup_phase, "started_at": utc_iso(self.started_at), "detail": self.startup_detail},
            "feed": {**feed, "simulated_time": self.now_iso() if self.sim_clock else None,
                     "dataset_version": self.dataset_version},
            "engine": {"kind": self.settings.gateway.value, "status": "UP" if engine_ok else "DOWN",
                       "last_heartbeat": hb_ts, "detail": engine_detail},
            "broker": {"kind": self.settings.gateway.value, "is_paper": True,
                       "connected": bool(get_state(self.db, "broker_connected", False)),
                       "last_reconciled_at": last.get("finished_at"),
                       "reconciliation_status": last.get("status")},
            "credentials": {
                "alpaca_paper": self.secrets.alpaca_paper_available(),
                "anthropic": self.secrets.has(SecretName.ANTHROPIC_API_KEY),
                "openai": self.secrets.has(SecretName.OPENAI_API_KEY),
            },
            "ai": {"transport": self.settings.ai_transport.upper(), "claude_model": self.settings.claude_model,
                   "sol_model": self.settings.sol_model},
            "policy_version": self.policy.version,
            "server_time": utc_iso(datetime.now(UTC)),
        }

    def market_bars(self, symbol: str, timeframe: str, limit: int) -> dict[str, Any]:
        now = self.now()
        start = now - (timedelta(days=5) if timeframe.endswith("Min") else timedelta(days=400))
        bars = self.store.load_bars([symbol], start, now, timeframe, feed=self.data_feed, asof=now)[-limit:]
        feed = bars[-1].feed.value if bars else ("SYNTHETIC" if self.feed_mode == FeedMode.SIMULATED else "IEX")
        return {
            "symbol": symbol, "timeframe": timeframe, "feed": feed, "feed_mode": self.feed_mode.value,
            "is_simulated": self.feed_mode == FeedMode.SIMULATED or any(b.is_synthetic for b in bars),
            "source_id": bars[-1].source_id if bars else None,
            "dataset_version": bars[-1].dataset_version if bars else self.dataset_version,
            "limitations": list(self.feed_monitor.status().get("limitations", [])),
            "simulated_time": self.now_iso() if self.sim_clock else None,
            "bars": [{"time": int(b.start_ts.timestamp()), "open": format(b.open, "f"), "high": format(b.high, "f"),
                      "low": format(b.low, "f"), "close": format(b.close, "f"), "volume": format(b.volume, "f"),
                      "available_at": utc_iso(b.available_at_ts)} for b in bars],
        }

    def market_quality(self) -> dict[str, Any]:
        from . import queries

        st = self.feed_monitor.status()
        return {**st, "events": queries.data_quality_events(self.db)}

    def risk_view(self) -> dict[str, Any]:
        from . import queries

        pol = self.policy
        util: list[dict[str, Any]] = []
        locked, _ = self.entries_status()
        open_orders = self.ledger.open_orders()
        positions = self.ledger.positions()
        def _u(name: str, used: Decimal, limit: Decimal) -> dict[str, Any]:
            pct = (used / limit * 100) if limit else Decimal(0)
            return {
                "name": name,
                "used": format(used, "f"),
                "limit": format(limit, "f"),
                "pct": format(pct.quantize(Decimal("0.1")), "f"),
            }

        util.append(_u("pending_orders", Decimal(len(open_orders)), Decimal(pol.max_pending_orders)))
        util.append(_u("positions", Decimal(len([p for p in positions if p.quantity != 0])), Decimal(pol.max_positions)))
        pos_notional = sum((p.cost_basis for p in positions), Decimal(0))
        util.append(_u("position_notional", pos_notional, pol.max_position_notional))
        pnl = get_state(self.db, "daily_pnl") or {}
        loss = -(Decimal(pnl.get("daily_realized", "0")) + Decimal(pnl.get("daily_unrealized", "0")))
        loss_util = _u("daily_loss", max(loss, Decimal(0)), pol.max_daily_loss)
        loss_util["available"] = pnl.get("available", True)
        loss_util["errors"] = pnl.get("errors", [])
        util.append(loss_util)
        today = self.now() - timedelta(days=1)
        util.append(_u("orders_today", Decimal(len(self.ledger.recent_submission_times(today))), Decimal(pol.max_orders_per_day)))
        return {
            "policy": pol.model_dump(mode="json"), "policy_version": pol.version,
            "history": queries.policy_history(self.db), "utilization": util,
            "entries_locked": locked, "recent_decisions": queries.risk_decisions(self.db),
        }

    # ------------------------------------------------------------------ control
    def halt(self, reason: str, source: str, scope: str = "ENTRIES") -> str:
        """ENTRIES (default): stop new entries, protective exits continue. ALL: kill switch, exits blocked too."""
        return self.db.write(lambda c: add_halt(c, reason=reason, source=source, mode=self.settings.mode.value,
                                                scope=scope))

    def resume(self, halt_id: str, reason: str, actor: str) -> dict[str, Any]:
        last = get_state(self.db, "last_reconciliation") or {}
        if last.get("status") != ReconciliationStatus.CLEAN.value:
            from fastapi import HTTPException

            raise HTTPException(409, "resume requires a CLEAN broker reconciliation first (POST /control/reconcile)")
        ok = self.db.write(lambda c: clear_halt(c, halt_id, by=actor, reason=reason, mode=self.settings.mode.value))
        locked, reasons = self.entries_status()
        return {"cleared": ok, "entries_locked": locked, "lock_reasons": reasons}

    def apply_policy(self, policy: RiskPolicy, operator_token: str, reason: str) -> str:
        version = self.policy_store.apply_change(policy, operator_token=operator_token, actor="operator", reason=reason)
        self._build_risk()
        return version

    async def reconcile(self) -> ReconciliationReport:
        report: ReconciliationReport = await self.reconciler.reconcile()
        self._after_reconcile(report)
        return report

    def _after_reconcile(self, report: ReconciliationReport) -> None:
        self.db.write(lambda c: set_state(c, "last_reconciliation", {
            "run_id": report.run_id, "status": report.status.value, "finished_at": utc_iso(report.finished_at)}))
        self.bus.publish("reconciliation", report.model_dump(mode="json"))

    async def cancel(self, client_order_id: str) -> dict[str, Any]:
        await self.execution.cancel(client_order_id)
        order = self.ledger.get_order(client_order_id)
        return {"client_order_id": client_order_id, "state": getattr(order, "state", None)}

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str,
                        session_id: str | None = None) -> dict[str, Any]:
        from .pipeline import DecisionCycle

        async with self._cycle_lock:  # one decision cycle at a time (sessions, agents and CLI share it)
            cycle = DecisionCycle(self, session_id=session_id)
            result = await cycle.run(symbols=symbols or list(self.settings.symbols), use_ai=use_ai, actor=actor)
        self.bus.publish("cycle", result)
        return result

    # ------------------------------------------------------------------ pipeline helpers
    def quote_spreads_bps(self, symbols: list[str]) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for sym in symbols:
            try:
                q = self._quote_source(sym)
            except Exception:
                q = None
            if q is not None:
                self.feed_monitor.observe_quote(q)
                out[sym] = q.spread_bps
        return out

    _engine_registry: str | None = None

    def current_signal_engine(self) -> Any:
        """Quant signal engine, upgraded to the walk-forward-selected model suite when a valid registry exists."""
        from services.research.autonomous.paper import selected_paper_model
        from services.research.signals import QuantSignalEngine

        svc = selected_paper_model(self.settings.experiments_dir, self.settings.mode.value, self.settings.gateway.value)
        if svc is None:
            svc = self._prediction_service()
        version = getattr(svc, "registry_version", None) or getattr(svc, "model_version", None)
        if version != self._engine_registry or getattr(self, "signal_engine", None) is None:
            self.signal_engine = QuantSignalEngine(self.cost_model, model=svc, clock=self.clock)
            self._engine_registry = version
        return self.signal_engine

    def model_suite_active(self) -> bool:
        return self._prediction_service() is not None

    def prediction_for(self, symbol: str | None, bars_df: Any) -> Any:
        if symbol is None:
            return None
        if str(getattr(self.signal_engine, "model_version", "")).startswith("autonomous:"):
            return None  # do not attach a different suite's predictions to an autonomous candidate
        svc = self._prediction_service()
        if svc is None:
            return None
        try:
            preds = svc.predict(bars_df, self.now(), [symbol])
        except Exception:
            return None
        self._record_predictions(preds)
        return preds[0] if preds else None

    async def ai_review(self, *, causal_id: str, candidate: Any, snapshot: Any, evidence: list[Any]) -> Any:
        """Model review of a candidate in modes where the app itself calls a model (API_KEYS, CHATGPT_OAUTH)."""
        from packages.contracts.enums import AiRuntimeMode

        if self.settings.ai_runtime == AiRuntimeMode.CHATGPT_OAUTH:
            from services.tools.backends import InProcessToolBackend

            return await self._ai_rt().chatgpt_research(
                candidate.symbol, tool_backend=InProcessToolBackend(self, actor="chatgpt"), candidate=candidate,
                snapshot=snapshot, evidence=evidence, policy=self.policy, causal_id=causal_id)
        return await self.ai.run(causal_id=causal_id, candidate=candidate, snapshot=snapshot, evidence=evidence,
                                 policy=self.policy)

    def health_report(self) -> dict[str, Any]:
        comps: list[dict[str, Any]] = []

        def add(name: str, status: str, detail: str, last_ok: str | None = None) -> None:
            comps.append({"name": name, "status": status, "detail": detail, "last_ok": last_ok})

        feed = self.feed_monitor.status()
        add("market_data", "OK" if feed.get("status") == "OK" else "DEGRADED",
            f"{feed.get('status')} {self.feed_mode.value} {self.provider_name}", feed.get("last_quote_ts"))
        broker_ok = bool(get_state(self.db, "broker_connected", False))
        add("broker", "OK" if broker_ok else "DOWN", f"{self.settings.gateway.value}: {get_state(self.db, 'broker_detail', '')}")
        last = get_state(self.db, "last_reconciliation") or {}
        add("reconciliation", "OK" if last.get("status") == "CLEAN" else "DEGRADED", str(last.get("status")),
            last.get("finished_at"))
        svc = self._prediction_service()
        add("models", "OK" if svc is not None else "DEGRADED",
            f"registry {svc.registry_version}" if svc is not None else "no valid model registry (quant baseline in use)")
        ai = self.ai_runtime_status()
        mode = ai.get("mode")
        ai_status = "DISABLED" if mode in ("NONE", "CLAUDE_CODE", "CODEX") else ("DEGRADED" if ai.get("error") else "OK")
        add("gpt_runtime", ai_status, f"mode {mode}" + (f": {ai['error']}" if ai.get("error") else ""))
        add("event_stream", "OK", f"{self.bus.subscriber_count} subscribers")
        try:
            row = self.db.query_one("PRAGMA quick_check")
            ok = bool(row) and next(iter(row.values())) == "ok"
            add("database", "OK" if ok else "DOWN", f"SQLite WAL {self.settings.db_path.name}: quick_check {'ok' if ok else row}")
        except Exception as exc:
            add("database", "DOWN", f"{type(exc).__name__}: {exc}")
        agent = self.db.query_one(
            "SELECT ts, source FROM audit_events WHERE event_type='AGENT_TOOL_CALL' ORDER BY seq DESC LIMIT 1"
        )
        add("agent", "OK" if agent else "DISABLED", f"last tool call {agent['ts']} by {agent['source']}" if agent
            else "no agent connected yet (sessions run without it)", agent["ts"] if agent else None)
        engine_ok, engine_detail, hb = self.engine_health()
        add("engine", "OK" if engine_ok else "DOWN", engine_detail, hb)
        sched = getattr(self, "sessions", None)
        sched_status = sched.scheduler_status() if sched is not None else {}
        sched_ok = bool(sched_status.get("loop_running") and sched_status.get("recovered"))
        add("scheduler", "OK" if sched_ok else "DEGRADED",
            "session scheduler running; startup recovery complete" if sched_ok
            else "session scheduler stopped or awaiting startup recovery")
        worst = "DOWN" if any(c["status"] == "DOWN" and c["name"] in ("database", "broker") for c in comps) else (
            "DEGRADED" if any(c["status"] in ("DOWN", "DEGRADED") for c in comps) else "OK")
        return {"overall": worst, "components": comps, "asof": utc_iso(datetime.now(UTC))}

    # ------------------------------------------------------------------ sessions + performance (services.sessions)
    sessions: Any = None

    def _build_sessions(self) -> None:
        try:
            from services.sessions.manager import SessionManager
        except ImportError:
            self.sessions = None
            return
        self.sessions = SessionManager(self)

    def performance(self) -> dict[str, Any]:
        from services.sessions.performance import compute_performance

        return compute_performance(self.db, now=self.now(), label=self.execution_label.value,
                                   is_simulated=self.execution_label.value == "SIMULATED", gateway=self.settings.gateway.value)

    def sessions_view(self, limit: int = 20) -> dict[str, Any]:
        if self.sessions is None:
            return {"active": None, "sessions": [], "schedules": [], "detail": "session manager not installed"}
        return {"active": self.sessions.active(), "sessions": self.sessions.list(limit),
                "schedules": self.sessions.list_schedules()}

    def session_report(self, session_id: str) -> dict[str, Any] | None:
        return None if self.sessions is None else self.sessions.report(session_id)

    # ------------------------------------------------------------------ tool-runtime facade
    # Used by services.tools handlers (MCP server via HTTP, in-app AI runtimes in-process). Read methods
    # return the same shapes as the public API; the only mutating paths go through the risk engine.
    def portfolio(self) -> dict[str, Any]:
        from . import queries

        return queries.portfolio(self.db, self.execution_label.value)

    def orders(self, limit: int = 100) -> dict[str, Any]:
        from . import queries

        return queries.orders(self.db, limit)

    def order_detail(self, client_order_id: str) -> dict[str, Any] | None:
        from . import queries

        return queries.order_detail(self.db, client_order_id)

    def signals(self, limit: int = 50) -> dict[str, Any]:
        from . import queries

        return queries.signals(self.db, limit)

    async def fresh_snapshot(self, symbol: str) -> Any:
        """Point-in-time snapshot for a decision right now (replay cutoffs in REPLAY), recorded in the ledger."""
        if self.sim_clock is None:
            snapshot = await self.market_snapshot(symbol)
        else:
            from services.market_data.snapshot import build_market_snapshot

            snapshot = await build_market_snapshot(self.provider, symbol, self.now(), self.feed_mode,
                                                   dataset_version=self.dataset_version)
        self.ledger.record_market_snapshot(snapshot)
        return snapshot

    def quantlab_view(self, kind: str, family: str | None = None) -> dict[str, Any]:
        from services.quantlab.views import view

        return view(self.settings.project_root, kind, family)

    def quantlab_experiment(self, spec: dict[str, Any], actor: str) -> dict[str, Any]:
        from services.quantlab.views import run_agent_experiment

        return run_agent_experiment(self.settings.project_root, spec, actor)

    def positions_view(self) -> dict[str, Any]:
        return {"execution_label": self.execution_label.value, "is_simulated": self.sim_clock is not None
                or self.execution_label == ExecutionLabel.SIMULATED, **self.positions.view()}

    async def market_snapshot(self, symbol: str) -> Any:
        from services.market_data.snapshot import build_market_snapshot

        observed = None
        if self.sim_clock is None:
            # Receive live data first, then establish the decision cutoff. Replay cutoffs never move.
            with contextlib.suppress(Exception):
                observed = await self.provider.get_latest_quote(symbol)
        snapshot = await build_market_snapshot(self.provider, symbol, self.now(), self.feed_mode,
                                              dataset_version=self.dataset_version, observed_quote=observed)
        if snapshot.quote is not None:
            self.feed_monitor.observe_quote(snapshot.quote)
        return snapshot

    def predictions(self, symbol: str | None = None) -> dict[str, Any]:
        """Latest point-in-time predictions from the walk-forward-selected local model suite."""
        svc = self._prediction_service()
        now = self.now()
        if svc is None:
            return {"predictions": [], "asof": utc_iso(now), "is_simulated": self.sim_clock is not None,
                    "detail": "no valid trained model registry (make models)"}
        universe = [symbol] if symbol else list(self.settings.symbols)
        # regime proxy symbols are always loaded so a single-symbol request still gets a regime estimate
        load = sorted(set(universe) | set(getattr(svc, "regime_symbols", [])))
        literal_har = (svc_meta := getattr(getattr(svc, "registry", None), "meta", {}) or {}) and \
            svc_meta.get("targets", {}).get("vol", {}).get("har", {}).get("base_bars") == 390
        lookback = timedelta(days=35 if literal_har else 10)
        bars_df = self.store.load_bars_df(load, now - lookback, now, "1Min", feed=self.data_feed, asof=now,
                                          dataset_version=self.dataset_version if self.sim_clock else None)
        preds = svc.predict(bars_df, now, universe)
        self._record_predictions(preds)
        return {"predictions": [p.model_dump(mode="json") for p in preds], "asof": utc_iso(now),
                "is_simulated": self.sim_clock is not None, "registry_version": svc.registry_version}

    def _record_predictions(self, predictions: Any) -> None:
        from packages.core.canonical import canonical_json
        def record(conn: Any) -> None:
            for prediction in predictions:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body) VALUES(?,?,?,?)",
                    ("predictions", prediction.prediction_id, utc_iso(prediction.asof),
                     canonical_json(prediction.model_dump(mode="json"))))
        if predictions:
            self.db.write(record)

    def search_memory(self, query: str, *, symbol: str | None = None,
                      kind: str | None = None, limit: int = 10) -> dict[str, Any]:
        from services.memory.store import Memory
        return Memory(self.db.path.parent / "memory.sqlite3").search(
            query, asof=self.now().isoformat(), symbol=symbol, kind=kind, limit=limit)

    def model_report(self) -> dict[str, Any]:
        try:
            from services.research.models.registry import load_registry
        except ImportError:
            return {"registry_version": None, "targets": [], "notes": ["model suite not installed"]}
        reg = load_registry(self.settings.experiments_dir)
        report = reg.report() if reg is not None else {"registry_version": None, "targets": [],
                                                      "notes": ["no trained registry (make models)"]}
        from services.research.autonomous.engine import snapshot

        try:
            research = snapshot(self.settings.experiments_dir / "autonomous")
            counts = research["counts"]
            note = (f"Autonomous research: {counts.get('COMPLETED', 0)} completed, "
                    f"{counts.get('RUNNING', 0)} running, {counts.get('FAILED', 0)} failed; "
                    f"promotion {research['promotion_status']}. Development results are not confirmed edge.")
        except Exception:
            note = "Autonomous research status unavailable; do not infer a promotion."
        report["notes"] = [note, *report.get("notes", [])]
        return report

    _pred_svc: Any = None

    def _prediction_service(self) -> Any:
        """Rebuild when a newer registry appears; a registry that fails verification yields None (fail closed)."""
        try:
            from services.research.models.predict import PredictionService
            from services.research.models.registry import list_registry_versions, load_registry
        except ImportError:
            return None
        versions = list_registry_versions(self.settings.experiments_dir)
        newest = versions[0] if versions else None
        if self._pred_svc is not None and newest == self._pred_svc.registry_version:
            return self._pred_svc
        reg = load_registry(self.settings.experiments_dir)
        self._pred_svc = PredictionService(reg, self.cost_model) if reg is not None else None
        return self._pred_svc

    def audit_tool_call(self, name: str, actor: str, mutating: bool, outcome: str, causal_id: str | None = None) -> None:
        self.db.write(lambda c: append_audit(c, event_type="AGENT_TOOL_CALL", source=f"agent:{actor}"[:80],
                                             mode=self.settings.mode.value, causal_id=causal_id,
                                             payload={"tool": name, "mutating": mutating, "outcome": outcome}))

    async def execute_proposal(self, proposal: Any, snapshot: Any) -> dict[str, Any]:
        """Risk-evaluate a proposal and, if approved, route the signed intent through the single order path."""
        self.ledger.record_proposal(proposal)
        out: dict[str, Any] = {"proposal": proposal.model_dump(mode="json"), "causal_id": proposal.causal_id}
        if proposal.action.value == "NO_TRADE":
            out.update(outcome="NO_TRADE", reason=proposal.no_trade_reason)
            return out
        ctx = await self.risk_context(snapshot)
        decision = self.risk.evaluate(proposal, ctx)
        self.ledger.record_decision(decision)
        out["decision"] = decision.model_dump(mode="json")
        if not decision.approved:
            out.update(outcome="REJECTED", reason="; ".join(decision.reasons))
            return out
        intent = self.risk.authorize(proposal, decision, ctx)
        self.ledger.record_intent(intent)
        out["client_order_id"] = intent.client_order_id
        if intent.requires_human_approval:
            out["outcome"] = "AWAITING_HUMAN_APPROVAL"
            return out
        async with self.submit_lock:
            await self.prepare_gate(snapshot)
            await self.execution.submit(intent)
        order = self.ledger.get_order(intent.client_order_id)
        state = getattr(getattr(order, "state", None), "value", getattr(order, "state", None))
        out["order"] = order.model_dump(mode="json") if order is not None else None
        out["outcome"] = "BLOCKED" if state in (None, "REJECTED") else "SUBMITTED"
        return out

    async def submit_agent_proposal(self, proposal: Any, actor: str) -> dict[str, Any]:
        # Sessions and concurrent agents share a single-use final-gate context. Serialize the
        # rate check, risk snapshot and submission with decision cycles, not just with other agents.
        async with self._cycle_lock:
            return await self._submit_agent_proposal(proposal, actor)

    async def _submit_agent_proposal(self, proposal: Any, actor: str) -> dict[str, Any]:
        from packages.contracts.enums import ExecutionLabel as _EL

        if self.execution_label == _EL.LIVE or self.settings.mode.value not in ("PAPER", "REPLAY"):
            return {"outcome": "REFUSED", "reason": "agent proposals are paper/replay only"}
        hour_ago = utc_iso(self.now() - timedelta(hours=1))
        n = (self.db.query_one("SELECT COUNT(*) n FROM proposals WHERE proposer='AGENT_TOOL' AND created_at >= ?",
                               (hour_ago,)) or {"n": 0})["n"]
        if n >= self.settings.agent_max_proposals_per_hour:
            return {"outcome": "RATE_LIMITED",
                    "reason": f"agent proposal limit {self.settings.agent_max_proposals_per_hour}/hour reached"}
        snapshot = await self.market_snapshot(str(proposal.symbol))
        self.ledger.record_market_snapshot(snapshot)
        result = await self.execute_proposal(proposal, snapshot)
        self.db.write(lambda c: append_audit(c, event_type="AGENT_PROPOSAL", source=f"agent:{actor}",
                                             mode=self.settings.mode.value,
                                             payload={k: result.get(k) for k in ("outcome", "reason", "client_order_id")},
                                             causal_id=proposal.causal_id, policy_version=self.policy.version))
        return result

    # ------------------------------------------------------------------ AI runtime modes
    _ai_runtime: Any = None

    def _ai_rt(self) -> Any:
        if self._ai_runtime is None:
            from services.ai.runtime_modes import AiRuntime

            self._ai_runtime = AiRuntime(self.settings, self.secrets, db=self.db, clock=self.clock)
        return self._ai_runtime

    def ai_runtime_status(self) -> dict[str, Any]:
        try:
            base: dict[str, Any] = dict(self._ai_rt().status())
        except Exception as exc:  # AI unavailable never affects the quant engine
            base = {"mode": self.settings.ai_runtime.value, "error": f"{type(exc).__name__}: {exc}"}
        base["quant_engine_independent"] = True
        try:
            from services.tools.registry import catalog

            tools = [{k: t.get(k) for k in ("name", "description", "category", "mutating")} for t in catalog()]
        except Exception:
            tools = []
        mcp_json = self.settings.project_root / ".mcp.json"
        codex_toml = self.settings.project_root / ".codex" / "config.toml"
        base["mcp"] = {
            "server": "local-ai-trader",
            "command": "uv run --frozen python -m services.mcp_server",
            "claude_code_registered": mcp_json.is_file() and "local-ai-trader" in mcp_json.read_text(),
            "codex_registered": codex_toml.is_file() and "local-ai-trader" in codex_toml.read_text(),
            "tools": tools,
        }
        hour_ago = utc_iso(self.now() - timedelta(hours=1))
        n = (self.db.query_one("SELECT COUNT(*) n FROM proposals WHERE proposer='AGENT_TOOL' AND created_at >= ?",
                               (hour_ago,)) or {"n": 0})["n"]
        base["limits"] = {"agent_max_proposals_per_hour": self.settings.agent_max_proposals_per_hour,
                          "agent_proposals_last_hour": n}
        return base

    async def chatgpt_login(self) -> dict[str, Any]:
        return dict(await self._ai_rt().chatgpt_login())

    async def chatgpt_logout(self) -> dict[str, Any]:
        await self._ai_rt().chatgpt_logout()
        return {"signed_in": False}

    async def chatgpt_research(self, symbol: str | None) -> dict[str, Any]:
        """One operator-initiated research turn on the ChatGPT plan; result goes through the normal risk path."""
        from packages.core.ids import new_causal_id
        from services.ai.evidence import build_internal_evidence
        from services.tools.backends import InProcessToolBackend

        causal_id = new_causal_id()
        now = self.now()
        universe = [symbol] if symbol else list(self.settings.symbols)
        bars_df = self.store.load_bars_df(universe, now - timedelta(days=10), now, "1Min", feed=self.data_feed, asof=now,
                                          dataset_version=self.dataset_version if self.sim_clock else None)
        signals = self.signal_engine.generate(bars_df, now, universe, causal_id)
        candidate = next((s for s in signals if s.is_candidate), None)
        snapshot = await self.market_snapshot(candidate.symbol if candidate else universe[0])
        self.ledger.record_market_snapshot(snapshot)
        evidence = build_internal_evidence(candidate, snapshot, self.policy) if candidate else []
        result = await self._ai_rt().chatgpt_research(
            symbol or (candidate.symbol if candidate else universe[0]),
            tool_backend=InProcessToolBackend(self, actor="chatgpt"), candidate=candidate, snapshot=snapshot,
            evidence=evidence, policy=self.policy, causal_id=causal_id)
        out = await self.execute_proposal(result.proposal, snapshot)
        out["model_runs"] = [r.model_dump(mode="json") for r in result.model_runs]
        return out

    # ------------------------------------------------------------------ engine bridge
    def bridge_next(self, engine_id: str, max_n: int) -> list[dict[str, Any]]:
        from services.order_ledger.bridge import claim_next_intents

        return claim_next_intents(self.ledger, engine_id, max_n)

    def bridge_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        from services.order_ledger.bridge import ingest_engine_event

        return ingest_engine_event(self.ledger, payload, gateway_kind=self.settings.gateway)

    def bridge_heartbeat(self, payload: dict[str, Any]) -> None:
        from services.order_ledger.bridge import record_heartbeat

        record_heartbeat(self.db, payload)

    def bridge_rejection(self, payload: dict[str, Any]) -> None:
        from services.order_ledger.bridge import record_engine_rejection

        record_engine_rejection(self.db, payload)

    # ------------------------------------------------------------------ lifecycle
    async def start(self, stop: asyncio.Event) -> list[asyncio.Task[Any]]:
        tasks: list[asyncio.Task[Any]] = []
        from services.memory.worker import run as run_memory
        tasks.append(asyncio.create_task(
            run_memory(self.db, self.settings.experiments_dir / "autonomous", stop), name="memory-projector"))
        tasks.append(asyncio.create_task(self.tailer.run(stop), name="db-tailer"))
        tasks.append(asyncio.create_task(self._startup(stop), name="startup-recovery"))
        tasks.append(asyncio.create_task(self._status_loop(stop), name="status-broadcast"))
        tasks.append(asyncio.create_task(self.execution.run_event_consumer(stop), name="order-events"))
        tasks.append(asyncio.create_task(
            self.positions.run(stop, interval_s=0.5 if self.sim_clock is not None else 1.0), name="position-manager"))
        if self.sim_clock is not None:
            tasks.append(asyncio.create_task(self.sim_clock.run(stop, self._on_sim_tick), name="sim-clock"))
        elif self.settings.data_pipeline_enabled:
            tasks.append(asyncio.create_task(self._data_pipeline_loop(stop), name="data-pipeline"))
        if self.sessions is not None:
            tasks.append(asyncio.create_task(self.sessions.run(stop), name="session-manager"))
        return tasks

    def _on_sim_tick(self, now: datetime) -> None:
        advance = getattr(self.gateway, "advance", None)
        for sym in self.settings.symbols:
            try:
                q = self._quote_source(sym)
            except Exception:
                q = None
            if q is not None:
                self.feed_monitor.observe_quote(q)
                if advance is not None:
                    advance(q)

    async def _startup(self, stop: asyncio.Event) -> None:
        self.startup_phase = "RECONCILING"
        try:
            report: ReconciliationReport = await self.reconciler.startup_recovery()
            self._after_reconcile(report)
            self.startup_phase = "READY" if report.status == ReconciliationStatus.CLEAN else "LOCKED"
            self.startup_detail = f"startup reconciliation {report.status.value}: {report.detail}"
            self.activity.activity("reconcile", f"Startup reconciliation {report.status.value}; entries "
                                   f"{'unlocked' if self.startup_phase == 'READY' else 'remain locked'}")
            if self.sessions is not None:  # automatic recovery only after broker-authoritative reconciliation
                await self.sessions.recover(clean=report.status == ReconciliationStatus.CLEAN)
        except Exception as exc:
            self.startup_phase = "LOCKED"
            self.startup_detail = f"startup reconciliation failed: {exc}"
            kind, detail = type(exc).__name__, str(exc)
            self.db.write(lambda c: record_failure(c, component="startup", kind=kind, detail=detail))
            log_event(log, "startup reconciliation failed", logging.ERROR, error=str(exc))
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.settings.reconcile_interval_s)
            if stop.is_set():
                break
            try:
                await self.reconcile()
            except Exception as exc:
                self.db.write(lambda c, e=exc: record_failure(c, component="reconcile", kind=type(e).__name__, detail=str(e)))

    async def _data_pipeline_loop(self, stop: asyncio.Event) -> None:
        """After every XNYS close (+ settle time) run one bounded history-pipeline pass, then hourly until the
        planned windows are exhausted. Real feeds only; never upgrades or purchases a feed."""
        from packages.contracts.policy import RiskPolicy as _Policy
        from services.market_data.pipeline import HistoryPipeline, PipelineConfig

        config = PipelineConfig(symbols=tuple(dict.fromkeys((*self.settings.symbols, *_Policy().symbol_allowlist))),
                                history_days=self.settings.data_pipeline_history_days,
                                max_requests=self.settings.data_pipeline_max_requests)
        pipe = HistoryPipeline(self.provider, self.store, self.db, self.clock, config)
        while not stop.is_set():
            try:
                pending = await asyncio.to_thread(pipe.plan)
                if pending:
                    run = await pipe.run()
                    self.activity.activity("info", f"Data pipeline: {len(run.done)} window(s), {run.rows_accepted} bars, "
                                           f"snapshot {run.snapshot_id}; research input {run.research.get('ready')}")
            except Exception as exc:
                self.db.write(lambda c, e=exc: record_failure(c, component="data_pipeline", kind=type(e).__name__,
                                                              detail=str(e)))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=3600)

    async def _status_loop(self, stop: asyncio.Event) -> None:
        tick = 0
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            if self.sim_clock is None:
                # Keep health current for every configured live symbol, independently of UI subscribers.
                quotes = await asyncio.gather(
                    *(self.provider.get_latest_quote(symbol) for symbol in self.settings.symbols),
                    return_exceptions=True,
                )
                for quote in quotes:
                    if isinstance(quote, Quote):
                        self.feed_monitor.observe_quote(quote)
            tick += 1
            self.bus.publish(
                "heartbeat", {"time": utc_iso(datetime.now(UTC)), "simulated_time": self.now_iso() if self.sim_clock else None}
            )
            if tick % 2 == 0 and self.bus.subscriber_count:
                with contextlib.suppress(Exception):
                    self.bus.publish("status", await asyncio.to_thread(self.status))

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            closer = getattr(self.gateway, "aclose", None)
            if closer is not None:
                await closer()
        self.db.close()


def build_runtime(settings: Settings) -> Runtime:
    return Runtime(settings)


class RuntimePositionPort:
    """Adapter giving the position manager exactly the runtime surface it needs (all orders via the risk engine)."""

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    @property
    def mode(self) -> Any:
        return self._rt.settings.mode

    @property
    def ledger(self) -> Any:
        return self._rt.ledger

    @property
    def entries(self) -> Any:
        return self._rt.entries

    @property
    def dataset_version(self) -> str:
        return self._rt.dataset_version

    def now(self) -> datetime:
        return self._rt.now()

    def risk_policy(self) -> RiskPolicy:
        return self._rt.policy

    async def snapshot(self, symbol: str) -> Any:
        return await self._rt.fresh_snapshot(symbol)

    async def submit_proposal(self, proposal: Any, snapshot: Any) -> dict[str, Any]:
        result = await self._rt.execute_proposal(proposal, snapshot)
        self._rt.activity.activity(
            "submit" if result.get("outcome") == "SUBMITTED" else "reject",
            f"Position manager {proposal.strategy_id}: {proposal.side} {proposal.quantity} {proposal.symbol} "
            f"{proposal.order_type.value} @ {proposal.limit_price}"
            + (f" stop {proposal.stop_price}" if proposal.stop_price is not None else "")
            + f" -> {result.get('outcome')}" + (f" ({result.get('reason')})" if result.get("reason") else ""),
            actor="position-manager", causal_id=proposal.causal_id)
        return result

    async def cancel_order(self, client_order_id: str) -> Any:
        return await self._rt.execution.cancel(client_order_id)

    async def request_reconcile(self) -> Any:
        return await self._rt.reconcile()


def _sessions_from_manifest(manifest: dict[str, Any]) -> list[tuple[datetime, datetime]]:
    """Regular-session (open, close) pairs of the synthetic fixture's XNYS trading days."""
    import exchange_calendars as xcals

    from services.market_data.fixtures import manifest_sessions

    cal = xcals.get_calendar("XNYS")
    out: list[tuple[datetime, datetime]] = []
    for day in manifest_sessions(manifest):
        out.append((cal.session_open(str(day)).to_pydatetime(), cal.session_close(str(day)).to_pydatetime()))
    if not out:
        raise RuntimeError("synthetic fixture manifest lists no sessions")
    return out


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code (worker threads). Inside a running loop thread, use a fresh loop thread."""
    if not asyncio.iscoroutine(coro):
        return coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result(timeout=30)
