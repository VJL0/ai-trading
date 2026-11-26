"""Replay engine: drives the real decision and execution stack over a bar snapshot on a simulated clock.

Per session (XNYS calendar: holidays and early closes from ``exchange_calendars``)::

    for every 5-minute bar:
        for every quote sample on the intrabar path (all symbols, time order; exact stop crossings inserted):
            clock -> sample time; quote published; FakeBroker matches resting orders; events -> ledger
            PositionManager.tick() when it has work (every sample) or on each minute (idle)
        clock -> bar end (the bar becomes available)
        decision cycle if the strategy decides at this bar (DecisionCycle -> risk -> signed intent -> ledger ->
        final gate -> broker; reconciliation after every cycle, as in the runtime)
        journal: tail the ledger tables, MARK (equity, exposure, resume cursors)
    at the close: DAY orders expire, the manager marks overnight holdings, reconciliation, SESSION_END

Nothing here decides or authorizes an order: the engine only moves the clock, publishes market data and calls the
runtime's own entry points (``run_cycle``, ``PositionManager.tick``, ``reconcile``, ``startup``).

Restart: a new engine on the same workspace with ``resume=True`` reopens the ledger and the FakeBroker state, runs
the runtime's startup recovery (SUBMITTING -> UNKNOWN, broker-authoritative reconciliation by client order id;
nothing is resubmitted) and continues at the first bar that starts after the last recorded activity. The journal
tail cursors come from the last ``MARK`` record, so records written to the ledger after it are journaled once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import OrderState, Side
from packages.contracts.models import Quote, RiskDecision
from packages.contracts.policy import RiskPolicy
from packages.core.config import PROJECT_ROOT
from packages.core.money import ZERO
from services.order_ledger.state_machine import OPEN, TERMINAL
from services.order_ledger.timeutil import NY_TZ, iso, parse_iso, trading_day
from services.positions.manager import ACKED
from services.positions.policy import PositionPolicy
from services.research.costs import CostModel

from .broker import FillModel
from .feed import MINUTE_TICKS, ReplayFeed
from .journal import EXECUTION_LABEL, RUNTIME_LABEL, Journal, write_summary
from .runtime import ReplayRuntime
from .snapshot import BAR, BarSnapshot, session_open_close
from .strategies import ReplayStrategy, get_strategy

ACTOR = "replay"
NEAR_STOP = Decimal("1.002")


@dataclass(frozen=True)
class ReplayConfig:
    strategy_id: str
    equity: Decimal = Decimal("5000")
    policy: RiskPolicy | None = None
    policy_source: str = "default"
    fill: FillModel = field(default_factory=FillModel)
    position_policy: PositionPolicy = field(default_factory=PositionPolicy)
    start: date | None = None
    end: date | None = None
    warmup_sessions: int = 1
    idle_tick_s: int = 60
    reconcile_interval_s: int = 900
    bar_latency_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        pol = self.policy or RiskPolicy()
        return {"strategy_id": self.strategy_id, "equity": str(self.equity), "policy_version": pol.version,
                "policy_name": pol.name, "policy_source": self.policy_source, "fill_model": self.fill.to_json(),
                "position_policy_version": self.position_policy.version,
                "start": None if self.start is None else self.start.isoformat(),
                "end": None if self.end is None else self.end.isoformat(), "warmup_sessions": self.warmup_sessions,
                "idle_tick_s": self.idle_tick_s, "reconcile_interval_s": self.reconcile_interval_s,
                "bar_latency_s": self.bar_latency_s}


Hook = Callable[["ReplayEngine", datetime], Awaitable[None] | None]


@dataclass
class ReplayHooks:
    """Test/fault-injection hooks (never part of the order path; they may only poke the broker's injector)."""

    before_decision: Hook | None = None
    after_bar: Hook | None = None


@dataclass
class ReplayResult:
    workspace: Path
    journal_path: Path
    summary_path: Path | None
    completed: bool
    stopped_at: datetime | None
    records: list[dict[str, Any]]
    summary: dict[str, Any]


class _Cursors:
    def __init__(self, d: dict[str, Any] | None = None) -> None:
        d = d or {}
        self.decisions = int(d.get("decisions", 0))
        self.transitions = int(d.get("transitions", 0))
        self.fills = int(d.get("fills", 0))
        self.actions = int(d.get("actions", 0))
        self.plans: set[str] = set(d.get("plans", []))
        self.costed: set[str] = set(d.get("costed", []))
        self.matched: set[str] = set(d.get("matched", []))

    def to_json(self) -> dict[str, Any]:
        return {"decisions": self.decisions, "transitions": self.transitions, "fills": self.fills,
                "actions": self.actions, "plans": sorted(self.plans), "costed": sorted(self.costed),
                "matched": sorted(self.matched)}


class ReplayEngine:
    def __init__(self, snapshot: BarSnapshot, config: ReplayConfig, workspace: Path, *,
                 journal_path: Path | None = None, hooks: ReplayHooks | None = None) -> None:
        snapshot.assert_outside_holdout()
        self.snapshot = snapshot
        self.config = config
        self.workspace = workspace.resolve()
        self.strategy: ReplayStrategy = get_strategy(config.strategy_id)
        self.journal_path = journal_path or workspace / "journal.jsonl"
        self.summary_path = self.journal_path.with_name("summary.json")
        self.hooks = hooks or ReplayHooks()
        self.days = snapshot.window(config.start, config.end, warmup_sessions=config.warmup_sessions)
        if not self.days:
            raise ValueError("no replay sessions in the requested window (after warm-up)")
        b = [session_open_close(d) for d in snapshot.sessions]
        self.bounds = [x for x in b if x is not None]
        self.cost_model = CostModel()
        self.rt: ReplayRuntime
        self.journal: Journal
        self.cursors = _Cursors()
        self._last_reconcile: datetime | None = None
        self._decision_clocks = set(self.strategy.decision_clocks or ())
        self._stops: dict[str, Decimal] = {}
        self._journaled: set[tuple[str, str | int]] = set()
        self._mark_prices: dict[str, Decimal] = {}

    # ================================================================== public
    @property
    def provenance(self) -> dict[str, Any]:
        p = self.snapshot.provenance
        return {"dataset_version": p.dataset_version, "bar_feed": p.feed.value, "quote_feed": "SYNTHETIC",
                "is_synthetic_bars": p.is_synthetic, "content_sha256": p.content_sha256, "source": p.source,
                "quotes": "synthesized from bars (intrabar path, cost-model spread)"}

    async def run(self, *, resume: bool = False, stop_after: datetime | None = None) -> ReplayResult:
        prior = self._prepare_workspace(resume)
        if prior and prior[-1]["kind"] == "RUN_END":
            return self._result(True, None, records=prior)
        feed = ReplayFeed(self.snapshot, step_s=self.config.fill.quote_step_s, bar_latency_s=self.config.bar_latency_s)
        self.rt = ReplayRuntime(self.workspace, feed=feed, strategy=self.strategy, equity=self.config.equity,
                                sessions=self.bounds, policy=self.config.policy, fill_model=self.config.fill,
                                position_policy=self.config.position_policy)
        self.journal = Journal(self.journal_path, self.provenance)
        rt = self.rt
        resume_from: datetime | None = None
        ended = {r["session"] for r in prior if r["kind"] == "SESSION_END"}
        started = {r["session"] for r in prior if r["kind"] == "SESSION_START"}
        close_marks = {parse_iso(r["at"]): r for r in prior if r["kind"] == "MARK" and r["session_end"]}
        if prior:
            marks = [r for r in prior if r["kind"] == "MARK"]
            if marks:
                self.cursors = _Cursors(marks[-1]["cursors"])
                self._mark_prices = {sym: Decimal(price) for sym, price in marks[-1].get("mark_prices", {}).items()}
            for r in prior:
                if marks and r["seq"] <= marks[-1]["seq"]:
                    continue
                key = r.get("source_key")
                if key is not None:
                    self._journaled.add((r["kind"], key))
            last = max([parse_iso(r["at"]) for r in prior] + [self._last_db_activity()])
            resume_from = last
            rt.sim_clock_set(last)
            self.journal.write("RESUMED", last, resume_after=iso(last), note=(
                "process restart: ledger and broker state reopened from the workspace; startup recovery reconciles "
                "in-doubt orders by client order id; nothing is resubmitted; bars before the restart point are skipped"))
        else:
            first = session_open_close(self.days[0])
            assert first is not None
            rt.sim_clock_set(first[0])
            self.journal.write("RUN_START", first[0], config=self.config.to_json(),
                               strategy=self.strategy.to_json(), policy=rt.policy.model_dump(mode="json"),
                               policy_version=rt.policy.version, position_policy=rt.position_policy.model_dump(mode="json"),
                               sessions=[d.isoformat() for d in self.days], symbols=list(self.snapshot.symbols),
                               workspace=str(self.workspace), live_execution_compiled=False)
        completed = True
        stopped_at: datetime | None = None
        try:
            report = await rt.startup()
            self.journal.write("STARTUP_RECONCILE", rt.now(), status=report.status.value, detail=report.detail)
            self._tail()
            for day in self.days:
                bounds = session_open_close(day)
                assert bounds is not None
                open_, close = bounds
                session = day.isoformat()
                if session in ended or (resume_from is not None and close < resume_from):
                    continue
                if close in close_marks:
                    self._write_session_end(day, close, close_marks[close])
                    continue
                if session not in started:
                    self._set(open_)
                    self.journal.write("SESSION_START", open_, session=day.isoformat(), close=iso(close),
                                       early_close=(close - open_) < timedelta(hours=6, minutes=30))
                t = open_
                while t + BAR <= close:
                    if resume_from is not None and t < resume_from:
                        t += BAR
                        continue
                    await self._bar(t, close)
                    end = t + BAR
                    if stop_after is not None and end >= stop_after:
                        completed, stopped_at = False, end
                        return self._result(completed, stopped_at)
                    t = end
                await self._close(day, close)
            self._tail()
            self.journal.write("RUN_END", self.rt.now(), completed=True)
        finally:
            self.journal.sync()
            if completed:
                self.journal.close()
                await rt.aclose()
        return self._result(completed, stopped_at)

    def _prepare_workspace(self, resume: bool) -> list[dict[str, Any]]:
        """Validate both destinations before opening a writer, policy store or key store."""
        operator_roots = {PROJECT_ROOT.resolve(), Path(os.environ.get("LAT_PROJECT_ROOT") or PROJECT_ROOT).resolve()}
        journal = self.journal_path.resolve()
        for root in operator_roots:
            var = (root / "var").resolve()
            if (self.workspace == root or self.workspace.is_relative_to(var)
                    or journal.is_relative_to(var) or self.summary_path.resolve().is_relative_to(var)):
                raise ValueError("replay destinations must be outside the operator project root and its var directory")
        if journal.is_relative_to(self.workspace / "var") or journal.is_relative_to(self.workspace / "datasets"):
            raise ValueError("the replay journal must be outside runtime state directories")
        for path in (self.journal_path, self.summary_path):
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
                raise ValueError("replay outputs must be regular, unlinked files")
        if not resume:
            if self.journal_path.exists() or self.summary_path.exists():
                raise ValueError("replay outputs already exist; use new paths or resume the original run")
            self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
            if any(self.workspace.iterdir()):
                raise ValueError("a new replay needs an empty disposable workspace")
            return []
        prior = list(Journal.read(self.journal_path))
        metadata = prior[0] if prior and prior[0]["kind"] == "RUN_START" else {}
        if (metadata.get("workspace") != str(self.workspace) or metadata.get("config") != self.config.to_json()
                or metadata.get("provenance") != self.provenance
                or metadata.get("position_policy") != self.config.position_policy.model_dump(mode="json")
                or any(r.get("execution_label") != EXECUTION_LABEL or r.get("provenance") != self.provenance
                       for r in prior)):
            raise ValueError("resume requires the original replay workspace, journal, snapshot and configuration")
        state = self.workspace / "var" / "replay"
        if not (state / "ledger.sqlite").is_file() or not (state / "keys").is_dir():
            raise ValueError("the original replay ledger and keys are required to resume")
        for path in self.workspace.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or (path.is_file() and path.stat().st_nlink == 1)):
                raise ValueError("replay workspace must contain only regular files and directories owned by the run")
        return prior

    def abandon(self) -> None:
        """Simulated crash: drop the process objects without a clean shutdown (the journal stays as written)."""
        self.journal.close()
        self.rt.db.close()

    # ================================================================== loop
    def _set(self, when: datetime) -> None:
        if when > self.rt.clock.now():
            self.rt.sim_clock_set(when)

    async def _bar(self, start: datetime, close: datetime) -> None:
        rt, feed, fm = self.rt, self.rt.replay_feed, self.config.fill
        rows = feed.bars_starting(start)
        paths = {r.symbol: r.path(fm.quote_step_s) for r in rows}
        offsets = sorted({off for p in paths.values() for off, _ in p}) if paths else MINUTE_TICKS
        by_sym = {s: dict(p) for s, p in paths.items()}
        prev: dict[str, tuple[datetime, float]] = {}
        for off in offsets:
            t = start + timedelta(seconds=off)
            events: list[tuple[datetime, int, Quote]] = []
            for sym, pts in by_sym.items():
                if off not in pts:
                    continue
                price = pts[off]
                if fm.stop_crossing and sym in prev:
                    pt, pp = prev[sym]
                    stops = rt.broker.resting_stops(sym)
                    events += [(q.exchange_ts, 0, q) for q in feed.crossing_quotes(sym, pt, pp, t, price, stops)]
                events.append((t, 1, feed.make_quote(sym, t, price)))
                prev[sym] = (t, price)
            events.sort(key=lambda e: (e[0], e[1], e[2].symbol))
            for when, _, q in events:
                self._set(when)
                feed.emit(q)
                self._mark_prices[q.symbol] = q.mid
                rt.feed_monitor.observe_quote(q)
                rt.broker.advance(q)
                if rt.pump():
                    self._dirty = True
                    self._tail()
            self._set(t)
            if self._pm_due(t):
                await self._pm_tick()
        end = start + BAR
        self._set(end)
        if self._decides_at(end, close):
            if self.hooks.before_decision is not None:
                await _maybe_await(self.hooks.before_decision(self, end))
            result = await rt.run_cycle(symbols=None, use_ai=False, actor=ACTOR)
            self._dirty = True
            self._tail()
            self.journal.write("DECISION", end, causal_id=result.get("causal_id"), outcome=result.get("outcome"),
                               reason=result.get("reason"), candidate_symbol=result.get("candidate_symbol"),
                               signals=result.get("signals"), client_order_id=result.get("client_order_id"),
                               entries_locked=result.get("entries_locked"), lock_reasons=result.get("lock_reasons"))
            self._last_reconcile = end
        elif self._last_reconcile is None or (end - self._last_reconcile).total_seconds() >= \
                self.config.reconcile_interval_s:
            await rt.reconcile()
            self._dirty = True
            self._last_reconcile = end
        if self.hooks.after_bar is not None:
            await _maybe_await(self.hooks.after_bar(self, end))
        self._tail()
        await self._mark(end)

    def _decides_at(self, end: datetime, close: datetime) -> bool:
        if end >= close:
            return False
        if not self._decision_clocks:
            return True
        ny = end.astimezone(NY_TZ)
        return f"{ny.hour:02d}:{ny.minute:02d}" in self._decision_clocks

    def _pm_due(self, t: datetime) -> bool:
        every = max(1, self.config.idle_tick_s // 60)
        if t.second == 1 and t.microsecond == 0 and t.minute % every == 0:
            self._dirty = True
            return True  # idle cadence: once a minute (EOD, TIME and holding deadlines fall on minute boundaries)
        return self._busy()

    _dirty = True
    _busy_state = True

    def _busy(self) -> bool:
        """Does the position manager have work right now? Recomputed from the ledger only after something changed
        (broker events, manager actions, a decision, a minute mark); the price test against open plans' stops runs
        on every sample from memory."""
        if self._dirty:
            self._dirty = False
            self._busy_state, self._stops = self._compute_busy()
        if self._busy_state:
            return True
        feed = self.rt.replay_feed
        for sym, stop in self._stops.items():
            q = feed.latest_quote_asof(sym, None)
            if q is not None and q.bid <= stop * NEAR_STOP:
                return True
        return False

    def _compute_busy(self) -> tuple[bool, dict[str, Decimal]]:
        led = self.rt.ledger
        records = led.records(sorted(OPEN, key=lambda s: s.value))
        stops = {p.symbol: p.stop_price for p in self.rt.position_store.open_plans()}
        if any(not (r.is_protective and r.state in ACKED) for r in records):
            return True, stops
        protected: dict[str, Decimal] = {}
        for r in records:
            protected[r.symbol] = protected.get(r.symbol, ZERO) + (r.quantity - r.filled_quantity)
        qty = {s: q for s, q in led.position_quantities().items() if q != 0}
        if any(protected.get(s, ZERO) != q for s, q in qty.items()):
            return True, stops
        if any(sym not in qty for sym in stops) or any(p.exit_reason for p in self.rt.position_store.open_plans()):
            return True, stops
        return False, stops

    async def _pm_tick(self) -> None:
        actions = await self.rt.positions.tick()
        if self.rt.pump() or actions:
            self._dirty = True
            self._tail()

    async def _close(self, day: date, close: datetime) -> None:
        rt = self.rt
        self._set(close)
        rt.broker.advance(None)  # DAY orders expire at the close
        rt.pump()
        await self._pm_tick()
        await rt.reconcile()
        self._last_reconcile = close
        self._tail()
        mark = await self._mark(close, session_end=True)
        self._write_session_end(day, close, mark)

    def _write_session_end(self, day: date, close: datetime, mark: dict[str, Any]) -> None:
        self.journal.write("SESSION_END", close, session=day.isoformat(), equity=mark["equity"], cash=mark["cash"],
                           positions=mark["positions"])
        self.journal.sync()

    async def _mark(self, at: datetime, *, session_end: bool = False) -> dict[str, Any]:
        acct = await self.rt.broker.get_account()
        pos = await self.rt.broker.get_positions()
        # Valuation carries forward the last observed mid across tape gaps. The gateway still ages
        # out stale quotes for execution and risk checks.
        value = sum((p.quantity * self._mark_prices[p.symbol] if p.symbol in self._mark_prices
                     else p.cost_basis for p in pos), ZERO)
        mark = self.journal.write("MARK", at, equity=str(acct.cash + value), cash=str(acct.cash),
                                  position_value=str(value), positions={p.symbol: str(p.quantity) for p in pos},
                                  mark_prices={sym: str(px) for sym, px in self._mark_prices.items()},
                                  session_end=session_end, cursors=self.cursors.to_json())
        self._journaled.clear()
        return mark

    def _last_db_activity(self) -> datetime:
        db = self.rt.db
        stamps: list[str] = []
        for sql in ("SELECT MAX(updated_at) m FROM orders", "SELECT MAX(ts) m FROM fills",
                    "SELECT MAX(at) m FROM position_actions", "SELECT MAX(at) m FROM order_state_transitions",
                    "SELECT MAX(evaluated_at) m FROM risk_decisions"):
            row = db.query_one(sql)
            if row is not None and row["m"]:
                stamps.append(str(row["m"]))
        return max((parse_iso(s) for s in stamps), default=self.rt.clock.now())

    # ================================================================== journal tail (ledger -> journal)
    def _write_source(self, kind: str, at: datetime, key: str | int, **payload: Any) -> None:
        if (kind, key) not in self._journaled:
            self.journal.write(kind, at, source_key=key, **payload)
            self._journaled.add((kind, key))

    def _tail(self) -> None:
        db, c = self.rt.db, self.cursors
        cost_candidates: dict[str, datetime] = {}
        for row in db.query("SELECT d.rowid AS rid, d.payload_json, p.payload_json AS prop FROM risk_decisions d "
                            "LEFT JOIN proposals p ON p.proposal_id=d.proposal_id WHERE d.rowid > ? ORDER BY d.rowid",
                            (c.decisions,)):
            c.decisions = int(row["rid"])
            dec = RiskDecision.model_validate_json(row["payload_json"])
            prop: dict[str, Any] = json.loads(row["prop"]) if row["prop"] else {}
            failed = [ch.name for ch in dec.checks if not ch.passed]
            self._write_source("RISK_DECISION" if dec.approved else "REJECTION", dec.evaluated_at, dec.decision_id,
                    decision_id=dec.decision_id,
                    proposal_id=dec.proposal_id, causal_id=dec.causal_id, verdict=dec.verdict.value,
                    approved=dec.approved, failed_checks=failed, reasons=list(dec.reasons)[:10],
                    policy_version=dec.policy_version, symbol=prop.get("symbol"), side=prop.get("side"),
                    order_type=prop.get("order_type"), quantity=prop.get("quantity"),
                    limit_price=prop.get("limit_price"), stop_price=prop.get("stop_price"),
                    proposer=prop.get("proposer"), strategy_id=prop.get("strategy_id"))
        for row in db.query("SELECT t.seq, t.client_order_id, t.from_state, t.to_state, t.at, t.reason, o.symbol, "
                            "o.side, o.order_type, o.quantity, o.limit_price, o.intent_json, o.filled_quantity, "
                            "o.avg_fill_price FROM order_state_transitions t JOIN orders o ON "
                            "o.client_order_id=t.client_order_id WHERE t.seq > ? ORDER BY t.seq", (c.transitions,)):
            c.transitions = int(row["seq"])
            intent = json.loads(row["intent_json"])
            base = {"client_order_id": row["client_order_id"], "symbol": row["symbol"], "side": row["side"],
                    "order_type": row["order_type"], "quantity": row["quantity"], "limit_price": row["limit_price"],
                    "stop_price": intent.get("stop_price"), "from_state": row["from_state"],
                    "to_state": row["to_state"], "reason": row["reason"]}
            at = parse_iso(str(row["at"]))
            if row["to_state"] == OrderState.RISK_APPROVED.value:
                self._write_source("INTENT", at, c.transitions, intent_id=intent.get("intent_id"),
                        key_id=intent.get("key_id"),
                        algorithm=intent.get("signature_alg"), signed=bool(intent.get("signature")),
                        expires_at=intent.get("expires_at"), policy_version=intent.get("policy_version"),
                        mode=intent.get("mode"), gateway=intent.get("gateway"), **base)
            else:
                self._write_source("ORDER", at, c.transitions, **base)
            if OrderState(row["to_state"]) in TERMINAL:
                cost_candidates[row["client_order_id"]] = at
        for row in db.query("SELECT f.rowid AS rid, f.fill_id, f.client_order_id, f.symbol, f.side, f.quantity, f.price,"
                            " f.ts, o.order_type FROM fills f JOIN orders o ON o.client_order_id=f.client_order_id "
                            "WHERE f.rowid > ? ORDER BY f.rowid", (c.fills,)):
            c.fills = int(row["rid"])
            self._write_source("FILL", parse_iso(str(row["ts"])), row["fill_id"],
                    fill_id=row["fill_id"], client_order_id=row["client_order_id"],
                    symbol=row["symbol"], side=row["side"], quantity=row["quantity"], price=row["price"],
                    order_type=row["order_type"], protective_stop_fill=row["order_type"] == "STOP_LIMIT",
                    entry_fill_model=self.rt.broker.entry_outcomes.get(row["client_order_id"]))
            # Reconciliation can discover a fill after an order was already terminal.
            cost_candidates.setdefault(row["client_order_id"], self.rt.now())
        for coid, at in cost_candidates.items():
            self._leg_cost(coid, at)
        for row in db.query("SELECT seq, plan_id, symbol, at, action, purpose, client_order_id, outcome, detail FROM "
                            "position_actions WHERE seq > ? ORDER BY seq", (c.actions,)):
            c.actions = int(row["seq"])
            action = str(row["action"])
            kind = ("STOP" if action in ("PROTECT", "TOP_UP") else "EXIT" if action in ("EXIT", "EXIT_TRIGGERED")
                    else "CANCEL" if action == "CANCEL" else "POSITION_ACTION")
            self._write_source(kind, parse_iso(str(row["at"])), c.actions,
                    action=action, plan_id=row["plan_id"], symbol=row["symbol"],
                    purpose=row["purpose"], client_order_id=row["client_order_id"], outcome=row["outcome"],
                    detail=(row["detail"] or "")[:400])
        for coid, outcome in sorted(self.rt.broker.entry_outcomes.items()):
            if coid not in c.matched:
                c.matched.add(coid)
                rec = self.rt.ledger.get_record(coid)
                self._write_source("ENTRY_MATCH", self.rt.now(), coid, client_order_id=coid, outcome=outcome,
                        symbol=None if rec is None else rec.symbol,
                        limit_price=None if rec is None or rec.limit_price is None else str(rec.limit_price))
        for plan in self.rt.position_store.closed_plans(limit=10_000):
            if plan.plan_id in c.plans:
                continue
            c.plans.add(plan.plan_id)
            pj = plan.to_json()
            self._write_source("ROUND_TRIP", plan.closed_at or self.rt.now(), plan.plan_id, **{k: pj.get(k) for k in (
                "plan_id", "symbol", "strategy_id", "opened_at", "closed_at", "trading_day", "entry_client_order_id",
                "stop_price", "stop_source", "horizon_minutes", "exit_deadline", "exit_reason", "close_reason",
                "entry_quantity", "entry_avg_price", "exit_quantity", "exit_avg_price", "realized_pl",
                "holding_sessions")})

    def _leg_cost(self, coid: str, at: datetime) -> None:
        """Runtime cost model, once per filled order leg: fees + fixed slippage + impact. The half spread is not
        charged again: replay fills happen at the synthesized bid/ask, so the spread is already in the fill price."""
        if coid in self.cursors.costed:
            return
        rec = self.rt.ledger.get_record(coid)
        if rec is None or rec.state not in TERMINAL or rec.filled_quantity <= 0 or rec.avg_fill_price is None:
            return
        self.cursors.costed.add(coid)
        day = trading_day(rec.submitted_at or rec.created_at)
        adv = self.snapshot.adv_shares(rec.symbol, day)
        spread = self.rt.replay_feed.spread_bps(rec.symbol, day)
        est = self.cost_model.estimate(rec.symbol, rec.side, rec.filled_quantity, rec.avg_fill_price,
                                       spread_bps=spread, adv_shares=None if adv is None else Decimal(f"{adv:.0f}"))
        charged = est.fees_usd + est.slippage_usd
        self._write_source("LEG_COST", at, coid, client_order_id=coid, symbol=rec.symbol, side=rec.side.value,
                           order_type=rec.order_type.value, quantity=str(rec.filled_quantity),
                           avg_price=str(rec.avg_fill_price), notional=str(est.notional),
                           charged_usd=str(charged), fees_usd=str(est.fees_usd), slippage_usd=str(est.slippage_usd),
                           impact_bps=str(est.impact_bps), spread_bps_in_fill=str(est.spread_cost_bps),
                           cost_model=self.cost_model.version, sell=rec.side == Side.SELL)

    # ================================================================== result
    def _result(self, completed: bool, stopped_at: datetime | None, *,
                records: list[dict[str, Any]] | None = None) -> ReplayResult:
        from .metrics import summarize

        if records is None:
            records = list(Journal.read(self.journal_path))
        summary = summarize(records, equity=self.config.equity)
        summary.update(config=self.config.to_json(), strategy=self.strategy.to_json(), provenance=self.provenance,
                       completed=completed, stopped_at=None if stopped_at is None else iso(stopped_at),
                       journal=str(self.journal_path), execution_label=EXECUTION_LABEL, runtime_label=RUNTIME_LABEL,
                       is_simulated=True)
        if completed and not self.summary_path.exists():
            write_summary(self.summary_path, summary)
        return ReplayResult(self.workspace, self.journal_path, self.summary_path if completed else None, completed,
                            stopped_at, records, summary)


async def _maybe_await(x: Awaitable[None] | None) -> None:
    if x is not None:
        await x
