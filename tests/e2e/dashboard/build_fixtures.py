"""Build the read-only dashboard's mock-API fixtures by RECORDING the real backend components.

The live-operations events are produced by the real ``services.api.activity.ActivityLog`` (stage /
activity / decision bodies and the exact ``/activity`` + ``/activity/current`` shapes), the order
lifecycle by the real ``services.order_ledger.ledger.OrderLedger`` (proposal -> decision -> signed
intent -> claim -> submit -> broker events -> fills), the REST order/audit bodies by the real
``services.api.queries`` functions and the ``audit`` / ``order_event`` / ``order`` SSE bodies by the
real ``services.api.events.DbTailer`` - all against a throw-away SQLite database and a ManualClock.
Contract objects are built through ``packages.contracts`` models. So when a backend shape changes, the
recorded fixtures change and ``--check`` (and test_dashboard_fixtures.py) fail instead of drifting.

Only the bodies whose producers do not exist in this build yet are constructed by hand against
docs/architecture/api-contract.md: /performance, /sessions + ``session`` events, the /health component
report and the ``cycle`` summary (its keys mirror services/api/pipeline.py).

Everything is explicitly SIMULATED: deterministic fake broker, synthetic data, FAKE model transport.

    uv run --frozen python tests/e2e/dashboard/build_fixtures.py          # (re)write fixtures
    uv run --frozen python tests/e2e/dashboard/build_fixtures.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.contracts import (  # noqa: E402
    ApprovedOrderIntent,
    BrokerOrder,
    OrderEvent,
    RiskCheck,
    RiskDecision,
    RiskPolicy,
    TradeProposal,
)
from packages.contracts.enums import (  # noqa: E402
    DecisionAction,
    EventSource,
    GatewayKind,
    OrderEventType,
    OrderState,
    OrderType,
    Proposer,
    RiskVerdict,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.core.audit import append_audit  # noqa: E402
from packages.core.clock import ManualClock  # noqa: E402
from packages.core.db import Database  # noqa: E402

FIXTURE_DIR = PROJECT_ROOT / "apps" / "dashboard" / "fixtures"

T0 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)  # 09:30 America/New_York, regular session open
DATASET = "synthetic-fixture-v1"
SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT")
CAUSAL_REJECT = "cz_01JBDASHFIXTURE0000000002"
CAUSAL_FILLED = "cz_01JBDASHFIXTURE0000000001"
CAUSAL_NOTRADE = "cz_01JBDASHFIXTURE0000000003"
FILLED_COID = "lat-sim-01jbdashfixture00000filled1"
SESSION_ID = "sess_01JBDASHFIXTURE00000001"
AGENT = "agent:claude-code"
POLICY = RiskPolicy()
SIM_LIMITATIONS = [
    "SIMULATED feed: deterministic synthetic bars from a fixture, not market prices",
    "Fills come from the deterministic fake broker; no market impact, queue position or latency",
]


def ts(minutes: float = 0, seconds: float = 0) -> datetime:
    return T0 + timedelta(minutes=minutes, seconds=seconds)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def dump(model: Any) -> Any:
    return model.model_dump(mode="json")


def fake_hex(seed: str, length: int) -> str:
    out, counter = "", 0
    while len(out) < length:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).hexdigest()
        counter += 1
    return out[:length]


# ------------------------------------------------------------------------------------------------
# Hand-built bodies (producers not in this build yet) - shapes from api-contract.md
# ------------------------------------------------------------------------------------------------


def status() -> dict[str, Any]:
    return {
        "mode": "PAPER",
        "execution_label": "SIMULATED",
        "gateway": "FAKE",
        "live_execution_compiled": False,
        "entries_locked": False,
        "lock_reasons": [],
        "halts": [],
        "startup": {"phase": "READY", "started_at": iso(ts(minutes=-95)), "detail": "startup reconciliation CLEAN: fake broker"},
        "feed": {
            "feed_mode": "SIMULATED",
            "provider": "replay:synthetic-fixture-v1",
            "is_simulated": True,
            "last_bar_ts": iso(ts()),
            "last_quote_ts": iso(ts(seconds=-3)),
            "staleness_s": "3.0",
            "status": "OK",
            "limitations": SIM_LIMITATIONS,
            "simulated_time": iso(ts()),
            "dataset_version": DATASET,
        },
        "engine": {
            "kind": "FAKE",
            "status": "UP",
            "last_heartbeat": iso(ts(seconds=-1)),
            "detail": "in-process deterministic fake broker",
        },
        "broker": {
            "kind": "FAKE",
            "is_paper": True,
            "connected": True,
            "last_reconciled_at": iso(ts(minutes=-1)),
            "reconciliation_status": "CLEAN",
        },
        "credentials": {"alpaca_paper": False, "anthropic": False, "openai": False},
        "ai": {"transport": "FAKE", "claude_model": "claude-opus-5-5", "sol_model": "gpt-6.1-sol"},
        "policy_version": POLICY.version,
        "server_time": iso(ts()),
    }


def health_simple() -> dict[str, Any]:
    """The original liveness body of GET /api/v1/health (still served first by some API builds)."""
    return {"status": "ok", "mode": "PAPER", "execution_label": "SIMULATED", "version": "0.1.0", "time": iso(ts())}


def health_report() -> dict[str, Any]:
    def c(name: str, st: str, detail: str, last_ok: datetime | None = None) -> dict[str, Any]:
        return {"name": name, "status": st, "detail": detail, "last_ok": iso(last_ok) if last_ok else None}

    return {
        "overall": "OK",
        "components": [
            c("market_data", "OK", "OK SIMULATED replay:synthetic-fixture-v1", ts(seconds=-3)),
            c("broker", "OK", "FAKE: connected (deterministic fake broker)"),
            c("reconciliation", "OK", "CLEAN", ts(minutes=-1)),
            c("models", "OK", "registry mreg_20261105T140000Z_dashfixture01"),
            c("gpt_runtime", "OK", "mode API_KEYS (FAKE transport)"),
            c("event_stream", "OK", "1 subscribers"),
            c("database", "OK", "SQLite WAL ledger.sqlite: quick_check ok"),
            c("agent", "OK", f"last tool call {iso(ts(minutes=-25))} by {AGENT}", ts(minutes=-25)),
            c("engine", "OK", "FAKE engine in-process", ts(seconds=-1)),
            c("scheduler", "OK", "session scheduler running"),
        ],
        "asof": iso(ts()),
    }


def session(cycles: int, trades: int, state: str = "RUNNING") -> dict[str, Any]:
    return {
        "session_id": SESSION_ID,
        "state": state,
        "started_at": iso(ts(minutes=-25)),
        "ends_at": iso(ts(minutes=35)),
        "ended_at": None,
        "symbols": list(SYMBOLS),
        "cadence_minutes": 5,
        "use_ai": True,
        "requested_by": AGENT,
        "cycles_run": cycles,
        "trades": trades,
        "stop_reason": None,
        "schedule_id": None,
    }


def schedules() -> list[dict[str, Any]]:
    return [
        {
            "schedule_id": "sched_01JBDASHFIXTURE00000001",
            "weekdays": ["MON", "TUE", "WED", "THU", "FRI"],
            "start_time_et": "09:45",
            "duration_minutes": 60,
            "symbols": list(SYMBOLS),
            "cadence_minutes": 5,
            "use_ai": True,
            "enabled": True,
            "next_run_at": iso(datetime(2026, 11, 6, 14, 45, tzinfo=UTC)),
            "created_by": AGENT,
        }
    ]


def sessions_body(cycles: int, trades: int) -> dict[str, Any]:
    previous = session(0, 0, "COMPLETED") | {
        "session_id": "sess_01JBDASHFIXTURE00000000",
        "started_at": iso(ts(minutes=-1440)),
        "ends_at": iso(ts(minutes=-1380)),
        "ended_at": iso(ts(minutes=-1380)),
        "cycles_run": 12,
        "trades": 0,
        "stop_reason": "session window ended",
    }
    return {"active": session(cycles, trades), "sessions": [session(cycles, trades), previous], "schedules": schedules()}


START_EQUITY = Decimal("100000.00")


def performance(
    fills: list[tuple[Decimal, Decimal]], mark: Decimal | None, open_orders: int, api_cost: Decimal
) -> dict[str, Any]:
    """Mark-to-market of the recorded fills (exact Decimal arithmetic; fixture values, SIMULATED)."""
    qty = sum((q for q, _ in fills), Decimal(0))
    cost = sum((q * p for q, p in fills), Decimal(0))
    value = (qty * mark) if (mark is not None and qty) else Decimal(0)
    unrealized = (value - cost).quantize(Decimal("0.01"))
    equity = (START_EQUITY + unrealized).quantize(Decimal("0.01"))
    cash = (START_EQUITY - cost).quantize(Decimal("0.01"))
    curve = [
        {"time": int(ts(minutes=-25 + 5 * i).timestamp()), "equity": format(START_EQUITY + Decimal(d), "f")}
        for i, d in enumerate(("0.00", "0.00", "-3.10", "1.25", "0.40"))
    ]
    if qty:
        curve.append({"time": int(ts(seconds=12).timestamp()), "equity": format(equity, "f")})
    return {
        "mode_label": "SIMULATED",
        "portfolio_value": format(equity, "f"),
        "cash": format(cash, "f"),
        "daily_pnl": format(unrealized, "f"),
        "total_pnl": format(unrealized, "f"),
        "realized_pnl": "0.00",
        "unrealized_pnl": format(unrealized, "f"),
        "win_rate": None,
        "round_trips": 0,
        "max_drawdown": "0.0000310",
        "current_drawdown": "0",
        "active_positions": 1 if qty else 0,
        "open_orders": open_orders,
        "api_cost_usd": format(api_cost, "f"),
        "equity_curve": curve,
        "asof": iso(ts(seconds=12) if qty else ts()),
        "is_simulated": True,
    }


# ------------------------------------------------------------------------------------------------
# Contract objects for the recorded cycles
# ------------------------------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str, observed: str | None = None, limit: str | None = None) -> RiskCheck:
    return RiskCheck(name=name, passed=passed, detail=detail, observed=observed, limit=limit)


def rejected_cycle_objects(at: datetime) -> tuple[TradeProposal, RiskDecision]:
    p = TradeProposal(
        proposal_id="tp_01JBDASHFIXTURE0000000002", causal_id=CAUSAL_REJECT, created_at=at, data_asof=at,
        action=DecisionAction.TRADE, proposer=Proposer.QUANT_MODEL, strategy_id="quant-model-suite",
        symbol="AAPL", side=Side.BUY, quantity=Decimal("9"), limit_price=Decimal("229.84"), stop_price=Decimal("226.40"),
        max_planned_loss=Decimal("30.96"), evidence_ids=("sig_01JBDASHFIXTURE0000000002",),
        signal_id="sig_01JBDASHFIXTURE0000000002", policy_version_seen=POLICY.version, dataset_version=DATASET,
        rationale="quant-model-suite: expected 9.8 bps, cost 6.1 bps, net edge 3.7 bps (runtime API_KEYS)",
    )
    d = RiskDecision(
        decision_id="rd_01JBDASHFIXTURE0000000002", proposal_id=p.proposal_id, causal_id=CAUSAL_REJECT,
        verdict=RiskVerdict.REJECTED,
        checks=(
            _check("regular_session", True, "regular session", "REGULAR"),
            _check("quote_age", True, "quote is fresh", "2.0", format(POLICY.max_quote_age_seconds, "f")),
            _check("spread", False, "spread 14.2 bps exceeds the 10 bps limit", "14.2", "10"),
            _check("order_notional", True, "recomputed notional within cap", "2068.56", format(POLICY.max_order_notional, "f")),
        ),
        reasons=("spread",), policy_version=POLICY.version, mode=TradingMode.PAPER,
        recomputed_quantity=Decimal("9"), recomputed_notional=Decimal("2068.56"), recomputed_max_loss=Decimal("30.96"),
        reference_price=Decimal("229.70"), evaluated_at=at + timedelta(seconds=1),
    )
    return p, d


def filled_cycle_objects(at: datetime) -> tuple[TradeProposal, RiskDecision, ApprovedOrderIntent]:
    p = TradeProposal(
        proposal_id="tp_01JBDASHFIXTURE0000000001", causal_id=CAUSAL_FILLED, created_at=at, data_asof=at,
        action=DecisionAction.TRADE, proposer=Proposer.AI_PIPELINE, strategy_id="quant-model-suite",
        symbol="SPY", side=Side.BUY, quantity=Decimal("3"), limit_price=Decimal("585.46"), stop_price=Decimal("576.68"),
        max_planned_loss=Decimal("26.34"), evidence_ids=("ev_sig_spy", "ev_mkt_spy"), signal_id="sig_01JBDASHFIXTURE0000000001",
        model_run_ids=("mr_01JBDASHFIXTURE00000CLAUDE", "mr_01JBDASHFIXTURE0000000SOL"),
        policy_version_seen=POLICY.version, dataset_version=DATASET,
        rationale="Net edge 6.2 bps after 5.2 bps estimated cost; model review supports the trade; limit at ask.",
    )
    d = RiskDecision(
        decision_id="rd_01JBDASHFIXTURE0000000001",
        proposal_id=p.proposal_id,
        causal_id=CAUSAL_FILLED,
        verdict=RiskVerdict.APPROVED,
        checks=(
            _check("mode_gateway", True, "PAPER mode with FAKE gateway is permitted", "PAPER/FAKE"),
            _check("regular_session", True, "regular session", "REGULAR"),
            _check("quote_age", True, "quote is fresh", "3.0", format(POLICY.max_quote_age_seconds, "f")),
            _check("spread", True, "spread within limit", "1.7", format(POLICY.max_spread_bps, "f")),
            _check("order_notional", True, "recomputed notional within cap", "1756.38", format(POLICY.max_order_notional, "f")),
            _check(
                "planned_loss", True, "recomputed max loss within cap", "26.34", format(POLICY.max_planned_loss_per_trade, "f")
            ),
        ),
        reasons=(),
        policy_version=POLICY.version,
        mode=TradingMode.PAPER,
        recomputed_quantity=Decimal("3"),
        recomputed_notional=Decimal("1756.38"),
        recomputed_max_loss=Decimal("26.34"),
        reference_price=Decimal("585.41"),
        evaluated_at=at + timedelta(seconds=1),
    )
    intent = ApprovedOrderIntent(
        intent_id="oi_01JBDASHFIXTURE00000001", client_order_id=FILLED_COID, proposal_id=p.proposal_id,
        decision_id=d.decision_id, causal_id=CAUSAL_FILLED, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE,
        symbol="SPY", side=Side.BUY, quantity=Decimal("3"), order_type=OrderType.LIMIT, limit_price=Decimal("585.46"),
        time_in_force=TimeInForce.DAY, policy_version=POLICY.version, created_at=at + timedelta(seconds=2),
        expires_at=at + timedelta(seconds=62), requires_human_approval=False, key_id="risk-key-fixture",
        signature=fake_hex(FILLED_COID + "sig", 128),
    )
    return p, d, intent


def no_trade_proposal(at: datetime) -> TradeProposal:
    return TradeProposal(
        proposal_id="tp_01JBDASHFIXTURE0000000003", causal_id=CAUSAL_NOTRADE, created_at=at, data_asof=at,
        action=DecisionAction.NO_TRADE, no_trade_reason="no candidate cleared the net-of-cost edge threshold",
        proposer=Proposer.QUANT_MODEL, strategy_id="quant-model-suite", policy_version_seen=POLICY.version,
        dataset_version=DATASET,
    )


# ------------------------------------------------------------------------------------------------
# Recorder: real ActivityLog + OrderLedger + DbTailer on a throw-away database
# ------------------------------------------------------------------------------------------------


class RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def publish(self, event: str, data: Any) -> None:
        self.events.append((event, json.loads(json.dumps(data, default=str))))

    @property
    def subscriber_count(self) -> int:
        return 1


class FakePerf:
    """Deterministic stand-in for time.perf_counter inside services.api.activity (stage durations)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def perf_counter(self) -> float:
        return self.t


@contextmanager
def recorder() -> Iterator[Recorder]:
    import services.api.activity as activity_mod

    saved = activity_mod.time
    ids = count(1)
    def fixed_id(prefix: str) -> str:
        return f"{prefix}_{next(ids):026d}"
    with (tempfile.TemporaryDirectory(prefix="lat-dash-fixtures-") as tmp,
          patch("packages.core.audit.new_id", fixed_id),
          patch("services.order_ledger.ledger.new_id", fixed_id)):
        rec = Recorder(Path(tmp) / "ledger.sqlite")
        activity_mod.time = rec.perf  # type: ignore[assignment]
        try:
            yield rec
        finally:
            activity_mod.time = saved
            rec.db.close()


class Recorder:
    def __init__(self, path: Path) -> None:
        from services.api.activity import ActivityLog
        from services.order_ledger.ledger import OrderLedger

        self.db = Database(path)
        self.clock = ManualClock(ts(minutes=-95))
        self.bus = RecordingBus()
        self.perf = FakePerf()
        self.log = ActivityLog(self.db, self.bus, self.clock)  # type: ignore[arg-type]
        self.ledger = OrderLedger(self.db, self.clock, TradingMode.PAPER)
        self.tailer: Any = None
        self.steps: list[dict[str, Any]] = []
        self._detail_sig: dict[str, str] = {}
        self._extra_state: dict[str, Any] = {}

    # -- helpers
    def at(self, when: datetime) -> None:
        self.clock.set(when)

    def wait(self, ms: int) -> None:
        """Advance simulated + perf time together (stage durations come from the perf counter)."""
        self.clock.advance(seconds=ms / 1000)
        self.perf.t += ms / 1000

    def stage(self, stage: str, status: str, ms: int = 0, *, item: str | None = None, detail: str = "") -> None:
        if ms:
            self.wait(ms)
        self.log.stage(stage, status, item=item, detail=detail)

    def audit(self, event_type: str, source: str, payload: dict[str, Any], causal_id: str | None = None) -> None:
        now = self.clock.now()
        self.db.write(lambda c: append_audit(c, event_type=event_type, source=source, mode="PAPER", payload=payload,
                                             causal_id=causal_id, policy_version=POLICY.version, ts=now))

    def start_tailer(self) -> None:
        from services.api.events import DbTailer

        self.bus.events.clear()
        self.tailer = DbTailer(self.db, self.bus)  # type: ignore[arg-type]

    def rest_state(self) -> dict[str, Any]:
        from services.api import queries

        return {"activity_current": self.log.current(), "orders": queries.orders(self.db, 20)}

    def flush(self, delay_ms: int, **state: Any) -> None:
        """Close one script step: the events published since the last step (+ DB tailer rows), then REST state."""
        from services.api import queries

        if self.tailer is not None:
            self.tailer.poll_once()
        events = []
        rows = {r["seq"]: r for r in self.log.recent(2000)}
        for name, data in self.bus.events:
            entry: dict[str, Any] = {"event": name, "data": data}
            if name in ("stage", "activity", "decision"):
                entry["row"] = rows[data["seq"]]  # the exact /activity row for this event
            events.append(entry)
        self.bus.events.clear()
        snap = self.rest_state()
        details = {}
        for o in snap["orders"]["orders"]:
            coid = o["client_order_id"]
            sig = f"{o['state']}|{o['updated_at']}|{o['filled_quantity']}"
            if self._detail_sig.get(coid) != sig:
                self._detail_sig[coid] = sig
                details[coid] = queries.order_detail(self.db, coid)
        step: dict[str, Any] = {"delay_ms": delay_ms, "events": events, "state": {**snap, **state}}
        if details:
            step["state"]["order_details"] = details
        self.steps.append(step)

    def take_steps(self) -> list[dict[str, Any]]:
        out, self.steps = self.steps, []
        return out


# ------------------------------------------------------------------------------------------------
# The recorded scenario
# ------------------------------------------------------------------------------------------------


def _summary(causal: str, ran_at: datetime, outcome: str, **extra: Any) -> dict[str, Any]:
    """Body of the SSE `cycle` event (keys as built in services/api/pipeline.py DecisionCycle.run)."""
    return {
        "causal_id": causal, "ran_at": iso(ran_at), "actor": f"session:{SESSION_ID}", "session_id": SESSION_ID,
        "execution_label": "SIMULATED", "symbols": list(SYMBOLS), "entries_locked": False, "lock_reasons": [],
        "outcome": outcome, **extra,
    }


def _card(rec: Recorder, **card: Any) -> None:
    base = {"ts": iso(rec.clock.now()), "confidence_calibrated": False, "is_simulated": True, "risk_failures": []}
    rec.log.decision(base | card)


def record_history(rec: Recorder) -> None:
    """Everything before the scripted cycles: startup, an agent-started session, a risk-rejected cycle."""
    rec.audit("RUNTIME_STARTED", "api", {"mode": "PAPER", "gateway": "FAKE", "execution_label": "SIMULATED"})
    rec.audit("STARTUP_RECOVERY", "reconciler", {"in_doubt": {}, "note": "SUBMITTING treated as UNKNOWN; nothing resubmitted"})
    rec.log.activity("reconcile", "Startup reconciliation CLEAN; entries unlocked")
    rec.at(ts(minutes=-25))
    rec.audit("AGENT_TOOL_CALL", AGENT, {"tool": "start_paper_session", "mutating": True, "outcome": "ok"})
    rec.log.activity("session", f"Paper trading session {SESSION_ID} started for 60 min on {', '.join(SYMBOLS)} "
                     "(every 5 min, AI review on)", actor=AGENT, session_id=SESSION_ID,
                     detail={"duration_minutes": 60, "cadence_minutes": 5})

    # a risk-rejected cycle (AAPL, spread too wide)
    at = ts(minutes=-20)
    rec.at(at)
    syms = ",".join(SYMBOLS)
    rec.log.begin_cycle(CAUSAL_REJECT, CAUSAL_REJECT, SESSION_ID)
    actor = f"session:{SESSION_ID}"
    rec.stage("MARKET_DATA", "running", item=syms)
    rec.log.activity("scan", f"Scanning {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}", actor=actor, causal_id=CAUSAL_REJECT)
    rec.stage("MARKET_DATA", "passed", 180, item=syms, detail="3840 bars, 4 fresh quotes, feed SIMULATED")
    rec.stage("FEATURES", "running", item=syms)
    rec.stage("FEATURES", "passed", 95, item=syms, detail="point-in-time features computed")
    rec.stage("MODELS", "running", item=syms)
    rec.log.activity(
        "predict", "Generating predictions with the walk-forward-selected model suite", actor=actor, causal_id=CAUSAL_REJECT
    )
    rec.stage("MODELS", "passed", 240, item="AAPL", detail="candidate AAPL net edge 3.7 bps")
    rec.log.activity(
        "evaluate", "Evaluating candidate AAPL: expected 9.8 bps, cost 6.1 bps", actor=actor, causal_id=CAUSAL_REJECT
    )
    rec.stage("GPT_REVIEW", "skipped", item="AAPL", detail="AI disabled for this cycle")
    p, d = rejected_cycle_objects(rec.clock.now())
    rec.ledger.record_proposal(p)
    rec.stage("RISK_ENGINE", "running", item="AAPL")
    rec.wait(12)
    rec.ledger.record_decision(d)
    failures = [f"{c.name}: {c.detail}" for c in d.checks if not c.passed]
    rec.stage("RISK_ENGINE", "rejected", item="AAPL", detail="; ".join(failures))
    rec.log.activity("reject", f"Risk engine rejected BUY 9 AAPL: {failures[0]}", actor="risk-engine", causal_id=CAUSAL_REJECT)
    rec.stage("ORDER_DECISION", "rejected", item="AAPL", detail="risk rejected")
    rec.stage("BROKER", "skipped", detail="nothing to submit")
    _card(rec, symbol="AAPL", side="BUY", action="BUY", status="REJECTED", expected_return_bps="9.8", expected_cost_bps="6.1",
          net_edge_bps="3.7", confidence="0.54", volatility_bps="18.4", regime="TURBULENT", reason="spread",
          risk_failures=failures, proposer="QUANT_MODEL")
    rec.stage("RECONCILIATION", "running")
    rec.stage("RECONCILIATION", "passed", 30, detail="CLEAN, 0 mismatches")
    rec.log.activity("reconcile", "Reconciled positions and orders with FAKE: CLEAN", causal_id=CAUSAL_REJECT)
    rec.audit("DECISION_CYCLE", "pipeline", {"causal_id": CAUSAL_REJECT, "outcome": "REJECTED", "reason": "spread",
                                            "candidate_symbol": "AAPL", "session_id": SESSION_ID}, CAUSAL_REJECT)
    rec.log.end_cycle()


def record_filled_cycle(rec: Recorder) -> list[dict[str, Any]]:
    """A full cycle (agent-triggered, inside the session) that ends in a FILLED paper order."""
    rec.start_tailer()
    at = ts()
    rec.at(at)
    actor = f"session:{SESSION_ID}"
    syms = ",".join(SYMBOLS)
    c = CAUSAL_FILLED
    rec.log.begin_cycle(c, c, SESSION_ID)
    rec.stage("MARKET_DATA", "running", item=syms)
    rec.log.activity("scan", f"Scanning {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}", actor=actor, causal_id=c)
    rec.flush(400)
    rec.stage("MARKET_DATA", "passed", 210, item=syms, detail="3840 bars, 4 fresh quotes, feed SIMULATED")
    rec.stage("FEATURES", "running", item=syms)
    rec.flush(700)
    rec.stage("FEATURES", "passed", 120, item=syms, detail="point-in-time features computed")
    rec.stage("MODELS", "running", item=syms)
    rec.log.activity("predict", "Generating predictions with the walk-forward-selected model suite", actor=actor, causal_id=c)
    rec.flush(600)
    rec.stage("MODELS", "passed", 260, item="SPY", detail="candidate SPY net edge 6.2 bps")
    rec.log.activity("evaluate", "Evaluating candidate SPY: expected 11.4 bps, cost 5.2 bps", actor=actor, causal_id=c)
    rec.stage("GPT_REVIEW", "running", item="SPY")
    rec.log.activity("gpt_call", "Requesting model review of SPY (API_KEYS, transport FAKE)", actor=actor, causal_id=c)
    rec.flush(700)
    p, d, intent = filled_cycle_objects(rec.clock.now())
    rec.ledger.record_proposal(p)
    rec.stage("GPT_REVIEW", "passed", 1450, item="SPY", detail="model review supports the trade")
    rec.stage("RISK_ENGINE", "running", item="SPY")
    rec.flush(700)
    rec.wait(15)
    rec.ledger.record_decision(d)
    rec.stage("RISK_ENGINE", "passed", item="SPY", detail=f"all {len(d.checks)} checks passed")
    rec.stage("ORDER_DECISION", "running", item="SPY")
    rec.wait(8)
    rec.clock.set(intent.created_at)
    rec.ledger.record_intent(intent)
    rec.stage("ORDER_DECISION", "passed", item="SPY", detail="BUY 3 SPY @ 585.46 (signed intent)")
    _card(rec, symbol="SPY", side="BUY", action="BUY", status="APPROVED", expected_return_bps="11.4", expected_cost_bps="5.2",
          net_edge_bps="6.2", confidence="0.58", volatility_bps="12.7", regime="CALM", reason=p.rationale or "",
          proposer="AI_PIPELINE")
    rec.flush(700, performance=performance([], None, 1, Decimal("0.0412")))
    rec.stage("BROKER", "running", item=FILLED_COID)
    rec.log.activity("submit", "Submitting BUY 3 SPY limit 585.46 to FAKE (SIMULATED)", actor=actor, causal_id=c)
    rec.wait(20)
    rec.ledger.claim_for_submission(FILLED_COID, "execution-service")
    rec.ledger.mark_submitted(FILLED_COID, BrokerOrder(
        client_order_id=FILLED_COID, broker_order_id="fake-ord-000001", intent_id=intent.intent_id, gateway=GatewayKind.FAKE,
        symbol="SPY", side=Side.BUY, quantity=Decimal("3"), order_type=OrderType.LIMIT, limit_price=Decimal("585.46"),
        time_in_force=TimeInForce.DAY, broker_status="pending_new", state=OrderState.SUBMITTED,
        submitted_at=rec.clock.now(), updated_at=rec.clock.now()))
    rec.stage("BROKER", "passed", item=FILLED_COID, detail="order state SUBMITTED")
    _card(rec, symbol="SPY", side="BUY", action="BUY", status="SUBMITTED", expected_return_bps="11.4", expected_cost_bps="5.2",
          net_edge_bps="6.2", confidence="0.58", volatility_bps="12.7", regime="CALM", reason=p.rationale or "",
          proposer="AI_PIPELINE")
    rec.flush(800)
    rec.stage("RECONCILIATION", "running")
    rec.flush(500)
    rec.stage("RECONCILIATION", "passed", 40, detail="CLEAN, 0 mismatches")
    rec.log.activity("reconcile", "Reconciled positions and orders with FAKE: CLEAN", causal_id=c)
    summary = _summary(c, at, "SUBMITTED", candidate_symbol="SPY", client_order_id=FILLED_COID, reason=None,
                       proposal=dump(p), decision=dump(d))
    rec.audit("DECISION_CYCLE", "pipeline", {k: summary.get(k) for k in ("causal_id", "ran_at", "candidate_symbol", "outcome",
                                                                          "reason", "client_order_id", "execution_label",
                                                                          "session_id")}, c)
    rec.log.end_cycle()
    rec.bus.publish("cycle", summary)
    rec.bus.publish("session", session(1, 1))
    rec.flush(900, sessions=sessions_body(1, 1))
    # broker events arrive on the fake broker's stream as the simulated market moves
    def ev(n: int, etype: OrderEventType, state: OrderState, fill: str | None = None, px: str | None = None,
           cum: str | None = None) -> OrderEvent:
        now = rec.clock.now()
        return OrderEvent(
            event_id=f"oe_{FILLED_COID[-6:]}_{n:03d}", client_order_id=FILLED_COID, broker_order_id="fake-ord-000001",
            event_type=etype, state=state, source=EventSource.STREAM, gateway=GatewayKind.FAKE,
            fill_quantity=Decimal(fill) if fill else None, fill_price=Decimal(px) if px else None,
            cum_filled_quantity=Decimal(cum) if cum else None, broker_exec_id=f"fx-{FILLED_COID[-6:]}-{n}" if fill else None,
            broker_ts=now, received_ts=now, sequence=n)

    rec.wait(1000)
    rec.ledger.apply_event(ev(2, OrderEventType.ACCEPTED, OrderState.ACKNOWLEDGED))
    rec.flush(900)
    rec.wait(2000)
    rec.ledger.apply_event(ev(3, OrderEventType.PARTIAL_FILL, OrderState.PARTIALLY_FILLED, "1", "585.41", "1"))
    rec.flush(900, performance=performance([(Decimal(1), Decimal("585.41"))], Decimal("585.44"), 1, Decimal("0.0412")))
    rec.wait(3000)
    rec.ledger.apply_event(ev(4, OrderEventType.FILL, OrderState.FILLED, "2", "585.46", "3"))
    rec.flush(900, performance=performance([(Decimal(1), Decimal("585.41")), (Decimal(2), Decimal("585.46"))],
                                           Decimal("585.52"), 0, Decimal("0.0412")))
    return rec.take_steps()


def record_no_trade_cycle(rec: Recorder) -> list[dict[str, Any]]:
    at = ts(minutes=5)
    rec.at(at)
    actor = f"session:{SESSION_ID}"
    syms = ",".join(SYMBOLS)
    c = CAUSAL_NOTRADE
    rec.log.begin_cycle(c, c, SESSION_ID)
    rec.stage("MARKET_DATA", "running", item=syms)
    rec.log.activity("scan", f"Scanning {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}", actor=actor, causal_id=c)
    rec.flush(400)
    rec.stage("MARKET_DATA", "passed", 190, item=syms, detail="3845 bars, 4 fresh quotes, feed SIMULATED")
    rec.stage("FEATURES", "running", item=syms)
    rec.flush(600)
    rec.stage("FEATURES", "passed", 110, item=syms, detail="point-in-time features computed")
    rec.stage("MODELS", "running", item=syms)
    rec.log.activity("predict", "Generating predictions with the walk-forward-selected model suite", actor=actor, causal_id=c)
    rec.flush(600)
    rec.stage("MODELS", "passed", 250, detail="4 scored, none cleared the net-of-cost threshold")
    rec.stage("GPT_REVIEW", "skipped", detail="no candidate")
    rec.ledger.record_proposal(no_trade_proposal(rec.clock.now()))
    for s in ("RISK_ENGINE", "BROKER"):
        rec.stage(s, "skipped", detail="no trade requested")
    reason = "no candidate cleared the net-of-cost edge threshold"
    rec.stage("ORDER_DECISION", "passed", item="SPY", detail=f"NO_TRADE: {reason}")
    _card(rec, symbol="SPY", side=None, action="NO_TRADE", status="NO_TRADE", expected_return_bps="3.1",
          expected_cost_bps="5.0", net_edge_bps="-1.9", confidence="0.51", volatility_bps="11.9", regime="CALM",
          reason=reason, proposer="QUANT_MODEL")
    rec.flush(700)
    rec.stage("RECONCILIATION", "running")
    rec.flush(400)
    rec.stage("RECONCILIATION", "passed", 35, detail="CLEAN, 0 mismatches")
    rec.log.activity("reconcile", "Reconciled positions and orders with FAKE: CLEAN", causal_id=c)
    summary = _summary(c, at, "NO_TRADE", candidate_symbol=None, reason=reason)
    rec.audit("DECISION_CYCLE", "pipeline", {k: summary.get(k) for k in ("causal_id", "ran_at", "candidate_symbol", "outcome",
                                                                          "reason", "client_order_id", "execution_label",
                                                                          "session_id")}, c)
    rec.log.end_cycle()
    rec.bus.publish("cycle", summary)
    rec.bus.publish("session", session(2, 1))
    rec.flush(600, sessions=sessions_body(2, 1))
    return rec.take_steps()


def build_all() -> dict[str, Any]:
    from services.api import queries

    with recorder() as rec:
        record_history(rec)
        initial = {
            "activity": {"events": rec.log.recent(200)},
            "activity_current": rec.log.current(),
            "orders": queries.orders(rec.db, 20),
            "audit": queries.audit(rec.db, 150),
        }
        filled = record_filled_cycle(rec)
        no_trade = record_no_trade_cycle(rec)
        final = {
            "activity": {"events": rec.log.recent(200)},
            "activity_current": rec.log.current(),
            "orders": queries.orders(rec.db, 20),
            "order_filled": queries.order_detail(rec.db, FILLED_COID),
            "audit": queries.audit(rec.db, 150),
        }
    return {"initial": initial, "filled": filled, "no_trade": no_trade, "final": final}


# ------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------


def outputs() -> dict[str, Any]:
    rec = build_all()
    return {
        "status.json": status(),
        "health.json": health_report(),
        "health_simple.json": health_simple(),
        # initial mock state (before any scripted cycle)
        "activity.json": rec["initial"]["activity"],
        "activity_current.json": rec["initial"]["activity_current"],
        "orders.json": rec["initial"]["orders"],
        "audit.json": rec["initial"]["audit"],
        "performance.json": performance([], None, 0, Decimal("0.0000")),
        "sessions.json": sessions_body(0, 0),
        # state after both scripted cycles (unit tests: a filled order, a NO_TRADE card, full timeline)
        "final_activity.json": rec["final"]["activity"],
        "final_activity_current.json": rec["final"]["activity_current"],
        "final_orders.json": rec["final"]["orders"],
        "final_audit.json": rec["final"]["audit"],
        f"order_{FILLED_COID}.json": rec["final"]["order_filled"],
        "final_performance.json": performance([(Decimal(1), Decimal("585.41")), (Decimal(2), Decimal("585.46"))],
                                              Decimal("585.52"), 0, Decimal("0.0412")),
        # scripted SSE sequences played by e2e/mock-api.mjs (POST /__mock/play {"script": ...})
        "live_script.json": {"cycle_filled": rec["filled"], "cycle_no_trade": rec["no_trade"]},
    }


def render() -> dict[Path, str]:
    return {
        FIXTURE_DIR / name: json.dumps(body, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
        for name, body in outputs().items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="record dashboard fixtures from the real backend components")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render()
    if args.check:
        stale = [p for p, text in rendered.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        for p in stale:
            print(f"stale fixture: {p.relative_to(PROJECT_ROOT)}")
        return 1 if stale else 0
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    keep = {p.name for p in rendered}
    for old in FIXTURE_DIR.glob("*.json"):
        if old.name not in keep:
            old.unlink()  # fixtures of the previous (tabbed) dashboard
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(rendered)} fixtures to {FIXTURE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

