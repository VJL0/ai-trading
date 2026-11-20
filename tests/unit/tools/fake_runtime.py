"""Deterministic fake ``ToolRuntime`` for the tool-interface tests (no broker, no model, no network).

It returns the same shapes as ``services.api.runtime.Runtime``'s tool facade, records every member access,
and evaluates agent proposals with the REAL deterministic ``services.risk.engine.RiskEngine`` (so a risky
proposal is genuinely rejected by policy, not by test logic). Money values are strings/Decimals only.
"""

from __future__ import annotations

import asyncio
import builtins
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts import (
    AccountSnapshot,
    Bar,
    HorizonForecast,
    MarketSnapshot,
    PredictionSet,
    QuantSignal,
    Quote,
    RiskContext,
    RiskPolicy,
    SystemHealth,
    TradeProposal,
)
from packages.contracts.enums import (
    DataFeed,
    ExecutionLabel,
    FeedMode,
    GatewayKind,
    MarketRegime,
    MarketSession,
    OrderState,
    Side,
    TradingMode,
)
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.signing import IntentSigner
from services.risk.engine import RiskEngine
from services.sessions.errors import SessionError

# Thursday 2026-11-05 10:00 America/New_York: inside the XNYS regular session.
NOW = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
QUOTES: dict[str, tuple[str, str]] = {
    "SPY": ("500.00", "500.04"), "QQQ": ("400.00", "400.04"), "AAPL": ("200.00", "200.02"), "MSFT": ("410.00", "410.04"),
}
SIGNATURE_HEX = "ab" * 64


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


class FakeToolRuntime:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        project_root: Path | None = None,
        mode: TradingMode = TradingMode.PAPER,
        label: ExecutionLabel = ExecutionLabel.SIMULATED,
    ) -> None:
        if settings is None:
            kwargs: dict[str, Any] = {"mode": mode, "use_keychain": False}
            if project_root is not None:
                kwargs["project_root"] = project_root
            settings = Settings(**kwargs)
        self.settings = settings
        self.policy = RiskPolicy()
        self.dataset_version = "synthetic-test-v1"
        self.execution_label = label
        self.clock = ManualClock(NOW)
        self.signer = IntentSigner.generate()
        self.risk = RiskEngine(self.policy, self.signer, self.clock)
        self.calls: list[str] = []
        self.fail: set[str] = set()
        self.rate_limited = False
        self.refuse = False
        self.corrupt_portfolio = False
        self.proposals: list[TradeProposal] = []
        self.decisions: list[Any] = []
        self.halts: list[dict[str, Any]] = []
        self.cancels: list[str] = []
        self.cycle_args: list[dict[str, Any]] = []
        self._snap_seq = 0
        self._orders: dict[str, dict[str, Any]] = {}
        self._seed_orders()
        self.sessions: FakeSessionControl | None = FakeSessionControl(self)
        self.activity = FakeActivity(self)

    # ------------------------------------------------------------------ helpers
    def search_memory(self, query: str, *, symbol: str | None = None,
                      kind: str | None = None, limit: int = 10) -> dict[str, Any]:
        self._rec("search_memory")
        return {"hits": [], "execution_authority": False}

    def _rec(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"injected failure in {name}")

    def _seed_orders(self) -> None:
        self._orders["lat-sim-open1"] = self._order_row("lat-sim-open1", "SPY", "BUY", "1", "500.04", "SUBMITTED")
        self._orders["lat-sim-done1"] = self._order_row("lat-sim-done1", "QQQ", "BUY", "2", "400.04", "FILLED",
                                                        filled="2", avg="400.03")

    def _order_row(self, coid: str, symbol: str, side: str, qty: str, limit: str, state: str, *,
                   filled: str = "0", avg: str | None = None) -> dict[str, Any]:
        return {"client_order_id": coid, "broker_order_id": None, "intent_id": f"int_{coid}", "symbol": symbol,
                "side": side, "quantity": qty, "filled_quantity": filled, "avg_fill_price": avg, "limit_price": limit,
                "state": state, "gateway": "FAKE", "execution_label": "SIMULATED", "created_at": _iso(NOW),
                "updated_at": _iso(NOW)}

    def _quote(self, symbol: str) -> Quote:
        bid, ask = QUOTES.get(symbol, ("50.00", "50.02"))
        ex = self.clock.now() - timedelta(seconds=1)
        return Quote(symbol=symbol, bid=Decimal(bid), ask=Decimal(ask), bid_size=Decimal(100), ask_size=Decimal(100),
                     exchange_ts=ex, received_ts=ex, available_at_ts=ex, feed=DataFeed.SYNTHETIC,
                     source_id="replay:synthetic-test-v1", is_synthetic=True)

    def _bar(self, symbol: str, start: datetime) -> Bar:
        bid, ask = QUOTES.get(symbol, ("50.00", "50.02"))
        mid = (Decimal(bid) + Decimal(ask)) / 2
        return Bar(symbol=symbol, timeframe="1Min", start_ts=start, end_ts=start + timedelta(minutes=1),
                   open=mid, high=mid + Decimal("0.10"), low=mid - Decimal("0.10"), close=mid, volume=Decimal(1000),
                   feed=DataFeed.SYNTHETIC, session=MarketSession.REGULAR, received_ts=start + timedelta(minutes=1),
                   available_at_ts=start + timedelta(minutes=1), source_id="replay:synthetic-test-v1",
                   dataset_version=self.dataset_version, is_synthetic=True)

    def build_snapshot(self, symbol: str) -> MarketSnapshot:
        self._snap_seq += 1
        now = self.clock.now()
        return MarketSnapshot(
            snapshot_id=f"snap_{symbol}_{self._snap_seq}", symbol=symbol, feed=DataFeed.SYNTHETIC,
            feed_mode=FeedMode.SIMULATED, session=MarketSession.REGULAR, quote=self._quote(symbol),
            last_bar=self._bar(symbol, now - timedelta(minutes=2)), data_asof=now, source_id="replay:synthetic-test-v1",
            dataset_version=self.dataset_version, is_simulated=True, limitations=("synthetic fixture, not market data",),
        )

    def _account(self) -> AccountSnapshot:
        return AccountSnapshot(snapshot_id="acct_fake", account_id_masked="PA****0001", is_paper=True, status="ACTIVE",
                               equity=Decimal("10000.00"), last_equity=Decimal("10000.00"), cash=Decimal("10000.00"),
                               buying_power=Decimal("10000.00"), trading_blocked=False, account_blocked=False,
                               asof=self.clock.now() - timedelta(seconds=1), source="FAKE_BROKER")

    # ------------------------------------------------------------------ ToolRuntime: reads
    def now(self) -> datetime:
        return self.clock.now()

    def status(self) -> dict[str, Any]:
        self._rec("status")
        return {"mode": self.settings.mode.value, "execution_label": self.execution_label.value, "gateway": "FAKE",
                "live_execution_compiled": False, "entries_locked": bool(self.halts),
                "lock_reasons": [f"halt: {h['reason']}" for h in self.halts],
                "halts": [{"halt_id": h["halt_id"], "reason": h["reason"], "source": h["source"],
                           "created_at": _iso(NOW), "cleared_at": None} for h in self.halts],
                "policy_version": self.policy.version}

    def market_bars(self, symbol: str, timeframe: str, limit: int) -> dict[str, Any]:
        self._rec("market_bars")
        now = self.clock.now()
        bars = [self._bar(symbol, now - timedelta(minutes=k + 1)) for k in range(60)][::-1][-limit:]
        return {"symbol": symbol, "timeframe": timeframe, "feed": "SYNTHETIC", "feed_mode": "SIMULATED",
                "is_simulated": True, "source_id": "replay:synthetic-test-v1", "dataset_version": self.dataset_version,
                "limitations": ["synthetic fixture"], "simulated_time": _iso(now),
                "bars": [{"time": int(b.start_ts.timestamp()), "open": format(b.open, "f"), "high": format(b.high, "f"),
                          "low": format(b.low, "f"), "close": format(b.close, "f"), "volume": format(b.volume, "f"),
                          "available_at": _iso(b.available_at_ts)} for b in bars]}

    def market_quality(self) -> dict[str, Any]:
        self._rec("market_quality")
        return {"feed_mode": "SIMULATED", "provider": "replay:synthetic-test-v1", "feed": "SYNTHETIC",
                "is_simulated": True, "last_bar_ts": _iso(NOW), "last_quote_ts": _iso(NOW), "last_trade_ts": None,
                "staleness_s": "1.0", "status": "OK", "detail": "fresh quotes for all tracked symbols",
                "connection_state": "CONNECTED", "max_staleness_s": "15", "limitations": ["synthetic fixture"],
                "symbols": [{"symbol": s, "last_bar_ts": _iso(NOW), "last_quote_ts": _iso(NOW), "staleness_s": "1.0",
                             "spread_bps": "0.80", "gaps_today": 0, "rejected_today": 0} for s in QUOTES],
                "checked_at": _iso(NOW),
                "events": [{"ts": _iso(NOW), "symbol": "SPY", "feed": "SYNTHETIC", "kind": "GAP", "severity": "WARN",
                            "detail": {"minutes": 1}}]}

    async def market_snapshot(self, symbol: str) -> MarketSnapshot:
        self._rec("market_snapshot")
        await asyncio.sleep(0)
        return self.build_snapshot(symbol)

    def predictions(self, symbol: str | None = None) -> dict[str, Any]:
        self._rec("predictions")
        universe = [symbol] if symbol else list(QUOTES)
        now = self.clock.now()
        preds = [PredictionSet(
            prediction_id=f"pred_{s}_1", symbol=s, asof=now, features_asof=now - timedelta(minutes=1),
            dataset_version=self.dataset_version, registry_version="reg_test_1",
            return_forecasts=(HorizonForecast(horizon_minutes=15, expected_return_bps=Decimal("4.2"), model_id="lgbm_reg",
                                              model_version="v1"),),
            success_probability=Decimal("0.54"), success_horizon_minutes=15, probability_calibrated=True,
            calibration_method="isotonic", rank=i + 1, rank_score=Decimal("0.5"), universe_size=len(universe),
            vol_forecast_bps=Decimal("35"), vol_horizon_minutes=15, vol_model="ewma", regime=MarketRegime.CALM,
            regime_probability=Decimal("0.8"), est_cost_bps=Decimal("3.1"), est_slippage_bps=Decimal("1.0"),
            cost_model_id="cost_v1", is_simulated=True,
        ).model_dump(mode="json") for i, s in enumerate(universe)]
        return {"predictions": preds, "asof": _iso(now), "is_simulated": True, "registry_version": "reg_test_1"}

    def model_report(self) -> dict[str, Any]:
        self._rec("model_report")
        return {
            "registry_version": "reg_test_1", "trained_at": _iso(NOW - timedelta(days=1)),
            "dataset_version": self.dataset_version, "is_simulated": True, "selection_metric": "net_sharpe",
            "targets": [{"target": "ret_15m", "selected_model": "lgbm_reg", "candidates": [
                {"model_id": "lgbm_reg", "net_sharpe": 0.41, "net_return": "0.012", "turnover": 3.2, "hit_rate": 0.52,
                 "brier": None, "ic": 0.03, "n_test": 1800, "selected": True, "rejected_reason": None},
                {"model_id": "ridge", "net_sharpe": -0.1, "net_return": "-0.004", "turnover": 4.0, "hit_rate": 0.49,
                 "brier": None, "ic": 0.0, "n_test": 1800, "selected": False,
                 "rejected_reason": "negative net sharpe after costs"},
            ], "calibration": {"method": "isotonic", "brier_raw": 0.26, "brier_calibrated": 0.24,
                               "reliability": [{"bin": 0, "pred": 0.4, "obs": 0.41, "n": 100}]}}],
            "walk_forward": {"folds": 5, "purge_minutes": 15, "embargo_minutes": 30,
                             "frozen_test": {"start": "2026-10-01", "end": "2026-10-31", "used": False}},
            "multiple_testing": {"method": "holm", "alpha": 0.05, "survivors": ["lgbm_reg"]},
            "notes": ["synthetic data: metrics are not evidence of an edge"],
        }

    def signals(self, limit: int = 50) -> dict[str, Any]:
        self._rec("signals")
        now = self.clock.now()
        sigs = [QuantSignal(
            signal_id=f"sig_{s}_1", strategy_id="quant-baseline-v1", model_version="baseline-1", symbol=s, side=Side.BUY,
            horizon_minutes=15, score=Decimal("1.5"), rank=i + 1, expected_return_bps=Decimal("12"),
            probability_up=None, est_spread_bps=Decimal("0.8"), est_fees_bps=Decimal("0.3"),
            est_slippage_bps=Decimal("1.0"), net_edge_bps=Decimal("9.9"), is_candidate=(i == 0),
            reason="momentum above threshold net of costs", features_asof=now - timedelta(minutes=1), generated_at=now,
            dataset_version=self.dataset_version, is_simulated=True,
        ).model_dump(mode="json") for i, s in enumerate(QUOTES)]
        rows: list[dict[str, Any]] = [*sigs, {"signal_id": "sig_broken", "symbol": "SPY", "score": 1.5}]
        return {"signals": rows[:limit],
                "latest_cycle": {"causal_id": "cz_prev", "ran_at": _iso(NOW), "candidate_symbol": "SPY",
                                 "outcome": "NO_TRADE", "reason": "entries locked", "client_order_id": None,
                                 "execution_label": "SIMULATED"}}

    def portfolio(self) -> dict[str, Any]:
        self._rec("portfolio")
        acct = self._account().model_dump(mode="json")
        if self.corrupt_portfolio:
            acct["equity"] = 10000.5  # a binary float in a money field: must be withheld (fail closed)
        return {"execution_label": self.execution_label.value, "account": acct,
                "broker_positions": [{"symbol": "QQQ", "quantity": "2", "avg_entry_price": "400.03", "cost_basis": "800.06",
                                      "market_value": "800.04", "current_price": "400.02", "unrealized_pl": "-0.02",
                                      "asof": _iso(NOW), "source": "FAKE_BROKER", "schema_version": "1.0.0"}],
                "ledger_positions": [{"symbol": "QQQ", "quantity": "2", "avg_cost": "400.03", "realized_pl": "0"}],
                "pnl": {"daily_realized": "0", "daily_unrealized": "-0.02", "start_of_day_equity": "10000.00"},
                "equity_curve": [{"time": int(NOW.timestamp()), "equity": "10000.00"}] * 500}

    def orders(self, limit: int = 100) -> dict[str, Any]:
        self._rec("orders")
        return {"orders": list(self._orders.values())[::-1][:limit]}

    def order_detail(self, client_order_id: str) -> dict[str, Any] | None:
        self._rec("order_detail")
        row = self._orders.get(client_order_id)
        if row is None:
            return None
        intent = {"intent_id": row["intent_id"], "client_order_id": client_order_id, "proposal_id": "prop_x",
                  "decision_id": "dec_x", "policy_version": self.policy.version, "expires_at": _iso(NOW + timedelta(minutes=1)),
                  "requires_human_approval": False, "key_id": "key_abc", "signature": SIGNATURE_HEX}
        return {"order": dict(row), "intent": intent,
                "events": [{"event_id": "ev1", "applied": True, "apply_note": None}],
                "transitions": [{"from_state": None, "to_state": "PROPOSED", "at": _iso(NOW), "reason": "created"},
                                {"from_state": "PROPOSED", "to_state": row["state"], "at": _iso(NOW), "reason": "gateway"}],
                "fills": ([{"fill_id": "fill_1", "quantity": row["filled_quantity"], "price": row["avg_fill_price"],
                            "ts": _iso(NOW), "broker_exec_id": "x1"}] if row["avg_fill_price"] else [])}

    def risk_view(self) -> dict[str, Any]:
        self._rec("risk_view")
        pol = self.policy
        return {"policy": pol.model_dump(mode="json"), "policy_version": pol.version, "history": [],
                "utilization": [{"name": "pending_orders", "used": "1", "limit": str(pol.max_pending_orders), "pct": "100.0"},
                                {"name": "daily_loss", "used": "0", "limit": format(pol.max_daily_loss, "f"), "pct": "0.0"}],
                "entries_locked": bool(self.halts),
                "recent_decisions": [d.model_dump(mode="json") for d in self.decisions[-5:]]}

    def performance(self) -> dict[str, Any]:
        self._rec("performance")
        return {"mode_label": "SIMULATED", "portfolio_value": "10000.00", "cash": "9199.94", "daily_pnl": "-0.02",
                "total_pnl": "12.48", "realized_pnl": "12.50", "unrealized_pnl": "-0.02", "win_rate": "0.5000",
                "round_trips": 2, "max_drawdown": "0.0120", "current_drawdown": "0.0040", "active_positions": 1,
                "open_orders": 1, "api_cost_usd": "0.0420",
                "equity_curve": [{"time": int(NOW.timestamp()) - 60 * k, "equity": "10000.00"} for k in range(40)][::-1],
                "asof": _iso(NOW), "is_simulated": True}

    def health_report(self) -> dict[str, Any]:
        self._rec("health_report")
        return {"overall": "DEGRADED", "asof": _iso(NOW), "components": [
            {"name": "market_data", "status": "OK", "detail": "OK SIMULATED replay:synthetic-test-v1", "last_ok": _iso(NOW)},
            {"name": "broker", "status": "OK", "detail": "FAKE: ok", "last_ok": None},
            {"name": "models", "status": "DEGRADED", "detail": "no valid model registry (quant baseline in use)",
             "last_ok": None},
            {"name": "gpt_runtime", "status": "DISABLED", "detail": "mode CLAUDE_CODE", "last_ok": None},
            {"name": "scheduler", "status": "OK", "detail": "session scheduler running", "last_ok": None}]}

    # ------------------------------------------------------------------ ToolRuntime: mutating
    async def submit_agent_proposal(self, proposal: TradeProposal, actor: str) -> dict[str, Any]:
        self._rec("submit_agent_proposal")
        await asyncio.sleep(0)
        if self.refuse or self.execution_label == ExecutionLabel.LIVE or self.settings.mode.value not in ("PAPER", "REPLAY"):
            return {"outcome": "REFUSED", "reason": "agent proposals are paper/replay only"}
        if self.rate_limited:
            return {"outcome": "RATE_LIMITED", "reason": "agent proposal limit 6/hour reached"}
        self.proposals.append(proposal)
        assert proposal.symbol is not None
        snapshot = self.build_snapshot(proposal.symbol)
        ctx = RiskContext(now=self.clock.now(), mode=self.settings.mode, gateway=GatewayKind.FAKE, account=self._account(),
                          positions=(), open_orders=(), start_of_day_equity=Decimal("10000.00"), market=snapshot,
                          health=SystemHealth(feed_ok=True, broker_connected=True, reconciled=True, engine_ok=True,
                                              entries_locked=bool(self.halts),
                                              lock_reasons=tuple(h["reason"] for h in self.halts)))
        decision = self.risk.evaluate(proposal, ctx)
        self.decisions.append(decision)
        out: dict[str, Any] = {"proposal": proposal.model_dump(mode="json"), "causal_id": proposal.causal_id,
                               "decision": decision.model_dump(mode="json")}
        if not decision.approved:
            out.update(outcome="REJECTED", reason="; ".join(decision.reasons))
            return out
        intent = self.risk.authorize(proposal, decision, ctx)
        row = self._order_row(intent.client_order_id, intent.symbol, intent.side.value, format(intent.quantity, "f"),
                              format(intent.limit_price, "f"), OrderState.SUBMITTED.value)
        self._orders[intent.client_order_id] = row
        out.update(outcome="SUBMITTED", client_order_id=intent.client_order_id,
                   order={**row, "state": OrderState.SUBMITTED.value})
        return out

    def halt(self, reason: str, source: str) -> str:
        self._rec("halt")
        halt_id = f"halt_{len(self.halts) + 1}"
        self.halts.append({"halt_id": halt_id, "reason": reason, "source": source})
        return halt_id

    async def cancel(self, client_order_id: str) -> dict[str, Any]:
        self._rec("cancel")
        self.cancels.append(client_order_id)
        self._orders[client_order_id]["state"] = OrderState.CANCELED.value
        return {"client_order_id": client_order_id, "state": OrderState.CANCELED}

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str) -> dict[str, Any]:
        self._rec("run_cycle")
        self.cycle_args.append({"symbols": symbols, "use_ai": use_ai, "actor": actor})
        return {"causal_id": "cz_cycle_1", "ran_at": _iso(self.clock.now()), "actor": actor,
                "execution_label": self.execution_label.value, "symbols": symbols or list(QUOTES),
                "entries_locked": False, "lock_reasons": [], "candidate_symbol": None, "signals": 4,
                "outcome": "NO_TRADE", "reason": "no candidate cleared the net-of-cost edge threshold"}

    # ------------------------------------------------------------------ FastAPI lifespan hooks (services.api.app)
    async def start(self, stop: asyncio.Event) -> list[asyncio.Task[Any]]:
        return []

    async def aclose(self) -> None:
        return None

    def now_iso(self) -> str:
        return _iso(self.clock.now())


def _session_row(sid: str, state: str, *, symbols: list[str], requested_by: str, cycles: int = 0) -> dict[str, Any]:
    return {"session_id": sid, "state": state, "started_at": _iso(NOW - timedelta(hours=1)),
            "ends_at": _iso(NOW if state != "RUNNING" else NOW + timedelta(hours=1)),
            "ended_at": None if state in ("RUNNING", "STOPPING") else _iso(NOW), "duration_minutes": 60,
            "symbols": symbols, "cadence_minutes": 5, "use_ai": False, "requested_by": requested_by,
            "cycles_run": cycles, "trades": 0, "stop_reason": None if state == "RUNNING" else "planned duration elapsed",
            "schedule_id": None, "created_at": _iso(NOW - timedelta(hours=1)), "next_cycle_at": None,
            "last_cycle_at": _iso(NOW) if cycles else None, "execution_label": "SIMULATED", "ai_suspended_until": None}


class FakeSessionControl:
    """Deterministic stand-in for ``services.sessions.manager.SessionManager`` (records ``sessions.<method>``)."""

    def __init__(self, parent: FakeToolRuntime) -> None:
        self._p = parent
        self.rows: dict[str, dict[str, Any]] = {
            "sess_fake_0": _session_row("sess_fake_0", "COMPLETED", symbols=["SPY"], requested_by="agent:seed", cycles=12)}
        self.schedules: dict[str, dict[str, Any]] = {"sched_fake_1": {
            "schedule_id": "sched_fake_1", "weekdays": ["MON", "WED"], "start_time_et": "09:45", "duration_minutes": 60,
            "symbols": ["SPY"], "cadence_minutes": 5, "use_ai": False, "enabled": True,
            "next_run_at": "2026-11-09T14:45:00Z", "created_by": "agent:seed", "created_at": _iso(NOW),
            "cancelled_at": None, "last_run": None}}
        self.start_args: list[dict[str, Any]] = []
        self.stop_args: list[dict[str, Any]] = []

    def search_memory(self, query: str, *, symbol: str | None = None,
                      kind: str | None = None, limit: int = 10) -> dict[str, Any]:
        self._rec("search_memory")
        return {"hits": [], "execution_authority": False}

    def _rec(self, name: str) -> None:
        self._p._rec(f"sessions.{name}")  # pyright: ignore[reportPrivateUsage]

    def _active(self) -> dict[str, Any] | None:
        return next((r for r in self.rows.values() if r["state"] in ("RUNNING", "STOPPING")), None)

    def start(self, duration_minutes: int, *, symbols: Any = None, cadence_minutes: int = 5, use_ai: bool = True,
              requested_by: str) -> dict[str, Any]:
        self._rec("start")
        self.start_args.append({"duration_minutes": duration_minutes, "symbols": symbols,
                                "cadence_minutes": cadence_minutes, "use_ai": use_ai, "requested_by": requested_by})
        act = self._active()
        if act is not None:
            raise SessionError(409, f"a paper session is already active ({act['session_id']}); stop it first")
        sid = f"sess_fake_{len(self.rows)}"
        row = _session_row(sid, "RUNNING", symbols=list(symbols or QUOTES), requested_by=requested_by)
        row.update(duration_minutes=duration_minutes, cadence_minutes=cadence_minutes, use_ai=use_ai,
                   started_at=_iso(NOW), ends_at=_iso(NOW + timedelta(minutes=duration_minutes)), next_cycle_at=_iso(NOW))
        self.rows[sid] = row
        return dict(row)

    def stop(self, session_id: str | None = None, *, reason: str, requested_by: str) -> dict[str, Any]:
        self._rec("stop")
        self.stop_args.append({"session_id": session_id, "reason": reason, "requested_by": requested_by})
        row = self.rows.get(session_id) if session_id else self._active()
        if row is None:
            if session_id:
                raise SessionError(404, f"unknown session {session_id}")
            return {"outcome": "NO_ACTIVE_SESSION", "session": None}
        if row["state"] not in ("RUNNING", "STOPPING"):
            return {"outcome": "ALREADY_ENDED", "session": dict(row)}
        row.update(state="STOPPED", ended_at=_iso(NOW), stop_reason=f"stopped by {requested_by}: {reason}")
        return {"outcome": "STOPPED", "session": dict(row)}

    def add_schedule(self, weekdays: Any, start_time_et: str, duration_minutes: int, symbols: Any = None,
                     cadence_minutes: int = 5, use_ai: bool = True, created_by: str = "system") -> dict[str, Any]:
        self._rec("add_schedule")
        sid = f"sched_fake_{len(self.schedules) + 1}"
        row = {**self.schedules["sched_fake_1"], "schedule_id": sid, "weekdays": list(weekdays),
               "start_time_et": start_time_et, "duration_minutes": duration_minutes, "symbols": list(symbols or QUOTES),
               "cadence_minutes": cadence_minutes, "use_ai": use_ai, "created_by": created_by, "enabled": True}
        self.schedules[sid] = row
        return dict(row)

    def cancel_schedule(self, schedule_id: str, *, requested_by: str = "system") -> dict[str, Any]:
        self._rec("cancel_schedule")
        row = self.schedules.get(schedule_id)
        if row is None:
            raise SessionError(404, f"unknown schedule {schedule_id}")
        if not row["enabled"]:
            return {"outcome": "ALREADY_CANCELLED", "schedule": dict(row)}
        row.update(enabled=False, next_run_at=None, cancelled_at=_iso(NOW))
        return {"outcome": "CANCELLED", "schedule": dict(row)}

    def active(self) -> dict[str, Any] | None:
        self._rec("active")
        act = self._active()
        return None if act is None else dict(act)

    def list(self, limit: int = 20) -> builtins.list[dict[str, Any]]:
        self._rec("list")
        return [dict(r) for r in builtins.list(self.rows.values())[::-1][:limit]]

    def report(self, session_id: str | None = None) -> dict[str, Any] | None:
        self._rec("report")
        sid = session_id or (self._active() or {}).get("session_id") or next(reversed(self.rows), None)
        row = self.rows.get(sid) if sid else None
        if row is None:
            return None
        return {"session": dict(row), "performance": self._p.performance(), "cycles": row["cycles_run"],
                "decisions": {"NO_TRADE": 10, "REJECTED": 2, "SUBMITTED": 0},
                "top_no_trade_reasons": [{"reason": "no candidate cleared the net-of-cost edge threshold", "count": 10}],
                "top_risk_rejections": [{"check": "max_spread_bps", "count": 2}],
                "trades": [{"client_order_id": "lat-sim-done1", "symbol": "QQQ", "side": "BUY", "qty": "2",
                            "filled_qty": "2", "avg_price": "400.03", "state": "FILLED", "pnl": "-0.02",
                            "pnl_basis": "unrealized"}],
                "api_cost_usd": "0", "narrative": "SIMULATED paper session sess_fake_0 on SPY completed its planned "
                                                  "window. It ran 12 decision cycle(s): 10 NO_TRADE, 2 rejected."}

    def list_schedules(self) -> builtins.list[dict[str, Any]]:
        self._rec("list_schedules")
        return [dict(r) for r in self.schedules.values()]

    def explain(self, limit: int = 10) -> dict[str, Any]:
        self._rec("explain")
        cyc = {"causal_id": "cz_prev", "ran_at": _iso(NOW), "session_id": "sess_fake_0", "outcome": "REJECTED",
               "candidate_symbol": "SPY", "reason": "max_spread_bps: spread 9.1 bps > 5", "no_trade_reason": None,
               "failed_checks": ["max_spread_bps: spread 9.1 bps > 5"], "entries_locked": False, "lock_reasons": [],
               "ai_review": "skipped: AI disabled for this cycle", "note": None, "client_order_id": None}
        return {"asof": _iso(NOW), "answer": "Last 1 decision cycle(s): 1 REJECTED. No order was submitted.",
                "cycles": [cyc][:limit],
                "aggregate": {"cycles": 1, "by_outcome": {"REJECTED": 1}, "top_no_trade_reasons": [],
                              "top_failed_checks": [{"check": "max_spread_bps", "count": 1}],
                              "entries_locked_now": False, "lock_reasons_now": []}}

    def scheduler_status(self) -> dict[str, Any]:
        self._rec("scheduler_status")
        act = self._active()
        return {"loop_running": True, "recovered": True, "last_tick_at": _iso(NOW), "market_open": True,
                "next_open": None, "active_session_id": act["session_id"] if act else None, "process_id": "proc_fake"}


class FakeActivity:
    def __init__(self, parent: FakeToolRuntime) -> None:
        self._p = parent

    def recent(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        self._p._rec("activity.recent")  # pyright: ignore[reportPrivateUsage]
        rows: list[dict[str, Any]] = [
            {"seq": 3, "ts": _iso(NOW), "kind": "session", "event": "activity", "actor": "system",
             "message": "Session completed: planned duration elapsed.", "causal_id": None, "session_id": "sess_fake_0",
             "detail": {}},
            {"seq": 2, "ts": _iso(NOW), "event": "decision", "kind": "decision", "action": "NO_TRADE", "symbol": "SPY",
             "status": "NO_TRADE", "reason": "no candidate cleared the net-of-cost edge threshold", "causal_id": "cz_prev",
             "session_id": "sess_fake_0"},
            {"seq": 1, "ts": _iso(NOW), "event": "stage", "kind": "stage", "stage": "MODELS", "status": "passed",
             "item": "SPY", "detail": "4 scored", "causal_id": "cz_prev", "session_id": "sess_fake_0"},
        ]
        return rows[:limit]


class GuardedRuntime:
    """Proxy that allows ONLY ``ToolRuntime`` protocol members and records every access."""

    def __init__(self, inner: FakeToolRuntime, allowed: frozenset[str]) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "accessed", [])

    def __getattr__(self, name: str) -> Any:
        self.accessed.append(name)
        if name not in self._allowed:
            raise AssertionError(f"tool handler touched runtime member outside the ToolRuntime protocol: {name}")
        return getattr(self._inner, name)
