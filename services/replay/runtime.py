"""``ReplayRuntime``: the real runtime composition, rebuilt on a disposable workspace and a simulated clock.

``ReplayRuntime`` subclasses :class:`services.api.runtime.Runtime` but deliberately does NOT call
``Runtime.__init__``: that constructor configures process logging, reads the keychain, builds the AI research
pipeline, the session scheduler and the bundled synthetic fixture. Instead it wires the same service objects the
runtime wires (same classes, same arguments) on the workspace given to it, in mode ``REPLAY`` with the ``FAKE``
gateway (client order ids tagged ``replay``), and inherits the runtime's own order-path methods unchanged:

``run_cycle``/``DecisionCycle`` -> ``execute_proposal`` -> ``risk_context`` (account, positions, open orders,
recent submissions and fingerprints, start-of-day equity, daily P&L) -> ``RiskEngine.evaluate`` -> ``authorize``
(ECDSA-signed intent, a key generated inside the workspace) -> ``OrderLedger.record_intent`` -> ``prepare_gate`` ->
``ExecutionService.submit`` (entries/exits lock, ``_final_gate`` = ``RiskEngine.final_gate`` on a fresh context,
ledger claim, gateway) -> ``ReplayBroker``. The ``PositionManager`` talks to it through the runtime's own
``RuntimePositionPort``.

Overrides (each a narrow, documented adapter):

* ``current_signal_engine``: the strategy's ``SignalModel`` inside the real ``QuantSignalEngine``;
* ``_prediction_service``: no trained registry in a replay workspace (``None``, like a fresh install);
* ``run_cycle``: ``ReplayDecisionCycle`` (the real ``DecisionCycle``; only the proposal's strategy id and
  ``holding_sessions`` are set from the strategy, then the proposal is re-validated);
* ``execute_proposal``/``reconcile``: the inherited method plus a synchronous pump of broker events into the ledger
  (the runtime streams them in an asyncio task; the replay applies them deterministically after every call).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from packages.contracts.enums import AiRuntimeMode, DecisionAction, FeedMode, GatewayKind, OrderState, TradingMode
from packages.contracts.models import QuantSignal, TradeProposal
from packages.contracts.policy import RiskPolicy
from packages.core import modes
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from packages.core.signing import IntentSigner
from services.api.events import EventBus
from services.api.pipeline import DecisionCycle
from services.api.runtime import Runtime, RuntimePositionPort
from services.api.simclock import SimulationClock
from services.order_ledger.live_gate import record_mode_decision
from services.positions.policy import PositionPolicy

from .broker import FillModel, ReplayBroker
from .feed import ReplayFeed
from .strategies import ReplayStrategy

REPLAY_ACTOR = "replay-harness"


class ReplaySettings(Settings):
    """Use runtime settings defaults and explicit replay overrides in a disposable workspace."""

    model_config = SettingsConfigDict(env_file=None)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class ReplayDecisionCycle(DecisionCycle):
    """The runtime's DecisionCycle; the strategy only names its proposal and its holding period."""

    def _quant_proposal(self, causal_id: str, candidate: QuantSignal | None, snapshot: Any, evidence: list[Any],
                        note: str) -> TradeProposal:
        proposal = super()._quant_proposal(causal_id, candidate, snapshot, evidence, note)
        strategy = cast(ReplayRuntime, self.rt).strategy
        update: dict[str, Any] = {}
        if strategy.proposal_strategy_id:
            update["strategy_id"] = strategy.proposal_strategy_id
        if proposal.action == DecisionAction.TRADE and strategy.holding_sessions > 1:
            update["holding_sessions"] = strategy.holding_sessions
        if not update:
            return proposal
        return TradeProposal.model_validate({**proposal.model_dump(), **update})


class ReplayRuntime(Runtime):
    def __init__(self, workspace: Path, *, feed: ReplayFeed, strategy: ReplayStrategy, equity: Decimal,
                 sessions: list[tuple[datetime, datetime]], policy: RiskPolicy | None = None,
                 fill_model: FillModel | None = None, position_policy: PositionPolicy | None = None) -> None:
        # Runtime.__init__ is intentionally not called (see the module docstring).
        from services.api.activity import ActivityLog
        from services.market_data.quality import FeedQualityMonitor
        from services.order_ledger.entries import EntriesLock
        from services.order_ledger.execution import ExecutionService
        from services.order_ledger.ledger import OrderLedger
        from services.order_ledger.reconcile import Reconciler
        from services.positions.manager import PositionManager
        from services.positions.store import PositionStore
        from services.research.costs import CostModel
        from services.risk.operator_auth import OperatorAuth
        from services.risk.policy_store import PolicyStore

        if not isinstance(equity, Decimal) or not equity.is_finite() or equity <= 0:
            raise ValueError("equity must be a positive Decimal")
        self.workspace = workspace
        self.strategy = strategy
        self.fill_model = fill_model or FillModel()
        symbols = tuple(feed.snapshot.symbols)
        self.settings = ReplaySettings(project_root=workspace, trading_mode=None, mode=TradingMode.REPLAY,
                                       gateway=GatewayKind.FAKE, feed_mode=FeedMode.SIMULATED, symbols=symbols,
                                       ai_runtime=AiRuntimeMode.NONE, ai_transport="fake", use_keychain=False,
                                       data_pipeline_enabled=False, quantlab_daily_enabled=False)
        settings = self.settings
        settings.ensure_dirs()
        self.secrets = SecretStore(workspace, use_keychain=False, env={})
        self.db = Database(settings.db_path)
        self.bus = EventBus()
        self.tailer = cast(Any, None)
        self.started_at = datetime.now(UTC)
        self.startup_phase = "STARTING"
        self.startup_detail = ""
        modes.assert_gateway_allowed(settings.mode, settings.gateway)
        self.execution_label = modes.execution_label(settings.mode, settings.gateway)
        self.feed_mode = FeedMode.SIMULATED
        # ---- market data (snapshot adapter in place of the DuckDB store and the replay provider)
        self.replay_feed = feed
        self.store = cast(Any, feed)
        self.sim_clock = SimulationClock(sessions, start_session=0, start_offset_min=0, speed=1.0)
        self.clock = self.sim_clock
        self.dataset_version = feed.dataset_version
        self.data_feed = feed.bar_feed
        self.provider = feed
        self.provider_name = f"replay:{feed.dataset_version}"
        self.activity = ActivityLog(self.db, self.bus, self.clock)
        self._cycle_lock = asyncio.Lock()
        self.submit_lock = asyncio.Lock()
        self.feed_monitor = FeedQualityMonitor(self.feed_mode, self.provider_name, self.clock, symbols=symbols)
        # ---- risk + ledger (a signing key and an operator token generated inside the workspace)
        self.operator_auth = OperatorAuth(settings.keys_dir)
        self.operator_auth.ensure_token()
        self.signer = IntentSigner.load_or_create(settings.keys_dir)
        self.verifier = self.signer.verifier
        self.policy_store = PolicyStore(self.db, settings.mode, operator_auth=self.operator_auth, clock=self.clock)
        self.policy_store.ensure_default()
        if policy is not None and self.policy_store.active().version != policy.version:
            token = self.operator_auth.token_path.read_text().strip()
            self.policy_store.apply_change(policy, operator_token=token, actor=REPLAY_ACTOR,
                                           reason="replay policy profile (disposable replay workspace only)")
        self.ledger = OrderLedger(self.db, self.clock, settings.mode, verifier=self.verifier)
        self.entries = EntriesLock(self.db, settings.mode, clock=self.clock)
        self._build_risk()
        # ---- execution: the FakeBroker wiring of services.order_ledger.gateway_factory.build_gateway (FAKE)
        ledger = self.ledger

        def claimed_in_ledger(coid: str) -> bool:
            rec = ledger.get_record(coid)
            return rec is not None and rec.state == OrderState.SUBMITTING and rec.gateway == GatewayKind.FAKE

        self.broker = ReplayBroker(self.clock, self._quote_source, starting_cash=equity, verifier=self.verifier,
                                   state_path=settings.var_dir / "fake_broker_state.json", poll_interval_s=0.02,
                                   claim_check=claimed_in_ledger, fill_model=self.fill_model)
        self.gateway = self.broker
        record_mode_decision(self.db, requested_mode=settings.mode.value, actor=REPLAY_ACTOR, result="ACCEPTED",
                             detail="gateway=FAKE label=SIMULATED: replay fill simulator (FakeBroker subclass)")
        self.execution = ExecutionService(self.ledger, self.gateway, self._final_gate, self.clock,
                                          submit_timeout_s=10.0, entries_lock=self.entries)
        self.reconciler = Reconciler(self.db, self.ledger, self.gateway, self.entries, self.clock, confirm_delay_s=0.0)
        # ---- positions + research (the runtime's builders)
        self.position_policy = position_policy or PositionPolicy()
        self.position_store = PositionStore(self.db, settings.mode.value)
        self.positions = PositionManager(RuntimePositionPort(self), self.position_store, self.position_policy)
        self.cost_model = CostModel()
        self.signal_engine = None
        self.sessions = None
        self._record_mode_decision()

    # ------------------------------------------------------------------ adapters
    def sim_clock_set(self, when: datetime) -> None:
        """Move the simulated clock (the engine only ever moves it forward)."""
        assert self.sim_clock is not None
        self.sim_clock.set(when)

    def current_signal_engine(self) -> Any:
        from services.research.signals import QuantSignalEngine

        if self.signal_engine is None:
            s = self.strategy
            self.signal_engine = QuantSignalEngine(
                self.cost_model, model=s.model_factory(), clock=self.clock, horizon_minutes=s.horizon_minutes,
                min_net_edge_bps=s.min_net_edge_bps, horizon_within_session=s.horizon_within_session)
        return self.signal_engine

    def _prediction_service(self) -> Any:
        return None

    def pump(self) -> int:
        """Apply every broker event emitted so far to the ledger (the runtime's order-events task, synchronously)."""
        n = 0
        for ev in self.broker.drain_events():
            self.execution.handle_event(ev)
            n += 1
        return n

    async def execute_proposal(self, proposal: Any, snapshot: Any) -> dict[str, Any]:
        out = await super().execute_proposal(proposal, snapshot)
        self.pump()
        return out

    async def reconcile(self) -> Any:
        self.pump()
        report = await super().reconcile()
        self.pump()
        return report

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str,
                        session_id: str | None = None) -> dict[str, Any]:
        async with self._cycle_lock:
            cycle = ReplayDecisionCycle(self, session_id=session_id)
            result = await cycle.run(symbols=symbols or list(self.settings.symbols), use_ai=False, actor=actor)
        self.pump()
        return result

    async def startup(self) -> Any:
        """The runtime's startup recovery: SUBMITTING -> UNKNOWN, broker-authoritative reconciliation."""
        report = await self.reconciler.startup_recovery()
        self.pump()
        self._after_reconcile(report)
        self.startup_phase = "READY" if report.status.value == "CLEAN" else "LOCKED"
        return report

    async def aclose(self) -> None:
        await self.broker.aclose()
        self.db.close()
