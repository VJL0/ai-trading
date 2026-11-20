"""Shared factories for services/ai tests (no network, no credentials, deterministic clocks)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts import MarketSnapshot, QuantSignal, Quote, RiskPolicy
from packages.contracts.enums import DataFeed, FeedMode, MarketSession, Side
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from services.ai.config import AiConfig
from services.ai.evidence import build_internal_evidence
from services.ai.fake_transport import FakeAnthropicHandler, FakeOpenAIHandler, FakeScenario, fake_http_client
from services.ai.pipeline import ResearchPipeline

T0 = datetime(2026, 11, 5, 15, 30, tzinfo=UTC)  # 10:30 America/New_York, regular session


def make_quote(symbol: str = "SPY", *, at: datetime = T0, bid: str = "500.00", ask: str = "500.04") -> Quote:
    return Quote(
        symbol=symbol, bid=Decimal(bid), ask=Decimal(ask), bid_size=Decimal("300"), ask_size=Decimal("200"),
        exchange_ts=at - timedelta(seconds=2), received_ts=at - timedelta(seconds=1),
        available_at_ts=at - timedelta(seconds=1), feed=DataFeed.SYNTHETIC, source_id="synthetic-fixture",
        is_synthetic=True,
    )


def make_snapshot(symbol: str = "SPY", *, at: datetime = T0, quote: Quote | str | None = "default",
                  snapshot_id: str = "snap_test_0001") -> MarketSnapshot:
    q = make_quote(symbol, at=at) if quote == "default" else quote
    assert not isinstance(q, str)
    return MarketSnapshot(
        snapshot_id=snapshot_id, symbol=symbol, feed=DataFeed.SYNTHETIC, feed_mode=FeedMode.SIMULATED,
        session=MarketSession.REGULAR, quote=q, last_trade_price=Decimal("500.02"), data_asof=at,
        source_id="synthetic-fixture", dataset_version="synthetic-v1", is_simulated=True,
        limitations=("SIMULATED synthetic fixture",),
    )


def make_signal(symbol: str = "SPY", *, at: datetime = T0, is_candidate: bool = True,
                signal_id: str = "sig_test_0001") -> QuantSignal:
    return QuantSignal(
        signal_id=signal_id, strategy_id="momentum-v1", model_version="baseline-1", symbol=symbol, side=Side.BUY,
        horizon_minutes=30, score=Decimal("1.25"), rank=1, expected_return_bps=Decimal("18.0"),
        probability_up=None, probability_calibrated=False, est_spread_bps=Decimal("0.8"), est_fees_bps=Decimal("0.3"),
        est_slippage_bps=Decimal("1.0"), net_edge_bps=Decimal("15.9"), is_candidate=is_candidate,
        reason="top momentum rank net of costs", features_asof=at - timedelta(minutes=1), generated_at=at,
        dataset_version="synthetic-v1", is_simulated=True,
    )


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(T0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False)


@pytest.fixture
def secrets(tmp_path: Path) -> SecretStore:
    return SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "no.env")


@pytest.fixture
def policy() -> RiskPolicy:
    return RiskPolicy()


@pytest.fixture
def ai_config() -> AiConfig:
    return AiConfig(min_wall_interval_per_symbol_s=0, min_interval_per_symbol_s=0)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    d = Database(tmp_path / "ledger.sqlite")
    yield d
    d.close()


class Harness:
    """A pipeline wired to explicit fake handlers so tests can choose scenarios and inspect request bodies."""

    def __init__(self, settings: Settings, secrets: SecretStore, clock: ManualClock, config: AiConfig,
                 db: Database | None, sol: FakeScenario, claude: FakeScenario) -> None:
        self.sol = FakeOpenAIHandler(sol)
        self.claude = FakeAnthropicHandler(claude)
        self.pipeline = ResearchPipeline(
            settings, secrets, transport="fake", db=db, clock=clock, config=config, wall_clock=clock,
            http_clients={"openai": fake_http_client(self.sol), "anthropic": fake_http_client(self.claude)},
        )

    async def run(self, *, signal: QuantSignal | None = None, snapshot: MarketSnapshot | None = None,
                  policy: RiskPolicy | None = None, extra_evidence: list[Any] | None = None,
                  evidence: list[Any] | None = None, causal_id: str = "cz_test") -> Any:
        sig = make_signal() if signal is None else signal
        snap = make_snapshot() if snapshot is None else snapshot
        pol = RiskPolicy() if policy is None else policy
        ev = build_internal_evidence(sig, snap, pol) if evidence is None else evidence
        return await self.pipeline.run(causal_id=causal_id, candidate=sig, snapshot=snap,
                                       evidence=[*ev, *(extra_evidence or [])], policy=pol)


@pytest.fixture
def harness_factory(settings: Settings, secrets: SecretStore, clock: ManualClock, ai_config: AiConfig):
    def _make(sol: FakeScenario = FakeScenario.TRADE, claude: FakeScenario = FakeScenario.TRADE,
              db: Database | None = None, config: AiConfig | None = None) -> Harness:
        return Harness(settings, secrets, clock, config or ai_config, db, sol, claude)

    return _make


@pytest.fixture
async def chatgpt_factory(tmp_path: Path, clock: ManualClock):
    """Build ChatGptHarness objects (fake server + AiRuntime); every harness is closed after the test."""
    from .chatgpt_helpers import ChatGptHarness

    made: list[Any] = []

    def _make(**kw: Any) -> ChatGptHarness:
        h = ChatGptHarness(tmp_path, clock, **kw)
        made.append(h)
        return h

    yield _make
    for h in made:
        await h.aclose()
