"""services/ai integrated with the real market-data fixture, quant signal engine, snapshot builder and SQLite ledger.

Model calls go through the explicit FAKE transport (no credentials exist); every run is recorded transport=FAKE.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.contracts import RiskPolicy
from packages.contracts.enums import DataFeed, DecisionAction, FeedMode, ModelTransport
from packages.core.audit import verify_chain
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from services.ai.config import AiConfig
from services.ai.evidence import build_internal_evidence
from services.ai.pipeline import ResearchPipeline
from tests.unit.ai.conftest import make_signal, make_snapshot

SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT")


async def test_real_signal_and_snapshot_through_fake_models(tmp_path: Path) -> None:
    from services.market_data.fixtures import ensure_synthetic_fixture
    from services.market_data.replay import ReplayMarketDataProvider
    from services.market_data.snapshot import build_market_snapshot
    from services.market_data.store import MarketDataStore
    from services.research.costs import CostModel
    from services.research.signals import QuantSignalEngine

    datasets = tmp_path / "datasets"
    store = MarketDataStore(datasets)
    manifest = ensure_synthetic_fixture(datasets, symbols=SYMBOLS, sessions=3, store=store)
    asof = datetime(2026, 11, 4, 19, 0, tzinfo=UTC)  # 14:00 America/New_York inside the fixture's last session
    clock = ManualClock(asof)
    bars = store.load_bars_df(list(SYMBOLS), asof - timedelta(days=10), asof, "1Min", asof=asof)
    signals = QuantSignalEngine(CostModel()).generate(bars, asof, list(SYMBOLS), "cz_integration")
    assert signals, "quant engine returned no signals"
    candidate = next((s for s in signals if s.is_candidate), None)
    if candidate is None:  # the engine found no edge: exercise the AI path on its top-ranked signal explicitly
        candidate = min(signals, key=lambda s: s.rank).model_copy(update={"is_candidate": True})
    provider = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    snapshot = await build_market_snapshot(provider, candidate.symbol, asof, FeedMode.SIMULATED,
                                           dataset_version=str(manifest["dataset_version"]))
    assert snapshot.quote is not None, snapshot.limitations
    policy = RiskPolicy()
    evidence = build_internal_evidence(candidate, snapshot, policy)
    settings = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False)
    secrets = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "none.env")
    with Database(tmp_path / "ledger.sqlite") as db:
        pipeline = ResearchPipeline(settings, secrets, transport="fake", db=db, clock=clock, wall_clock=clock)
        result = await pipeline.run(causal_id="cz_integration", candidate=candidate, snapshot=snapshot,
                                    evidence=evidence, policy=policy)
        await pipeline.aclose()
        p = result.proposal
        assert p.action == DecisionAction.TRADE, p.no_trade_reason
        assert p.symbol == candidate.symbol and p.signal_id == candidate.signal_id
        assert p.dataset_version == snapshot.dataset_version and p.data_asof == asof
        assert p.limit_price is not None and snapshot.quote is not None
        assert p.limit_price <= snapshot.quote.ask  # request near the ask; risk engine recomputes anyway
        assert all(r.transport == ModelTransport.FAKE for r in result.model_runs)
        assert len(db.query("SELECT 1 FROM model_runs WHERE causal_id='cz_integration'")) == 2
        assert verify_chain(db)[0]


async def test_gate_state_survives_pipeline_restart(tmp_path: Path) -> None:
    clock = ManualClock(datetime(2026, 11, 5, 15, 30, tzinfo=UTC))
    settings = Settings(project_root=tmp_path, ai_transport="fake", use_keychain=False)
    secrets = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "none.env")
    cfg = AiConfig(min_interval_per_symbol_s=0, min_wall_interval_per_symbol_s=300)
    sig, snap, pol = make_signal(at=clock.now()), make_snapshot(at=clock.now()), RiskPolicy()
    with Database(tmp_path / "ledger.sqlite") as db:
        first = ResearchPipeline(settings, secrets, transport="fake", db=db, clock=clock, config=cfg, wall_clock=clock)
        r1 = await first.run(causal_id="cz_a", candidate=sig, snapshot=snap,
                             evidence=build_internal_evidence(sig, snap, pol), policy=pol)
        await first.aclose()
        assert r1.proposal.action == DecisionAction.TRADE
        clock.advance(60)
        sig2, snap2 = make_signal(at=clock.now(), signal_id="sig_2"), make_snapshot(at=clock.now(), snapshot_id="snap_2")
        restarted = ResearchPipeline(settings, secrets, transport="fake", db=db, clock=clock, config=cfg, wall_clock=clock)
        r2 = await restarted.run(causal_id="cz_b", candidate=sig2, snapshot=snap2,
                                 evidence=build_internal_evidence(sig2, snap2, pol), policy=pol)
        await restarted.aclose()
        assert r2.proposal.action == DecisionAction.NO_TRADE and "gated" in (r2.proposal.no_trade_reason or "")
        assert r2.model_runs == []
        assert restarted.fake_handlers["openai"].exchanges == []  # no model call after the restart
