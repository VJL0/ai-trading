"""REAL Alpaca market-data call (free IEX feed) — runs only when Alpaca PAPER keys are configured, else skips.

When it runs it exercises the real network API (data endpoints only; never orders, never the SIP feed).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.clock import SystemClock
from packages.core.config import load_settings
from packages.core.secrets import SecretStore
from services.market_data.alpaca_provider import AlpacaMarketDataProvider
from services.market_data.ingestion import IngestionService
from services.market_data.quality import FeedQualityMonitor
from services.market_data.store import MarketDataStore

pytestmark = pytest.mark.paper_broker


def _secrets() -> SecretStore:
    settings = load_settings()
    return SecretStore(settings.project_root, use_keychain=settings.use_keychain)


@pytest.fixture
def provider() -> AlpacaMarketDataProvider:
    secrets = _secrets()
    if not secrets.alpaca_paper_available():
        pytest.skip("ALPACA_PAPER_KEY_ID/ALPACA_PAPER_SECRET_KEY not configured: real IEX fetch not exercised")
    return AlpacaMarketDataProvider(secrets, FeedMode.IEX_FREE, SystemClock())


async def test_real_iex_bars(provider: AlpacaMarketDataProvider) -> None:
    now = provider.clock.now()
    bars = await provider.get_bars("SPY", now - timedelta(days=5), now, "1Min")
    assert bars, "expected at least one IEX minute bar in the last 5 days"
    assert all(b.feed == DataFeed.IEX and not b.is_synthetic and b.available_at_ts <= now for b in bars)
    assert provider.alpaca_feed.value == "iex"


async def test_real_iex_backfill_into_store(provider: AlpacaMarketDataProvider, tmp_path: Path) -> None:
    clock = provider.clock
    monitor = FeedQualityMonitor(FeedMode.IEX_FREE, provider.source_id, clock, symbols=("SPY",))
    svc = IngestionService(provider, MarketDataStore(tmp_path), None, clock, monitor, symbols=["SPY"],
                           backfill_lookback=timedelta(days=3))
    result = await svc.backfill(reason="paper_broker integration test")
    assert result.ok and result.rows_accepted > 0 and result.dataset_version
