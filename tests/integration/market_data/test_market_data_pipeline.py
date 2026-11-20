"""Market-data vertical slice on the default SYNTHETIC fixture: store -> replay -> snapshot/monitor -> universe -> LEAN.

This is the exact composition the API runtime uses in SIMULATED mode (no credentials, no network).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, FeedMode, MarketSession
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from services.market_data.calendar import get_calendar
from services.market_data.fixtures import ensure_synthetic_fixture, manifest_sessions
from services.market_data.lean_export import export_bars_to_lean
from services.market_data.lean_sample import read_lean_equity_dir
from services.market_data.quality import FeedQualityMonitor
from services.market_data.replay import ReplayMarketDataProvider
from services.market_data.snapshot import build_market_snapshot
from services.market_data.store import MarketDataStore
from services.market_data.universe import build_universe_snapshot

CAL = get_calendar()
SYMS = ["SPY", "QQQ", "AAPL", "MSFT"]


@pytest.fixture(scope="module")
def fixture_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarketDataStore, dict[str, Any]]:
    d = tmp_path_factory.mktemp("pipeline")
    return MarketDataStore(d), ensure_synthetic_fixture(d)


async def test_simulated_session_snapshots_and_monitor(fixture_store: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = fixture_store
    day = manifest_sessions(m)[-1]
    open_, close = CAL.session_open(day), CAL.session_close(day)
    clock = ManualClock(open_ + timedelta(seconds=20))
    provider = ReplayMarketDataProvider(store, clock, DataFeed.SYNTHETIC, FeedMode.SIMULATED)
    monitor = FeedQualityMonitor(FeedMode.SIMULATED, f"replay:{m['dataset_version']}", clock)
    checked = 0
    while clock.now() < close:
        for s in SYMS:
            q = await provider.get_latest_quote(s)
            assert q is not None and q.available_at_ts <= clock.now()
            monitor.observe_quote(q)
        assert monitor.status()["status"] == "OK"
        snap = await build_market_snapshot(provider, "AAPL", clock.now(), FeedMode.SIMULATED,
                                           dataset_version=m["dataset_version"])
        assert snap.quote is not None and snap.is_simulated and snap.session == MarketSession.REGULAR
        assert snap.last_bar is None or snap.last_bar.available_at_ts <= clock.now()
        checked += 1
        clock.advance(minutes=29, seconds=13)
    assert checked >= 13
    clock.set(close + timedelta(seconds=30))
    for s in SYMS:
        q = await provider.get_latest_quote(s)
        assert q is not None
        monitor.observe_quote(q)
    st = monitor.status()
    assert st["status"] == "STALE" and st["is_simulated"] and "Synthetic data — not market prices" in st["limitations"]
    snap = await build_market_snapshot(provider, "AAPL", clock.now(), FeedMode.SIMULATED, dataset_version=m["dataset_version"])
    assert snap.quote is None and snap.session == MarketSession.POST  # fail closed after the close


def test_research_frame_point_in_time(fixture_store: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = fixture_store
    day = manifest_sessions(m)[-1]
    asof = CAL.session_open(day) + timedelta(hours=2, seconds=30)
    df = store.load_bars_df(SYMS, asof - timedelta(days=10), asof, "1Min", asof=asof)
    assert set(df["symbol"]) == set(SYMS) and bool((df["available_at_ts"] <= asof).all())
    assert bool(df["is_synthetic"].all()) and set(df["dataset_version"]) == {m["dataset_version"]}
    last: Any = df.loc[df["symbol"] == "SPY", "end_ts"].max()
    # at open+2h00m30s the bar [open+1h59, open+2h) is known (available at +1 s); the still-forming next bar is not
    assert last == CAL.session_open(day) + timedelta(hours=2)
    assert not bool((df["start_ts"] >= CAL.session_open(day) + timedelta(hours=2)).any())


def test_universe_over_fixture(fixture_store: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = fixture_store
    asof = datetime.fromisoformat(m["end"].replace("Z", "+00:00")) + timedelta(hours=1)
    snap = build_universe_snapshot(store, RiskPolicy(), asof, candidates=[*SYMS, "SQQQ"])
    assert snap.eligible_symbols == sorted(SYMS) and snap.dataset_versions == [m["dataset_version"]]


def test_full_fixture_lean_round_trip(fixture_store: tuple[MarketDataStore, dict[str, Any]], tmp_path: Path) -> None:
    store, m = fixture_store
    bars = [b for tf in ("1Min", "1Day") for b in store.load_bars(SYMS, None, None, tf, dataset_version=m["dataset_version"])]
    summary = export_bars_to_lean(bars, tmp_path)
    assert summary["rows"]["minute"] == m["row_counts"]["bars_1Min"] and summary["rows"]["daily"] == 80
    back = read_lean_equity_dir(tmp_path / "equity" / "usa", source_id="rt", received=datetime(2030, 1, 1, tzinfo=UTC))
    key = ("open", "high", "low", "close", "volume")
    ours = sorted((b.symbol, b.timeframe, b.start_ts, *(getattr(b, k) for k in key)) for b in bars)
    theirs = sorted((r["symbol"], r["timeframe"], r["start_ts"], *(r[k] for k in key)) for r in back.bar_records)
    assert ours == theirs
