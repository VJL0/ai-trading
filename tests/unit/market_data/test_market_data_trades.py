"""Trade prints: validation, storage (raw + DuckDB), Alpaca normalization, monitor and snapshot last-trade price."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from alpaca.data.models.trades import Trade as AlpacaTrade
from pydantic import ValidationError

from packages.contracts.enums import DataFeed, FeedMode
from packages.contracts.models import Bar, Quote
from packages.core.clock import ManualClock
from packages.core.secrets import SecretStore
from services.market_data.alpaca_provider import AlpacaMarketDataProvider, normalize_trade
from services.market_data.manifest import DatasetSpec
from services.market_data.quality import FeedQualityMonitor, validate_trades
from services.market_data.snapshot import build_market_snapshot
from services.market_data.store import MarketDataStore
from services.market_data.trades import Trade
from tests.unit.market_data.md_helpers import DAY, make_bar, make_quote, session_open

OPEN = session_open(DAY)


def trade_record(ts: datetime, *, price: str = "100.01", size: str = "100", latency_s: float = 0.05,
                 **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "symbol": "SPY", "price": Decimal(price), "size": Decimal(size), "exchange": "V", "trade_id": None,
        "conditions": (), "tape": "B", "exchange_ts": ts, "provider_ts": None,
        "received_ts": ts + timedelta(seconds=latency_s), "available_at_ts": ts + timedelta(seconds=latency_s),
        "feed": DataFeed.IEX, "source_id": "test:trades", "is_synthetic": False,
    }
    rec.update(over)
    return rec


def make_trade(ts: datetime, **kw: Any) -> Trade:
    return Trade.model_validate(trade_record(ts, **kw))


def test_trade_model_rejects_floats_and_bad_values() -> None:
    with pytest.raises(ValidationError, match="float"):
        Trade.model_validate({**trade_record(OPEN), "price": 100.0})
    with pytest.raises(ValidationError, match="non-positive"):
        make_trade(OPEN, price="0")


def test_validate_trades_checks() -> None:
    t = OPEN + timedelta(minutes=5)
    recs = [
        trade_record(t, trade_id="1"),
        trade_record(t, trade_id="1"),  # duplicate print
        trade_record(t - timedelta(seconds=10), trade_id="2"),  # out of order
        trade_record(t + timedelta(seconds=1), trade_id="3", received_ts=t - timedelta(seconds=5)),  # clock skew
        trade_record(OPEN - timedelta(minutes=30), trade_id="4"),  # pre-market print
        trade_record(t + timedelta(seconds=2), trade_id="5", price="-1"),
        {"symbol": "SPY", "price": 1.5, "size": Decimal(1), "exchange_ts": t, "received_ts": t, "available_at_ts": t},
    ]
    rep = validate_trades(recs)
    assert [t.trade_id for t in rep.accepted] == ["1"]
    reasons = [r.reasons for r in rep.rejected]
    assert ("DUPLICATE",) in reasons and ("OUT_OF_ORDER",) in reasons
    assert any("OUTSIDE_REGULAR_SESSION" in r for r in reasons)
    assert any("CLOCK_SKEW" in r for r in reasons) and ("NON_POSITIVE_PRICE",) in reasons
    assert ("INVALID_RECORD",) in reasons  # float price never becomes a Decimal silently


def test_store_trades_roundtrip_raw_and_pit(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path)
    trades = [make_trade(OPEN + timedelta(seconds=i), trade_id=str(i), price=str(Decimal("100.00") + i)) for i in range(5)]
    rep = validate_trades([*(t.model_dump() for t in trades), trade_record(OPEN - timedelta(hours=1), trade_id="pre")])
    m = store.write_trades(rep.accepted, manifest=DatasetSpec(source_id="t", feed=DataFeed.IEX, generator="tests",
                                                              generator_version="1", is_synthetic=False),
                           extras=rep.raw_extras())
    assert m["row_counts"]["trades"] == 5 and m["row_counts"]["trades_rejected"] == 1
    raw = pq.read_table(tmp_path / "raw" / "IEX" / "trades" / "symbol=SPY" / "date=2026-11-03" / "part.parquet")
    assert sorted(raw.column("quality_status").to_pylist()) == ["ACCEPTED"] * 5 + ["REJECTED"]
    assert store.load_trades(["SPY"], None, None) == trades
    latest = store.latest_trade("SPY", OPEN + timedelta(seconds=2, milliseconds=60))
    assert latest is not None and latest.trade_id == "2" and latest.price == Decimal("102")
    assert store.latest_trade("SPY", OPEN) is None
    assert store.verify_dataset(m["dataset_version"])["ok"]


def secrets(tmp_path: Path) -> SecretStore:
    return SecretStore(tmp_path, use_keychain=False, env={"ALPACA_PAPER_KEY_ID": "PK", "ALPACA_PAPER_SECRET_KEY": "s"},
                       dotenv_path=tmp_path / "none")


class LatestTradeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[Any] = []

    def get_stock_latest_trade(self, req: Any) -> dict[str, Any]:
        self.requests.append(req)
        return self.payload


async def test_alpaca_trade_normalization_and_latest(tmp_path: Path) -> None:
    now = OPEN + timedelta(minutes=30)
    raw = {"t": "2026-11-03T15:00:00.123456789Z", "x": "V", "p": 452.13, "s": 25, "c": ["@", "I"], "i": 52983525029461,
           "z": "B"}
    rec = normalize_trade(raw, "spy", feed=DataFeed.IEX, source_id="alpaca:iex", received=now)
    assert rec["price"] == Decimal("452.13") and rec["size"] == Decimal(25) and rec["trade_id"] == "52983525029461"
    assert rec["exchange_ts"] == OPEN + timedelta(minutes=30, microseconds=123456) and rec["conditions"] == ("@", "I")
    model = normalize_trade(AlpacaTrade("SPY", raw), "SPY", feed=DataFeed.IEX, source_id="x", received=now)
    assert (model["price"], model["exchange"], model["tape"]) == (Decimal("452.13"), "V", "B")
    client = LatestTradeClient({"SPY": {**raw, "t": "2026-11-03T14:59:59Z"}})
    p = AlpacaMarketDataProvider(secrets(tmp_path), FeedMode.IEX_FREE, ManualClock(now), client=client)
    trade = await p.get_latest_trade("SPY")
    assert trade is not None and trade.feed == DataFeed.IEX and trade.available_at_ts == now
    assert client.requests[0].feed.value == "iex"


def test_monitor_tracks_trades_and_fails_closed_on_future_quotes() -> None:
    clock = ManualClock(OPEN + timedelta(minutes=10))
    mon = FeedQualityMonitor(FeedMode.IEX_FREE, "alpaca:iex", clock, symbols=("SPY",))
    mon.observe_trade(make_trade(clock.now() - timedelta(seconds=3)))
    mon.observe_quote(make_quote("SPY", clock.now() - timedelta(seconds=1)))
    st = mon.status()
    assert st["status"] == "OK" and st["last_trade_ts"] is not None
    future = make_quote("SPY", clock.now() + timedelta(seconds=30))  # a local clock ~30 s behind the exchange
    mon.observe_quote(future)
    st = mon.status()
    assert st["status"] == "STALE" and "clock skew" in st["detail"]


class TradeProvider:
    def __init__(self, quote: Quote | None, bars: Sequence[Bar], trade: Trade | None) -> None:
        self.quote, self.bars, self.trade = quote, list(bars), trade
        self.feed_mode = FeedMode.IEX_FREE
        self.source_id = "stub:iex"

    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]:
        return list(self.bars)

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        return self.quote

    async def get_latest_trade(self, symbol: str) -> Trade | None:
        return self.trade

    async def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]:
        if False:
            yield  # pragma: no cover


async def test_snapshot_last_trade_price_point_in_time() -> None:
    asof = OPEN + timedelta(minutes=10)
    q = make_quote("SPY", asof - timedelta(seconds=1))
    bar = make_bar("SPY", asof - timedelta(minutes=2), c="99.90")
    trade = make_trade(asof - timedelta(seconds=2), price="100.07")
    snap = await build_market_snapshot(TradeProvider(q, [bar], trade), "SPY", asof, FeedMode.IEX_FREE, dataset_version="v")
    assert snap.last_trade_price == Decimal("100.07")
    late = make_trade(asof, price="100.50", latency_s=5)  # printed at asof but only received 5 s later
    snap2 = await build_market_snapshot(TradeProvider(q, [bar], late), "SPY", asof, FeedMode.IEX_FREE, dataset_version="v")
    assert snap2.last_trade_price == Decimal("99.90")  # fell back to the PIT bar close, never the future print
    assert any("not available at the decision time" in x for x in snap2.limitations)
    assert any("close of the last point-in-time bar" in x for x in snap2.limitations)
