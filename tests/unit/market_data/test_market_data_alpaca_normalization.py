"""Alpaca provider normalization with handcrafted alpaca-py objects and mocked clients (no network, no keys)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from alpaca.data.enums import DataFeed as AlpacaDataFeed
from alpaca.data.models.bars import Bar as AlpacaBar
from alpaca.data.models.quotes import Quote as AlpacaQuote

from packages.contracts import MarketDataProvider
from packages.contracts.enums import DataFeed, FeedMode, MarketSession
from packages.core.clock import ManualClock
from packages.core.secrets import SecretStore
from services.market_data.alpaca_provider import (
    AlpacaMarketDataProvider,
    FeedEntitlementError,
    MarketDataUnavailable,
    MissingCredentials,
    StreamDisconnected,
    normalize_bar,
    normalize_corporate_actions,
    normalize_quote,
)
from services.market_data.corporate_actions import CorporateActionType
from tests.unit.market_data.md_helpers import utc

KEY = "PKTESTKEYID000000000"
SECRET = "test-secret-value-never-printed"
T0 = utc(2026, 11, 3, 15, 0)  # 10:00 ET, regular session


def secrets(tmp_path: Path, *, keys: bool = True) -> SecretStore:
    env = {"ALPACA_PAPER_KEY_ID": KEY, "ALPACA_PAPER_SECRET_KEY": SECRET} if keys else {}
    return SecretStore(tmp_path, use_keychain=False, env=env, dotenv_path=tmp_path / "missing.env")


class FakeHistClient:
    def __init__(self, bars: dict[str, list[dict[str, Any]]] | None = None, quote: dict[str, Any] | None = None,
                 exc: Exception | None = None) -> None:
        self.bars = bars or {}
        self.quote = quote or {}
        self.exc = exc
        self.requests: list[Any] = []

    def get_stock_bars(self, req: Any) -> dict[str, list[dict[str, Any]]]:
        self.requests.append(req)
        if self.exc:
            raise self.exc
        return self.bars

    def get_stock_latest_quote(self, req: Any) -> dict[str, Any]:
        self.requests.append(req)
        if self.exc:
            raise self.exc
        return self.quote


class HTTPError(Exception):
    def __init__(self, status: int, msg: str) -> None:
        super().__init__(msg)
        self.status_code = status


def provider(tmp_path: Path, clock: ManualClock, mode: FeedMode = FeedMode.IEX_FREE, **kw: Any) -> AlpacaMarketDataProvider:
    return AlpacaMarketDataProvider(secrets(tmp_path), mode, clock, **kw)


def test_missing_credentials_raise_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(MissingCredentials, match="ALPACA_PAPER_KEY_ID") as exc:
        AlpacaMarketDataProvider(secrets(tmp_path, keys=False), FeedMode.IEX_FREE, ManualClock(T0))
    assert "SIMULATED" in str(exc.value)


def test_ambiguous_upstream_key_names_are_ignored(tmp_path: Path) -> None:
    store = SecretStore(tmp_path, use_keychain=False, env={"APCA_API_KEY_ID": KEY, "APCA_API_SECRET_KEY": SECRET},
                        dotenv_path=tmp_path / "none")
    with pytest.raises(MissingCredentials):
        AlpacaMarketDataProvider(store, FeedMode.IEX_FREE, ManualClock(T0))


def test_feed_mapping_and_protocol(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    iex = provider(tmp_path, clock)
    sip = provider(tmp_path, clock, FeedMode.SIP_PAID)
    assert iex.alpaca_feed == AlpacaDataFeed.IEX and iex.alpaca_feed.value == "iex" and iex.feed == DataFeed.IEX
    assert sip.alpaca_feed == AlpacaDataFeed.SIP and sip.alpaca_feed.value == "sip" and sip.feed == DataFeed.SIP
    assert isinstance(iex, MarketDataProvider)
    assert SECRET not in repr(iex) and KEY not in repr(iex)
    with pytest.raises(ValueError, match="replay provider"):
        provider(tmp_path, clock, FeedMode.SIMULATED)


def test_normalize_raw_dict_and_model_bar() -> None:
    raw = {"t": "2026-11-03T15:00:00Z", "o": 452.12, "h": 452.5, "l": 451.9, "c": 452.3, "v": 12000, "n": 87, "vw": 452.21}
    rec = normalize_bar(raw, "spy", "1Min", feed=DataFeed.IEX, source_id="alpaca:iex", received=T0 + timedelta(seconds=70))
    assert rec["symbol"] == "SPY" and rec["start_ts"] == T0 and rec["end_ts"] == T0 + timedelta(minutes=1)
    assert rec["open"] == Decimal("452.12") and isinstance(rec["close"], Decimal) and rec["vwap"] == Decimal("452.21")
    assert rec["session"] == MarketSession.REGULAR and rec["available_at_ts"] == T0 + timedelta(minutes=1, seconds=2)
    model = AlpacaBar("SPY", raw)
    rec2 = normalize_bar(model, "SPY", "1Min", feed=DataFeed.IEX, source_id="alpaca:iex", received=T0 + timedelta(minutes=5))
    assert {k: rec2[k] for k in ("open", "high", "low", "close", "volume", "trade_count", "start_ts")} == \
        {k: rec[k] for k in ("open", "high", "low", "close", "volume", "trade_count", "start_ts")}
    live = normalize_bar(raw, "SPY", "1Min", feed=DataFeed.IEX, source_id="x", received=T0 + timedelta(minutes=3), live=True)
    assert live["available_at_ts"] == T0 + timedelta(minutes=3)  # never usable before we actually had it
    pre = normalize_bar({**raw, "t": "2026-11-03T13:00:00Z"}, "SPY", "1Min", feed=DataFeed.IEX, source_id="x", received=T0)
    assert pre["session"] == MarketSession.PRE
    daily = normalize_bar({**raw, "t": "2026-11-03T05:00:00Z"}, "SPY", "1Day", feed=DataFeed.IEX, source_id="x",
                          received=utc(2026, 11, 4))
    assert daily["start_ts"] == utc(2026, 11, 3, 14, 30) and daily["end_ts"] == utc(2026, 11, 3, 21, 0)


def test_nanosecond_timestamps_truncate_never_round_up() -> None:
    raw = {"t": "2026-11-03T15:00:01.999999999Z", "bp": 1.0, "ap": 1.01, "bs": 1, "as": 1}
    q = normalize_quote(raw, "SPY", feed=DataFeed.IEX, source_id="x", received=T0 + timedelta(seconds=2))
    assert q["exchange_ts"] == T0 + timedelta(seconds=1, microseconds=999999)


def test_normalize_quote_model() -> None:
    model = AlpacaQuote("AAPL", {"t": "2026-11-03T15:00:00.5Z", "bp": 260.11, "bs": 3, "bx": "V", "ap": 260.13, "as": 5,
                                 "ax": "V", "c": ["R"], "z": "C"})
    q = normalize_quote(model, "AAPL", feed=DataFeed.IEX, source_id="alpaca:iex", received=T0 + timedelta(seconds=1))
    assert (q["bid"], q["ask"], q["bid_size"], q["ask_size"]) == (Decimal("260.11"), Decimal("260.13"), Decimal(3), Decimal(5))
    assert q["bid_exchange"] == "V" and q["conditions"] == ("R",) and q["tape"] == "C"
    assert q["available_at_ts"] == q["received_ts"] and q["provider_ts"] is None


async def test_get_bars_validates_and_hides_forming_bar(tmp_path: Path) -> None:
    clock = ManualClock(T0 + timedelta(minutes=3, seconds=30))
    raw = [{"t": f"2026-11-03T15:0{i}:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10, "n": 1, "vw": 100.2}
           for i in range(4)]
    raw.append({"t": "2026-11-03T15:01:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10})  # duplicate
    client = FakeHistClient(bars={"SPY": raw})
    p = provider(tmp_path, clock, client=client)
    bars = await p.get_bars("SPY", T0, T0 + timedelta(minutes=10), "1Min")
    assert [b.start_ts for b in bars] == [T0 + timedelta(minutes=i) for i in range(3)]  # 15:03 bar still forming
    assert all(b.feed == DataFeed.IEX and not b.is_synthetic for b in bars)
    req = client.requests[0]
    assert req.feed == AlpacaDataFeed.IEX and str(req.adjustment.value) == "raw"
    assert p.last_bar_report is not None and len(p.last_bar_report.rejected) == 1


async def test_latest_quote_and_out_of_session_rejected(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    quote = {"SPY": {"t": "2026-11-03T14:59:59.9Z", "bp": 680.1, "bs": 2, "ap": 680.12, "as": 4, "bx": "V", "ax": "V"}}
    p = provider(tmp_path, clock, client=FakeHistClient(quote=quote))
    q = await p.get_latest_quote("SPY")
    assert q is not None and q.ask == Decimal("680.12") and q.received_ts == T0
    clock.set(utc(2026, 11, 3, 13, 0))
    pre = {"SPY": {**quote["SPY"], "t": "2026-11-03T12:59:59Z"}}
    p2 = provider(tmp_path, clock, client=FakeHistClient(quote=pre))
    assert await p2.get_latest_quote("SPY") is None  # pre-market quote never served for regular-only trading


async def test_sip_entitlement_error_never_falls_back(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    client = FakeHistClient(exc=HTTPError(403, "subscription does not permit querying recent SIP data"))
    p = provider(tmp_path, clock, FeedMode.SIP_PAID, client=client)
    with pytest.raises(FeedEntitlementError, match=r"never|Nothing is purchased"):
        await p.get_bars("SPY", T0, T0 + timedelta(minutes=5), "1Min")
    assert len(client.requests) == 1 and client.requests[0].feed == AlpacaDataFeed.SIP  # no silent IEX retry
    iex = provider(tmp_path, clock, client=FakeHistClient(exc=HTTPError(429, "too many")))
    with pytest.raises(MarketDataUnavailable, match="429"):
        await iex.get_latest_quote("SPY")


def test_corporate_actions_normalization() -> None:
    raw = {"forward_splits": [{"symbol": "NVDA", "ex_date": "2024-06-10", "new_rate": 10, "old_rate": 1}],
           "cash_dividends": [{"symbol": "AAPL", "ex_date": "2024-08-12", "rate": 0.25}, {"symbol": "X", "ex_date":
                                                                                          "2024-08-12", "rate": 0}]}
    acts = normalize_corporate_actions(raw, source_id="alpaca:iex:ca", feed=DataFeed.IEX)
    assert [(a.symbol, a.action_type, a.split_factor, a.cash_amount) for a in acts] == [
        ("NVDA", CorporateActionType.SPLIT, Decimal("0.1"), None), ("AAPL", CorporateActionType.DIVIDEND, None, Decimal("0.25"))]


class FakeStream:
    """Mimics StockDataStream's subscribe/_run_forever/stop_ws surface."""

    def __init__(self, messages: list[Any], *, end_with_error: bool = True) -> None:
        self.messages = messages
        self.handler: Callable[[Any], Any] | None = None
        self.symbols: tuple[str, ...] = ()
        self.end_with_error = end_with_error
        self.stopped = False

    def subscribe_quotes(self, handler: Callable[[Any], Any], *symbols: str) -> None:
        self.handler, self.symbols = handler, symbols

    async def _run_forever(self) -> None:
        assert self.handler is not None
        for m in self.messages:
            await self.handler(m)
        if self.end_with_error:
            raise ConnectionError("socket closed")

    async def stop_ws(self) -> None:
        self.stopped = True


async def test_stream_quotes_normalizes_and_surfaces_disconnect(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    msgs = [AlpacaQuote("SPY", {"t": "2026-11-03T15:00:00Z", "bp": 680.1, "bs": 1, "ap": 680.12, "as": 1}),
            AlpacaQuote("SPY", {"t": "2026-11-03T15:00:00.1Z", "bp": 680.2, "bs": 1, "ap": 680.1, "as": 1})]  # crossed
    fake = FakeStream(msgs)
    p = provider(tmp_path, clock, stream_factory=lambda: fake)
    got: list[Any] = []

    async def drain() -> None:
        stream: AsyncIterator[Any] = p.stream_quotes(["SPY"])
        async for q in stream:
            got.append(q)

    with pytest.raises(StreamDisconnected, match="ConnectionError"):
        await drain()
    assert fake.symbols == ("SPY",) and fake.stopped
    assert len(got) == 1 and got[0].bid == Decimal("680.1")  # the crossed quote was rejected
