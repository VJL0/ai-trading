"""Alpaca market-data provider (alpaca-py) implementing ``packages.contracts.MarketDataProvider``.

* Feed mapping: ``IEX_FREE -> DataFeed.IEX ('iex')`` (Alpaca Basic, free) and ``SIP_PAID -> DataFeed.SIP ('sip')``
  (Algo Trader Plus, paid). SIP must be configured explicitly; the provider never upgrades, falls back between
  feeds or purchases anything. A SIP entitlement error is raised as :class:`FeedEntitlementError`.
* Credentials: PAPER market-data keys only (``ALPACA_PAPER_KEY_ID``/``ALPACA_PAPER_SECRET_KEY`` via
  :class:`packages.core.secrets.SecretStore`); constructing without them raises :class:`MissingCredentials`.
  Secrets are held as ``SecretStr`` and never logged or included in ``repr``.
* Timestamps: ``exchange_ts`` = Alpaca ``t`` (exchange/SIP participant time); Alpaca exposes no separate provider
  timestamp so ``provider_ts`` is None; ``received_ts`` = local clock at receipt; bars ``available_at_ts`` =
  ``end_ts + bar_latency`` (and never before receipt for live data); quotes ``available_at_ts`` = ``received_ts``.
* Session: classified with the XNYS calendar (PRE/REGULAR/POST/CLOSED). Records are validated with the standard
  :class:`QualityPolicy` before being returned; the last reports are kept for the ingestion service.
* Streaming: quotes, trades and minute bars share ONE ``StockDataStream`` websocket (``stream_events``; Alpaca allows a single
  market-data connection per account). alpaca-py reconnects silently inside ``_run_forever``; every reconnect after the
  first connection is surfaced as :class:`StreamDisconnected` so the ingestion service backfills before entries resume.
* Historical bars still forming when the response was produced (``end_ts > received_ts``) are excluded (provisional;
  fetched by the next backfill) and counted in ``last_forming_excluded``.
* This module is data-only: it never imports or calls any Alpaca *trading/order* API.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from pydantic import SecretStr

from packages.contracts.enums import Adjustment, DataFeed, FeedMode, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.clock import Clock
from packages.core.secrets import SecretName, SecretStore

from .calendar import CalendarOutOfRange, MarketCalendar, get_calendar
from .common import PENDING_DATASET_VERSION, canonical_timeframe, is_daily, timeframe_delta, to_decimal, to_utc
from .corporate_actions import CorporateAction, CorporateActionType
from .quality import QualityPolicy, ValidationReport, validate_bars, validate_quotes, validate_trades
from .trades import Trade

log = logging.getLogger("lat.market_data.alpaca")

DEFAULT_BAR_LATENCY = timedelta(seconds=2)


class MissingCredentials(RuntimeError):
    """Alpaca PAPER market-data keys are not configured."""


class FeedEntitlementError(RuntimeError):
    """The account is not entitled to the configured feed (e.g. SIP without a paid plan). Never auto-resolved."""


class MarketDataUnavailable(RuntimeError):
    """Provider call failed; callers must treat data as unknown (fail closed)."""


class StreamDisconnected(ConnectionError):
    """The streaming connection ended; the ingestion service must backfill before resuming entries."""


MISSING_KEYS_MESSAGE = (
    "Alpaca PAPER market-data credentials are missing. Add them to the macOS Keychain "
    "(security add-generic-password -s local-ai-trader -a ALPACA_PAPER_KEY_ID -w, which prompts for the value; repeat "
    "with -a ALPACA_PAPER_SECRET_KEY) "
    "or to the git-ignored .env as ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY (paper keys only; the ambiguous "
    "APCA_API_* names are ignored). Without keys use LAT_FEED_MODE=SIMULATED (synthetic fixture replay)."
)


def _alpaca_feed(feed_mode: FeedMode) -> Any:
    from alpaca.data.enums import DataFeed as AlpacaDataFeed

    if feed_mode == FeedMode.IEX_FREE:
        return AlpacaDataFeed.IEX
    if feed_mode == FeedMode.SIP_PAID:
        return AlpacaDataFeed.SIP
    raise ValueError(f"AlpacaMarketDataProvider does not serve feed mode {feed_mode}; use the replay provider")


def _our_feed(feed_mode: FeedMode) -> DataFeed:
    return DataFeed.IEX if feed_mode == FeedMode.IEX_FREE else DataFeed.SIP


def _alpaca_timeframe(timeframe: str) -> Any:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    unit: Any = TimeFrameUnit  # members are annotated ``str`` in alpaca-py; keep them as the enum
    delta = timeframe_delta(timeframe)
    if delta >= timedelta(days=1):
        if delta != timedelta(days=1):
            raise ValueError("only 1Day daily bars are supported")
        return TimeFrame(1, unit.Day)
    minutes = int(delta.total_seconds() // 60)
    if minutes % 60 == 0:
        return TimeFrame(minutes // 60, unit.Hour)
    return TimeFrame(minutes, unit.Minute)


# ------------------------------------------------------------------------------------------ normalization
def _get(obj: Any, raw_key: str, attr: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(raw_key, obj.get(attr))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return getattr(obj, attr, None)


def _parse_ts(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing timestamp")
    if hasattr(value, "to_datetime") and not isinstance(value, datetime):
        value = value.to_datetime()
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        # Alpaca sends nanoseconds; Python keeps microseconds (truncate, never round up into the future)
        if "." in s:
            head, frac = s.split(".", 1)
            digits = "".join(c for c in frac if c.isdigit())
            tz = frac[len(digits):]
            s = f"{head}.{digits[:6].ljust(6, '0')}{tz}"
        value = datetime.fromisoformat(s)
    if not isinstance(value, datetime):
        raise ValueError(f"unparseable timestamp {value!r}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)  # alpaca-py documents naive values as UTC
    return value.astimezone(UTC)


def _raw_json(obj: Any) -> str:
    if isinstance(obj, Mapping):
        data: Any = dict(obj)  # pyright: ignore[reportUnknownArgumentType]
    elif hasattr(obj, "model_dump"):
        data = obj.model_dump()
    else:
        data = {"repr": repr(obj)}
    return json.dumps(data, sort_keys=True, default=str)


def _opt_dec(v: Any) -> Decimal | None:
    return None if v is None else to_decimal(v)


def normalize_bar(raw: Any, symbol: str, timeframe: str, *, feed: DataFeed, source_id: str, received: datetime,
                  latency: timedelta = DEFAULT_BAR_LATENCY, cal: MarketCalendar | None = None,
                  live: bool = False) -> dict[str, Any]:
    """Alpaca bar (REST raw dict ``{t,o,h,l,c,v,n,vw}``, stream dict, or alpaca-py ``Bar`` model) -> record dict.

    ``available_at_ts`` = ``end_ts + latency``: the earliest instant a subscriber could have known the bar (used for
    historical/backfilled bars so point-in-time research matches publication time). For ``live`` stream bars it is
    ``max(end_ts + latency, received_ts)``: a bar is never usable before this process actually had it.
    """
    cal = cal or get_calendar()
    tf = canonical_timeframe(timeframe)
    t = _parse_ts(_get(raw, "t", "timestamp"))
    session: MarketSession
    if is_daily(tf):
        day = cal.ny_date(t)
        try:
            start, end = cal.session_open(day), cal.session_close(day)
            session = MarketSession.REGULAR
        except (ValueError, CalendarOutOfRange):
            start, end = t, t + timedelta(days=1)
            session = MarketSession.CLOSED
    else:
        start = t
        end = t + timeframe_delta(tf)
        try:
            session = cal.classify_bar(start, end)
        except CalendarOutOfRange:
            session = MarketSession.CLOSED
    n = _get(raw, "n", "trade_count")
    return {
        "symbol": symbol.upper(), "timeframe": tf, "start_ts": start, "end_ts": end,
        "open": to_decimal(_get(raw, "o", "open")), "high": to_decimal(_get(raw, "h", "high")),
        "low": to_decimal(_get(raw, "l", "low")), "close": to_decimal(_get(raw, "c", "close")),
        "volume": to_decimal(_get(raw, "v", "volume")), "vwap": _opt_dec(_get(raw, "vw", "vwap")),
        "trade_count": int(n) if n is not None else None, "feed": feed, "adjustment": Adjustment.RAW,
        "session": session, "provider_ts": None, "received_ts": received,
        "available_at_ts": max(end + latency, received) if live else end + latency,
        "source_id": source_id, "dataset_version": PENDING_DATASET_VERSION, "is_synthetic": False,
    }


def normalize_quote(raw: Any, symbol: str, *, feed: DataFeed, source_id: str, received: datetime,
                    available_at: datetime | None = None) -> dict[str, Any]:
    """Alpaca quote (REST raw ``{t,bp,bs,bx,ap,as,ax,c,z}``, stream dict or ``Quote`` model) -> record dict."""
    cond = _get(raw, "c", "conditions")
    if isinstance(cond, str):
        cond = [cond]
    return {
        "symbol": symbol.upper(), "bid": to_decimal(_get(raw, "bp", "bid_price")),
        "ask": to_decimal(_get(raw, "ap", "ask_price")), "bid_size": to_decimal(_get(raw, "bs", "bid_size")),
        "ask_size": to_decimal(_get(raw, "as", "ask_size")),
        "bid_exchange": _str_or_none(_get(raw, "bx", "bid_exchange")),
        "ask_exchange": _str_or_none(_get(raw, "ax", "ask_exchange")),
        "exchange_ts": _parse_ts(_get(raw, "t", "timestamp")), "provider_ts": None, "received_ts": received,
        "available_at_ts": available_at or received, "feed": feed,
        "conditions": tuple(str(c) for c in (cond or ())), "tape": _str_or_none(_get(raw, "z", "tape")),
        "source_id": source_id, "is_synthetic": False,
    }


def normalize_trade(raw: Any, symbol: str, *, feed: DataFeed, source_id: str, received: datetime) -> dict[str, Any]:
    """Alpaca trade (REST/stream raw ``{t,x,p,s,c,i,z}`` or alpaca-py ``Trade`` model) -> record dict.
    ``available_at_ts`` = ``received_ts`` (a print is usable once this process has it)."""
    cond = _get(raw, "c", "conditions")
    if isinstance(cond, str):
        cond = [cond]
    tid = _get(raw, "i", "id")
    return {
        "symbol": symbol.upper(), "price": to_decimal(_get(raw, "p", "price")), "size": to_decimal(_get(raw, "s", "size")),
        "exchange": _str_or_none(_get(raw, "x", "exchange")), "trade_id": None if tid is None else str(tid),
        "conditions": tuple(str(c) for c in (cond or ())), "tape": _str_or_none(_get(raw, "z", "tape")),
        "exchange_ts": _parse_ts(_get(raw, "t", "timestamp")), "provider_ts": None, "received_ts": received,
        "available_at_ts": received, "feed": feed, "source_id": source_id, "is_synthetic": False,
    }


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    return str(v.value) if hasattr(v, "value") else str(v)


def normalize_corporate_actions(raw: Any, *, source_id: str, feed: DataFeed,
                                cal: MarketCalendar | None = None) -> list[CorporateAction]:
    """Alpaca ``/corporate-actions`` raw response -> splits and cash dividends (other types are ignored)."""
    cal = cal or get_calendar()
    data = raw if isinstance(raw, Mapping) else getattr(raw, "data", {}) or {}
    out: list[CorporateAction] = []

    def _date(v: Any) -> date:
        return v if isinstance(v, date) else date.fromisoformat(str(v))

    for kind in ("forward_splits", "reverse_splits"):
        for item in data.get(kind, []) or []:
            ex = _date(_get(item, "ex_date", "ex_date"))
            new_rate, old_rate = to_decimal(_get(item, "new_rate", "new_rate")), to_decimal(_get(item, "old_rate", "old_rate"))
            if new_rate <= 0 or old_rate <= 0:
                continue
            out.append(CorporateAction(
                symbol=str(_get(item, "symbol", "symbol")).upper(), action_type=CorporateActionType.SPLIT, ex_date=ex,
                available_at_ts=_ex_open(cal, ex), split_factor=old_rate / new_rate, feed=feed, source_id=source_id,
                raw=_raw_json(item)))
    for item in data.get("cash_dividends", []) or []:
        ex = _date(_get(item, "ex_date", "ex_date"))
        rate = to_decimal(_get(item, "rate", "rate"))
        if rate <= 0:
            continue
        out.append(CorporateAction(
            symbol=str(_get(item, "symbol", "symbol")).upper(), action_type=CorporateActionType.DIVIDEND, ex_date=ex,
            available_at_ts=_ex_open(cal, ex), cash_amount=rate, feed=feed, source_id=source_id, raw=_raw_json(item)))
    return out


def _ex_open(cal: MarketCalendar, ex: date) -> datetime:
    try:
        return cal.session_open(ex) if cal.is_session(ex) else datetime(ex.year, ex.month, ex.day, 13, 30, tzinfo=UTC)
    except CalendarOutOfRange:
        return datetime(ex.year, ex.month, ex.day, 13, 30, tzinfo=UTC)


def _classify_error(exc: BaseException, feed_mode: FeedMode) -> Exception:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if feed_mode == FeedMode.SIP_PAID and (status == 403 or "subscription" in text or "not permit" in text):
        return FeedEntitlementError(
            "SIP feed not permitted for this account (requires the paid Algo Trader Plus plan). Nothing is purchased "
            "automatically; set LAT_FEED_MODE=IEX_FREE to use the free IEX feed.")
    if status in (401, 403):
        return MarketDataUnavailable(f"Alpaca rejected the market-data credentials (HTTP {status})")
    if status == 429:
        return MarketDataUnavailable("Alpaca market-data rate limit hit (HTTP 429); back off and retry later")
    return MarketDataUnavailable(f"Alpaca market-data call failed: {type(exc).__name__}")


# ------------------------------------------------------------------------------------------ provider
class AlpacaMarketDataProvider:
    """Real Alpaca market-data adapter (REST historical + websocket stream). PAPER keys, data endpoints only."""

    def __init__(
        self,
        secret_store: SecretStore,
        feed_mode: FeedMode,
        clock: Clock,
        *,
        client: Any | None = None,
        stream_factory: Callable[[], Any] | None = None,
        corporate_actions_client: Any | None = None,
        bar_latency: timedelta = DEFAULT_BAR_LATENCY,
        policy: QualityPolicy | None = None,
        calendar: MarketCalendar | None = None,
        stream_queue_size: int = 10_000,
    ) -> None:
        self.feed_mode = FeedMode(feed_mode)
        self.alpaca_feed = _alpaca_feed(self.feed_mode)
        self.feed = _our_feed(self.feed_mode)
        self.source_id = f"alpaca:{self.alpaca_feed.value}"
        self.clock = clock
        self.bar_latency = bar_latency
        self.policy = policy or QualityPolicy()
        self.calendar = calendar or get_calendar()
        self.stream_queue_size = stream_queue_size
        key = secret_store.get(SecretName.ALPACA_PAPER_KEY_ID)
        secret = secret_store.get(SecretName.ALPACA_PAPER_SECRET_KEY)
        if key is None or secret is None:
            raise MissingCredentials(MISSING_KEYS_MESSAGE)
        self._key: SecretStr = key
        self._secret: SecretStr = secret
        self._client = client
        self._ca_client = corporate_actions_client
        self._stream_factory = stream_factory
        self.last_bar_report: ValidationReport | None = None
        self.last_quote_report: ValidationReport | None = None
        self.last_trade_report: ValidationReport | None = None
        self.last_raw_bars: dict[int, str] = {}
        self.last_forming_excluded = 0
        self.on_stream_report: Callable[[ValidationReport], None] | None = None
        """Called with every streamed record's validation report that has issues (rejects/flags); the ingestion
        service uses it to count rejects and persist data-quality events. Rejected stream records are never yielded."""

    def __repr__(self) -> str:
        return f"AlpacaMarketDataProvider(feed_mode={self.feed_mode.value}, feed={self.alpaca_feed.value}, keys=***)"

    # ------------------------------------------------------------ clients (lazy; never logged)
    def _historical(self) -> Any:
        if self._client is None:
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self._key.get_secret_value(), self._secret.get_secret_value(),
                                                     raw_data=True)
        return self._client

    def _corporate(self) -> Any:
        if self._ca_client is None:
            from alpaca.data.historical.corporate_actions import CorporateActionsClient

            self._ca_client = CorporateActionsClient(self._key.get_secret_value(), self._secret.get_secret_value(),
                                                     raw_data=True)
        return self._ca_client

    def _stream(self) -> Any:
        if self._stream_factory is not None:
            return self._stream_factory()
        from alpaca.data.live.stock import StockDataStream

        return StockDataStream(self._key.get_secret_value(), self._secret.get_secret_value(), raw_data=False,
                               feed=self.alpaca_feed)

    # ------------------------------------------------------------ REST
    async def fetch_bar_records(self, symbol: str, start: datetime, end: datetime,
                                timeframe: str) -> tuple[list[dict[str, Any]], dict[int, str]]:
        """Raw normalized records (+ verbatim payload JSON by index) without validation."""
        from alpaca.data.enums import Adjustment as AlpacaAdjustment
        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=_alpaca_timeframe(timeframe), start=to_utc(start),
                               end=to_utc(end), feed=self.alpaca_feed, adjustment=AlpacaAdjustment.RAW)
        try:
            resp = await asyncio.to_thread(self._historical().get_stock_bars, req)
        except Exception as exc:
            raise _classify_error(exc, self.feed_mode) from None
        received = to_utc(self.clock.now())
        items = _items_for(resp, symbol)
        records: list[dict[str, Any]] = []
        raw: dict[int, str] = {}
        forming = 0
        for item in items:
            rec: dict[str, Any]
            try:
                rec = normalize_bar(item, symbol, timeframe, feed=self.feed, source_id=self.source_id,
                                    received=received, latency=self.bar_latency, cal=self.calendar)
            except (ValueError, TypeError, ArithmeticError) as exc:
                rec = {"symbol": symbol.upper(), "timeframe": timeframe, "error": str(exc)}
            if isinstance(rec.get("end_ts"), datetime) and rec["end_ts"] > received:
                # The bar was still forming when the response was produced: it is not knowable yet and its values are
                # provisional. It is not stored now (nor counted as a defect); the next backfill fetches the final bar.
                forming += 1
                continue
            raw[len(records)] = _raw_json(item)
            records.append(rec)
        self.last_forming_excluded = forming
        return records, raw

    async def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> list[Bar]:
        records, raw = await self.fetch_bar_records(symbol, start, end, timeframe)
        now = to_utc(self.clock.now())
        report = validate_bars(records, self.calendar, self.policy, expected_range=(to_utc(start), min(to_utc(end), now)),
                               expected_symbols=[symbol], validated_at=now)
        self.last_bar_report = report
        self.last_raw_bars = raw
        # never hand out an incomplete (still forming) bar: it is not knowable yet
        return [b for b in report.accepted if b.available_at_ts <= now]

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        from alpaca.data.requests import StockLatestQuoteRequest

        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.alpaca_feed)
        try:
            resp = await asyncio.to_thread(self._historical().get_stock_latest_quote, req)
        except Exception as exc:
            raise _classify_error(exc, self.feed_mode) from None
        received = to_utc(self.clock.now())
        item = resp.get(symbol) if isinstance(resp, Mapping) else None
        if item is None:
            return None
        try:
            rec = normalize_quote(item, symbol, feed=self.feed, source_id=self.source_id, received=received)
        except (ValueError, TypeError, ArithmeticError):
            return None
        report = validate_quotes([rec], self.calendar, self.policy, validated_at=received)
        self.last_quote_report = report
        return report.accepted[0] if report.accepted else None

    async def get_latest_trade(self, symbol: str) -> Trade | None:
        from alpaca.data.requests import StockLatestTradeRequest

        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.alpaca_feed)
        try:
            resp = await asyncio.to_thread(self._historical().get_stock_latest_trade, req)
        except Exception as exc:
            raise _classify_error(exc, self.feed_mode) from None
        received = to_utc(self.clock.now())
        item = resp.get(symbol) if isinstance(resp, Mapping) else None
        if item is None:
            return None
        try:
            rec = normalize_trade(item, symbol, feed=self.feed, source_id=self.source_id, received=received)
        except (ValueError, TypeError, ArithmeticError):
            return None
        report = validate_trades([rec], self.calendar, self.policy, validated_at=received)
        self.last_trade_report = report
        return report.accepted[0] if report.accepted else None

    async def get_corporate_actions(self, symbols: Sequence[str], start: date, end: date) -> list[CorporateAction]:
        from alpaca.data.requests import CorporateActionsRequest

        req = CorporateActionsRequest(symbols=[s.upper() for s in symbols], start=start, end=end)
        try:
            resp = await asyncio.to_thread(self._corporate().get_corporate_actions, req)
        except Exception as exc:
            raise _classify_error(exc, self.feed_mode) from None
        return normalize_corporate_actions(resp, source_id=f"{self.source_id}:corporate-actions", feed=self.feed,
                                           cal=self.calendar)

    # ------------------------------------------------------------ streaming
    async def _stream_messages(self, symbols: Sequence[str], *, quotes: bool = True, bars: bool = False,
                               trades: bool = False) -> AsyncIterator[tuple[str, Any]]:
        """ONE websocket for every requested channel (Alpaca allows a single market-data connection per account).

        Yields ``("quote"|"bar"|"trade", message)``. Raises :class:`StreamDisconnected` when the SDK task ends, when the
        local queue overflows, or when the SDK silently re-opens its websocket after the first connection (alpaca-py
        reconnects internally; any reconnect means data continuity is unknown and a backfill is required).
        """
        if not quotes and not bars and not trades:
            raise ValueError("subscribe to at least one channel")
        stream = self._stream()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=self.stream_queue_size)
        overflow = asyncio.Event()

        def put(item: tuple[str, Any]) -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                overflow.set()

        async def on_quote(msg: Any) -> None:
            put(("quote", msg))

        async def on_bar(msg: Any) -> None:
            put(("bar", msg))

        async def on_trade(msg: Any) -> None:
            put(("trade", msg))

        if quotes:
            stream.subscribe_quotes(on_quote, *symbols)
        if bars:
            stream.subscribe_bars(on_bar, *symbols)
        if trades:
            stream.subscribe_trades(on_trade, *symbols)
        _watch_reconnects(stream, lambda: put(("reconnect", None)))
        task = asyncio.create_task(stream._run_forever(), name="alpaca-market-data-stream")
        try:
            while True:
                if overflow.is_set():
                    raise StreamDisconnected("stream queue overflow; consumer too slow — data continuity unknown")
                try:
                    kind, msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    if task.done():
                        exc = task.exception() if not task.cancelled() else None
                        raise StreamDisconnected(f"stream ended: {type(exc).__name__ if exc else 'closed'}") from None
                    continue
                if kind == "reconnect":
                    raise StreamDisconnected("SDK websocket reconnected; data continuity unknown — backfill required")
                yield kind, msg
        finally:
            with contextlib.suppress(Exception):
                await stream.stop_ws()
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    def _report(self, report: ValidationReport) -> None:
        if report.issues and self.on_stream_report is not None:
            try:
                self.on_stream_report(report)
            except Exception:  # a monitoring hook must never break the stream
                log.exception("on_stream_report hook failed")

    def _stream_quote(self, msg: Any, received: datetime) -> list[Quote]:
        sym = str(_get(msg, "S", "symbol") or "").upper()
        try:
            rec: dict[str, Any] = normalize_quote(msg, sym, feed=self.feed, source_id=self.source_id, received=received)
        except (ValueError, TypeError, ArithmeticError) as exc:  # malformed: reject (raw kept), never silently skip
            rec = {"symbol": sym, "error": str(exc), "raw_json": _raw_json(msg)}
        report = validate_quotes([rec], self.calendar, self.policy, validated_at=received)
        self.last_quote_report = report
        self._report(report)
        return list(report.accepted)

    def _stream_bar(self, msg: Any, received: datetime) -> list[Bar]:
        sym = str(_get(msg, "S", "symbol") or "").upper()
        try:
            rec: dict[str, Any] = normalize_bar(msg, sym, "1Min", feed=self.feed, source_id=self.source_id,
                                                received=received, latency=self.bar_latency, cal=self.calendar, live=True)
        except (ValueError, TypeError, ArithmeticError) as exc:
            rec = {"symbol": sym, "timeframe": "1Min", "error": str(exc), "raw_json": _raw_json(msg)}
        report = validate_bars([rec], self.calendar, self.policy, validated_at=received)
        self.last_bar_report = report
        self._report(report)
        return list(report.accepted)

    def _stream_trade(self, msg: Any, received: datetime) -> list[Trade]:
        sym = str(_get(msg, "S", "symbol") or "").upper()
        try:
            rec: dict[str, Any] = normalize_trade(msg, sym, feed=self.feed, source_id=self.source_id, received=received)
        except (ValueError, TypeError, ArithmeticError) as exc:
            rec = {"symbol": sym, "error": str(exc), "raw_json": _raw_json(msg)}
        report = validate_trades([rec], self.calendar, self.policy, validated_at=received)
        self.last_trade_report = report
        self._report(report)
        return list(report.accepted)

    async def stream_events(self, symbols: Sequence[str], *, quotes: bool = True, bars: bool = True,
                            trades: bool = True) -> AsyncIterator[Quote | Bar | Trade]:
        """Validated quotes, trades and 1-minute bars from a single websocket connection, in arrival order."""
        async for kind, msg in self._stream_messages(symbols, quotes=quotes, bars=bars, trades=trades):
            received = to_utc(self.clock.now())
            items: list[Quote] | list[Bar] | list[Trade]
            if kind == "quote":
                items = self._stream_quote(msg, received)
            elif kind == "trade":
                items = self._stream_trade(msg, received)
            else:
                items = self._stream_bar(msg, received)
            for item in items:
                yield item

    async def stream_quotes(self, symbols: Sequence[str]) -> AsyncIterator[Quote]:
        async for item in self.stream_events(symbols, quotes=True, bars=False, trades=False):
            if isinstance(item, Quote):
                yield item

    async def stream_bars(self, symbols: Sequence[str]) -> AsyncIterator[Bar]:
        async for item in self.stream_events(symbols, quotes=False, bars=True, trades=False):
            if isinstance(item, Bar):
                yield item

    async def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[Trade]:
        async for item in self.stream_events(symbols, quotes=False, bars=False, trades=True):
            if isinstance(item, Trade):
                yield item


def _watch_reconnects(stream: Any, on_reconnect: Callable[[], None]) -> None:
    """Wrap the SDK's websocket (re)connect so every connection after the first is reported (alpaca-py's
    ``DataStream._start_ws``; streams without it, e.g. test fakes, are left untouched)."""
    found = getattr(stream, "_start_ws", None)
    if found is None or not callable(found):
        return
    original = cast(Callable[[], Awaitable[None]], found)
    connections = 0

    async def start_ws() -> None:
        nonlocal connections
        connections += 1
        if connections > 1:
            on_reconnect()
        await original()

    stream._start_ws = start_ws


def _items_for(resp: Any, symbol: str) -> list[Any]:
    if isinstance(resp, Mapping):
        items = resp.get(symbol) or resp.get(symbol.upper()) or []  # pyright: ignore[reportUnknownMemberType]
        return list(items)  # pyright: ignore[reportUnknownArgumentType]
    data = getattr(resp, "data", None)
    if isinstance(data, Mapping):
        return list(data.get(symbol, []) or [])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    return []
