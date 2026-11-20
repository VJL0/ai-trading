"""Build a point-in-time ``MarketSnapshot`` for one symbol (fail closed on stale/unknown data)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import timedelta
from decimal import Decimal
from typing import Any, cast

from packages.contracts.enums import DataFeed, FeedMode, MarketSession
from packages.contracts.models import Bar, MarketSnapshot, Quote
from packages.core.ids import new_id

from .calendar import CalendarOutOfRange, get_calendar
from .common import FEED_MODE_TO_FEED, feed_limitations, iso, to_utc
from .trades import Trade

log = logging.getLogger("lat.market_data.snapshot")

DEFAULT_MAX_QUOTE_AGE_S = Decimal("15")
DEFAULT_MAX_CLOCK_SKEW_S = Decimal("2")
DEFAULT_BAR_LOOKBACK = timedelta(minutes=30)


async def build_market_snapshot(
    provider: Any,
    symbol: str,
    asof: Any,
    feed_mode: FeedMode,
    *,
    dataset_version: str,
    observed_quote: Quote | None = None,
    max_quote_age_s: Decimal = DEFAULT_MAX_QUOTE_AGE_S,
    max_clock_skew_s: Decimal = DEFAULT_MAX_CLOCK_SKEW_S,
    bar_timeframe: str = "1Min",
    bar_lookback: timedelta = DEFAULT_BAR_LOOKBACK,
) -> MarketSnapshot:
    """Snapshot as of ``asof`` using only point-in-time-available data.

    * The quote must satisfy ``available_at_ts <= asof`` and be no older than ``max_quote_age_s`` (by exchange and
      receive time) and not in the future beyond ``max_clock_skew_s``; otherwise ``quote=None`` (fail closed) and
      the reason is appended to ``limitations``.
    * ``last_bar`` is the latest bar with ``available_at_ts <= asof`` inside ``bar_lookback``.
    * ``last_trade_price`` is the latest point-in-time trade print when the provider exposes trades, else the last
      bar's close (stated in ``limitations``).
    * ``is_simulated`` is true for the SIMULATED feed mode, SYNTHETIC data or any replay provider.
    """
    asof = to_utc(asof)
    limitations: list[str] = []
    quote: Quote | None = None
    try:
        latest_asof = getattr(provider, "latest_quote_asof", None)
        raw_quote = observed_quote
        if raw_quote is None:
            raw_quote = latest_asof(symbol, asof) if callable(latest_asof) else await provider.get_latest_quote(symbol)
    except Exception as exc:  # provider failure => unknown quote => fail closed
        log.warning("quote lookup failed for %s: %r", symbol, exc)
        raw_quote = None
        limitations.append(f"Quote unavailable ({type(exc).__name__}); fail closed")
    if isinstance(raw_quote, Quote):
        age_ex = Decimal(str((asof - raw_quote.exchange_ts).total_seconds()))
        age_rx = Decimal(str((asof - raw_quote.received_ts).total_seconds()))
        if raw_quote.symbol != symbol:
            limitations.append("Provider returned a quote for another symbol; withheld")
        elif raw_quote.available_at_ts > asof:
            limitations.append("Latest quote not yet available at the decision time (look-ahead); withheld")
        elif age_ex < -max_clock_skew_s or age_rx < -max_clock_skew_s:
            limitations.append("Quote timestamp is in the future relative to the decision time (clock skew); withheld")
        elif max(age_ex, age_rx) > max_quote_age_s:
            limitations.append(
                f"No quote within {max_quote_age_s}s (latest {max(age_ex, age_rx):.1f}s old at {iso(raw_quote.exchange_ts)}); "
                "quote withheld (fail closed)")
        else:
            quote = raw_quote
    elif raw_quote is None and not limitations:
        limitations.append("No point-in-time quote available; fail closed")

    last_bar: Bar | None = None
    try:
        get_bars_asof: Any = getattr(provider, "get_bars_asof", None)
        start = asof - bar_lookback
        bars: list[Any]
        if callable(get_bars_asof):
            fn = cast(Callable[..., Iterable[Any]], get_bars_asof)
            bars = list(fn(symbol, start, asof, bar_timeframe, asof))
        else:
            bars = list(await provider.get_bars(symbol, start, asof, bar_timeframe))
        pit = [b for b in bars if isinstance(b, Bar) and b.symbol == symbol and b.available_at_ts <= asof]
        last_bar = max(pit, key=lambda b: b.end_ts) if pit else None
    except Exception as exc:
        log.warning("bar lookup failed for %s: %r", symbol, exc)
        limitations.append(f"Bars unavailable ({type(exc).__name__})")

    last_trade_price, trade_note = await _last_trade_price(provider, symbol, asof, max_clock_skew_s)
    if trade_note:
        limitations.append(trade_note)
    if last_trade_price is None and last_bar is not None:
        last_trade_price = last_bar.close
        limitations.append("last_trade_price is the close of the last point-in-time bar (no trade print available)")

    try:
        session = get_calendar().classify(asof)
    except CalendarOutOfRange:
        session = MarketSession.CLOSED
        limitations.append("Decision time outside the loaded exchange calendar; session treated as CLOSED")

    default_feed = FEED_MODE_TO_FEED.get(feed_mode, DataFeed.SYNTHETIC)
    feed = quote.feed if quote is not None else (last_bar.feed if last_bar is not None else default_feed)
    from .replay import ReplayMarketDataProvider

    is_replay = isinstance(provider, ReplayMarketDataProvider)
    is_simulated = feed_mode == FeedMode.SIMULATED or feed == DataFeed.SYNTHETIC or is_replay
    base = list(feed_limitations(feed_mode, feed))
    if is_replay and feed_mode != FeedMode.SIMULATED:
        base.append("Replay of stored data at a simulated clock — not live")
    if session != MarketSession.REGULAR:
        base.append(f"Market session is {session.value}; entries are regular-session only")
    return MarketSnapshot(
        snapshot_id=new_id("snap"),
        symbol=symbol,
        feed=feed,
        feed_mode=feed_mode,
        session=session,
        quote=quote,
        last_trade_price=last_trade_price,
        last_bar=last_bar,
        data_asof=asof,
        source_id=str(getattr(provider, "source_id", "unknown")),
        dataset_version=dataset_version,
        is_simulated=is_simulated,
        limitations=tuple(dict.fromkeys(base + limitations)),
    )


async def _last_trade_price(provider: Any, symbol: str, asof: Any, max_clock_skew_s: Decimal) -> tuple[Decimal | None, str]:
    """Latest PIT trade price, or (None, reason). Never raises: unknown means unknown."""
    try:
        trade: Any
        asof_fn = getattr(provider, "latest_trade_asof", None)
        live_fn = getattr(provider, "get_latest_trade", None)
        if callable(asof_fn):
            trade = cast(Callable[[str, Any], Any], asof_fn)(symbol, asof)
        elif callable(live_fn):
            trade = await cast(Callable[[str], Awaitable[Any]], live_fn)(symbol)
        else:
            return None, ""
    except Exception as exc:
        log.warning("trade lookup failed for %s: %r", symbol, exc)
        return None, f"Last trade unavailable ({type(exc).__name__})"
    if not isinstance(trade, Trade) or trade.symbol != symbol:
        return None, ""
    if trade.available_at_ts > asof or (trade.exchange_ts - asof).total_seconds() > float(max_clock_skew_s):
        return None, "Latest trade print not available at the decision time; withheld"
    return trade.price, ""
