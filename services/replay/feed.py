"""Point-in-time market data for the replay runtime: the store and provider adapter over a :class:`BarSnapshot`.

The real runtime reads bars through ``MarketDataStore.load_bars_df`` and quotes/bars through a provider
(``latest_quote_asof``, ``get_bars_asof``) inside ``build_market_snapshot``. ``ReplayFeed`` serves both from the
snapshot with the same point-in-time rule (``available_at_ts <= asof``), so the unchanged runtime code decides on
exactly what was knowable.

Quotes. A bar snapshot has no NBBO, so quotes are SYNTHESIZED (feed ``SYNTHETIC``, labelled simulated):

* an intrabar price path through the bar's anchors, open (+1 s) -> first extreme (+100 s) -> second extreme (+200 s)
  -> close (+299 s), low first on an up bar and high first on a down bar, sampled every ``step_s`` seconds;
* a quoted spread equal to the runtime cost model's liquidity-bucket spread for the symbol's point-in-time ADV
  (previous completed sessions), so the risk engine's spread/band checks and the signal engine's cost estimate see
  the spread the cost model assumes; bid/ask are rounded outward to the tick;
* between two consecutive bars the path JUMPS (a gap is never interpolated), and inside a bar the path is continuous:
  when it crosses the stop price of a resting protective STOP_LIMIT the feed inserts an exact quote at that stop
  (``crossing_quote``), like a tape that prints every price level on the way down.

The runtime requests ``1Min`` bars; the replay serves the snapshot's native 5-minute bars (documented adapter
difference: features and horizons are computed in 5-minute bars, exactly as in QuantLab's session panel).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

import numpy as np
import pandas as pd

from packages.contracts.enums import DataFeed, MarketSession
from packages.contracts.models import Bar, Quote
from packages.core.money import price_tick
from services.research.costs import SpreadModel

from .snapshot import BAR, TIMEFRAME, BarSnapshot

ANCHORS_S = (1.0, 100.0, 200.0, 299.0)
MINUTE_TICKS = tuple(range(1, int(BAR.total_seconds()), 60))
QUOTE_SIZE = Decimal("100000")
_HISTORY = 400


def dec(x: float) -> Decimal:
    return Decimal(f"{x:.6f}")


def floor_tick(p: Decimal) -> Decimal:
    t = price_tick(p)
    return (p / t).to_integral_value(rounding=ROUND_FLOOR) * t


def ceil_tick(p: Decimal) -> Decimal:
    t = price_tick(p)
    return (p / t).to_integral_value(rounding=ROUND_CEILING) * t


@dataclass(frozen=True)
class BarRow:
    symbol: str
    start: datetime
    o: float
    h: float
    l: float  # noqa: E741 - OHLC naming
    c: float
    v: float
    vw: float

    @property
    def end(self) -> datetime:
        return self.start + BAR

    def anchors(self) -> list[tuple[float, float]]:
        first, second = (self.l, self.h) if self.c >= self.o else (self.h, self.l)
        return list(zip(ANCHORS_S, (self.o, first, second, self.c), strict=True))

    def path(self, step_s: int) -> list[tuple[float, float]]:
        """(offset seconds from bar start, price) samples; always includes the open and the close."""
        anchors = self.anchors()
        xs = [a[0] for a in anchors]
        ys = [a[1] for a in anchors]
        offs = sorted({*np.arange(ANCHORS_S[0], ANCHORS_S[-1], step_s).tolist(), *ANCHORS_S, *MINUTE_TICKS})
        return [(float(t), float(np.interp(t, xs, ys))) for t in offs]


class ReplayFeed:
    """Bars + synthesized quotes for one snapshot. Quotes are emitted forward in time by the replay engine."""

    def __init__(self, snapshot: BarSnapshot, *, step_s: int = 5, bar_latency_s: float = 0.0,
                 spread_model: SpreadModel | None = None) -> None:
        if step_s < 1 or step_s > 60:
            raise ValueError("quote step must be 1..60 s")
        if not math.isfinite(bar_latency_s) or bar_latency_s < 0:
            raise ValueError("bar latency must be finite and non-negative")
        self.snapshot = snapshot
        self.step_s = step_s
        self.bar_latency = timedelta(seconds=bar_latency_s)
        self.spread_model = spread_model or SpreadModel()
        prov = snapshot.provenance
        self.dataset_version = prov.dataset_version
        self.bar_feed = prov.feed
        self.source_id = f"replay-snapshot:{prov.dataset_version}"
        b = snapshot.bars
        self._rows: dict[str, dict[datetime, BarRow]] = {}
        for rec in b.itertuples(index=False):
            r: Any = rec
            start: datetime = pd.Timestamp(r.t).to_pydatetime()  # pyright: ignore[reportAssignmentType]
            self._rows.setdefault(str(r.symbol), {})[start] = BarRow(str(r.symbol), start, float(r.o), float(r.h),
                                                                     float(r.l), float(r.c), float(r.v), float(r.vw))
        self.frame = self._runtime_frame()
        self._avail = self.frame["available_at_ts"].to_numpy(dtype="datetime64[ns]")
        self._start = self.frame["start_ts"].to_numpy(dtype="datetime64[ns]")
        self._end = self.frame["end_ts"].to_numpy(dtype="datetime64[ns]")
        self._sym = self.frame["symbol"].to_numpy()
        self._quotes: dict[str, list[Quote]] = {}
        self._quote_avail: dict[str, list[datetime]] = {}
        self._bar_cache: dict[tuple[str, datetime], Bar] = {}
        self._spread_cache: dict[tuple[str, date], Decimal] = {}
        self.quotes_emitted = 0

    # ------------------------------------------------------------------ bars
    def _runtime_frame(self) -> pd.DataFrame:
        b = self.snapshot.bars
        prov = self.snapshot.provenance
        start = pd.to_datetime(b["t"], utc=True)
        end = start + pd.Timedelta(BAR)
        return pd.DataFrame({
            "symbol": b["symbol"].astype(str).to_numpy(), "timeframe": TIMEFRAME, "start_ts": start, "end_ts": end,
            "open": b["o"].to_numpy(dtype=float), "high": b["h"].to_numpy(dtype=float),
            "low": b["l"].to_numpy(dtype=float), "close": b["c"].to_numpy(dtype=float),
            "volume": b["v"].to_numpy(dtype=float), "vwap": b["vw"].to_numpy(dtype=float), "trade_count": 0,
            "feed": prov.feed.value, "session": MarketSession.REGULAR.value,
            "available_at_ts": end + pd.Timedelta(self.bar_latency), "source_id": self.source_id,
            "dataset_version": prov.dataset_version, "is_synthetic": prov.is_synthetic,
        })

    def bar(self, symbol: str, start: datetime) -> BarRow | None:
        return self._rows.get(symbol, {}).get(start)

    def bars_starting(self, start: datetime) -> list[BarRow]:
        return [r for s in self.snapshot.symbols if (r := self.bar(s, start)) is not None]

    # MarketDataStore.load_bars_df surface (the runtime's DecisionCycle reads bars through it)
    def load_bars_df(self, symbols: Sequence[str], start: datetime | None, end: datetime | None, timeframe: str,
                     feed: DataFeed | None = None, asof: datetime | None = None, *,
                     dataset_version: str | None = None, sessions: Any = None, prices: str = "float") -> pd.DataFrame:
        del timeframe, feed, sessions, prices  # the snapshot has one native timeframe (5Min) and one feed
        if dataset_version is not None and dataset_version != self.dataset_version:
            return self.frame.iloc[0:0].copy()
        m = np.isin(self._sym, list(symbols))
        if asof is not None:
            m &= self._avail <= np.datetime64(pd.Timestamp(asof).tz_convert(None).to_datetime64())
        if start is not None:
            m &= self._start >= np.datetime64(pd.Timestamp(start).tz_convert(None).to_datetime64())
        if end is not None:
            m &= self._end <= np.datetime64(pd.Timestamp(end).tz_convert(None).to_datetime64())
        return self.frame.loc[m].reset_index(drop=True)

    # provider surface used by build_market_snapshot
    def get_bars_asof(self, symbol: str, start: datetime, end: datetime, timeframe: str,
                      asof: datetime | None) -> list[Bar]:
        del timeframe
        cap = end if asof is None else asof
        out: list[Bar] = []
        t = start + (-(start - datetime.min.replace(tzinfo=start.tzinfo))) % BAR
        while t + BAR <= min(end, cap):
            row = self.bar(symbol, t)
            if row is not None and row.end + self.bar_latency <= cap:
                out.append(self._bar_model(row))
            t += BAR
        return out

    def _bar_model(self, row: BarRow) -> Bar:
        key = (row.symbol, row.start)
        hit = self._bar_cache.get(key)
        if hit is None:
            prov = self.snapshot.provenance
            avail = row.end + self.bar_latency
            hit = Bar(symbol=row.symbol, timeframe=TIMEFRAME, start_ts=row.start, end_ts=row.end, open=dec(row.o),
                      high=dec(row.h), low=dec(row.l), close=dec(row.c), volume=dec(row.v), vwap=dec(row.vw),
                      feed=prov.feed, session=MarketSession.REGULAR, received_ts=avail, available_at_ts=avail,
                      source_id=self.source_id, dataset_version=prov.dataset_version, is_synthetic=prov.is_synthetic)
            self._bar_cache[key] = hit
        return hit

    # ------------------------------------------------------------------ quotes
    def spread_bps(self, symbol: str, day: date) -> Decimal:
        """Full quoted spread from the runtime cost model's liquidity buckets at the point-in-time ADV."""
        key = (symbol, day)
        if key not in self._spread_cache:
            adv = self.snapshot.adv_shares(symbol, day)
            px = self.snapshot.prev_close(symbol, day)
            adv_usd = None if adv is None or px is None else Decimal(f"{adv * px:.0f}")
            self._spread_cache[key] = self.spread_model.full_spread_bps(adv_usd)[0]
        return self._spread_cache[key]

    def make_quote(self, symbol: str, at: datetime, price: float, *, bid: Decimal | None = None) -> Quote:
        from services.order_ledger.timeutil import trading_day

        half = self.spread_bps(symbol, trading_day(at)) / Decimal(20000)
        p = dec(price)
        b = bid if bid is not None else floor_tick(p * (1 - half))
        a = ceil_tick((b if bid is not None else p) * (1 + (2 * half if bid is not None else half)))
        if a <= b:
            a = b + price_tick(b)
        return Quote(symbol=symbol, bid=b, ask=a, bid_size=QUOTE_SIZE, ask_size=QUOTE_SIZE, exchange_ts=at,
                     received_ts=at, available_at_ts=at, feed=DataFeed.SYNTHETIC, source_id=self.source_id,
                     is_synthetic=True)

    def emit(self, quote: Quote) -> Quote:
        lst = self._quotes.setdefault(quote.symbol, [])
        av = self._quote_avail.setdefault(quote.symbol, [])
        if av and quote.available_at_ts < av[-1]:
            raise ValueError("replay quotes must be emitted in time order")
        lst.append(quote)
        av.append(quote.available_at_ts)
        if len(lst) > _HISTORY:
            del lst[: len(lst) - _HISTORY]
            del av[: len(av) - _HISTORY]
        self.quotes_emitted += 1
        return quote

    def latest_quote_asof(self, symbol: str, asof: datetime | None) -> Quote | None:
        lst = self._quotes.get(symbol)
        if not lst:
            return None
        if asof is None:
            return lst[-1]
        i = bisect.bisect_right(self._quote_avail[symbol], asof) - 1
        return lst[i] if i >= 0 else None

    async def get_latest_quote(self, symbol: str) -> Quote | None:
        return self.latest_quote_asof(symbol, None)

    def last_price(self, symbol: str) -> Decimal | None:
        q = self.latest_quote_asof(symbol, None)
        return None if q is None else q.mid

    def crossing_quotes(self, symbol: str, t0: datetime, p0: float, t1: datetime, p1: float,
                        stops: Sequence[Decimal]) -> Iterator[Quote]:
        """Exact quotes at every resting stop price the continuous path crosses on the way DOWN from p0 to p1."""
        if p1 >= p0:
            return
        from services.order_ledger.timeutil import trading_day

        span = (t1 - t0).total_seconds()
        half = self.spread_bps(symbol, trading_day(t0)) / Decimal(20000)
        raw_bid0, raw_bid1 = dec(p0) * (1 - half), dec(p1) * (1 - half)
        for stop in sorted(stops, reverse=True):
            if floor_tick(raw_bid1) <= stop < floor_tick(raw_bid0):
                # A rounded bid reaches the stop as it drops through the next higher tick.
                frac = float((raw_bid0 - stop - price_tick(stop)) / (raw_bid0 - raw_bid1))
                at = t0 + timedelta(seconds=max(1e-3, min(span - 1e-3, frac * span)))
                yield self.make_quote(symbol, at, float(stop), bid=stop)
