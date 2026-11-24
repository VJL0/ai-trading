"""Phase 2: scheduled history pipeline, immutable research snapshots, coverage and the research trigger.

Offline: a deterministic in-memory provider serves IEX-labelled minute bars; DuckDB/Parquet are real (tmp dirs).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, FeedMode
from packages.contracts.models import Bar
from packages.core.clock import ManualClock
from services.market_data.calendar import get_calendar
from services.market_data.common import PENDING_DATASET_VERSION
from services.market_data.feed_compare import compare_bar_sets
from services.market_data.manifest import DatasetSpec
from services.market_data.pipeline import HistoryPipeline, PipelineConfig, ready_snapshot
from services.market_data.research_snapshots import (
    SnapshotError,
    build_snapshot,
    get_snapshot,
    list_snapshots,
    verify_snapshot,
)
from services.market_data.store import MarketDataStore
from tests.unit.market_data.md_helpers import FakeHistoricalProvider, bar_record, make_bar, utc

D = Decimal
CAL = get_calendar()
DAYS = [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4), date(2026, 11, 5)]
AFTER_CLOSE = utc(2026, 11, 5, 22, 0)  # Thursday 17:00 ET


def session_bars(symbol: str, d: date, *, feed: DataFeed = DataFeed.IEX, keep: int | None = None,
                 base: str = "100.00") -> list[Bar]:
    starts = CAL.regular_bar_starts(d)[: keep]
    px = D(base)
    return [make_bar(symbol, t, o=str(px), h=str(px + D("0.05")), lo=str(px - D("0.05")), c=str(px + D("0.01")),
                     feed=feed) for t in starts]


def history(symbols: tuple[str, ...] = ("SPY", "QQQ"), days: list[date] = DAYS, **kw: Any) -> list[Bar]:
    return [b for s in symbols for d in days for b in session_bars(s, d, **kw)]


class Provider(FakeHistoricalProvider):
    def __init__(self, clock: ManualClock, bars: list[Bar]) -> None:
        super().__init__(clock, bars, feed=DataFeed.IEX, feed_mode=FeedMode.IEX_FREE)
        self.source_id = "fake:iex"


def pipe(tmp: Path, clock: ManualClock, provider: Provider, **cfg: Any) -> HistoryPipeline:
    config = PipelineConfig(**{"symbols": ("SPY", "QQQ"), "history_days": 4, "chunk_sessions": 2,
                               "request_pause_s": 0.0, **cfg})
    return HistoryPipeline(provider, MarketDataStore(tmp / "datasets"), None, clock, config)


# ------------------------------------------------------------------------------------------------ pipeline
async def test_initial_backfill_is_chunked_newest_first_and_builds_a_ready_snapshot(tmp_path: Path) -> None:
    clock = ManualClock(AFTER_CLOSE)
    p = pipe(tmp_path, clock, Provider(clock, history()))
    plan = p.plan()
    assert [(w.symbol, w.kind, w.start.date(), w.end.date()) for w in plan] == [
        ("QQQ", "initial", DAYS[2], DAYS[3]), ("SPY", "initial", DAYS[2], DAYS[3]),
        ("QQQ", "initial", DAYS[0], DAYS[1]), ("SPY", "initial", DAYS[0], DAYS[1])]
    run = await p.run()
    assert run.as_dict()["ok"] and run.rows_accepted == 4 * 2 * 390
    snap = get_snapshot(p.store, run.snapshot_id or "")
    assert snap is not None and snap["feed"] == "IEX" and len(snap["constituents"]) == 4
    cov = snap["coverage"]["SPY"]
    assert cov["sessions"] == 4 and cov["missing_sessions"] == 0 and cov["incomplete_sessions"] == 0
    assert cov["stale_sessions_at_cutoff"] == 0 and cov["median_regular_minute_coverage"] == 1.0
    assert run.research["ready"] == run.snapshot_id and run.research["reason"] == "first snapshot"
    assert ready_snapshot(tmp_path / "datasets", DataFeed.IEX) == run.snapshot_id
    assert verify_snapshot(p.store, run.snapshot_id or "")["ok"]
    assert p.plan() == []  # nothing left to fetch: idempotent


async def test_increments_rebuild_the_snapshot_but_research_waits_for_enough_new_sessions(tmp_path: Path) -> None:
    clock = ManualClock(AFTER_CLOSE)
    bars = history()
    provider = Provider(clock, bars)
    p = pipe(tmp_path, clock, provider, min_new_sessions=2)
    first = await p.run()
    # Friday: one more complete session
    friday = date(2026, 11, 6)
    provider.bars += history(days=[friday])
    clock.set(utc(2026, 11, 6, 22, 0))
    plan = p.plan()
    assert {(w.kind, w.symbol, w.start.date()) for w in plan} == {("increment", "SPY", friday), ("increment", "QQQ", friday)}
    second = await p.run()
    assert second.snapshot_id != first.snapshot_id
    assert second.research["new_complete_sessions"] == 1
    assert second.research["ready"] == first.snapshot_id  # 1 < 2: research keeps the previous snapshot
    # Monday: a second new session makes the newest snapshot the research input
    monday = date(2026, 11, 9)
    provider.bars += history(days=[monday])
    clock.set(utc(2026, 11, 9, 22, 0))
    third = await p.run()
    assert third.research["new_complete_sessions"] == 2 and third.research["ready"] == third.snapshot_id
    # an old snapshot still reproduces exactly what it contained
    assert verify_snapshot(p.store, first.snapshot_id or "")["ok"]
    old = p.store.load_bars(["SPY"], None, None, "1Min", dataset_version=first.snapshot_id)
    assert len(old) == 4 * 390 and old[-1].start_ts.date() == DAYS[-1]
    assert len(list_snapshots(p.store)) == 3


async def test_settle_window_never_requests_a_still_forming_session(tmp_path: Path) -> None:
    clock = ManualClock(utc(2026, 11, 5, 21, 5))  # 5 minutes after Thursday's close: not settled yet
    p = pipe(tmp_path, clock, Provider(clock, history()))
    assert max(w.end for w in p.plan()) == CAL.session_close(DAYS[2])  # Wednesday is the last complete session
    clock.set(utc(2026, 11, 5, 15, 0))  # mid-session
    assert max(w.end for w in p.plan()) == CAL.session_close(DAYS[2])


async def test_window_errors_are_recorded_and_a_rate_limit_stops_the_run(tmp_path: Path) -> None:
    clock = ManualClock(AFTER_CLOSE)
    provider = Provider(clock, history())
    p = pipe(tmp_path, clock, provider)
    provider.fail_next = RuntimeError("Alpaca market-data rate limit hit (HTTP 429)")
    run = await p.run()
    assert len(run.errors) == 1 and run.done == [] and not run.as_dict()["ok"]
    again = await p.run()  # the next scheduled run picks everything up
    assert again.as_dict()["ok"] and len(again.done) == 4


# ------------------------------------------------------------------------------------------------ snapshots
def _write(store: MarketDataStore, bars: list[Bar], *, feed: DataFeed, created: datetime, kind: str = "test_bars",
           synthetic: bool = False) -> str:
    spec = DatasetSpec(source_id=f"test:{feed.value.lower()}", feed=feed, generator="test", generator_version="1",
                       is_synthetic=synthetic, kind=kind, description="test", params={"n": len(bars)})
    pending = [b.model_copy(update={"dataset_version": PENDING_DATASET_VERSION}) for b in bars]
    return str(store.write_bars(pending, manifest=spec, created_at=created)["dataset_version"])


def test_snapshot_never_splices_feeds_and_honours_the_knowledge_cutoff(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path / "ds")
    iex = _write(store, history(("SPY",), DAYS[:2]), feed=DataFeed.IEX, created=utc(2026, 11, 3, 22))
    _write(store, history(("SPY",), DAYS[:2], feed=DataFeed.SIP), feed=DataFeed.SIP, created=utc(2026, 11, 3, 22))
    later = _write(store, history(("SPY",), DAYS[2:3]), feed=DataFeed.IEX, created=utc(2026, 11, 4, 22))
    s1 = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 3, 23))
    assert s1["constituents"] == [iex] and s1["source_ids"] == ["test:iex"]
    s2 = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 4, 23))
    assert s2["constituents"] == [iex, later] and s2["coverage"]["SPY"]["sessions"] == 3
    sip = build_snapshot(store, feed=DataFeed.SIP, cutoff=utc(2026, 11, 4, 23))
    assert all(store.get_manifest(v)["feed"] == "SIP" for v in sip["constituents"])
    # identical definition -> identical id (idempotent)
    assert build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 3, 23))["snapshot_id"] == s1["snapshot_id"]
    with pytest.raises(SnapshotError):
        build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 1))


def test_revisions_after_the_cutoff_never_leak_into_a_reproduced_snapshot(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path / "ds")
    _write(store, history(("SPY",), DAYS[:1]), feed=DataFeed.IEX, created=utc(2026, 11, 2, 22))
    snap = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 2, 23))
    t0 = CAL.session_open(DAYS[0])
    revised = make_bar("SPY", t0, o="100.00", h="101.00", lo="99.00", c="100.90",
                       received_ts=utc(2026, 11, 3, 12), available_at_ts=utc(2026, 11, 3, 12))
    _write(store, [revised], feed=DataFeed.IEX, created=utc(2026, 11, 3, 12), kind="test_revision")
    first = store.load_bars(["SPY"], t0, t0 + timedelta(minutes=1), "1Min", dataset_version=snap["snapshot_id"])
    assert first[0].close == D("100.01")  # the original print, as known at the cutoff
    assert verify_snapshot(store, snap["snapshot_id"])["ok"]
    new = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 3, 13))
    now_view = store.load_bars(["SPY"], t0, t0 + timedelta(minutes=1), "1Min", dataset_version=new["snapshot_id"])
    assert now_view[0].close == D("100.90") and new["coverage"]["SPY"]["revised_bars"] == 1


def test_synthetic_rows_never_enter_a_real_feed_snapshot_and_tampering_is_detected(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path / "ds")
    _write(store, history(("SPY",), DAYS[:1]), feed=DataFeed.IEX, created=utc(2026, 11, 2, 22))
    synth = [Bar.model_validate(bar_record("SPY", t, feed=DataFeed.SYNTHETIC)) for t in CAL.regular_bar_starts(DAYS[1])]
    _write(store, synth, feed=DataFeed.SYNTHETIC, created=utc(2026, 11, 3, 22), synthetic=True)
    snap = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 4))
    assert snap["is_synthetic"] is False and snap["coverage"]["SPY"]["sessions"] == 1
    with store._con() as con:
        con.execute("UPDATE bars SET close = 1.23 WHERE feed = 'IEX' AND start_ts = ?",
                    [CAL.session_open(DAYS[0]).replace(tzinfo=None)])
    report = verify_snapshot(store, snap["snapshot_id"])
    assert not report["ok"] and any("content hash" in e for e in report["errors"])


def test_coverage_reports_missing_incomplete_and_stale_sessions(tmp_path: Path) -> None:
    store = MarketDataStore(tmp_path / "ds")
    bars = session_bars("QQQ", DAYS[0]) + session_bars("QQQ", DAYS[2], keep=100)  # Tue missing, Wed incomplete
    _write(store, bars, feed=DataFeed.IEX, created=utc(2026, 11, 4, 22))
    cov = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 6, 22))["coverage"]["QQQ"]
    assert cov["missing_sessions"] == 1 and cov["missing_session_dates"] == ["2026-11-03"]
    assert cov["incomplete_sessions"] == 1 and cov["incomplete_session_dates"] == ["2026-11-04"]
    assert cov["stale_sessions_at_cutoff"] == 2  # Thu and Fri closed before the cutoff but are absent


def test_research_loads_a_snapshot_like_a_dataset(tmp_path: Path) -> None:
    from services.research.autonomous.datasets import pick_history
    from services.research.cli import load_data

    store = MarketDataStore(tmp_path / "ds")
    _write(store, history(("SPY", "QQQ"), DAYS), feed=DataFeed.IEX, created=utc(2026, 11, 5, 22))
    snap = build_snapshot(store, feed=DataFeed.IEX, cutoff=utc(2026, 11, 5, 23))
    data = load_data(source="store", dataset=snap["snapshot_id"], timeframe="1Min", symbols=None,
                     prefer_timeframe="1Min", datasets_dir=tmp_path / "ds")
    assert data.dataset_version == snap["snapshot_id"] and len(data.bars) == 2 * 4 * 390
    assert pick_history(store)["dataset_version"] == snap["snapshot_id"]  # snapshots preferred over raw batches


# ------------------------------------------------------------------------------------------------ feed comparison
def test_iex_vs_sip_comparison_metrics() -> None:
    d = DAYS[0]
    starts = CAL.regular_bar_starts(d)
    sip = [bar_record("SPY", t, c="100.10", v="10000", feed=DataFeed.SIP) for t in starts]
    iex = [bar_record("SPY", t, c="100.11", v="250") for i, t in enumerate(starts) if i % 3]  # 2/3 of minutes
    out = compare_bar_sets(iex, sip, start=CAL.session_open(d), end=CAL.session_close(d), calendar=CAL)
    assert out["expected_regular_minutes"] == 390 and out["sip_minute_coverage"] == 1.0
    assert out["iex_minute_coverage"] == pytest.approx(260 / 390, abs=1e-3)
    assert out["minutes_missing_in_iex_present_in_sip"] == 130
    assert out["iex_volume_share"] == pytest.approx(260 * 250 / (390 * 10000), abs=1e-4)
    assert out["close_diff_bps"]["median"] == pytest.approx(0.999, abs=1e-2)
