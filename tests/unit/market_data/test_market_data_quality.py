"""Point-in-time validation checks (reject vs flag, never fill gaps) and the live feed-quality monitor."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, FeedMode
from packages.core.clock import ManualClock
from packages.core.db import Database
from services.market_data.calendar import get_calendar
from services.market_data.quality import (
    CheckAction,
    FeedConnectionState,
    FeedQualityMonitor,
    IssueKind,
    QualityPolicy,
    record_quality_events,
    validate_bars,
    validate_quotes,
)
from tests.unit.market_data.md_helpers import DAY, bar_record, make_quote, minute_bars, quote_record, session_open

CAL = get_calendar()
OPEN = session_open(DAY)


def kinds(report: object, action: CheckAction | None = None) -> list[str]:
    """Issue kinds excluding calendar GAPs (short handcrafted series are expected to leave session gaps)."""
    return [i.kind.value for i in report.issues  # type: ignore[attr-defined]
            if (action is None or i.action == action) and i.kind != IssueKind.GAP]


def test_clean_minute_series_is_fully_accepted_without_gaps() -> None:
    bars = minute_bars("SPY", OPEN, 390)
    rep = validate_bars(bars, CAL)
    assert len(rep.accepted) == 390 and not rep.rejected and not rep.gaps and rep.ok


def test_gaps_are_reported_never_filled() -> None:
    bars = minute_bars("SPY", OPEN, 390, skip=(100, 101, 102, 200))
    rep = validate_bars(bars, CAL)
    assert len(rep.accepted) == 386  # nothing synthesized
    assert [(g.start, g.missing) for g in rep.gaps] == [(OPEN + timedelta(minutes=100), 3), (OPEN + timedelta(minutes=200), 1)]
    assert {i.kind for i in rep.issues} == {IssueKind.GAP}
    assert not rep.ok


def test_gap_vs_expected_range_and_missing_symbol() -> None:
    bars = minute_bars("SPY", OPEN, 10)
    rng = (OPEN, OPEN + timedelta(minutes=15))
    rep = validate_bars(bars, CAL, expected_range=rng, expected_symbols=["SPY", "QQQ"])
    by_sym = {g.symbol: g.missing for g in rep.gaps}
    assert by_sym == {"SPY": 5, "QQQ": 15}


def test_out_of_order_and_duplicates_rejected() -> None:
    recs = [bar_record("SPY", OPEN + timedelta(minutes=i)) for i in range(5)]
    recs[2], recs[3] = recs[3], recs[2]  # minute 2 arrives after minute 3
    recs.append(dict(recs[0]))  # identical duplicate
    conflict = dict(recs[1])
    conflict["close"] = Decimal("100.30")
    recs.append(conflict)  # same key, different values
    rep = validate_bars(recs, CAL)
    reasons = sorted(r for rej in rep.rejected for r in rej.reasons)
    assert reasons == ["DUPLICATE", "DUPLICATE_CONFLICT", "OUT_OF_ORDER"]
    assert len(rep.accepted) == 4


@pytest.mark.parametrize(
    ("override", "kind"),
    [
        ({"high": Decimal("99.00")}, "OHLC_INCONSISTENT"),
        ({"low": Decimal("0"), "open": Decimal("0.5"), "close": Decimal("0.6"), "high": Decimal("1")}, "NON_POSITIVE_PRICE"),
        ({"low": Decimal("-1")}, "NON_POSITIVE_PRICE"),
        ({"volume": Decimal("-5")}, "NEGATIVE_VOLUME"),
        ({"provider_ts": OPEN + timedelta(minutes=5)}, "CLOCK_SKEW"),  # provider ts far after received
        ({"available_at_ts": OPEN + timedelta(seconds=30)}, "LOOK_AHEAD"),  # usable before the bar closed
    ],
)
def test_single_bar_anomalies_rejected(override: dict[str, Any], kind: str) -> None:
    rec = bar_record("SPY", OPEN, **override)
    rep = validate_bars([rec], CAL)
    assert not rep.accepted
    assert kind in rep.rejected[0].reasons


def test_bar_received_before_close_is_clock_skew() -> None:
    rec = bar_record("SPY", OPEN, received_ts=OPEN + timedelta(seconds=20), available_at_ts=OPEN + timedelta(minutes=1))
    rep = validate_bars([rec], CAL)
    assert "CLOCK_SKEW" in rep.rejected[0].reasons


def test_float_prices_are_invalid_records() -> None:
    rec = bar_record("SPY", OPEN)
    rec["close"] = 100.2  # binary float must never enter the store
    rep = validate_bars([rec], CAL)
    assert rep.rejected and rep.rejected[0].reasons == ("INVALID_RECORD",)


def test_outside_regular_session_and_weekend() -> None:
    pre = bar_record("SPY", OPEN - timedelta(minutes=30))
    straddle_close = bar_record("SPY", CAL.session_close(DAY) - timedelta(seconds=30))
    saturday = bar_record("SPY", OPEN + timedelta(days=4))
    rep = validate_bars([pre, straddle_close, saturday], CAL)
    assert [r.reasons for r in rep.rejected] == [("OUTSIDE_REGULAR_SESSION",)] * 3


def test_outlier_jump_pct_flag_vs_reject_and_ignore() -> None:
    bars = [bar_record("SPY", OPEN + timedelta(minutes=i)) for i in range(3)]
    bars[2].update(open=Decimal("115"), high=Decimal("116"), low=Decimal("114"), close=Decimal("115.5"))  # +15% in a minute
    flag = validate_bars(bars, CAL)
    assert len(flag.accepted) == 3 and kinds(flag, CheckAction.FLAG) == ["OUTLIER_JUMP"]
    assert flag.flags_by_accepted_position() == {2: ["OUTLIER_JUMP"]}
    reject = validate_bars(bars, CAL, QualityPolicy().with_actions(outlier_jump="REJECT"))
    assert len(reject.accepted) == 2 and reject.rejected[0].reasons == ("OUTLIER_JUMP",)
    ignore = validate_bars(bars, CAL, QualityPolicy().with_actions(outlier_jump="IGNORE"))
    assert len(ignore.accepted) == 3 and not kinds(ignore)


def test_outlier_robust_zscore() -> None:
    bars = minute_bars("SPY", OPEN, 60, step=Decimal("0.01"))
    recs = [b.model_dump() for b in bars]
    # +1% move: far below the 8% pct limit but a huge robust z-score versus 1-cent steps
    r = recs[50]
    prev_close = recs[49]["close"]
    r.update(open=prev_close, close=(prev_close * Decimal("1.01")).quantize(Decimal("0.01")))
    r["high"] = max(r["high"], r["close"])
    rep = validate_bars(recs, CAL, QualityPolicy(max_jump_pct=Decimal("0.5")))
    flagged = [i for i in rep.issues if i.kind == IssueKind.OUTLIER_JUMP]
    assert flagged and "robust z" in flagged[0].detail


def test_corporate_action_dates_suppress_outlier() -> None:
    d1, d2 = DAY, CAL.next_session(DAY)
    recs = [bar_record("AAPL", CAL.session_open(d1), timeframe="1Day", end_ts=CAL.session_close(d1)),
            bar_record("AAPL", CAL.session_open(d2), timeframe="1Day", end_ts=CAL.session_close(d2),
                       o="25.00", h="25.10", lo="24.90", c="25.05")]  # 4:1 split
    assert kinds(validate_bars(recs, CAL)) == ["OUTLIER_JUMP"]
    assert not validate_bars(recs, CAL, corporate_action_dates={"AAPL": {d2}}).issues  # consecutive sessions: no gap


def test_quote_checks() -> None:
    t = OPEN + timedelta(minutes=5)
    recs = [
        quote_record("SPY", t),
        quote_record("SPY", t + timedelta(seconds=1), bid="100.05", ask="100.01"),  # crossed
        quote_record("SPY", t + timedelta(seconds=2), bid="100.01", ask="100.01"),  # locked
        quote_record("SPY", t + timedelta(seconds=3), bid="0", ask="100.01"),  # non-positive
        quote_record("SPY", t + timedelta(seconds=4), latency_s=30),  # stale (flag)
        quote_record("SPY", t + timedelta(seconds=5), provider_ts=t + timedelta(seconds=60)),  # clock skew
        quote_record("SPY", t - timedelta(seconds=10)),  # out of order
        quote_record("SPY", t),  # identical duplicate of the first
        quote_record("SPY", OPEN - timedelta(minutes=10)),  # pre-market
        quote_record("SPY", t + timedelta(seconds=6), available_at_ts=t),  # look-ahead
    ]
    rep = validate_quotes(recs, CAL)
    reasons = [r.reasons for r in rep.rejected]
    assert reasons == [("CROSSED_QUOTE",), ("LOCKED_QUOTE",), ("NON_POSITIVE_PRICE",), ("CLOCK_SKEW",),
                       ("OUT_OF_ORDER",), ("DUPLICATE",), ("OUTSIDE_REGULAR_SESSION", "OUT_OF_ORDER"),
                       ("LOOK_AHEAD",)]
    assert len(rep.accepted) == 2
    assert kinds(rep, CheckAction.FLAG) == ["STALE_QUOTE"]


def test_stale_quote_can_be_configured_to_reject() -> None:
    rec = quote_record("SPY", OPEN + timedelta(minutes=1), latency_s=30)
    rep = validate_quotes([rec], CAL, QualityPolicy().with_actions(stale_quote="REJECT"))
    assert rep.rejected[0].reasons == ("STALE_QUOTE",)


def test_malformed_records_never_crash() -> None:
    rep = validate_quotes([{"symbol": "SPY"}, "garbage", {"symbol": "SPY", "bid": "abc"}], CAL)  # type: ignore[list-item]
    assert len(rep.rejected) == 3 and all(r.reasons == ("INVALID_RECORD",) for r in rep.rejected)


def test_record_quality_events_writes_rows(tmp_path: Path) -> None:
    db = Database(tmp_path / "ledger.sqlite")
    try:
        bars = minute_bars("SPY", OPEN, 390, skip=(10,))
        recs = [b.model_dump() for b in bars] + [bar_record("SPY", OPEN + timedelta(minutes=400), volume=Decimal(-1))]
        rep = validate_bars(recs, CAL)
        n = record_quality_events(db, rep, DataFeed.IEX, context={"dataset_version": "ds_test"})
        rows = db.query("SELECT kind, severity, feed, symbol, detail_json FROM data_quality_events ORDER BY seq")
        # one row per issue (negative volume, after-close bar, the one-minute gap) + one summary row
        assert n == len(rows) == len(rep.issues) + 1
        assert [r["kind"] for r in rows] == ["NEGATIVE_VOLUME", "OUTSIDE_REGULAR_SESSION", "GAP", "VALIDATION_SUMMARY"]
        assert [r["severity"] for r in rows] == ["ERROR", "ERROR", "INFO", "ERROR"]
        summary = json.loads(rows[-1]["detail_json"])
        assert summary["rejected"] == 1 and summary["gaps"] == 1 and summary["context"]["dataset_version"] == "ds_test"
        assert all(r["feed"] == "IEX" for r in rows)
    finally:
        db.close()


def test_record_quality_events_truncates_per_kind(tmp_path: Path) -> None:
    db = Database(tmp_path / "ledger.sqlite")
    try:
        recs = [bar_record("SPY", OPEN + timedelta(minutes=i), volume=Decimal(-1)) for i in range(80)]
        rep = validate_bars(recs, CAL, QualityPolicy(report_gaps=False))
        record_quality_events(db, rep, DataFeed.IEX, max_rows_per_kind=10)
        rows = db.query("SELECT kind, detail_json FROM data_quality_events ORDER BY seq")
        assert len(rows) == 11
        assert json.loads(rows[-1]["detail_json"])["truncated_kinds"] == {"NEGATIVE_VOLUME": 80}
    finally:
        db.close()


# ------------------------------------------------------------------------------------------ monitor
def test_monitor_fail_closed_until_fresh_quotes() -> None:
    clock = ManualClock(OPEN + timedelta(minutes=10))
    mon = FeedQualityMonitor(FeedMode.IEX_FREE, "alpaca:iex", clock, symbols=("SPY", "QQQ"))
    st = mon.status()
    assert st["status"] == "STALE" and "no quote observed" in st["detail"]
    mon.observe_quote(make_quote("SPY", clock.now() - timedelta(seconds=1)))
    assert mon.status()["status"] == "STALE"  # QQQ still unknown
    mon.observe_quote(make_quote("QQQ", clock.now() - timedelta(seconds=2)))
    st = mon.status()
    assert st["status"] == "OK" and Decimal(st["staleness_s"]) <= Decimal("2.05")
    clock.advance(30)
    assert mon.status()["status"] == "STALE"
    mon.observe_quote(make_quote("SPY", clock.now()))
    mon.observe_quote(make_quote("QQQ", clock.now()))
    mon.set_connection_state(FeedConnectionState.DISCONNECTED, "socket closed")
    assert mon.status()["status"] == "DISCONNECTED"
    mon.set_connection_state(FeedConnectionState.BACKFILLING)
    assert mon.status()["status"] == "BACKFILLING"
    mon.set_connection_state(FeedConnectionState.LIVE)
    assert mon.status()["status"] == "OK"


def test_monitor_status_shape_matches_api_contract() -> None:
    clock = ManualClock(OPEN + timedelta(minutes=10))
    mon = FeedQualityMonitor(FeedMode.SIMULATED, "replay:ds_x", clock)
    mon.observe_quote(make_quote("SPY", clock.now(), feed=DataFeed.SYNTHETIC))
    mon.observe_bar(minute_bars("SPY", OPEN, 1, feed=DataFeed.SYNTHETIC)[0])
    mon.record_gaps("SPY", 2)
    mon.record_rejects("SPY", 3)
    st = mon.status()
    for key in ("feed_mode", "provider", "is_simulated", "last_bar_ts", "last_quote_ts", "staleness_s", "status",
                "limitations", "symbols"):
        assert key in st
    assert st["is_simulated"] is True and st["feed_mode"] == "SIMULATED"
    assert "Synthetic data — not market prices" in st["limitations"]
    row = st["symbols"][0]
    assert set(row) == {"symbol", "last_bar_ts", "last_quote_ts", "staleness_s", "spread_bps", "gaps_today", "rejected_today"}
    assert row["gaps_today"] == 2 and row["rejected_today"] == 3 and row["spread_bps"] == "2.00"
    assert st["last_quote_ts"].endswith("Z")


def test_monitor_iex_limitation_text() -> None:
    mon = FeedQualityMonitor(FeedMode.IEX_FREE, "alpaca:iex", ManualClock(OPEN))
    lim = mon.status()["limitations"]
    assert "Real-time quotes/trades from IEX only (~2-3% of consolidated volume); NBBO not guaranteed" in lim
    assert mon.status()["is_simulated"] is False


def test_monitor_daily_counters_reset() -> None:
    clock = ManualClock(OPEN)
    mon = FeedQualityMonitor(FeedMode.IEX_FREE, "alpaca:iex", clock)
    mon.record_rejects("SPY", 4)
    clock.advance(days=1)
    assert mon.status()["symbols"][0]["rejected_today"] == 0
