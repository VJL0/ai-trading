"""LEAN on-disk format: parsing, corporate actions from factor/map files, export and export->import round trip."""

from __future__ import annotations

import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import DataFeed, MarketSession
from services.market_data.calendar import NY, get_calendar
from services.market_data.corporate_actions import CorporateActionType
from services.market_data.fixtures import ensure_synthetic_fixture
from services.market_data.lean_export import LeanExportError, export_bars_to_lean
from services.market_data.lean_sample import (
    LeanFormatError,
    corporate_actions_from_factors,
    corporate_actions_from_map,
    parse_daily_csv,
    parse_factor_file,
    parse_map_file,
    parse_minute_quote_csv,
    parse_minute_trade_csv,
    read_lean_equity_dir,
    read_zip_text,
)
from services.market_data.quality import validate_bars
from services.market_data.store import MarketDataStore
from tests.unit.market_data.md_helpers import DAY, make_bar, session_open, utc

CAL = get_calendar()
RECV = utc(2026, 1, 1)


def test_parse_daily_and_minute_lines() -> None:
    daily = parse_daily_csv("20131007 00:00,1674100,1675300,1665300,1670100,2150000\n", "SPY", cal=CAL,
                            source_id="t", received=RECV)
    d = daily[0]
    assert d["start_ts"] == CAL.session_open(date(2013, 10, 7)) and d["end_ts"] == CAL.session_close(date(2013, 10, 7))
    assert (d["open"], d["high"], d["low"], d["close"], d["volume"]) == (
        Decimal("167.41"), Decimal("167.53"), Decimal("166.53"), Decimal("167.01"), Decimal("2150000"))
    assert d["available_at_ts"] > d["end_ts"] and d["feed"] == DataFeed.LEAN_SAMPLE and d["is_synthetic"] is False
    minute = parse_minute_trade_csv("34200000,1674100,1674100,1672500,1672700,3197\n14400000,1,2,1,1,5\n", "SPY",
                                    date(2013, 10, 7), cal=CAL, source_id="t", received=RECV)
    assert minute[0]["start_ts"] == datetime(2013, 10, 7, 9, 30, tzinfo=NY) and minute[0]["session"] == MarketSession.REGULAR
    assert minute[1]["session"] == MarketSession.PRE  # 04:00 ET extended hours
    assert minute[0]["close"] == Decimal("167.27")


def test_parse_quote_bar_one_sided_and_timestamps() -> None:
    text = "34200000,1672700,1672700,1671500,1672500,300,1680000,1680000,1672600,1672700,20000\n" \
           "34260000,1672500,1672500,1671500,1672100,300,,,,,\n"
    qs = parse_minute_quote_csv(text, "SPY", date(2013, 10, 7), source_id="t", received=RECV)
    q = qs[0]
    assert q["bid"] == Decimal("167.25") and q["ask"] == Decimal("167.27") and q["ask_size"] == Decimal(20000)
    assert q["exchange_ts"] == datetime(2013, 10, 7, 9, 31, tzinfo=NY) - timedelta(microseconds=1)
    assert q["available_at_ts"] > datetime(2013, 10, 7, 9, 31, tzinfo=NY)
    assert qs[1]["ask"] is None  # one-sided -> rejected later as INVALID_RECORD, never filled


def test_bad_price_field_raises() -> None:
    with pytest.raises(LeanFormatError):
        parse_minute_trade_csv("34200000,16741.5,1,1,1,1\n", "SPY", date(2013, 10, 7), cal=CAL, source_id="t", received=RECV)


def test_factor_file_corporate_actions_follow_lean_formula() -> None:
    rows = parse_factor_file("20140605,0.9,0.25,400\n20150101,0.92,1,100\n20501231,1,1,0\n")
    acts = corporate_actions_from_factors("XYZ", rows, CAL, "t")
    got = [(a.action_type, a.ex_date, a.split_factor, a.cash_amount) for a in acts]
    assert got == [
        (CorporateActionType.SPLIT, date(2014, 6, 6), Decimal("0.25"), None),
        (CorporateActionType.DIVIDEND, date(2014, 6, 6), None, Decimal("8.70")),  # 400 - 400*0.9/0.92
        (CorporateActionType.DIVIDEND, date(2015, 1, 2), None, Decimal("8.00")),
    ]
    assert all(a.available_at_ts == CAL.session_open(a.ex_date) for a in acts)  # known no later than the ex-date open


def test_map_file_rename_and_delisting() -> None:
    # LEAN map rows mean "ticker in use up to and including this date": AAA until 2010-06-01, BBB afterwards
    rows = parse_map_file("19980102,aaa,Q\n20100601,aaa,Q\n20120103,bbb,Q\n")
    acts = corporate_actions_from_map("AAA", rows, CAL, "t")
    assert [(a.action_type, a.ex_date, a.old_symbol, a.new_symbol) for a in acts] == [
        (CorporateActionType.RENAME, date(2010, 6, 2), "AAA", "BBB"),
        (CorporateActionType.DELISTING, date(2012, 1, 3), "BBB", None),
    ]
    live = corporate_actions_from_map("SPY", parse_map_file("19980102,spy,P\n20501231,spy,P\n"), CAL, "t")
    assert live == []


def test_export_round_trip_identical_decimal_ohlcv(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path / "ds", symbols=("SPY", "AAPL"), sessions=2)
    store = MarketDataStore(tmp_path / "ds")
    minute = store.load_bars(["SPY", "AAPL"], None, None, "1Min", dataset_version=m["dataset_version"])
    daily = store.load_bars(["SPY", "AAPL"], None, None, "1Day", dataset_version=m["dataset_version"])
    quotes = store.load_quotes(["SPY"], None, None, dataset_version=m["dataset_version"])
    out = tmp_path / "lean"
    summary = export_bars_to_lean(minute + daily, out, quotes=quotes)
    assert summary["rows"] == {"minute": len(minute), "daily": len(daily), "quote_minutes": 2 * 390}
    assert summary["is_synthetic"] is True and summary["dataset_versions"] == [m["dataset_version"]]
    eq = out / "equity" / "usa"
    assert (eq / "minute" / "spy" / f"{m['extra']['sessions'][0].replace('-', '')}_trade.zip").is_file()
    assert (eq / "map_files" / "spy.csv").read_text().splitlines()[-1] == "20501231,spy,P"
    assert (eq / "factor_files" / "aapl.csv").read_text().splitlines()[-1] == "20501231,1,1,0"
    read = read_lean_equity_dir(eq, source_id="roundtrip", received=utc(2026, 12, 31), cal=CAL)
    rep = validate_bars(read.bar_records, CAL)
    assert not rep.rejected

    def key(b: object) -> tuple[object, ...]:
        return (b.symbol, b.timeframe, CAL.ny_date(b.start_ts), b.start_ts if b.timeframe == "1Min" else None,  # type: ignore[attr-defined]
                b.open, b.high, b.low, b.close, b.volume)  # type: ignore[attr-defined]

    assert sorted(map(key, rep.accepted)) == sorted(map(key, minute + daily))
    # quote bars: the imported close equals the last stored quote of each minute
    imported = {(q["exchange_ts"].replace(second=0, microsecond=0)): (q["bid"], q["ask"]) for q in read.quote_records}
    last_per_min: dict[datetime, tuple[Decimal, Decimal]] = {}
    for q in quotes:
        last_per_min[q.exchange_ts.replace(second=0, microsecond=0)] = (q.bid, q.ask)
    assert imported == last_per_min


def test_export_is_byte_deterministic(tmp_path: Path) -> None:
    bars = [make_bar("SPY", session_open(DAY) + timedelta(minutes=i)) for i in range(3)]
    export_bars_to_lean(bars, tmp_path / "a")
    export_bars_to_lean(bars, tmp_path / "b")
    for f in sorted((tmp_path / "a").rglob("*.zip")):
        twin = tmp_path / "b" / f.relative_to(tmp_path / "a")
        assert f.read_bytes() == twin.read_bytes()
        with zipfile.ZipFile(f) as zf:
            assert len(zf.namelist()) == 1
    assert read_zip_text(tmp_path / "a" / "equity" / "usa" / "daily" / "spy.zip").startswith("20261103 00:00,")


def test_export_refuses_unrepresentable_values(tmp_path: Path) -> None:
    with pytest.raises(LeanExportError, match="x10000"):
        export_bars_to_lean([make_bar("SPY", session_open(DAY), o="100.00001", h="100.5", lo="99.5", c="100.2")], tmp_path)
    with pytest.raises(LeanExportError, match="fractional"):
        export_bars_to_lean([make_bar("SPY", session_open(DAY), v="10.5")], tmp_path)
    with pytest.raises(LeanExportError, match="unsupported timeframe"):
        export_bars_to_lean([make_bar("SPY", session_open(DAY), timeframe="5Min",
                                      end_ts=session_open(DAY) + timedelta(minutes=5))], tmp_path)


def test_zip_with_multiple_entries_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.csv", "1")
        zf.writestr("b.csv", "2")
    with pytest.raises(LeanFormatError, match="exactly one entry"):
        read_zip_text(p)
