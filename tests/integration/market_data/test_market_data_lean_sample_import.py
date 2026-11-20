"""Import of the REAL sample data bundled in the pinned LEAN clone (trading/lean/Lean); skipped when absent."""

from __future__ import annotations

import subprocess
import zipfile
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import DataFeed, MarketSession
from packages.core.config import PROJECT_ROOT
from services.market_data.calendar import NY, get_calendar
from services.market_data.corporate_actions import CorporateActionType
from services.market_data.lean_export import export_bars_to_lean
from services.market_data.lean_sample import import_lean_sample, read_lean_equity_dir
from services.market_data.store import MarketDataStore

LEAN_ROOT = PROJECT_ROOT / "trading" / "lean" / "Lean"
EQUITY = LEAN_ROOT / "Data" / "equity" / "usa"
pytestmark = pytest.mark.skipif(not (EQUITY / "daily" / "spy.zip").is_file(), reason="pinned LEAN clone not present")
CAL = get_calendar()


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> tuple[MarketDataStore, dict[str, Any]]:
    d = tmp_path_factory.mktemp("lean_sample")
    manifest = import_lean_sample(d, symbols=["SPY", "AAPL"])
    return MarketDataStore(d), manifest


def _raw_lines(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        return [line.split(",") for line in zf.read(zf.namelist()[0]).decode().splitlines() if line.strip()]


def test_manifest_provenance(imported: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = imported
    sha = subprocess.run(["git", "-C", str(LEAN_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert m["feed"] == "LEAN_SAMPLE" and m["is_synthetic"] is False and m["lean_commit"] == sha
    assert m["source_id"] == f"lean-sample:QuantConnect/Lean@{sha[:12]}"
    lim = " ".join(m["limitations"])
    assert "Adjustment provenance unknown" in lim and "old historical sample" in lim and "2013" in lim
    assert 2013 in m["extra"]["coverage"]["minute_years"]
    assert m["adjustment"] == "raw" and store.verify_dataset(m["dataset_version"])["ok"]


def test_daily_values_match_source_exactly(imported: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = imported
    raw = {r[0][:8]: r for r in _raw_lines(EQUITY / "daily" / "spy.zip")}
    day = date(2013, 10, 7)
    bars = store.load_bars(["SPY"], CAL.session_open(day), CAL.session_close(day), "1Day", dataset_version=m["dataset_version"])
    assert len(bars) == 1
    b, r = bars[0], raw["20131007"]
    scale = Decimal(10_000)
    assert (b.open, b.high, b.low, b.close) == tuple(Decimal(int(x)) / scale for x in r[1:5])
    assert b.volume == Decimal(int(r[5])) and b.feed == DataFeed.LEAN_SAMPLE and not b.is_synthetic
    assert (b.start_ts, b.end_ts) == (CAL.session_open(day), CAL.session_close(day))
    assert b.available_at_ts > b.end_ts  # never knowable before the session closed
    assert len(store.load_bars(["SPY"], None, None, "1Day", dataset_version=m["dataset_version"])) == len(raw)


def test_minute_bars_regular_session_and_exact(imported: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = imported
    day = date(2013, 10, 7)
    rows = _raw_lines(EQUITY / "minute" / "spy" / "20131007_trade.zip")
    bars = store.load_bars(["SPY"], CAL.session_open(day), CAL.session_close(day), "1Min",
                           dataset_version=m["dataset_version"])
    assert bars and all(b.session == MarketSession.REGULAR for b in bars)
    first = bars[0]
    midnight = datetime(2013, 10, 7, tzinfo=NY)
    by_start = {(midnight + timedelta(milliseconds=int(r[0]))).astimezone(UTC): r for r in rows}
    src = by_start[first.start_ts]
    assert first.start_ts == CAL.session_open(day)
    assert first.close == Decimal(int(src[4])) / Decimal(10_000) and first.volume == Decimal(int(src[5]))
    # extended-hours rows exist in the source but never reach the normalized table
    assert any(t < CAL.session_open(day) for t in by_start) and len(bars) <= 390
    quotes = store.load_quotes(["SPY"], CAL.session_open(day), CAL.session_close(day), dataset_version=m["dataset_version"])
    assert quotes and all(q.ask >= q.bid > 0 for q in quotes)


def test_corporate_actions_from_factor_files_point_in_time(imported: tuple[MarketDataStore, dict[str, Any]]) -> None:
    store, m = imported
    acts = store.load_corporate_actions(["AAPL"], dataset_version=m["dataset_version"])
    splits = {a.ex_date: a.split_factor for a in acts if a.action_type == CorporateActionType.SPLIT}
    seven_for_one = splits[date(2014, 6, 9)]
    assert seven_for_one is not None and seven_for_one.quantize(Decimal("0.0001")) == Decimal("0.1429")  # 7-for-1
    assert splits[date(2020, 8, 31)] == Decimal("0.25")  # 4-for-1
    divs = {a.ex_date: a.cash_amount for a in acts if a.action_type == CorporateActionType.DIVIDEND}
    assert divs[date(2014, 2, 6)] == Decimal("3.05")
    before = store.load_corporate_actions(["AAPL"], asof=CAL.session_open(date(2014, 6, 9)) - timedelta(seconds=1),
                                          dataset_version=m["dataset_version"])
    assert date(2014, 6, 9) not in {a.ex_date for a in before}  # unknown before the ex-date open


def test_lean_sample_roundtrips_through_exporter(imported: tuple[MarketDataStore, dict[str, Any]], tmp_path: Path) -> None:
    store, m = imported
    bars = store.load_bars(["SPY"], None, None, "1Min", dataset_version=m["dataset_version"])
    export_bars_to_lean(bars, tmp_path, derive_daily_if_missing=False)
    back = read_lean_equity_dir(tmp_path / "equity" / "usa", source_id="x", received=datetime(2030, 1, 1, tzinfo=UTC))
    key = ("open", "high", "low", "close", "volume")
    assert sorted((b.start_ts, *(getattr(b, k) for k in key)) for b in bars) == \
        sorted((r["start_ts"], *(r[k] for k in key)) for r in back.bar_records)
