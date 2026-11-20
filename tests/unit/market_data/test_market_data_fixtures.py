"""Deterministic SYNTHETIC fixture: labeled, seeded, regular-session-only on real XNYS days, idempotent."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from packages.contracts.enums import DataFeed, MarketSession
from services.market_data.calendar import get_calendar
from services.market_data.fixtures import SYNTHETIC_LABEL, ensure_synthetic_fixture, manifest_sessions
from services.market_data.store import MarketDataStore

CAL = get_calendar()


@pytest.fixture(scope="module")
def default_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    d = tmp_path_factory.mktemp("fixture_default")
    return d, ensure_synthetic_fixture(d)


def test_default_fixture_labels_and_shape(default_fixture: tuple[Path, dict[str, Any]]) -> None:
    _d, m = default_fixture
    assert m["feed"] == "SYNTHETIC" and m["is_synthetic"] is True and m["series_mode"] == "snapshot"
    assert SYNTHETIC_LABEL == "SYNTHETIC — not market data" and SYNTHETIC_LABEL in m["limitations"]
    assert m["symbols"] == ["AAPL", "MSFT", "QQQ", "SPY"] and m["params"]["seed"] == 20261105
    sessions = manifest_sessions(m)
    assert len(sessions) == 20 and sessions[-1] == date(2026, 11, 4)
    assert all(CAL.is_session(d) for d in sessions)
    assert m["row_counts"]["bars_1Min"] == 4 * sum(CAL.regular_minutes_count(d) for d in sessions)
    assert m["row_counts"]["bars_1Day"] == 80 and m["row_counts"]["quotes"] == m["row_counts"]["bars_1Min"] * 12
    assert m["extra"]["validation"]["bars"]["rejected"] == 0 and m["extra"]["validation"]["bars"]["gaps"] == 0


def test_bars_regular_session_only_and_realistic(default_fixture: tuple[Path, dict[str, Any]]) -> None:
    d, m = default_fixture
    store = MarketDataStore(d)
    bars = store.load_bars(["SPY"], None, None, "1Min", dataset_version=m["dataset_version"])
    assert all(b.session == MarketSession.REGULAR and b.is_synthetic and b.feed == DataFeed.SYNTHETIC for b in bars)
    for b in bars[:: 97]:
        day = CAL.ny_date(b.start_ts)
        assert CAL.session_open(day) <= b.start_ts and b.end_ts <= CAL.session_close(day)
        assert b.available_at_ts > b.end_ts  # knowable only after close + latency
    closes = [b.close for b in bars]
    assert Decimal("400") < min(closes) and max(closes) < Decimal("1000")  # SPY-like level
    # intraday U-shape in volume: first/last 30 minutes are heavier than midday on average
    by_min: dict[int, list[Decimal]] = defaultdict(list)
    for b in bars:
        k = int((b.start_ts - CAL.session_open(CAL.ny_date(b.start_ts))).total_seconds() // 60)
        by_min[k].append(b.volume)
    avg: dict[int, Decimal] = {k: sum(v, Decimal(0)) / len(v) for k, v in by_min.items()}
    edge = (sum((avg[k] for k in range(30)), Decimal(0)) + sum((avg[k] for k in range(360, 390)), Decimal(0))) / 60
    mid = sum((avg[k] for k in range(165, 225)), Decimal(0)) / 60
    assert edge > 2 * mid


def test_quotes_plausible(default_fixture: tuple[Path, dict[str, Any]]) -> None:
    d, m = default_fixture
    store = MarketDataStore(d)
    day = manifest_sessions(m)[0]
    qs = store.load_quotes(["MSFT"], CAL.session_open(day), CAL.session_close(day), dataset_version=m["dataset_version"])
    assert len(qs) == 390 * 12
    spreads = [q.ask - q.bid for q in qs]
    assert min(spreads) >= Decimal("0.01") and max(spreads) <= Decimal("0.10")
    assert all(q.available_at_ts > q.exchange_ts and q.is_synthetic for q in qs)
    bps = sorted(q.spread_bps for q in qs)
    assert bps[len(bps) // 2] < Decimal("5")  # median spread < 5 bps for a mega-cap


def test_deterministic_and_idempotent(tmp_path: Path, default_fixture: tuple[Path, dict[str, Any]]) -> None:
    d1, m1 = default_fixture
    m2 = ensure_synthetic_fixture(tmp_path / "other")
    assert m1["dataset_version"] == m2["dataset_version"]
    h1 = {f["path"]: f["sha256"] for f in m1["files"]}
    h2 = {f["path"]: f["sha256"] for f in m2["files"]}
    assert h1 == h2 and len(h1) > 100
    again = ensure_synthetic_fixture(d1)
    assert again["dataset_version"] == m1["dataset_version"] and again["created_at"] == m1["created_at"]
    assert len(MarketDataStore(d1).list_manifests(DataFeed.SYNTHETIC)) == 1


def test_seed_changes_data(tmp_path: Path) -> None:
    a = ensure_synthetic_fixture(tmp_path / "a", symbols=("SPY",), sessions=2, seed=1)
    b = ensure_synthetic_fixture(tmp_path / "b", symbols=("SPY",), sessions=2, seed=2)
    assert a["dataset_version"] != b["dataset_version"]


def test_holidays_and_early_close(tmp_path: Path) -> None:
    m = ensure_synthetic_fixture(tmp_path, symbols=("QQQ",), sessions=3, end_date=date(2026, 11, 28))
    days = manifest_sessions(m)
    assert days == [date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 27)]  # Thanksgiving + weekend skipped
    bars = MarketDataStore(tmp_path).load_bars(["QQQ"], None, None, "1Min")
    per_day: dict[date, int] = defaultdict(int)
    for b in bars:
        per_day[CAL.ny_date(b.start_ts)] += 1
    assert per_day == {date(2026, 11, 24): 390, date(2026, 11, 25): 390, date(2026, 11, 27): 210}


def test_injected_defects_are_caught_and_excluded(tmp_path: Path) -> None:
    clean = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1)
    bad = ensure_synthetic_fixture(tmp_path, symbols=("SPY",), sessions=1, inject_defects=True)
    assert bad["dataset_version"] != clean["dataset_version"] and bad["defects_injected"]
    truth = bad["defects_injected"]
    expected_reject = {t["kind"] for t in truth if t["expected_action"] == "REJECT"}
    assert expected_reject >= {"OHLC_INCONSISTENT", "NON_POSITIVE_PRICE", "CLOCK_SKEW", "NEGATIVE_VOLUME", "LOOK_AHEAD",
                               "OUT_OF_ORDER", "DUPLICATE", "OUTSIDE_REGULAR_SESSION", "CROSSED_QUOTE", "LOCKED_QUOTE"}
    counts = {**bad["extra"]["validation"]["bars"]["issue_counts"], **{
        f"q:{k}": v for k, v in bad["extra"]["validation"]["quotes"]["issue_counts"].items()}}
    for kind in ("OHLC_INCONSISTENT", "CLOCK_SKEW", "NEGATIVE_VOLUME", "LOOK_AHEAD", "OUT_OF_ORDER",
                 "OUTSIDE_REGULAR_SESSION"):
        assert counts.get(f"reject:{kind}", 0) >= 1, kind
    assert counts.get("reject:DUPLICATE", 0) + counts.get("reject:DUPLICATE_CONFLICT", 0) >= 1
    assert counts["flag:OUTLIER_JUMP"] >= 1 and counts["flag:GAP"] >= 1
    for kind in ("CROSSED_QUOTE", "LOCKED_QUOTE", "CLOCK_SKEW", "NON_POSITIVE_PRICE", "OUT_OF_ORDER", "DUPLICATE"):
        assert counts.get(f"q:reject:{kind}", 0) >= 1, kind
    assert counts["q:flag:STALE_QUOTE"] >= 1
    gap_truth = sorted(t["start_ts"] for t in truth if t["kind"] == "GAP")
    # the deleted 5-minute run is reported as one gap; rejected bars also leave (reported) holes, never filled
    assert {"start": gap_truth[0], "missing": 5} in [{"start": g["start"], "missing": g["missing"]} for g in bad["extra"]["gaps"]]
    rejected_starts = {g["start"] for g in bad["extra"]["gaps"] if g["missing"] == 1}
    assert {t["start_ts"] for t in truth if t["kind"] in ("OHLC_INCONSISTENT", "NEGATIVE_VOLUME")} <= rejected_starts
    store = MarketDataStore(tmp_path)
    # default loads never see the defective dataset; the clean one stays the active series
    assert {b.dataset_version for b in store.load_bars(["SPY"], None, None, "1Min")} == {clean["dataset_version"]}
    rejected_raw = pq.read_table(next((tmp_path / "raw" / "SYNTHETIC" / "bars" / "symbol=SPY").glob("date=*/part.parquet")))
    assert "REJECTED" in rejected_raw.column("quality_status").to_pylist()
