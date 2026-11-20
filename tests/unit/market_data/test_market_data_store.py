"""MarketDataStore: Decimal storage, raw partitions, manifests, point-in-time loads and series selection."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from packages.contracts.enums import DataFeed
from packages.contracts.models import Bar
from services.market_data.corporate_actions import CorporateAction, CorporateActionType
from services.market_data.manifest import SERIES_SNAPSHOT, DatasetSpec, compute_dataset_version
from services.market_data.quality import validate_bars, validate_quotes
from services.market_data.store import BAR_DF_COLUMNS, MarketDataStore, StoreError
from tests.unit.market_data.md_helpers import DAY, bar_record, make_bar, make_quote, minute_bars, quote_record, session_open

OPEN = session_open(DAY)


def spec(feed: DataFeed = DataFeed.IEX, **kw: object) -> DatasetSpec:
    base: dict[str, object] = {"source_id": f"test:{feed.value}", "feed": feed, "generator": "tests",
                               "generator_version": "1", "is_synthetic": feed == DataFeed.SYNTHETIC}
    base.update(kw)
    return DatasetSpec(**base)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> MarketDataStore:
    return MarketDataStore(tmp_path / "datasets")


def test_roundtrip_exact_decimals_and_duckdb_types(store: MarketDataStore) -> None:
    bars = [make_bar("SPY", OPEN, o="452.1234", h="452.9999", lo="451.0001", c="452.5", v="1234567")]
    m = store.write_bars(bars, manifest=spec())
    out = store.load_bars(["SPY"], None, None, "1Min")
    assert len(out) == 1
    b = out[0]
    assert (b.open, b.high, b.low, b.close, b.volume) == (Decimal("452.1234"), Decimal("452.9999"), Decimal("451.0001"),
                                                          Decimal("452.5"), Decimal("1234567"))
    assert all(isinstance(x, Decimal) for x in (b.open, b.high, b.low, b.close, b.volume))
    assert b.dataset_version == m["dataset_version"] and b.start_ts.tzinfo is not None
    import duckdb

    con = duckdb.connect(str(store.duckdb_path), read_only=True)
    try:
        types = {r[0]: r[1] for r in con.execute("DESCRIBE bars").fetchall()}
    finally:
        con.close()
    assert types["open"] == "DECIMAL(18,6)" and types["close"] == "DECIMAL(18,6)" and types["volume"] == "DECIMAL(24,6)"
    assert not any("FLOAT" in t or "DOUBLE" in t for t in types.values())


def test_floats_cannot_reach_the_store() -> None:
    with pytest.raises(ValidationError, match="float"):
        Bar.model_validate(bar_record("SPY", OPEN, close=100.2))


def test_raw_partitions_manifest_and_hashes(store: MarketDataStore) -> None:
    bars = minute_bars("SPY", OPEN, 5) + minute_bars("QQQ", OPEN, 3)
    daily = [make_bar("SPY", OPEN, timeframe="1Day", end_ts=OPEN + timedelta(hours=6, minutes=30))]
    rep = validate_bars([*(b.model_dump() for b in bars), bar_record("SPY", OPEN, volume=Decimal(-1))])
    m = store.write_bars(rep.accepted + daily, manifest=spec(), extras=rep.raw_extras())
    raw = store.datasets_dir / "raw" / "IEX"
    p = raw / "bars" / "symbol=SPY" / "date=2026-11-03" / "part.parquet"
    assert p.is_file() and (raw / "bars" / "symbol=QQQ" / "date=2026-11-03" / "part.parquet").is_file()
    assert (raw / "bars_daily" / "symbol=SPY" / "year=2026" / "part.parquet").is_file()
    t = pq.read_table(p)
    assert t.schema.field("open").type == pa.decimal128(18, 6)
    statuses = sorted(t.column("quality_status").to_pylist())
    assert statuses == ["ACCEPTED"] * 5 + ["REJECTED"]  # the rejected raw event is preserved, not dropped
    rejected = next(r for r in t.to_pylist() if r["quality_status"] == "REJECTED")
    assert set(rejected["quality_reasons"].split(",")) == {"NEGATIVE_VOLUME", "DUPLICATE_CONFLICT"}
    assert json.loads(rejected["raw_json"])["volume"] == "-1"
    # manifest
    assert re.fullmatch(r"ds_[0-9a-f]{16}", m["dataset_version"])
    assert m["dataset_version"] == compute_dataset_version(m)
    path = store.datasets_dir / "processed" / "manifests" / f"{m['dataset_version']}.json"
    on_disk = json.loads(path.read_text())
    for key in ("source_id", "feed", "symbols", "start", "end", "row_counts", "files", "generator", "created_at",
                "is_synthetic", "limitations", "lean_commit"):
        assert key in on_disk
    assert on_disk["row_counts"]["bars"] == 9 and on_disk["row_counts"]["bars_rejected"] == 1
    for f in on_disk["files"]:
        assert hashlib.sha256((store.datasets_dir / f["path"]).read_bytes()).hexdigest() == f["sha256"]
    assert store.verify_dataset(m["dataset_version"])["ok"]


def test_idempotent_write_same_version_no_duplicates(store: MarketDataStore) -> None:
    bars = minute_bars("SPY", OPEN, 10)
    m1 = store.write_bars(bars, manifest=spec())
    m2 = store.write_bars(bars, manifest=spec())
    assert m1["dataset_version"] == m2["dataset_version"] and m1["created_at"] == m2["created_at"]
    assert store.dataset_row_counts(m1["dataset_version"])["bars"] == 10
    assert len(store.list_manifests()) == 1


def test_point_in_time_asof_and_revisions(store: MarketDataStore) -> None:
    first = make_bar("SPY", OPEN, c="100.20")
    store.write_bars([first], manifest=spec(description="initial"))
    revised = make_bar("SPY", OPEN, c="100.30", received_ts=OPEN + timedelta(hours=2),
                       available_at_ts=OPEN + timedelta(hours=2))
    store.write_bars([revised], manifest=spec(description="late revision"))
    before = OPEN + timedelta(seconds=30)
    assert store.load_bars(["SPY"], None, None, "1Min", asof=before) == []  # bar not closed yet
    mid = store.load_bars(["SPY"], None, None, "1Min", asof=OPEN + timedelta(minutes=5))
    assert [b.close for b in mid] == [Decimal("100.20")]  # the revision was not knowable yet
    late = store.load_bars(["SPY"], None, None, "1Min", asof=OPEN + timedelta(hours=3))
    assert [b.close for b in late] == [Decimal("100.30")]
    assert store.latest_bar("SPY", "1Min", asof=OPEN + timedelta(minutes=5)).close == Decimal("100.20")  # type: ignore[union-attr]


def test_feeds_never_spliced(store: MarketDataStore) -> None:
    store.write_bars(minute_bars("SPY", OPEN, 5, feed=DataFeed.IEX), manifest=spec(DataFeed.IEX))
    store.write_bars(minute_bars("SPY", OPEN, 10, feed=DataFeed.SYNTHETIC, base=Decimal("50")),
                     manifest=spec(DataFeed.SYNTHETIC))
    rows = store.load_bars(["SPY"], None, None, "1Min")
    assert len(rows) == 5 and {b.feed for b in rows} == {DataFeed.IEX}
    synth = store.load_bars(["SPY"], None, None, "1Min", feed=DataFeed.SYNTHETIC)
    assert len(synth) == 10 and all(b.is_synthetic for b in synth)


def test_snapshot_series_are_never_spliced(store: MarketDataStore) -> None:
    a = store.write_bars(minute_bars("SPY", OPEN, 10, feed=DataFeed.SYNTHETIC), manifest=spec(
        DataFeed.SYNTHETIC, series_mode=SERIES_SNAPSHOT, description="A"))
    b = store.write_bars(minute_bars("SPY", OPEN + timedelta(minutes=5), 10, feed=DataFeed.SYNTHETIC, base=Decimal("200")),
                         manifest=spec(DataFeed.SYNTHETIC, series_mode=SERIES_SNAPSHOT, description="B"))
    rows = store.load_bars(["SPY"], None, None, "1Min")
    assert {r.dataset_version for r in rows} == {b["dataset_version"]} and len(rows) == 10
    store.activate_dataset(a["dataset_version"])
    rows = store.load_bars(["SPY"], None, None, "1Min")
    assert {r.dataset_version for r in rows} == {a["dataset_version"]} and len(rows) == 10
    explicit = store.load_bars(["SPY"], None, None, "1Min", dataset_version=b["dataset_version"])
    assert len(explicit) == 10 and explicit[0].close > Decimal("199")
    assert store.active_series()["SYNTHETIC"]["SPY"] == a["dataset_version"]


def test_incremental_batches_compose(store: MarketDataStore) -> None:
    store.write_bars(minute_bars("SPY", OPEN, 5), manifest=spec(description="batch1"))
    store.write_bars(minute_bars("SPY", OPEN + timedelta(minutes=5), 5, base=Decimal("100.05")), manifest=spec(description="b2"))
    rows = store.load_bars(["SPY"], OPEN, OPEN + timedelta(minutes=10), "1Min")
    assert [r.start_ts for r in rows] == [OPEN + timedelta(minutes=i) for i in range(10)]
    with pytest.raises(StoreError, match="incremental"):
        store.activate_dataset(rows[0].dataset_version)


def test_defect_datasets_excluded_by_default(store: MarketDataStore) -> None:
    store.write_bars(minute_bars("SPY", OPEN, 3), manifest=spec(defects_injected=({"kind": "X"},)))
    assert store.load_bars(["SPY"], None, None, "1Min") == []
    version = store.list_manifests()[0]["dataset_version"]
    assert len(store.load_bars(["SPY"], None, None, "1Min", dataset_version=version)) == 3
    assert store.latest_manifest() is None


def test_load_bars_df_columns_and_float_caveat(store: MarketDataStore) -> None:
    store.write_bars(minute_bars("SPY", OPEN, 4) + minute_bars("QQQ", OPEN, 4), manifest=spec())
    df = store.load_bars_df(["SPY", "QQQ"], OPEN, OPEN + timedelta(minutes=3), "1Min", asof=OPEN + timedelta(hours=1))
    assert tuple(df.columns) == BAR_DF_COLUMNS
    assert len(df) == 6 and str(df["close"].dtype) == "float64"
    assert str(df["start_ts"].dt.tz) == "UTC" and str(df["available_at_ts"].dt.tz) == "UTC"
    exact = store.load_bars_df(["SPY"], None, None, "1Min", prices="decimal")
    assert isinstance(exact["close"].iloc[0], Decimal)
    assert bool((df["available_at_ts"] <= OPEN + timedelta(hours=1)).all())


def test_quotes_roundtrip_and_latest_pit(store: MarketDataStore) -> None:
    qs = [make_quote("SPY", OPEN + timedelta(seconds=5 * i), bid=str(Decimal("100.00") + i), ask=str(Decimal("100.02") + i))
          for i in range(5)]
    store.write_quotes(qs, manifest=spec())
    got = store.load_quotes(["SPY"], None, None)
    assert [q.bid for q in got] == [q.bid for q in qs]
    latest = store.latest_quote("SPY", OPEN + timedelta(seconds=11))
    assert latest is not None and latest.bid == Decimal("102.00")  # quote at +10s available at +10.05s
    assert store.latest_quote("SPY", OPEN) is None


def test_corporate_actions_point_in_time(store: MarketDataStore) -> None:
    ca = CorporateAction(symbol="AAPL", action_type=CorporateActionType.SPLIT, ex_date=DAY, available_at_ts=OPEN,
                         split_factor=Decimal("0.25"), feed=DataFeed.IEX, source_id="test")
    store.write_dataset(spec(), corporate_actions=[ca])
    assert store.load_corporate_actions(["AAPL"], asof=OPEN - timedelta(days=1)) == []
    got = store.load_corporate_actions(["AAPL"], asof=OPEN)
    assert len(got) == 1 and got[0].split_factor == Decimal("0.25")


def test_write_guards(store: MarketDataStore) -> None:
    with pytest.raises(StoreError, match="empty"):
        store.write_bars([], manifest=spec())
    dup = minute_bars("SPY", OPEN, 2)
    with pytest.raises(StoreError, match="duplicate"):
        store.write_bars([dup[0], dup[0]], manifest=spec())
    with pytest.raises(StoreError, match="do not match"):
        store.write_bars(minute_bars("SPY", OPEN, 2, feed=DataFeed.IEX), manifest=spec(DataFeed.SIP))
    with pytest.raises(ValueError, match="is_synthetic"):
        spec(DataFeed.IEX, is_synthetic=True)


def test_verify_detects_tampering(store: MarketDataStore) -> None:
    m = store.write_bars(minute_bars("SPY", OPEN, 5), manifest=spec())
    version = m["dataset_version"]
    part = store.datasets_dir / m["files"][0]["path"]
    part.write_bytes(part.read_bytes() + b"x")
    res = store.verify_dataset(version)
    assert not res["ok"] and any("sha256 mismatch" in e for e in res["errors"])
    mpath = store.manifests_dir / f"{version}.json"
    data = json.loads(mpath.read_text())
    data["limitations"] = ["edited"]
    mpath.write_text(json.dumps(data))
    assert any("differs from the DuckDB registry" in e for e in store.verify_dataset(version)["errors"])
    assert store.verify_dataset("ds_0000000000000000")["errors"] == ["unknown dataset_version"]


def test_rejected_quotes_preserved_raw(store: MarketDataStore) -> None:
    recs = [quote_record("SPY", OPEN + timedelta(seconds=1)),
            quote_record("SPY", OPEN + timedelta(seconds=2), bid="101", ask="100")]
    rep = validate_quotes(recs)
    store.write_quotes(rep.accepted, manifest=spec(), extras=rep.raw_extras())
    t = pq.read_table(store.datasets_dir / "raw" / "IEX" / "quotes" / "symbol=SPY" / "date=2026-11-03" / "part.parquet")
    assert sorted(t.column("quality_status").to_pylist()) == ["ACCEPTED", "REJECTED"]
    assert len(store.load_quotes(["SPY"], None, None)) == 1
