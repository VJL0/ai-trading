"""Research against the REAL services.market_data store + its deterministic SYNTHETIC fixture (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import services.research.cli as cli
from services.research.costs import CostModel
from services.research.signals import QuantSignalEngine, persist_signals


def to_dt(value: object) -> datetime:
    out = pd.Timestamp(value).to_pydatetime()  # type: ignore[arg-type]
    assert isinstance(out, datetime) and out.tzinfo is not None
    return out


fixtures = pytest.importorskip("services.market_data.fixtures")
store_mod = pytest.importorskip("services.market_data.store")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    d = tmp_path_factory.mktemp("datasets")
    manifest = fixtures.ensure_synthetic_fixture(d, symbols=["SPY", "QQQ"])
    return d, manifest


def _cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, Any]:
    code = cli.main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else None)


def test_engine_on_store_frame_as_the_api_pipeline_loads_it(dataset: tuple[Path, dict[str, Any]]) -> None:
    d, manifest = dataset
    st = store_mod.MarketDataStore(d)
    end = pd.Timestamp(manifest["end"])
    asof = to_dt(end - pd.Timedelta(hours=3))
    # exactly the call shape used by services/api/pipeline.py (float prices, 10-day lookback, PIT asof)
    df = st.load_bars_df(["SPY", "QQQ"], asof - timedelta(days=10), asof, "1Min", asof=asof)
    assert not df.empty and (df["available_at_ts"] <= pd.Timestamp(asof)).all()
    sigs = QuantSignalEngine(CostModel()).generate(df, asof, ["SPY", "QQQ"], "cz_it")
    assert len(sigs) == 2
    assert all(s.is_simulated and s.dataset_version == manifest["dataset_version"] for s in sigs)
    assert sum(s.is_candidate for s in sigs) <= 1
    assert all(s.features_asof <= asof for s in sigs)


def test_cli_commands_on_stored_fixture(dataset: tuple[Path, dict[str, Any]], tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    d, manifest = dataset
    ver = manifest["dataset_version"]
    common = ["--datasets-dir", str(d), "--dataset", ver, "--out", str(tmp_path)]
    code, out = _cli(capsys, ["baselines", *common, "--timeframe", "1Min"])
    assert code == 0 and out["dataset_version"] == ver and out["is_simulated"] is True
    assert out["aggregate"]["intraday_mean_reversion"]["symbols"] == 2
    asof = (pd.Timestamp(manifest["end"]) - pd.Timedelta(hours=2)).isoformat()
    code, out = _cli(capsys, ["signals", *common, "--asof", asof])
    assert code == 0 and out["timeframe"] == "1Min" and out["outcome"] in {"CANDIDATE", "NO_TRADE"}
    code, out = _cli(capsys, ["walkforward", *common, "--timeframe", "1Min", "--horizon-bars", "30", "--folds", "3"])
    assert code == 0 and out["final_test"]["status"] == "EVALUATED_ONCE"
    code, out = _cli(capsys, ["qlib", *common])
    assert code == 0 and out["status"] in {"OK", "SKIPPED"}


def test_persist_signals_into_core_schema(tmp_path: Path, dataset: tuple[Path, dict[str, Any]]) -> None:
    from packages.core.db import Database

    d, manifest = dataset
    st = store_mod.MarketDataStore(d)
    asof = to_dt(pd.Timestamp(manifest["end"]) - pd.Timedelta(hours=3))
    df = st.load_bars_df(["SPY", "QQQ"], None, asof, "1Min", asof=asof)
    sigs = QuantSignalEngine(CostModel()).generate(df, asof, ["SPY", "QQQ"], "cz_db")
    with Database(tmp_path / "ops.db") as db:
        assert persist_signals(db, sigs, "cz_db") == 2
        assert persist_signals(db, sigs, "cz_db") == 0  # idempotent replay
        rows = db.query("SELECT signal_id, is_candidate, net_edge_bps, payload_json FROM quant_signals ORDER BY signal_id")
    assert {r["signal_id"] for r in rows} == {s.signal_id for s in sigs}
    for r in rows:
        payload = json.loads(r["payload_json"])
        assert payload["net_edge_bps"] == r["net_edge_bps"]  # Decimal string, never a float
