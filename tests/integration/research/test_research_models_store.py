"""Model suite against the REAL services.market_data store + its deterministic SYNTHETIC fixture (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import services.research.cli as cli
from packages.contracts.models import PredictionSet
from services.research.costs import CostModel
from services.research.models.predict import PredictionService
from services.research.models.registry import load_registry
from services.research.signals import QuantSignalEngine

fixtures = pytest.importorskip("services.market_data.fixtures")
store_mod = pytest.importorskip("services.market_data.store")

UNIVERSE = ["AAPL", "QQQ", "SPY"]


def _dt(value: object) -> datetime:
    out = pd.Timestamp(value).to_pydatetime()  # type: ignore[arg-type]
    assert isinstance(out, datetime) and out.tzinfo is not None
    return out


def _cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, Any, str]:
    code = cli.main(argv)
    cap = capsys.readouterr()
    return code, (json.loads(cap.out) if cap.out.strip() else None), cap.err


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    d = tmp_path_factory.mktemp("datasets")
    manifest = fixtures.ensure_synthetic_fixture(d, symbols=UNIVERSE, sessions=8)
    return d, manifest


def test_train_and_predict_on_the_stored_fixture(dataset: tuple[Path, dict[str, Any]], tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    d, manifest = dataset
    ver = manifest["dataset_version"]
    common = ["--source", "store", "--datasets-dir", str(d), "--dataset", ver, "--out", str(tmp_path)]
    code, rep, err = _cli(capsys, ["train-models", *common, "--folds", "3"])
    assert code == 0, err
    assert rep["dataset_version"] == ver and rep["is_simulated"] is True and rep["timeframe"] == "1Min"
    assert rep["symbols"] == UNIVERSE
    assert rep["spread_source"].startswith("stored quotes"), "PIT stored quote spreads feed the cost labels"
    assert rep["walk_forward"]["frozen_test"]["status"] == "EVALUATED_ONCE"
    reg = load_registry(tmp_path)
    assert reg is not None and reg.registry_version == rep["registry_version"]
    assert reg.meta["regime_proxy_symbols"] == UNIVERSE
    assert "SIMULATED" in reg.meta["targets"]["cost"]["realized_slippage"]
    assert reg.meta["targets"]["vol"]["har"]["base_period"].startswith("FALLBACK"), "8 sessions < 32 needed"

    # exactly the call shape of services/api/runtime.py predictions(): float prices, 10-day lookback, PIT asof
    st = store_mod.MarketDataStore(d)
    asof = _dt(pd.Timestamp(manifest["end"]) - pd.Timedelta(hours=2))
    df = st.load_bars_df(UNIVERSE, asof - timedelta(days=10), asof, "1Min", asof=asof, dataset_version=ver)
    assert not df.empty and (pd.to_datetime(df["available_at_ts"], utc=True) <= pd.Timestamp(asof)).all()
    svc = PredictionService(reg, CostModel())
    preds = svc.predict(df, asof, UNIVERSE)
    assert [p.symbol for p in preds] == UNIVERSE
    for p in preds:
        PredictionSet.model_validate_json(p.model_dump_json())
        assert p.dataset_version == ver and p.is_simulated and p.features_asof <= asof
        assert p.est_cost_bps is not None and p.est_cost_bps > 0
    if reg.selected("regime") == "markov_2state":
        assert all(p.regime.value in ("CALM", "TURBULENT") for p in preds)
        assert len({p.regime for p in preds}) == 1
    else:
        assert all(p.regime.value == "UNKNOWN" for p in preds)
    # the signal engine runs with the suite as its SignalModel; a zero-baseline selection can only be NO_TRADE
    report = QuantSignalEngine(CostModel(), model=svc).generate_report(df, asof, UNIVERSE, "cz_models_it")
    assert len(report.signals) == 3 and report.signals[0].model_version.startswith("msuite:")
    if reg.selected("ret_60m") == "zero_baseline":
        assert report.outcome == "NO_TRADE"
        assert all("expected return not positive" in s.reason for s in report.signals)

    code, out, err = _cli(capsys, ["predict", *common, "--asof", asof.isoformat()])
    assert code == 0, err
    assert out["registry_version"] == reg.registry_version and len(out["predictions"]) == 3
    assert out["dataset_version"] == ver

    # same dataset again: the frozen test was already looked at -> refused -> trade models demoted (fail closed)
    code, rep2, err = _cli(capsys, ["train-models", *common, "--folds", "3"])
    assert code == 0, err
    assert rep2["walk_forward"]["frozen_test"]["status"] == "REFUSED_ALREADY_EVALUATED"
    assert all(rep2["selected"][t] == "zero_baseline" for t in ("ret_5m", "ret_15m", "ret_60m", "rank"))
    newest = load_registry(tmp_path)
    assert newest is not None and newest.registry_version == rep2["registry_version"]
