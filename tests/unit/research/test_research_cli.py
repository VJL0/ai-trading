"""Research CLI: synthetic fallback labelling, store path (fake store), errors fail closed."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import services.research.cli as cli

from .bar_factory import make_bars


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, Any, str]:
    code = cli.main(argv)
    cap = capsys.readouterr()
    out = json.loads(cap.out) if cap.out.strip() else None
    return code, out, cap.err


class FakeStore:
    """Mimics the MarketDataStore methods the CLI uses (registry + load_bars_df with PIT asof)."""

    def __init__(self, df: pd.DataFrame, manifest: dict[str, Any]) -> None:
        self.df = df
        self.manifest = manifest
        self.calls: list[dict[str, Any]] = []

    def list_manifests(self, feed: Any = None, *, include_defective: bool = True) -> list[dict[str, Any]]:
        return [self.manifest]

    def get_manifest(self, dataset_version: str) -> dict[str, Any] | None:
        return self.manifest if dataset_version == self.manifest["dataset_version"] else None

    def load_bars_df(self, symbols: list[str], start: Any, end: Any, timeframe: str, feed: Any = None,
                     asof: datetime | None = None, *, dataset_version: str | None = None, sessions: Any = None,
                     prices: str = "float") -> pd.DataFrame:
        self.calls.append({"symbols": symbols, "timeframe": timeframe, "asof": asof, "dataset_version": dataset_version,
                           "prices": prices})
        mask = (self.df["symbol"].isin(symbols) & (self.df["timeframe"] == timeframe)).to_numpy()
        if asof is not None:
            mask = mask & (self.df["available_at_ts"] <= pd.Timestamp(asof)).to_numpy()
        return pd.DataFrame(self.df.loc[mask]).reset_index(drop=True)


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    df = make_bars(("SPY", "QQQ"), timeframe="1Day", n=320, seed=31, dataset_version="ds_fake123",
                   is_synthetic=False)
    store = FakeStore(df, {"dataset_version": "ds_fake123", "timeframes": ["1Day"], "symbols": ["QQQ", "SPY"],
                           "feed": "LEAN_SAMPLE", "is_synthetic": False})
    monkeypatch.setattr(cli, "_store", lambda datasets_dir=None: store)
    return store


def test_baselines_synthetic_fallback_is_labelled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, ["baselines", "--source", "synthetic", "--out", str(tmp_path)])
    assert code == 0
    assert out["is_simulated"] is True and "SYNTHETIC" in out["data"]
    assert "NO EDGE CLAIMED" in out["no_edge_note"]
    run_dir = Path(out["run_dir"])
    run_dir = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
    assert (run_dir / "results.json").exists() and (run_dir / "manifest.json").exists()


def test_baselines_read_store_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_store: FakeStore) -> None:
    code, out, _ = _run(capsys, ["baselines", "--dataset", "ds_fake123", "--out", str(tmp_path)])
    assert code == 0
    assert out["dataset_version"] == "ds_fake123" and out["is_simulated"] is False
    assert fake_store.calls[0]["prices"] == "decimal" and fake_store.calls[0]["dataset_version"] == "ds_fake123"
    # implicit selection (no --dataset) also uses the stored dataset, never the synthetic fallback
    code, out, _ = _run(capsys, ["baselines", "--out", str(tmp_path)])
    assert code == 0 and out["dataset_version"] == "ds_fake123"


def test_signals_are_point_in_time(tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_store: FakeStore) -> None:
    asof = pd.Timestamp(sorted(fake_store.df["available_at_ts"].unique())[200]).isoformat()
    code, out, _ = _run(capsys, ["signals", "--dataset", "ds_fake123", "--asof", asof, "--out", str(tmp_path),
                                 "--horizon-minutes", "1950"])
    assert code == 0
    assert out["outcome"] in {"CANDIDATE", "NO_TRADE"}
    assert fake_store.calls[-1]["asof"] == pd.Timestamp(asof).to_pydatetime()
    assert all(pd.Timestamp(s["features_asof"]) <= pd.Timestamp(asof) for s in out["signals"])
    assert sum(1 for s in out["signals"] if s["is_candidate"]) <= 1
    assert all(s["is_simulated"] is False for s in out["signals"])
    if out["outcome"] == "NO_TRADE":
        assert out["no_trade_reason"]


def test_walkforward_command(tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_store: FakeStore) -> None:
    code, out, _ = _run(capsys, ["walkforward", "--dataset", "ds_fake123", "--out", str(tmp_path), "--folds", "3"])
    assert code == 0
    assert out["final_test"]["status"] == "EVALUATED_ONCE"
    assert out["final_test"]["final_test_hash"].startswith("sha256:")
    code2, out2, _ = _run(capsys, ["walkforward", "--dataset", "ds_fake123", "--out", str(tmp_path), "--folds", "3"])
    assert code2 == 0 and out2["final_test"]["status"] == "REFUSED_ALREADY_EVALUATED"


def test_errors_fail_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_store: FakeStore) -> None:
    code, _, err = _run(capsys, ["baselines", "--dataset", "ds_missing", "--out", str(tmp_path)])
    assert code == 2 and "not found" in err
    code, _, err = _run(capsys, ["signals", "--asof", "2024-01-05T15:00:00", "--out", str(tmp_path)])
    assert code == 2 and "timezone-aware" in err
    code, _, err = _run(capsys, ["baselines", "--timeframe", "1Min", "--out", str(tmp_path)])
    assert code == 2 and "timeframes" in err


def test_store_source_never_falls_back_to_synthetic(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_store", lambda datasets_dir=None: None)
    code, _, err = _run(capsys, ["baselines", "--source", "store", "--out", str(tmp_path)])
    assert code == 2 and "not available" in err
    code, out, _ = _run(capsys, ["baselines", "--out", str(tmp_path)])  # auto: explicit, labelled fallback
    assert code == 0 and out["is_simulated"] is True and "SYNTHETIC" in out["data"]
