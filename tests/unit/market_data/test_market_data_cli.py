"""Market-data CLI end-to-end on a temporary datasets dir (no network, no credentials, no repo state touched)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.market_data import cli
from services.market_data.lean_sample import read_lean_equity_dir
from services.market_data.store import MarketDataStore

SECRET_NAMES = ("ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LAT_FEED_MODE")


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("LAT_USE_KEYCHAIN", "false")
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    return code, out, captured.err


@pytest.fixture
def dsdir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    d = tmp_path / "datasets"
    code, out, _ = run(capsys, "--datasets-dir", str(d), "fixture", "--symbols", "SPY", "QQQ", "--sessions", "5")
    assert code == 0
    return d, out["dataset_version"]


def test_fixture_is_labeled_and_idempotent(dsdir: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    code, out, _ = run(capsys, "--datasets-dir", str(d), "fixture", "--symbols", "SPY", "QQQ", "--sessions", "5")
    assert code == 0 and out["dataset_version"] == ver
    assert out["label"] == "SYNTHETIC — not market data" and out["is_synthetic"] is True and out["feed"] == "SYNTHETIC"
    assert (d / "processed" / "manifests" / f"{ver}.json").is_file()


def test_status_reports_presence_only(dsdir: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    code, out, _ = run(capsys, "--datasets-dir", str(d), "status")
    assert code == 0
    assert [m["dataset_version"] for m in out["datasets"]] == [ver]
    assert out["credentials_present"] == {"alpaca_paper": False, "anthropic": False, "openai": False}
    assert out["feed_mode_effective"] == "SIMULATED" and out["active_series"]["SYNTHETIC"]["SPY"] == ver
    assert "never selected or purchased automatically" in out["sip_note"]


def test_validate_ok_then_detects_tampering(dsdir: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    code, out, _ = run(capsys, "--datasets-dir", str(d), "validate", "--dataset", ver)
    assert code == 0 and out["ok"] and out["revalidation"]["bars"]["rejected"] == 0
    assert out["revalidation"]["quotes"]["accepted"] == 2 * 5 * 390 * 12
    part = next((d / "raw" / "SYNTHETIC" / "quotes" / "symbol=SPY").glob("date=*/part.parquet"))
    part.write_bytes(part.read_bytes()[:-20] + b"\x00" * 20)
    code, out, _ = run(capsys, "--datasets-dir", str(d), "validate", "--dataset", ver)
    assert code == 1 and not out["ok"] and any("sha256 mismatch" in e for e in out["integrity"]["errors"])


def test_export_lean_round_trip(dsdir: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    out_dir = tmp_path / "lean-data"
    code, out, _ = run(capsys, "--datasets-dir", str(d), "export-lean", "--dataset", ver, "--out", str(out_dir),
                       "--no-metadata")
    assert code == 0 and out["is_synthetic"] is True and out["dataset_version"] == ver
    assert json.loads((out_dir / "lat-export-manifest.json").read_text())["dataset_version"] == ver
    assert (out_dir / "equity" / "usa" / "daily" / "spy.zip").is_file()
    assert (out_dir / "equity" / "usa" / "factor_files" / "spy.csv").is_file()
    back = read_lean_equity_dir(out_dir / "equity" / "usa", source_id="t", received=datetime(2027, 1, 1, tzinfo=UTC))
    store = MarketDataStore(d)
    orig = store.load_bars(["SPY", "QQQ"], None, None, "1Min", dataset_version=ver)
    key = ("open", "high", "low", "close", "volume")
    ours = sorted((b.symbol, b.start_ts, *(getattr(b, k) for k in key)) for b in orig)
    theirs = sorted((r["symbol"], r["start_ts"], *(r[k] for k in key)) for r in back.bar_records if r["timeframe"] == "1Min")
    assert ours == theirs  # identical Decimal OHLCV after export -> re-import
    assert len(back.quote_records) == 2 * 5 * 390


def test_import_lean_sample_from_exported_tree(dsdir: tuple[Path, str], tmp_path: Path,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    root = tmp_path / "MiniLean"
    run(capsys, "--datasets-dir", str(d), "export-lean", "--dataset", ver, "--out", str(root / "Data"), "--no-metadata")
    code, out, _ = run(capsys, "--datasets-dir", str(tmp_path / "ds2"), "import-lean-sample", "--lean-root", str(root),
                       "--no-db")
    assert code == 0 and out["feed"] == "LEAN_SAMPLE" and out["is_synthetic"] is False
    assert out["lean_commit"] == "unknown"  # not a git checkout: provenance says so instead of guessing
    assert out["row_counts"]["bars_1Min"] == 2 * 5 * 390
    code, _, err = run(capsys, "--datasets-dir", str(tmp_path / "ds3"), "import-lean-sample", "--lean-root",
                       str(tmp_path / "nowhere"), "--no-db")
    assert code == 2 and "LEAN sample data not found" in err


def test_ingest_alpaca_without_keys_is_actionable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "--datasets-dir", str(tmp_path / "d"), "ingest-alpaca", "--symbols", "SPY", "--no-db")
    assert code == 2 and out is None
    assert "ALPACA_PAPER_KEY_ID" in err and "security add-generic-password" in err and "SIMULATED" in err
    assert not (tmp_path / "d" / "raw" / "IEX").exists()


def test_universe_from_fixture(dsdir: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    d, ver = dsdir
    code, out, _ = run(capsys, "--datasets-dir", str(d), "universe", "--candidates", "SPY", "QQQ", "TQQQ", "--no-db")
    assert code == 0
    assert out["eligible_symbols"] == ["QQQ", "SPY"] and out["dataset_versions"] == [ver]
    tqqq = next(r for r in out["rows"] if r["symbol"] == "TQQQ")
    assert "DENYLISTED_LEVERAGED_INVERSE" in tqqq["reasons"] and not tqqq["eligible"]


def test_unknown_dataset_is_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, "--datasets-dir", str(tmp_path), "validate", "--dataset", "ds_0000000000000000")
    assert code == 2 and "unknown dataset" in err
