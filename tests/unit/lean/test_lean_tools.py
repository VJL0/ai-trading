"""LEAN run tooling: pins, cross-language intent vectors, per-run configs and backtest result parsing.

No LEAN process is started here (see tests/integration/lean and tests/failure_injection/lean for real runs).
"""

from __future__ import annotations

import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.models import INTENT_SIGNING_FIELDS, ApprovedOrderIntent
from packages.core.signing import IntentVerifier
from trading.lean.tools import gen_intent_vectors, lat_lean

ROOT = lat_lean.ROOT
VECTORS = ROOT / "trading" / "lean" / "LatStrategies.Tests" / "vectors" / "intent_vectors.json"


# ------------------------------------------------------------------------------------------ contract parity


def test_csharp_signed_field_order_matches_the_python_contract() -> None:
    src = (ROOT / "trading" / "lean" / "LatStrategies" / "Gate" / "SignedIntent.cs").read_text()
    block = re.search(r"public static readonly string\[\] Fields =\s*\{(.*?)\};", src, re.S)
    assert block, "Fields array not found in SignedIntent.cs"
    fields = tuple(re.findall(r'"([a-z_]+)"', block.group(1)))
    assert fields == INTENT_SIGNING_FIELDS
    assert 'public const string Prefix = "lat-intent-v1";' in src


def test_committed_vectors_verify_identically_in_python() -> None:
    doc = json.loads(VECTORS.read_text())
    ours = IntentVerifier.from_pem(doc["public_key_pem"].encode())
    other = IntentVerifier.from_pem(doc["other_public_key_pem"].encode())
    assert ours.key_id == doc["key_id"] and other.key_id == doc["other_key_id"]
    names = set()
    for c in doc["cases"]:
        names.add(c["name"])
        assert ours.verify(c["signing_string"], c["signature"]) is c["expect_signature_valid"], c["name"]
        assert c["signing_string"].startswith("lat-intent-v1|intent_id=" + c["intent_id"] + "|client_order_id=")
    assert other.verify(next(c for c in doc["cases"] if c["name"] == "signed_by_other_key")["signing_string"],
                        next(c for c in doc["cases"] if c["name"] == "signed_by_other_key")["signature"])
    assert {"valid_buy", "tampered_quantity", "signed_by_other_key", "expired", "market_order_intent"} <= names


def test_vector_generator_uses_the_real_contract_signing_string() -> None:
    doc = gen_intent_vectors.build()
    v = IntentVerifier.from_pem(doc["public_key_pem"].encode())
    buy = next(c for c in doc["cases"] if c["name"] == "valid_buy")
    assert v.verify(buy["signing_string"], buy["signature"])
    # the string is exactly what ApprovedOrderIntent.signing_string() produces for those fields
    parts = dict(p.split("=", 1) for p in buy["signing_string"].split("|")[1:])
    fields: dict[str, Any] = {k: (val or None) for k, val in parts.items() if k in INTENT_SIGNING_FIELDS}
    intent = ApprovedOrderIntent.model_validate({**fields, "signature": buy["signature"]})
    assert intent.signing_string() == buy["signing_string"]
    assert "-----BEGIN PUBLIC KEY-----" in doc["public_key_pem"] and "PRIVATE" not in json.dumps(doc)


# ------------------------------------------------------------------------------------------ pins


def _git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def test_pins_match_the_checked_out_sources_and_lock_files() -> None:
    pins = lat_lean.pins()
    lean_dir, plugin_dir = ROOT / "trading" / "lean" / "Lean", ROOT / "trading" / "alpaca_plugin" / "Lean.Brokerages.Alpaca"
    if not (lean_dir / ".git").exists() or not (plugin_dir / ".git").exists():
        pytest.skip("upstream clones absent")
    assert _git(lean_dir, "rev-parse", "HEAD") == pins["lean"]["commit"]
    assert _git(plugin_dir, "rev-parse", "HEAD") == pins["alpaca_plugin"]["commit"]
    assert _git(lean_dir, "status", "--porcelain", "--untracked-files=no") == "", "upstream LEAN tree modified"
    assert _git(plugin_dir, "status", "--porcelain", "--untracked-files=no") == "", "upstream plugin tree modified"
    assert pins["lean"]["local_patches"] == [] and pins["alpaca_plugin"]["local_patches"] == []
    for name, info in pins["lat_projects"].items():
        lock = ROOT / info["lock_file"]
        assert lock.is_file(), f"{name}: lock file missing"
        deps = json.loads(lock.read_text())["dependencies"]["net10.0"]
        direct = {k: v["resolved"] for k, v in deps.items() if v.get("type") == "Direct"}
        assert direct == info.get("direct_packages", direct), name
    source_build = ROOT / "trading" / "alpaca_plugin" / "source-build"
    plugin_csproj = (source_build / "QuantConnect.Brokerages.Alpaca.SourceBuild.csproj").read_text()
    assert "QuantConnect.Lean.Engine\" Version" not in plugin_csproj, "the source build must not use the NuGet LEAN packages"
    assert "..\\..\\lean\\Lean\\Engine\\QuantConnect.Lean.Engine.csproj" in plugin_csproj
    resolved = pins.get("resolved_nuget_packages", {})
    assert resolved, "PINS.json must record the resolved NuGet closure"
    assert all(re.fullmatch(r"\d+(\.\d+){1,3}(-[0-9A-Za-z.\-]+)?", v) for v in resolved.values())
    union: dict[str, str] = {}
    for info in pins["lat_projects"].values():
        for pkg, meta in json.loads((ROOT / info["lock_file"]).read_text())["dependencies"]["net10.0"].items():
            if meta.get("type") != "Project":
                assert union.setdefault(pkg, meta["resolved"]) == meta["resolved"], f"{pkg}: two versions resolved"
    assert resolved == union, "PINS.json resolved_nuget_packages out of sync with the lock files"


def test_dotnet_sdk_matches_pin() -> None:
    exe = ROOT / ".tools" / "dotnet" / "dotnet"
    if not exe.is_file():
        pytest.skip(".NET SDK not installed locally")
    out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    assert out == lat_lean.pins()["dotnet"]["sdk_version"]


# ------------------------------------------------------------------------------------------ per-run configs


def test_backtest_config_is_flat_per_run_and_accepts_a_data_folder_override(tmp_path: Path) -> None:
    spec = lat_lean.BacktestSpec(run_id="lean-bt-unit", data_folder=tmp_path, dataset="SYNTHETIC",
                                 parameters={"resolution": "Daily", "symbol": "SYN"})
    cfg = lat_lean.backtest_config(spec)
    assert "environment" not in cfg and "environments" not in cfg
    assert cfg["live-mode"] is False
    assert cfg["algorithm-type-name"] == "Lat.Lean.Algorithms.LatBaselineAlgorithm"
    assert cfg["data-folder"] == str(tmp_path.resolve()) + "/"
    assert cfg["results-destination-folder"] == str(lat_lean.RUNS_DIR / "lean-bt-unit")
    assert cfg["parameters"] == {"resolution": "Daily", "symbol": "SYN"}
    assert cfg["lat-dataset"] == "SYNTHETIC"
    with pytest.raises(ValueError, match="illegal run id"):
        lat_lean.backtest_config(lat_lean.BacktestSpec(run_id="../escape"))


def test_live_sim_config_is_simulated_gated_and_loopback(tmp_path: Path) -> None:
    spec = lat_lean.LiveSimSpec(run_id="lean-live-sim-unit", bridge_url="http://127.0.0.1:9999",
                                pubkey_path=tmp_path / "k.pem")
    cfg = lat_lean.live_sim_config(spec)
    assert cfg["live-mode"] is True
    assert cfg["live-mode-brokerage"] == "GatedPaperBrokerage"
    assert cfg["data-queue-handler"] == ["Lat.Lean.Data.LatSimulatedDataQueue"]
    assert cfg["lat-gateway"] == "LEAN_PAPER_SIM" and cfg["lat-execution-label"] == "SIMULATED"
    assert cfg["algorithm-type-name"] == "Lat.Lean.Algorithms.LatIntentBridgeAlgorithm"
    assert "alpaca-api-key" not in cfg and "alpaca-paper-trading" not in cfg
    assert cfg["job-user-id"] == "0" and cfg["api-access-token"] == ""  # no QuantConnect cloud account involved
    rogue = lat_lean.live_sim_config(lat_lean.LiveSimSpec(run_id="lean-live-sim-unit", bridge_url="http://127.0.0.1:1",
                                                          pubkey_path=tmp_path / "k.pem", rogue=True))
    assert rogue["algorithm-type-name"] == "Lat.Lean.TestAlgorithms.LatRogueStrategyAlgorithm"
    assert rogue["algorithm-location"].endswith("LatTestAlgorithms.dll")


def test_upstream_launcher_config_is_never_edited() -> None:
    lean_dir = ROOT / "trading" / "lean" / "Lean"
    if not (lean_dir / ".git").exists():
        pytest.skip("LEAN clone absent")
    assert _git(lean_dir, "status", "--porcelain", "Launcher/config.json") == ""


# ------------------------------------------------------------------------------------------ result parsing


def _fake_lean_result(run_dir: Path, run_id: str) -> None:
    run_dir.mkdir(parents=True)
    day = 86400
    values = [[1_600_000_000 + i * day, 0, 0, 0, 100000 + i * 10] for i in range(40)]
    result = {
        "statistics": {"Total Orders": "4", "Total Fees": "$1.23", "Sharpe Ratio": "0.5", "Drawdown": "2.5%",
                       "Net Profit": "0.39%", "Compounding Annual Return": "3.1%"},
        "runtimeStatistics": {"Equity": "$100,390.00"},
        "charts": {"Strategy Equity": {"series": {"Equity": {"values": values}}}},
    }
    (run_dir / f"{run_id}.json").write_text(json.dumps(result))
    (run_dir / f"{run_id}-log.txt").write_text("Launching analysis for x with LEAN Engine v2.5.0.0\nLAT baseline done\n")


def test_summarize_backtest_maps_lean_statistics_to_the_api_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lat_lean, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(lat_lean, "ROOT", tmp_path)
    run_id = "lean-bt-unit-1"
    _fake_lean_result(tmp_path / "runs" / run_id, run_id)
    out = lat_lean.summarize_backtest(run_id, strategy="LatBaselineAlgorithm", dataset="LEAN_SAMPLE",
                                      started_at="2026-11-07T00:00:00Z", notes="n", out_dir=tmp_path / "exp")
    m = out["metrics"]
    assert m == {**m, "trades": 4, "fees": "1.23", "sharpe": "0.5", "max_drawdown": "0.025", "start_equity": "100000",
                 "end_equity": "100390"}
    assert Decimal(m["total_return"]) == Decimal("0.003900")
    assert out["engine"] == "LEAN" and out["is_simulated"] is False and out["dataset"] == "LEAN_SAMPLE"
    assert all(isinstance(p["equity"], str) and isinstance(p["time"], int) for p in out["equity_curve"])
    assert json.loads((tmp_path / "exp" / run_id / "summary.json").read_text())["run_id"] == run_id
    assert (tmp_path / "exp" / run_id / "lean-log-excerpt.txt").is_file()


def test_summarize_refuses_missing_or_empty_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lat_lean, "RUNS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        lat_lean.summarize_backtest("lean-bt-none", strategy="s", dataset="d", started_at="t", notes="", out_dir=tmp_path)
    (tmp_path / "lean-bt-empty").mkdir()
    (tmp_path / "lean-bt-empty" / "lean-bt-empty.json").write_text(json.dumps({"statistics": {}, "charts": {}}))
    with pytest.raises(ValueError, match="no equity curve"):
        lat_lean.summarize_backtest("lean-bt-empty", strategy="s", dataset="d", started_at="t", notes="", out_dir=tmp_path)


def _summaries() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "experiments" / "lean").glob("*/summary.json"))]


def test_recorded_lean_backtests_are_real_engine_runs_on_labelled_data() -> None:
    summaries = _summaries()
    if not summaries:
        pytest.skip("no LEAN backtest has been run yet (scripts/lean_backtest.sh)")
    for s in summaries:
        assert s["engine"] == "LEAN" and s["lean_commit"] == lat_lean.pins()["lean"]["commit"]
        assert s["is_simulated"] is (s["dataset"] != "LEAN_SAMPLE")
        result = ROOT / s["source_path"]
        assert result.is_file(), result
        stats = json.loads(result.read_text())["statistics"]
        assert int(stats["Total Orders"]) == s["metrics"]["trades"]
        excerpt = (ROOT / s["log_excerpt_path"]).read_text()
        assert "with LEAN Engine v" in excerpt and "LAT baseline" in excerpt
        assert len(s["equity_curve"]) >= 2
        for key in ("total_return", "sharpe", "max_drawdown", "fees"):
            Decimal(s["metrics"][key])  # decimal strings, never floats


# ------------------------------------------------------------------------------------------ engine for the real local API


def _api_namespace(tmp_path: Path) -> dict[str, str]:
    from packages.core.signing import IntentSigner
    from services.order_ledger.bridge import EngineTokenAuth

    keys = tmp_path / "var" / "paper" / "keys"
    keys.mkdir(parents=True)
    IntentSigner.load_or_create(keys)
    EngineTokenAuth(keys).ensure()
    return {"LAT_MODE": "PAPER", "LAT_GATEWAY": "LEAN_PAPER_SIM", "LAT_PROJECT_ROOT": str(tmp_path), "LAT_API_PORT": "18999"}


def test_api_live_sim_config_uses_the_namespace_key_token_file_and_durable_journal(tmp_path: Path) -> None:
    env = _api_namespace(tmp_path)
    token = (tmp_path / "var" / "paper" / "keys" / "engine.token").read_text().strip()
    run_id = lat_lean.new_run_id("lean-unit-api")
    try:
        cfg_path, token_path = lat_lean.write_api_live_sim_config(run_id, exit_after_seconds=5, env=env)
        cfg = json.loads(cfg_path.read_text())
        assert token_path == tmp_path / "var" / "paper" / "keys" / "engine.token"
        assert token not in cfg_path.read_text(), "the engine token value never goes into a config file"
        assert cfg["lat-bridge-url"] == "http://127.0.0.1:18999"
        assert cfg["lat-intent-pubkey-path"] == str(tmp_path / "var" / "paper" / "keys" / "intent_signing.pub.pem")
        assert cfg["lat-gate-journal-path"] == str(tmp_path / "var" / "paper" / "lean" / "used-intents.tsv")
        assert cfg["live-mode-brokerage"] == "GatedPaperBrokerage" and cfg["lat-gateway"] == "LEAN_PAPER_SIM"
        assert cfg["algorithm-type-name"] == "Lat.Lean.Algorithms.LatIntentBridgeAlgorithm"
        assert cfg["lat-execution-label"] == "SIMULATED" and cfg["lat-exit-after-seconds"] == "5"
        quotes = json.loads(cfg["lat-sim-quotes"])
        assert set(quotes) == set(cfg["lat-symbols"].split(","))
        assert all(Decimal(q["bid"]) < Decimal(q["last"]) < Decimal(q["ask"]) for q in quotes.values())
        assert not any("alpaca" in k for k in cfg)
    finally:
        import shutil

        shutil.rmtree(lat_lean.RUNS_DIR / run_id, ignore_errors=True)


@pytest.mark.parametrize(("override", "exc", "text"), [
    ({"LAT_MODE": "REPLAY"}, "Refused", "PAPER API only"),
    ({"LAT_GATEWAY": "FAKE"}, "Refused", "LAT_GATEWAY=LEAN_PAPER_SIM"),
    ({"LAT_GATEWAY": "LEAN_ALPACA_PAPER"}, "Refused", "lean_paper_alpaca.sh"),
])
def test_api_live_sim_config_refuses_other_modes_and_gateways(tmp_path: Path, override: dict[str, str], exc: str,
                                                              text: str) -> None:
    env = {**_api_namespace(tmp_path), **override}
    with pytest.raises(getattr(lat_lean, exc), match=re.escape(text)):
        lat_lean.write_api_live_sim_config("lean-unit-api-refused", env=env)
    assert not (lat_lean.RUNS_DIR / "lean-unit-api-refused").exists()


def test_api_live_sim_config_is_blocked_without_the_api_namespace_keys(tmp_path: Path) -> None:
    env = {"LAT_MODE": "PAPER", "LAT_GATEWAY": "LEAN_PAPER_SIM", "LAT_PROJECT_ROOT": str(tmp_path)}
    with pytest.raises(lat_lean.Blocked, match="start the API once"):
        lat_lean.write_api_live_sim_config("lean-unit-api-blocked", env=env)
    assert not (lat_lean.RUNS_DIR / "lean-unit-api-blocked").exists()
