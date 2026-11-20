"""Alpaca PAPER configuration of the LEAN path: paper-only by construction, no secrets in the repo, fail closed.

Proves (by parsing and grepping the committed files) that no live-trading config or launcher exists and that
``alpaca-paper-trading=false`` cannot be produced by these tools. No network, no LEAN process, no real secrets.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from trading.lean.tools import lat_lean

ROOT = lat_lean.ROOT
TEMPLATE = lat_lean.PAPER_TEMPLATE
LAUNCHER = ROOT / "scripts" / "lean_paper_alpaca.sh"
FAKE_PAPER_KEY = "PKTESTONLY0000000000"  # not a credential: shape of a paper key id, used offline only
FAKE_SECRET = "fake-secret-value-for-unit-tests-only"  # test value, never sent anywhere


def _template() -> dict[str, Any]:
    return json.loads(TEMPLATE.read_text())


def _owned_text_files() -> list[Path]:
    files = [*sorted((ROOT / "scripts").glob("lean_*.sh")), *sorted((ROOT / "trading" / "lean" / "config").glob("*")),
             *sorted((ROOT / "trading" / "lean" / "tools").glob("*.py"))]
    for proj in ("LatStrategies", "LatAlpacaGate", "LatTestAlgorithms"):
        files += sorted((ROOT / "trading" / "lean" / proj).rglob("*.cs"))
    return [f for f in files if "/bin/" not in str(f) and "/obj/" not in str(f)]


# ------------------------------------------------------------------------------------------ static proofs


def test_template_hardcodes_paper_gated_brokerage_and_has_no_secrets() -> None:
    t = _template()
    assert t["alpaca-paper-trading"] is True
    assert t["live-mode-brokerage"] == "GatedAlpacaBrokerage"
    assert t["data-queue-handler"] == ["GatedAlpacaBrokerage"]
    assert t["transaction-handler"].endswith("BrokerageTransactionHandler")
    assert t["lat-gateway"] == "LEAN_ALPACA_PAPER" and t["lat-mode"] == "PAPER"
    assert t["force-exchange-always-open"] is False
    assert t["algorithm-type-name"] == "Lat.Lean.Algorithms.LatIntentBridgeAlgorithm"
    for key in ("alpaca-api-key", "alpaca-api-secret", "alpaca-access-token"):
        assert t[key] == "", key
    assert not re.search(r"\b(PK|AK)[A-Z0-9]{16,}\b", TEMPLATE.read_text()), "key-id shaped string in template"


def test_no_live_trading_config_or_launcher_exists() -> None:
    configs = sorted((ROOT / "trading" / "lean" / "config").glob("*"))
    assert [c.name for c in configs] == ["live-alpaca-paper.template.json"]
    scripts = sorted(p.name for p in (ROOT / "scripts").glob("lean_*"))
    assert scripts == ["lean_backtest.sh", "lean_build.sh", "lean_live_sim.sh", "lean_paper_alpaca.sh"]
    for f in _owned_text_files():
        text = f.read_text(errors="replace")
        # any assignment of the paper flag must be literally true
        for m in re.finditer(r'"?alpaca-paper-trading"?\s*[:=,]\s*"?([A-Za-z]+)', text):
            assert m.group(1).lower() == "true", f"{f}: alpaca-paper-trading set to {m.group(1)}"
        assert "isPaperTrading: false" not in text, f
        assert '"live-mode-brokerage": "AlpacaBrokerage"' not in text, f
        assert "LEAN_ALPACA_LIVE" not in text, f


def test_gated_alpaca_brokerage_hardcodes_paper_in_source() -> None:
    src = (ROOT / "trading" / "lean" / "LatAlpacaGate" / "GatedAlpacaBrokerage.cs").read_text()
    assert "public const bool PaperTradingHardCoded = true;" in src
    assert "isPaperTrading: PaperTradingHardCoded" in src
    assert '{ "alpaca-paper-trading", "true" }' in src
    assert 'key.StartsWith("AK", StringComparison.Ordinal)' in src
    # LatBuild.LiveExecutionCompiled mirrors packages/core/modes.py and is compiled off
    gate = (ROOT / "trading" / "lean" / "LatStrategies" / "Gate" / "IntentGate.cs").read_text()
    assert "public const bool LiveExecutionCompiled = false;" in gate
    assert 'public const string AcceptedMode = "PAPER";' in gate


def test_launcher_script_has_no_live_switch() -> None:
    text = LAUNCHER.read_text()
    assert "REFUSED: LEAN execution is retired" in text
    assert "exit 2" in text
    assert "dotnet" not in text
    assert "set -x" not in text and "xtrace" not in text
    for line in text.splitlines():
        if re.search(r"\b(echo|printf|tee)\b", line):
            assert "LAT_ENGINE_TOKEN" not in line and "SECRET" not in line, line
    assert not re.search(r"\b(cat|less|head|tail)\b[^\n]*\$CFG", text), "config (holds the secret) must never be printed"


# ------------------------------------------------------------------------------------------ config generation


def _paper(**kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"key_id": FAKE_PAPER_KEY, "secret": FAKE_SECRET, "bridge_url": "http://127.0.0.1:8765",
                            "pubkey_path": Path("/nonexistent/intent_signing.pub.pem")}
    args.update(kw)
    return lat_lean.paper_config("lean-paper-unit", **args)


def test_paper_config_is_paper_only_and_carries_lat_settings() -> None:
    cfg = _paper()
    assert cfg["alpaca-paper-trading"] is True
    assert cfg["live-mode"] is True and cfg["live-mode-brokerage"] == "GatedAlpacaBrokerage"
    assert cfg["force-exchange-always-open"] is False
    assert cfg["ignore-unknown-asset-holdings"] is False  # unexpected broker holdings stop the engine (fail closed)
    assert cfg["lat-gate-journal-path"].endswith("var/paper/lean/used-intents.tsv")
    assert "_comment" not in cfg
    lat_lean.assert_paper_only(cfg)


def test_paper_config_refuses_live_key_ids() -> None:
    with pytest.raises(lat_lean.Refused, match="LIVE key"):
        _paper(key_id="AKLIVE00000000000000")


@pytest.mark.parametrize(("mutation", "match"), [
    ({"alpaca-paper-trading": False}, "hard-code"),
    ({"alpaca-paper-trading": "true"}, "hard-code"),
    ({"live-mode-brokerage": "AlpacaBrokerage"}, "GatedAlpacaBrokerage"),
    ({"lat-gateway": "LEAN_ALPACA_LIVE"}, "LEAN_ALPACA_PAPER"),
    ({"alpaca-api-secret": "oops"}, "secrets"),
    ({"alpaca-access-token": "oops"}, "secrets"),
])
def test_tampered_templates_are_refused(tmp_path: Path, mutation: dict[str, Any], match: str) -> None:
    t = _template()
    t.update(mutation)
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps(t))
    with pytest.raises(lat_lean.Refused, match=match):
        _paper(template_path=bad)


@pytest.mark.parametrize(("mutation", "match"), [
    ({"alpaca-paper-trading": False}, "must be true"),
    ({"live-mode-brokerage": "AlpacaBrokerage"}, "gated"),
    ({"data-queue-handler": ["AlpacaBrokerage"]}, "data queue"),
    ({"lat-mode": "LIVE_APPROVAL"}, "PAPER"),
    ({"algorithm-type-name": "Lat.Lean.TestAlgorithms.LatRogueStrategyAlgorithm"}, "LatIntentBridgeAlgorithm"),
    ({"force-exchange-always-open": True}, "SIMULATION-only"),
    ({"alpaca-api-key": "AKLIVE00000000000000"}, "LIVE"),
])
def test_final_structural_check_rejects_live_or_ungated_configs(mutation: dict[str, Any], match: str) -> None:
    cfg = _paper()
    cfg.update(mutation)
    with pytest.raises(lat_lean.Refused, match=match):
        lat_lean.assert_paper_only(cfg)


# ------------------------------------------------------------------------------------------ secret handling


def _write(env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pub: Path | None = None) -> Path:
    monkeypatch.setattr(lat_lean, "RUNS_DIR", tmp_path / "runs")
    if pub is None:
        pub = tmp_path / "intent_signing.pub.pem"
        pub.write_text("-----BEGIN PUBLIC KEY-----\nMFk=\n-----END PUBLIC KEY-----\n")
    return lat_lean.write_paper_config_from_secret_store("lean-paper-unit", bridge_url="http://127.0.0.1:8765",
                                                         pubkey_path=pub, env=env, use_keychain=False)


def _full_env() -> dict[str, str]:
    return {"LAT_MODE": "PAPER", "ALPACA_PAPER_KEY_ID": FAKE_PAPER_KEY, "ALPACA_PAPER_SECRET_KEY": FAKE_SECRET,
            lat_lean.QC_USER_ENV: "12345", lat_lean.QC_TOKEN_ENV: "qc-token-test-only", lat_lean.QC_ACK_ENV: "1"}


def test_refuses_non_paper_mode_before_reading_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("LIVE_APPROVAL", "BOUNDED_LIVE", "REPLAY"):
        with pytest.raises(lat_lean.Refused, match="PAPER mode only"):
            _write({**_full_env(), "LAT_MODE": mode}, tmp_path, monkeypatch)
    assert not (tmp_path / "runs").exists()


def test_blocked_without_paper_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"LAT_MODE": "PAPER"}
    monkeypatch.setattr(lat_lean, "ROOT", tmp_path)  # no project .env is consulted
    with pytest.raises(lat_lean.Blocked, match="BLOCKED: Alpaca PAPER credentials not found"):
        _write(env, tmp_path, monkeypatch)
    assert not (tmp_path / "runs").exists()


def test_blocked_without_quantconnect_licence_credentials_or_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _full_env()
    for k in (lat_lean.QC_USER_ENV, lat_lean.QC_TOKEN_ENV):
        partial = {kk: v for kk, v in env.items() if kk != k}
        with pytest.raises(lat_lean.Blocked, match="ValidateSubscription"):
            _write(partial, tmp_path, monkeypatch)
    no_ack = {k: v for k, v in env.items() if k != lat_lean.QC_ACK_ENV}
    with pytest.raises(lat_lean.Blocked, match="MAC/IP"):
        _write(no_ack, tmp_path, monkeypatch)


def test_live_key_from_secret_store_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(lat_lean.Refused, match="LIVE"):
        _write({**_full_env(), "ALPACA_PAPER_KEY_ID": "AKLIVE00000000000000"}, tmp_path, monkeypatch)


def test_ambiguous_upstream_env_names_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lat_lean, "ROOT", tmp_path)
    env = {"LAT_MODE": "PAPER", "APCA_API_KEY_ID": FAKE_PAPER_KEY, "APCA_API_SECRET_KEY": FAKE_SECRET}
    with pytest.raises(lat_lean.Blocked):
        _write(env, tmp_path, monkeypatch)


def test_writes_private_config_and_never_prints_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(_full_env(), tmp_path, monkeypatch)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    cfg = json.loads(path.read_text())
    assert cfg["alpaca-paper-trading"] is True and cfg["alpaca-api-key"] == FAKE_PAPER_KEY
    assert cfg["job-user-id"] == "12345"
    out = capsys.readouterr()
    assert FAKE_SECRET not in out.out + out.err and "qc-token-test-only" not in out.out + out.err


def test_retired_cli_refuses_to_generate_execution_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    import packages.core.secrets as secrets_mod

    # hermetic: no Keychain lookups (SecretStore disables them when `security` is absent), no project .env
    monkeypatch.setattr(secrets_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(lat_lean, "ROOT", tmp_path)
    monkeypatch.setattr(lat_lean, "RUNS_DIR", tmp_path / "runs")
    pub = tmp_path / "k.pem"
    pub.write_text("x")
    for k, v in _full_env().items():
        monkeypatch.setenv(k, v)
    assert lat_lean.main(["paper-config", "--run-id", "lean-paper-cli", "--pubkey", str(pub), "--symbols", "spy,qqq"]) == 2
    out = capsys.readouterr()
    cfg_path = tmp_path / "runs" / "lean-paper-cli" / "config.json"
    assert "execution is retired" in out.err
    assert not cfg_path.exists()
    assert FAKE_SECRET not in out.out + out.err and FAKE_PAPER_KEY not in out.out + out.err


# ------------------------------------------------------------------------------------------ launcher script


def _run_launcher(env_overrides: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ALPACA_", "APCA_", "LAT_QC_"))}
    env.update(env_overrides)
    return subprocess.run(["bash", str(LAUNCHER), *args], capture_output=True, text=True, env=env, timeout=120,
                          check=False, cwd=ROOT)


@pytest.mark.parametrize("mode", ["LIVE_APPROVAL", "BOUNDED_LIVE", "RESEARCH_ONLY"])
def test_launcher_refuses_non_paper_mode(mode: str) -> None:
    proc = _run_launcher({"LAT_MODE": mode})
    assert proc.returncode == 2 and "REFUSED" in proc.stderr


@pytest.mark.parametrize("args", [("--live",), ("--alpaca-paper-trading", "false"), ("--bridge-url", "http://10.0.0.2:8765")])
def test_launcher_refuses_overrides(args: tuple[str, ...]) -> None:
    proc = _run_launcher({"LAT_MODE": "PAPER"}, *args)
    assert proc.returncode == 2 and "REFUSED" in proc.stderr


def test_launcher_reports_blocked_without_keys_and_leaves_no_config() -> None:
    from packages.core.secrets import SecretStore

    if SecretStore(ROOT).alpaca_paper_available():
        pytest.skip("Alpaca PAPER keys are configured on this Mac; the BLOCKED path cannot be observed")
    before = set((ROOT / "trading" / "lean" / "runs").glob("lean-paper-alpaca-*"))
    proc = _run_launcher({"LAT_MODE": "PAPER"})
    assert proc.returncode == 2, proc.stderr
    assert "LEAN execution is retired" in proc.stderr
    after = set((ROOT / "trading" / "lean" / "runs").glob("lean-paper-alpaca-*"))
    assert not [p for p in after - before if (p / "config.json").exists()]
