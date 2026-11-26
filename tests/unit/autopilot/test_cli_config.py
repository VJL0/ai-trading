"""CLI entrypoints (`run`, `run --once`, `status`, `plan`) and LAT_AUTOPILOT_* configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.autopilot import cli
from services.autopilot.config import AutopilotConfig, ConfigError, env_names
from services.autopilot.store import instance_lock

from .fakes import FakeApi, FakeClock, et


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for k in list(__import__("os").environ):
        if k.startswith("LAT_AUTOPILOT_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("LAT_USE_KEYCHAIN", "false")
    monkeypatch.setenv("LAT_FEED_MODE", "SIMULATED")
    monkeypatch.setenv("LAT_MODE", "PAPER")
    monkeypatch.setenv("LAT_GATEWAY", "FAKE")
    monkeypatch.setenv("LAT_AUTOPILOT_LEAD", "false")  # never launch a real lead child (subscription quota) in a test
    return tmp_path


def test_config_defaults_bounds_and_names(tmp_path: Path) -> None:
    cfg = AutopilotConfig.from_env({}, root=tmp_path)
    assert not cfg.enabled and cfg.intraday_interval_min == 15 and cfg.crew_ticks == 2
    assert cfg.state_dir == tmp_path / "experiments" / "autopilot"
    assert cfg.report_dir == tmp_path / "experiments" / "autopilot" / "reports"
    assert cfg.session_minutes is None
    assert cfg.lead and cfg.lead_session_actions and not cfg.lead_fake and cfg.lead_max_cycles == 10
    assert cfg.lead_briefing_path == tmp_path / "experiments" / "autopilot" / "lead" / "briefing.json"
    cfg = AutopilotConfig.from_env({"LAT_AUTOPILOT_ENABLED": "true", "LAT_AUTOPILOT_SYMBOLS": "spy, qqq",
                                    "LAT_AUTOPILOT_SESSION_MINUTES": "120"}, root=tmp_path)
    assert cfg.enabled and cfg.symbols == ("SPY", "QQQ") and cfg.session_minutes == 120
    for bad in ({"LAT_AUTOPILOT_INTRADAY_INTERVAL_MIN": "1"}, {"LAT_AUTOPILOT_ENABLED": "maybe"},
                {"LAT_AUTOPILOT_SYMBOLS": "spy;rm"}, {"LAT_AUTOPILOT_CREW_TICKS": "100"}):
        with pytest.raises(ConfigError):
            AutopilotConfig.from_env(bad, root=tmp_path)
    assert "LAT_AUTOPILOT_ENABLED" in env_names() and all(n.startswith("LAT_AUTOPILOT_") for n in env_names())


def test_run_refuses_when_disabled(env: Path) -> None:
    assert cli.main(["run", "--once"]) == cli.EXIT_DISABLED


def test_run_refuses_live(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    monkeypatch.setenv("LAT_TRADING_MODE", "LIVE")
    assert cli.main(["run", "--once"]) == cli.EXIT_REFUSED


def test_dotenv_configuration_supports_process_overrides(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (env / ".env").write_text("LAT_AUTOPILOT_ENABLED=true\nLAT_AUTOPILOT_CREW_TICKS=7\n")
    monkeypatch.setenv("LAT_AUTOPILOT_CREW_TICKS", "1")
    cfg = AutopilotConfig.from_env()
    assert cfg.enabled and cfg.crew_ticks == 1


def test_single_instance_lock(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    cfg = AutopilotConfig.from_env()
    clock = FakeClock(et(2026, 11, 25, 7, 0))
    with instance_lock(cfg.lock_path):
        rc = cli.main(["run", "--once"], clock=clock, backend_factory=lambda s: FakeApi(clock),
                      ensure_api=lambda s: None)
    assert rc == cli.EXIT_LOCKED


def test_alternate_state_dirs_cannot_bypass_workspace_lock(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    cfg = AutopilotConfig.from_env()
    clock = FakeClock(et(2026, 11, 25, 7))
    with instance_lock(cfg.lock_path):
        monkeypatch.setenv("LAT_AUTOPILOT_STATE_DIR", "alternate")
        rc = cli.main(["run", "--once"], clock=clock, backend_factory=lambda s: FakeApi(clock))
    assert rc == cli.EXIT_LOCKED


def test_lock_refuses_links_without_truncating_target(env: Path) -> None:
    cfg = AutopilotConfig.from_env()
    cfg.lock_path.parent.mkdir(parents=True)
    target = env / "other-file"
    target.write_text("preserve")
    cfg.lock_path.symlink_to(target)
    with pytest.raises(OSError), instance_lock(cfg.lock_path):
        pytest.fail("linked lock was acquired")
    assert target.read_text() == "preserve"


def test_corrupt_state_refuses_to_run_and_preserves_file(env: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    cfg = AutopilotConfig.from_env()
    cfg.store_path.parent.mkdir(parents=True)
    cfg.store_path.write_bytes(b"corrupt fixture")
    clock = FakeClock(et(2026, 11, 25, 8, 45))
    api = FakeApi(clock)
    assert cli.main(["run", "--once"], clock=clock, backend_factory=lambda s: api) == cli.EXIT_STATE_ERROR
    assert "state error" in capsys.readouterr().err
    assert cfg.store_path.read_bytes() == b"corrupt fixture" and api.calls == []


def test_run_once_status_and_plan(env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    clock = FakeClock(et(2026, 11, 27, 8, 45))
    api = FakeApi(clock)
    started: list[str] = []
    assert cli.main(["plan", "--json"], clock=clock) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [a["kind"] for a in plan["actions"]] == ["PREOPEN"]
    assert plan["trading_day"]["early_close"] is True
    assert not (env / "experiments" / "autopilot" / "autopilot.sqlite3").exists()  # plan writes nothing
    rc = cli.main(["run", "--once"], clock=clock, backend_factory=lambda s: api,
                  ensure_api=lambda s: started.append("api"))
    capsys.readouterr()
    assert rc == 0 and started == ["api"] and api.count("start_paper_session") == 1
    assert (env / "experiments" / "autopilot" / "reports" / "2026-11-27.json").is_file()
    assert cli.main(["status", "--json"], clock=clock) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["phase"] == "PREOPEN" and st["running_pid"] is None
    assert {(j["job"], j["status"]) for j in st["jobs"]} >= {("preopen", "DONE")}
    assert cli.main(["plan", "--at", "2026-11-26T12:00:00-05:00"], clock=clock) == 0
    assert "not an XNYS session" in capsys.readouterr().out
