"""Shared helpers for the crew tests (FakeProvider, fake experiment runner, controllable clock) plus config tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from services.agents.config import COOLDOWN_CAP, ConfigError, CrewConfig, env_names
from services.agents.contracts import Invocation, Provider, RunOutcome, RunResult, TaskKind
from services.agents.crew import Crew
from services.agents.fake import FakeProvider, demo_output, fake_mcp_spec
from services.agents.schemas import kind_of_schema
from services.agents.store import CrewStore

T0 = datetime(2026, 11, 25, 14, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


class FakeRunner:
    """Stands in for run_agent_experiment: records calls, returns a ledger-shaped result."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[Path, dict[str, Any], str]] = []
        self.result = result or {"passed_dev_gate": False, "fail_reasons": ["dsr below bar"], "dsr": 0.41,
                                 "wf_oos_sharpe": 0.37, "n_trials": 12}
        self.responses: list[dict[str, Any]] = []

    def __call__(self, root: Path, spec: dict[str, Any], actor: str) -> dict[str, Any]:
        self.calls.append((root, spec, actor))
        if self.responses:
            return self.responses.pop(0)
        return {"accepted": True, "status": "DONE", "spec": spec, "result": dict(self.result)}


def make_config(tmp_path: Path, **kw: Any) -> CrewConfig:
    cfg = CrewConfig.from_env({}, project_root=tmp_path)
    return replace(cfg, **kw) if kw else cfg


def ok(kind: TaskKind, **overrides: Any) -> RunResult:
    def answer(inv: Invocation) -> RunResult:
        out = demo_output(kind, inv)
        out.update(overrides)
        return RunResult(RunOutcome.OK, output=out)

    return answer  # type: ignore[return-value]


def auto_ok() -> Any:
    def answer(inv: Invocation) -> RunResult:
        kind = kind_of_schema(dict(inv.output_schema))
        assert kind is not None
        return RunResult(RunOutcome.OK, output=demo_output(kind, inv))

    return answer


def make_crew(tmp_path: Path, script_claude: list[Any] | None = None, script_codex: list[Any] | None = None, *,
              clock: Clock | None = None, runner: FakeRunner | None = None, providers: tuple[Provider, ...] | None = None,
              per_hour: int = 6, **cfg: Any) -> tuple[Crew, dict[Provider, FakeProvider], FakeRunner, Clock]:
    config = make_config(tmp_path, **cfg)
    if providers is not None:
        config = replace(config, providers=providers)
    fakes = {Provider.CLAUDE: FakeProvider(Provider.CLAUDE, script_claude or []),
             Provider.CODEX: FakeProvider(Provider.CODEX, script_codex or [])}
    fakes = {p: f for p, f in fakes.items() if p in config.providers}
    clock = clock or Clock()
    runner = runner or FakeRunner()
    crew = Crew(config, CrewStore(config.store_path), fakes, experiment_runner=runner, clock=clock,
                mcp_spec_factory=fake_mcp_spec, experiments_per_hour=per_hour)
    return crew, fakes, runner, clock


# ------------------------------------------------------------------------------------------- config tests
def test_config_defaults_and_env_names(tmp_path: Path) -> None:
    cfg = CrewConfig.from_env({}, project_root=tmp_path)
    assert cfg.providers == (Provider.CLAUDE, Provider.CODEX)
    assert (cfg.max_concurrent, cfg.daily_runs_per_provider, cfg.run_timeout_s, cfg.max_turns) == (1, 12, 900, 12)
    assert (cfg.specs_per_task, cfg.cooldown_minutes, cfg.review_wait_hours, cfg.idle_sleep_s) == (3, 60, 24, 600)
    assert cfg.store_path == tmp_path / "experiments" / "agents" / "crew.sqlite3"
    assert cfg.model_claude is None and cfg.model_codex is None
    names = env_names()
    assert "LAT_AGENTS_PROVIDERS" in names and "LAT_AGENTS_STORE" in names and "LAT_AGENTS_MODEL_CODEX" in names
    assert all(n.startswith("LAT_AGENTS_") for n in names)


def test_config_from_env_overrides_and_errors(tmp_path: Path) -> None:
    cfg = CrewConfig.from_env({"LAT_AGENTS_PROVIDERS": "codex", "LAT_AGENTS_DAILY_RUNS_PER_PROVIDER": "3",
                               "LAT_AGENTS_STORE": "x/crew.db", "LAT_AGENTS_MODEL_CLAUDE": "opus"},
                              project_root=tmp_path)
    assert cfg.providers == (Provider.CODEX,)
    assert cfg.daily_runs_per_provider == 3
    assert cfg.store_path == tmp_path / "x" / "crew.db"
    assert cfg.model_for(Provider.CLAUDE) == "opus"
    with pytest.raises(ConfigError):
        CrewConfig.from_env({"LAT_AGENTS_PROVIDERS": "gemini"}, project_root=tmp_path)
    with pytest.raises(ConfigError):
        CrewConfig.from_env({"LAT_AGENTS_MAX_CONCURRENT": "lots"}, project_root=tmp_path)


def test_backoff_doubles_and_caps(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    assert [cfg.backoff(n) for n in (1, 2, 3, 4)] == [timedelta(minutes=m) for m in (60, 120, 240, 480)]
    assert cfg.backoff(5) == COOLDOWN_CAP == timedelta(hours=12)
    assert cfg.backoff(50) == COOLDOWN_CAP
