"""Authority limits, enforced in code: the autopilot never reads the operator token, never reaches policy writes,
gateways, the ledger or a broker, never selects Live, and can call only allowlisted tools and subprocesses."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from packages.contracts.enums import FeedMode
from packages.core.config import Settings
from services.autopilot.backend import (
    ALLOWED_TOOLS,
    FAIL_SAFE_TOOLS,
    MUTATING_TOOLS,
    AuthorityError,
    GuardedBackend,
    make_http_backend,
    require_paper_runtime,
)
from services.autopilot.checks import halt_state
from services.autopilot.research import CREW_ARGV, check_argv, holdout_argv, lead_argv, run_child
from services.tools.handlers import ToolError
from services.tools.registry import TOOL_REGISTRY

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "services" / "autopilot"
FORBIDDEN_MODULES = (
    "services.risk.policy_store", "services.risk.operator_auth", "services.risk.engine", "services.api.security",
    "services.api.app", "services.api.runtime", "services.order_ledger", "services.positions", "services.agents",
    "services.mcp_server", "alpaca", "services.capabilities.policy",
)
FORBIDDEN_TOOL_NAMES = sorted(n for n, s in TOOL_REGISTRY.items() if s.mutating and n not in MUTATING_TOOLS)
FORBIDDEN_TEXT = ("operator.token", "operator_token", "x-lat-operator-token", "OPERATOR_TOKEN", "/control/",
                  "LAT_TRADING_MODE", "LAT_MODE", "LAT_GATEWAY", "policy_store", "apply_change")


def _sources() -> list[Path]:
    files = sorted(PKG.rglob("*.py"))
    assert len(files) >= 9
    return files


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in ("__import__", "import_module"):
                out.add(str(node.args[0].value))
    return out


ALLOWED_GATEWAY_NAMES = {"packages.contracts.enums.GatewayKind"}  # the enum used to refuse non-paper gateways


def test_package_source_never_imports_privileged_modules() -> None:
    offenders = [f"{p.name}: {m}" for p in _sources() for m in _imports(p)
                 if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)
                 or ("gateway" in m.lower() and m not in ALLOWED_GATEWAY_NAMES)]
    assert offenders == []


def test_package_source_never_names_operator_token_policy_writes_mode_or_forbidden_tools() -> None:
    hits = []
    for p in _sources():
        text = p.read_text()
        hits += [f"{p.name}: {t}" for t in FORBIDDEN_TEXT if t.lower() in text.lower()]
        hits += [f"{p.name}: tool {t}" for t in FORBIDDEN_TOOL_NAMES if f'"{t}"' in text or f"'{t}'" in text]
    assert hits == []


_PROBE = """
import json, sys
import services.autopilot, services.autopilot.cli, services.autopilot.supervisor, services.autopilot.backend
import services.autopilot.research, services.autopilot.report, services.autopilot.checks, services.autopilot.store
import services.tools.backends, services.api.launcher, services.quantlab.daily  # what it imports lazily at run time
print(json.dumps(sorted(sys.modules)))
"""


def test_importing_the_autopilot_and_its_lazy_paths_loads_no_privileged_module() -> None:
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, capture_output=True, text=True, timeout=120,
                          check=False, env={"PATH": "/usr/bin:/bin", "LAT_USE_KEYCHAIN": "false",
                                            "LAT_FEED_MODE": "SIMULATED", "PYTHONPATH": str(REPO)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    mods = json.loads(proc.stdout.strip().splitlines()[-1])
    loaded = [m for m in mods if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]
    assert loaded == []
    assert not [m for m in mods if "gateway" in m.lower() and m.startswith(("services", "packages"))]


def test_guarded_backend_allows_only_reads_the_session_start_and_fail_safe_tools() -> None:
    assert set(TOOL_REGISTRY) >= ALLOWED_TOOLS
    # start_paper_session (deterministic pre-open only) plus two tools that can only reduce trading, used for
    # validated lead-agent requests. Nothing that proposes, cancels, schedules, resumes or changes policy.
    assert {n for n in ALLOWED_TOOLS if TOOL_REGISTRY[n].mutating} == {"start_paper_session", "stop_session",
                                                                       "request_halt"}
    assert {"stop_session", "request_halt"} == FAIL_SAFE_TOOLS
    for name in ("propose_paper_trade", "cancel_paper_order", "schedule_session", "cancel_schedule",
                 "run_quant_cycle", "run_research_experiment"):
        assert name in TOOL_REGISTRY and name not in ALLOWED_TOOLS

    class Inner:
        calls: list[str] = []

        async def call(self, name: str, args: Any) -> dict[str, Any]:
            self.calls.append(name)
            return {}

    inner = Inner()
    guarded = GuardedBackend(inner)
    for name in TOOL_REGISTRY:
        if name in ALLOWED_TOOLS:
            continue
        with pytest.raises(AuthorityError):
            asyncio.run(guarded.call(name, {}))
    with pytest.raises(AuthorityError):
        asyncio.run(guarded.call("start_paper_session", {"duration_minutes": 60, "mode": "LIVE"}))
    with pytest.raises(AuthorityError):
        asyncio.run(guarded.call("resume_trading", {}))
    with pytest.raises(AuthorityError):
        asyncio.run(guarded.call("request_halt", {"reason": "x", "scope": "ALL"}))
    with pytest.raises(AuthorityError):
        asyncio.run(guarded.call("stop_session", {"reason": "x", "flatten": True}))
    assert inner.calls == []
    asyncio.run(guarded.call("request_halt", {"reason": "x"}))
    asyncio.run(guarded.call("stop_session", {"reason": "x", "session_id": "sess_1"}))
    assert inner.calls == ["request_halt", "stop_session"]


def test_subprocess_allowlist() -> None:
    assert check_argv(CREW_ARGV) == CREW_ARGV
    briefing = Path("/tmp/ws/experiments/autopilot/lead/briefing.json")
    assert lead_argv(2700, briefing)[:4] == ("-m", "services.agents", "lead", "--json")
    assert lead_argv(2700, briefing, fake=True)[-1] == "--fake"
    for bad_lead in (("--budget-s", "2700"), ("--budget-s", "10", "--briefing", str(briefing)),
                     ("--budget-s", "2700", "--briefing", "relative/lead/briefing.json"),
                     ("--budget-s", "2700", "--briefing", "/tmp/ws/../etc/lead/briefing.json"),
                     ("--budget-s", "2700", "--briefing", "/tmp/ws/agent.token"),
                     ("--budget-s", "2700", "--briefing", str(briefing), "--store", "/x"),
                     ("--budget-s", "-1", "--briefing", str(briefing))):
        with pytest.raises(AuthorityError):
            check_argv(("-m", "services.agents", "lead", "--json", *bad_lead))
    assert holdout_argv("J_volman")[-1] == "J_volman"
    for bad in (("-m", "services.api"), ("-m", "services.api.cli", "resume", "h1"), ("-m", "services.agents", "run"),
                ("-m", "services.quantlab.cli", "research", "--force"), ("-c", "print(1)"),
                ("-m", "services.quantlab.cli", "holdout", "--family", "A", "--again"),
                ("-m", "services.quantlab.cli", "holdout", "--family", "../x"),
                ("-m", "services.quantlab.cli", "build-panel", "--what", "session"),
                ("-m", "services.quantlab.cli", "fetch-delta", "--end", "2000-01-01")):
        with pytest.raises(AuthorityError):
            check_argv(bad)
    for fam in ("", "A;rm", "../x", "A B"):
        with pytest.raises(AuthorityError):
            holdout_argv(fam)


def _settings(tmp: Path, **kw: Any) -> Settings:
    return Settings(project_root=tmp, use_keychain=False, feed_mode=FeedMode.SIMULATED, **kw)


def test_never_runs_with_live_or_outside_paper(tmp_path: Path) -> None:
    with pytest.raises(AuthorityError, match="Live"):
        require_paper_runtime(_settings(tmp_path, trading_mode="LIVE"))
    with pytest.raises(AuthorityError):
        require_paper_runtime(_settings(tmp_path, mode="REPLAY"))
    with pytest.raises(AuthorityError):
        require_paper_runtime(_settings(tmp_path, mode="RESEARCH_ONLY"))
    require_paper_runtime(_settings(tmp_path, mode="PAPER", gateway="FAKE"))
    with pytest.raises(AuthorityError):
        make_http_backend(_settings(tmp_path, trading_mode="LIVE"))


def test_http_backend_sends_only_the_agent_token_and_never_reads_the_operator_token(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, mode="PAPER", gateway="FAKE")
    keys = settings.keys_dir
    keys.mkdir(parents=True)
    for name, value in (("agent.token", "agent-test-value"), ("operator.token", "operator-test-value")):
        (keys / name).write_text(value)
        os.chmod(keys / name, 0o600)
    read: list[str] = []
    real_read_text = Path.read_text

    def spy(self: Path, *a: Any, **k: Any) -> str:
        read.append(self.name)
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"execution_label": "SIMULATED", "is_simulated": True,
                                         "asof": "2026-11-25T13:45:00Z", "overall": "OK", "components": []})

    backend = make_http_backend(settings)
    backend._inner._transport = httpx.MockTransport(handler)  # type: ignore[attr-defined]
    out = asyncio.run(backend.call("get_system_health", {}))
    assert out["overall"] == "OK"
    (req,) = seen
    assert req.url.host == "127.0.0.1"
    assert req.headers["x-lat-agent-token"] == "agent-test-value"
    assert req.headers["x-lat-agent-name"] == "autopilot"
    assert "x-lat-operator-token" not in req.headers
    assert "operator-test-value" not in json.dumps(dict(req.headers))
    assert "operator.token" not in read and "agent.token" in read


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_agent_backend_refuses_linked_token(tmp_path: Path, link: str) -> None:
    settings = _settings(tmp_path)
    settings.keys_dir.mkdir(parents=True)
    secret = settings.keys_dir / "operator.token"
    secret.touch(mode=0o600)
    agent = settings.keys_dir / "agent.token"
    (agent.symlink_to if link == "symlink" else agent.hardlink_to)(secret)
    backend = make_http_backend(settings)
    with pytest.raises(AuthorityError, match="agent token"):
        asyncio.run(backend.call("get_system_health"))


def test_guarded_session_request_never_implicitly_launches_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from services.api import launcher

    settings = _settings(tmp_path)
    settings.keys_dir.mkdir(parents=True)
    agent = settings.keys_dir / "agent.token"
    agent.write_text("agent-fixture")
    agent.chmod(0o600)
    calls: list[httpx.Request] = []

    def forbidden_start(*args: object, **kwargs: object) -> None:
        pytest.fail("the HTTP backend tried to launch the API")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(409, json={"detail": "session already exists"})

    monkeypatch.setattr(launcher, "ensure_api", forbidden_start)
    backend = make_http_backend(settings)
    backend._inner._transport = httpx.MockTransport(handler)  # type: ignore[attr-defined]
    with pytest.raises(ToolError) as refused:
        asyncio.run(backend.call("start_paper_session", {"duration_minutes": 90}))
    assert refused.value.status == 409 and len(calls) == 1


def test_research_timeout_kills_child_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = tmp_path / "grandchild.started"
    finished = tmp_path / "grandchild.finished"
    code = (f"from pathlib import Path; import time; Path({str(started)!r}).touch(); "
            f"time.sleep(2); Path({str(finished)!r}).touch()")
    tree = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {code!r}], check=True)"
    spawn = asyncio.create_subprocess_exec

    async def launch(*_command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        return await spawn(sys.executable, "-c", tree, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)

    async def exercise() -> None:
        outcome = await asyncio.wait_for(run_child(("-m", "services.quantlab.cli", "scan"), tmp_path, 1), timeout=5)
        assert outcome.returncode is None
        assert started.exists()
        await asyncio.sleep(1.3)
        assert not finished.exists()

    asyncio.run(exercise())


def test_research_uses_isolated_python_and_ignores_crew_executable_overrides(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spawn = asyncio.create_subprocess_exec

    async def launch(*command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        assert command == ("/usr/bin/nice", "-n", "10", sys.executable, "-I", *CREW_ARGV)
        assert "LAT_AGENTS_CLAUDE_BIN" not in kwargs["env"]
        assert "LAT_AGENTS_CODEX_BIN" not in kwargs["env"]
        return await spawn(sys.executable, "-c", "pass", **kwargs)

    monkeypatch.setenv("LAT_AGENTS_CLAUDE_BIN", "unapproved")
    monkeypatch.setenv("LAT_AGENTS_CODEX_BIN", "unapproved")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    assert asyncio.run(run_child(CREW_ARGV, tmp_path, 5)).ok


def test_kill_switch_marker_inside_an_unrelated_lock_reason_is_data() -> None:
    assert halt_state({"exit_lock_reasons": ["flag: comment includes scope=ALL"]}) == (False, False, [])
