"""Static checks of the official registrations (.mcp.json, .codex/config.toml) and scripts/register_mcp.sh."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from services.tools.registry import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, TOOL_NAMES, TOOL_REGISTRY

ROOT = Path(__file__).resolve().parents[3]
SERVER_ARGS = ["run", "--frozen", "--directory", str(ROOT), "python", "-m", "services.mcp_server"]


def test_claude_code_project_registration() -> None:
    cfg = json.loads((ROOT / ".mcp.json").read_text())
    assert set(cfg) == {"mcpServers"}
    server = cfg["mcpServers"]["local-ai-trader"]
    assert server["type"] == "stdio" and server["command"] == "uv"
    assert server["args"] == SERVER_ARGS
    assert set(server.get("env", {})) <= {"LAT_AGENT_NAME"}  # no secrets, no mode/gateway overrides
    blob = json.dumps(cfg).lower()
    for forbidden in ("token", "key", "secret", "password", "lat_mode", "lat_gateway", "live"):
        assert forbidden not in blob


def test_codex_project_registration_and_approval_modes() -> None:
    cfg = tomllib.loads((ROOT / ".codex" / "config.toml").read_text())
    server = cfg["mcp_servers"]["local-ai-trader"]
    assert server["command"] == "uv" and server["args"] == SERVER_ARGS and server["cwd"] == str(ROOT)
    assert server["startup_timeout_sec"] > 0 and server["tool_timeout_sec"] >= 60
    assert server["default_tools_approval_mode"] == "prompt"
    assert server["env"] == {"LAT_AGENT_NAME": "codex"}
    tools = server["tools"]
    # New memory reader inherits the existing prompt default; adding a reader must not silently
    # broaden the user's persisted security permissions.
    assert set(tools) == set(TOOL_NAMES) - {"search_memory"}
    for name in set(READ_ONLY_TOOL_NAMES) - {"search_memory"}:
        assert tools[name] == {"approval_mode": "approve"}, name
    for name in MUTATING_TOOL_NAMES:
        assert tools[name] == {"approval_mode": "prompt"}, name
    documented = {"auto", "prompt", "writes", "approve"}  # Codex config reference, approval mode values
    assert {t["approval_mode"] for t in tools.values()} | {server["default_tools_approval_mode"]} <= documented
    assert server["startup_timeout_sec"] >= 10  # uv start-up is slower than Codex's 10 s default


def test_tool_input_schemas_fit_codex_default_schema_budget() -> None:
    # Codex's mcp_servers.<id>.tool_input_schema_max_bytes defaults to 5,000 bytes per tool input schema.
    for name in TOOL_NAMES:
        size = len(json.dumps(TOOL_REGISTRY[name].input_schema(), separators=(",", ":")))
        assert size < 5000, (name, size)


def _run_script(*args: str, path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/bash", str(ROOT / "scripts" / "register_mcp.sh"), *args], capture_output=True,
                          text=True, timeout=60, env={"PATH": path, "HOME": os.environ.get("HOME", "/tmp")}, check=False)


def test_register_script_without_any_cli_prints_instructions_and_installs_nothing(tmp_path: Path) -> None:
    proc = _run_script(path="/usr/bin:/bin")  # neither claude nor codex on PATH
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "claude mcp add --scope project" in out and "codex mcp add local-ai-trader" in out
    assert "NOT installed" in out and "does not install anything" in out and "Sign in with ChatGPT" in out
    assert "services.mcp_server" in out


def test_register_script_dry_run_never_calls_vendor_add(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    for cli in ("claude", "codex"):
        stub = bindir / cli
        stub.write_text(f'#!/bin/bash\necho "{cli} $*" >> "{log}"\nexit 1\n')
        stub.chmod(0o755)
    proc = _run_script(path=f"{bindir}:/usr/bin:/bin")
    assert proc.returncode == 0, proc.stderr
    assert "already registered in .mcp.json" in proc.stdout
    assert not log.exists() or "mcp add" not in log.read_text()  # dry run: no registration executed


def test_register_script_apply_uses_official_commands_only(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    (bindir / "codex").write_text(f'#!/bin/bash\necho "codex $*" >> "{log}"\n[ "$2" = get ] && exit 1\nexit 0\n')
    (bindir / "codex").chmod(0o755)
    proc = _run_script("--apply", path=f"{bindir}:/usr/bin:/bin")
    assert proc.returncode == 0, proc.stderr
    calls = log.read_text().splitlines()
    assert any(c.startswith("codex mcp add local-ai-trader --env LAT_AGENT_NAME=codex -- uv run --frozen --directory")
               for c in calls), calls
    assert all(c.split()[1] == "mcp" for c in calls)  # nothing but `codex mcp ...`


def test_register_script_rejects_unknown_args() -> None:
    proc = _run_script("--force", path="/usr/bin:/bin")
    assert proc.returncode == 2


def test_script_is_bash_syntax_clean() -> None:
    bash = shutil.which("bash") or "/bin/bash"
    subprocess.run([bash, "-n", str(ROOT / "scripts" / "register_mcp.sh")], check=True, timeout=30)
