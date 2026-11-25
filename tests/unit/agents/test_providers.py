from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.agents.contracts import RESEARCH_TOOL_NAMES, Invocation, Provider, RunOutcome
from services.agents.providers import ClaudeCodeProvider, CodexProvider, research_mcp_spec

FIXTURE = Path(__file__).parent / "fixtures" / "fake_provider.py"
SCHEMA = {
    "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False,
}


@pytest.fixture(params=[ClaudeCodeProvider, CodexProvider], ids=["claude", "codex"])
def provider(request: pytest.FixtureRequest) -> ClaudeCodeProvider | CodexProvider:
    return request.param(str(FIXTURE))


@pytest.fixture
def invocation(provider: ClaudeCodeProvider | CodexProvider, tmp_path: Path) -> Invocation:
    return Invocation(provider.name, "ok", SCHEMA, research_mcp_spec(tmp_path, "fixture-agent"), workdir=tmp_path, timeout_s=5)


def test_success_and_usage(provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation) -> None:
    assert provider.availability() == (True, "")
    result = provider.run(invocation)
    assert result.outcome is RunOutcome.OK
    assert result.output == {"ok": True}
    assert result.usage["input_tokens"] == 13 and result.usage["output_tokens"] == 5
    if provider.name is Provider.CLAUDE:
        assert result.session_id == "fixture-session-id"
        assert result.usage["duration_s"] == 1.25 and result.usage["cost_usd"] == 0.01
    else:
        assert result.session_id == "fixture-thread-id"
        assert result.usage["reasoning_output_tokens"] == 0


@pytest.mark.parametrize(("prompt", "expected"), [
    ("invalid", RunOutcome.INVALID_OUTPUT),
    ("missing-output", RunOutcome.INVALID_OUTPUT),
    ("limit-no-time", RunOutcome.USAGE_LIMIT),
    ("auth", RunOutcome.AUTH_REQUIRED),
    ("error", RunOutcome.ERROR),
])
def test_failure_outcomes(
    provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation, prompt: str, expected: RunOutcome,
) -> None:
    result = provider.run(replace(invocation, prompt=prompt))
    assert result.outcome is expected
    assert result.output is None and result.retry_at is None
    assert 0 < len(result.detail) <= 2000 and "fake-test-value" not in result.detail


def test_usage_limit_with_reset(provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation) -> None:
    before = datetime.now(UTC)
    result = provider.run(replace(invocation, prompt="limit-time"))
    assert result.outcome is RunOutcome.USAGE_LIMIT
    assert result.retry_at is not None
    if provider.name is Provider.CLAUDE:
        offset = timedelta(hours=1, minutes=30)
        assert before + offset <= result.retry_at <= datetime.now(UTC) + offset
    else:
        assert result.retry_at == datetime(2099, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_timeout_kills_process_group(
    provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation, tmp_path: Path,
) -> None:
    result = provider.run(replace(invocation, prompt="timeout", timeout_s=1))
    assert result.outcome is RunOutcome.TIMEOUT
    pid = (tmp_path / "child.pid").read_text()
    process = subprocess.run(["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True, check=False, timeout=5)
    assert process.returncode != 0 or process.stdout.strip().startswith("Z")


def test_missing_binary(provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation, tmp_path: Path) -> None:
    provider.binary = str(tmp_path / "missing")
    assert provider.availability()[0] is False
    assert provider.run(invocation).outcome is RunOutcome.UNAVAILABLE


def test_argv_mcp_and_scrubbed_env(
    provider: ClaudeCodeProvider | CodexProvider, invocation: Invocation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CLAUDE_CODE_USE_BEDROCK",
                 "CLAUDE_CODE_USE_VERTEX", "LAT_OPERATOR_TOKEN", "LAT_SECRET"}
    for name in forbidden:
        monkeypatch.setenv(name, "fake-test-value")
    monkeypatch.setenv("LAT_USE_KEYCHAIN", "false")
    spec = research_mcp_spec(tmp_path, "fixture-agent")
    assert spec.args == ("run", "--frozen", "--directory", str(tmp_path), "python", "-m", "services.mcp_server",
                         "--profile", "research")
    result = provider.run(replace(invocation, mcp_server=replace(spec, env={**spec.env, "LAT_SECRET": "fake-test-value"}),
                                  allowed_tools=("get_research_results",), model="test-model"))
    assert result.outcome is RunOutcome.OK
    inspection = json.loads((tmp_path / "inspection.json").read_text())
    argv = inspection["argv"]
    assert forbidden.isdisjoint(inspection["env_keys"])
    assert str(tmp_path.resolve()) == Path(inspection["cwd"]).resolve().as_posix()
    assert argv[argv.index("--model") + 1] == "test-model"
    assert all("dangerously" not in arg for arg in argv)
    if provider.name is Provider.CLAUDE:
        assert argv[0] == "-p"
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--max-turns") + 1] == str(invocation.max_turns)
        assert argv[argv.index("--allowedTools") + 1] == f"mcp__{spec.name}__get_research_results"
        denied = set(argv[argv.index("--disallowedTools") + 1].split(","))
        assert {"Bash", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task"} <= denied
        assert {f"mcp__{spec.name}__{tool}" for tool in RESEARCH_TOOL_NAMES if tool != "get_research_results"} <= denied
        assert inspection["mcp"] == {
            "mcpServers": {spec.name: {"type": "stdio", "command": "uv", "args": list(spec.args), "env": dict(spec.env)}},
        }
        assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    else:
        assert argv[0] == "exec" and "--json" in argv and "--skip-git-repo-check" in argv and argv[-1] == "-"
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert argv[argv.index("--cd") + 1] == str(tmp_path.resolve())
        assert {argv[i + 1] for i, arg in enumerate(argv) if arg == "--disable"} >= {"shell_tool", "unified_exec"}
        cfg = tomllib.loads("\n".join(argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"))
        assert cfg["approval_policy"] == "never" and cfg["forced_login_method"] == "chatgpt"
        assert cfg["mcp_servers"] == {spec.name: {
            "command": "uv", "args": list(spec.args), "env": dict(spec.env), "cwd": str(tmp_path.resolve()),
            "enabled_tools": ["get_research_results"],
            "default_tools_approval_mode": "prompt",
            "tools": {"get_research_results": {"approval_mode": "approve"}},
        }}
        assert inspection["schema"] == SCHEMA


def test_research_spec_does_not_add_keychain_setting_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LAT_USE_KEYCHAIN", raising=False)
    assert research_mcp_spec(tmp_path, "test").env == {"LAT_AGENT_NAME": "test"}


# --- review regressions: limit/auth detection must read error channels only, never the agent's own answer ----
def _script_cli(tmp_path: Path, stdout_lines: list[str], exit_code: int = 0) -> str:
    script = tmp_path / "scripted-cli"
    body = "\n".join(f"print({line!r})" for line in stdout_lines)
    script.write_text(f"#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\n{body}\nsys.exit({exit_code})\n")
    script.chmod(0o755)
    return str(script)


_PROSE = ("Short-term reversal resets after earnings; the rate limit of turnover matters. "
          "Try again in 5 days with a weekly rebalance. A 401 error would mean the data is not signed in.")


@pytest.mark.parametrize("claude_result", [
    {"type": "result", "subtype": "success", "is_error": False, "result": _PROSE},
    {"type": "result", "subtype": "error_max_structured_output_retries", "is_error": True, "result": _PROSE},
], ids=["prose-answer", "structured-output-retries-exhausted"])
def test_claude_agent_prose_is_not_a_usage_limit_or_auth_failure(tmp_path: Path, claude_result: dict) -> None:
    binary = _script_cli(tmp_path, [json.dumps(claude_result)], exit_code=0 if not claude_result["is_error"] else 1)
    inv = Invocation(Provider.CLAUDE, "x", SCHEMA, research_mcp_spec(tmp_path, "t"), workdir=tmp_path, timeout_s=10)
    result = ClaudeCodeProvider(binary).run(inv)
    assert result.outcome is RunOutcome.INVALID_OUTPUT
    assert result.retry_at is None


def test_claude_real_limit_in_error_result_still_detected(tmp_path: Path) -> None:
    out = {"type": "result", "subtype": "success", "is_error": True,
           "result": "You've hit your limit · resets in 2h"}
    binary = _script_cli(tmp_path, [json.dumps(out)], exit_code=1)
    inv = Invocation(Provider.CLAUDE, "x", SCHEMA, research_mcp_spec(tmp_path, "t"), workdir=tmp_path, timeout_s=10)
    result = ClaudeCodeProvider(binary).run(inv)
    assert result.outcome is RunOutcome.USAGE_LIMIT and result.retry_at is not None


def test_codex_transient_connection_reset_is_not_a_usage_limit(tmp_path: Path) -> None:
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected: connection reset by peer)"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "not json"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    binary = _script_cli(tmp_path, [json.dumps(e) for e in events])
    inv = Invocation(Provider.CODEX, "x", SCHEMA, research_mcp_spec(tmp_path, "t"), workdir=tmp_path, timeout_s=10)
    assert CodexProvider(binary).run(inv).outcome is RunOutcome.INVALID_OUTPUT


@pytest.mark.parametrize("message", [
    "You've hit your weekly limit \u00b7 resets Nov 28, 3pm (America/New_York)",
    "5-hour limit reached \u2219 resets 3pm (Europe/Berlin)",
    "Claude usage limit reached. Your limit will reset at 7pm.",
    "You've hit your session limit. Try again in 3 hours.",
    "Rate limit exceeded (429)",
])
def test_claude_limit_phrasings_detected(tmp_path: Path, message: str) -> None:
    out = {"type": "result", "subtype": "success", "is_error": True, "result": message}
    binary = _script_cli(tmp_path, [json.dumps(out)], exit_code=1)
    inv = Invocation(Provider.CLAUDE, "x", SCHEMA, research_mcp_spec(tmp_path, "t"), workdir=tmp_path, timeout_s=10)
    assert ClaudeCodeProvider(binary).run(inv).outcome is RunOutcome.USAGE_LIMIT


def test_retry_at_named_date_without_year_rolls_into_next_year() -> None:
    from services.agents.providers import _retry_at

    now = datetime(2026, 12, 29, 18, 0, tzinfo=UTC)
    assert _retry_at("weekly limit reached · resets Jan 2, 3am (America/New_York)", now) == \
        datetime(2027, 1, 2, 8, 0, tzinfo=UTC)
    assert _retry_at("resets Dec 31, 11pm (America/New_York)", now) == datetime(2027, 1, 1, 4, 0, tzinfo=UTC)
    # an explicit year in the past stays in the past (no retry_at; the crew falls back to its backoff)
    assert _retry_at("resets Jan 2, 2026, 3am (America/New_York)", now) is None
