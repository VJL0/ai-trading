"""Headless subscription CLI adapters for the research crew."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Event
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jsonschema import ValidationError, validate

from services.agents.contracts import (
    RESEARCH_MCP_SERVER_NAME,
    RESEARCH_TOOL_NAMES,
    Invocation,
    McpServerSpec,
    Provider,
    RunOutcome,
    RunResult,
)
from services.tools.sanitize import truncate

_CLI_ENV = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "TERM", "SHELL", "CODEX_HOME", "CLAUDE_CONFIG_DIR")
_MCP_ENV = ("LAT_AGENT_NAME", "LAT_USE_KEYCHAIN")
_MCP_NAME = re.compile(r"[A-Za-z0-9_-]+")
_LIMIT = re.compile(
    # No bare "reset": transport errors ("connection reset by peer") are not usage limits.
    r"usage[\s_-]+limit|rate[\s_-]+limit|limit reached|hit your (?:[\w-]+ ){0,3}limit"
    r"|\b(?:\d+[- ]hour|weekly|daily|monthly|session) limit|limit will reset"
    r"|try again (?:at|in)",
    re.IGNORECASE,
)
_AUTH = re.compile(r"not (?:logged|signed) in|please (?:log|sign) in|/login\b|\b401\b|authentication required",
                   re.IGNORECASE)
_RESET_MARKER = re.compile(r"(?:try again|retry)\s+(at|in)\s+|\bresets?\b(?:\s+(at|in))?\s+", re.IGNORECASE)
_DURATION = re.compile(r"\s*(\d{1,5})\s*(days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", re.I)
_CLOCK = re.compile(r"(?<!\d)(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
_ZONE = re.compile(r"\(((?:[A-Za-z_]+/)+[A-Za-z0-9_+\-]+|[A-Z]{2,5})\)|\s+(UTC|GMT)\b")
_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})", re.I)
_MONTH = re.compile(r"^([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?[, ]+(.+)", re.I)
_CLAUDE_DENIED = ("Bash", "PowerShell", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Task", "Agent")


def research_mcp_spec(project_root: Path, agent_name: str) -> McpServerSpec:
    env = {"LAT_AGENT_NAME": agent_name}
    if "LAT_USE_KEYCHAIN" in os.environ:
        env["LAT_USE_KEYCHAIN"] = os.environ["LAT_USE_KEYCHAIN"]
    return McpServerSpec(
        RESEARCH_MCP_SERVER_NAME, "uv",
        ("run", "--frozen", "--directory", str(project_root.resolve()), "python", "-m", "services.mcp_server",
         "--profile", "research"),
        env,
    )


def _mcp_env(spec: McpServerSpec) -> dict[str, str]:
    return {name: spec.env[name] for name in _MCP_ENV if name in spec.env}


def _child_env() -> dict[str, str]:
    return {name: os.environ[name] for name in _CLI_ENV if name in os.environ}


def _execute(
    argv: list[str], *, prompt: str = "", cwd: Path | None = None, timeout: int, stop: Event | None = None,
) -> subprocess.CompletedProcess[str]:
    if stop is not None and stop.is_set():
        raise _InvocationCancelled
    # A file descriptor preserves stdin across repeated communicate timeouts, even when
    # the CLI delays reading a prompt larger than a pipe buffer.
    with TemporaryFile(mode="w+t", encoding="utf-8") as stdin:
        stdin.write(prompt)
        stdin.seek(0)
        with subprocess.Popen(
            argv, cwd=cwd, env=_child_env(), stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", start_new_session=True,
        ) as process:
            try:
                if stop is None:
                    stdout, stderr = process.communicate(timeout=timeout)
                else:
                    deadline = time.monotonic() + timeout
                    while True:
                        if stop.is_set():
                            raise _InvocationCancelled
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(argv, timeout)
                        try:
                            stdout, stderr = process.communicate(timeout=min(remaining, 0.2))
                            break
                        except subprocess.TimeoutExpired:
                            continue
            except (subprocess.TimeoutExpired, _InvocationCancelled):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


class _InvocationCancelled(Exception):
    pass


def _duration(text: str) -> timedelta:
    seconds = 0
    factors = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    while match := _DURATION.match(text):
        seconds += int(match[1]) * factors[match[2][0].lower()]
        text = text[match.end():].lstrip(" ,")
        text = re.sub(r"^and\s+", "", text, flags=re.I)
    return timedelta(seconds=seconds)


def _clock_reset(text: str, now: datetime) -> datetime | None:
    zone_match = _ZONE.search(text)
    zone = ZoneInfo(zone_match[1] or zone_match[2]) if zone_match else None
    local_now = now.astimezone(zone)
    date = local_now.date()
    named_date = _MONTH.match(text)
    if named_date:
        month, day, year, text = named_date.groups()
        date = datetime.strptime(f"{month[:3]} {day} {year or local_now.year}", "%b %d %Y").replace(tzinfo=UTC).date()
    time_match = _CLOCK.match(text)
    if time_match is None:
        return None
    hour, minute, period = time_match.groups()
    if period:
        if not 1 <= int(hour) <= 12:
            return None
        hour = str(int(hour) % 12 + (12 if period.lower() == "pm" else 0))
    elif minute is None:
        return None
    reset = datetime.combine(date, datetime.min.time()).replace(hour=int(hour), minute=int(minute or 0))
    reset = reset.replace(tzinfo=zone) if zone else reset.astimezone()
    if not named_date and reset < local_now:
        reset += timedelta(days=1)
    elif named_date and not named_date[3] and reset < local_now:
        reset = reset.replace(year=reset.year + 1)  # "resets Jan 2" seen in late December means next year
    return reset.astimezone(UTC)


def _retry_at(text: str, now: datetime) -> datetime | None:
    for marker in _RESET_MARKER.finditer(text):
        tail = text[marker.end():].lstrip(" :")[:100]
        try:
            if iso := _ISO.match(tail):
                reset = datetime.fromisoformat(iso[0].upper()).astimezone(UTC)
            else:
                direction = (marker[1] or marker[2] or "").lower()
                elapsed = _duration(tail) if direction != "at" else timedelta()
                if direction == "in" and not elapsed:
                    continue
                reset = now + elapsed if elapsed else _clock_reset(tail, now)
        except (ValueError, OverflowError, ZoneInfoNotFoundError):
            continue
        if reset and reset > now:
            return reset
    return None


def _object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _answer(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```json") and value.endswith("```"):
            value = value[7:-3].strip()
        return _object(value)
    return None


def _usage(raw: Any) -> dict[str, Any]:
    keys = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens", "cache_write_input_tokens", "reasoning_output_tokens")
    return {key: raw[key] for key in keys if type(raw.get(key)) is int} if isinstance(raw, dict) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


@dataclass
class _Parsed:
    output: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    error: str = ""
    failed: bool = False
    invalid_schema_output: bool = False


def _result(outcome: RunOutcome, detail: str = "", parsed: _Parsed | None = None,
            retry_at: datetime | None = None) -> RunResult:
    parsed = parsed or _Parsed()
    # Diagnostics are generated locally. Provider text and validation values can contain secrets.
    return RunResult(outcome, output=parsed.output if outcome is RunOutcome.OK else None, usage=parsed.usage,
                     retry_at=retry_at, detail=truncate(detail, 2000) or "", session_id=parsed.session_id)


class _CliProvider:
    name: Provider
    _auth_args: tuple[str, ...]

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def availability(self) -> tuple[bool, str]:
        try:
            binary = shutil.which(self.binary)
            if binary is None:
                return False, "CLI executable was not found"
            proc = _execute([binary, "--version"], timeout=10)
            return (True, "") if proc.returncode == 0 else (False, f"CLI version check exited with status {proc.returncode}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return False, f"CLI version check failed ({type(exc).__name__})"

    def run(self, invocation: Invocation) -> RunResult:
        return self._run(invocation, None)

    def run_cancellable(self, invocation: Invocation, stop: Event) -> RunResult:
        return self._run(invocation, stop)

    def auth_available(self) -> tuple[bool, str]:
        """Ask the CLI for login status without issuing a model request or returning account data."""
        try:
            proc = _execute([self.binary, *self._auth_args], timeout=10)
            if proc.returncode == 0 and self._subscription_signed_in(proc):
                return True, ""
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        return False, "CLI requires subscription sign-in"

    def _subscription_signed_in(self, proc: subprocess.CompletedProcess[str]) -> bool:
        raise NotImplementedError

    def _run(self, invocation: Invocation, stop: Event | None) -> RunResult:
        try:
            binary = shutil.which(self.binary)
            if binary is None:
                return _result(RunOutcome.UNAVAILABLE, "CLI executable was not found")
            names = (invocation.mcp_server.name, *invocation.allowed_tools)
            if any(not _MCP_NAME.fullmatch(name) for name in names):
                return _result(RunOutcome.ERROR, "Invalid MCP server or tool name")
            with TemporaryDirectory(dir=invocation.workdir, prefix="lat-agent-") as temp:
                files = Path(temp).resolve()
                cwd = invocation.workdir.resolve() if invocation.workdir else files
                argv = [str(Path(binary).absolute()), *self._args(invocation, files, cwd)]
                try:
                    proc = _execute(argv, prompt=invocation.prompt, cwd=cwd, timeout=invocation.timeout_s, stop=stop)
                except (FileNotFoundError, PermissionError) as exc:
                    if exc.filename == argv[0]:
                        return _result(RunOutcome.UNAVAILABLE, "CLI executable could not be launched")
                    raise
            parsed = self._parse(proc.stdout)
            if proc.returncode or parsed.failed or parsed.output is None:
                diagnostics = "\n".join((parsed.error, proc.stderr))
                if _LIMIT.search(diagnostics):
                    return _result(RunOutcome.USAGE_LIMIT, "CLI reported a usage limit", parsed,
                                   _retry_at(diagnostics, datetime.now(UTC)))
                if _AUTH.search(diagnostics):
                    return _result(RunOutcome.AUTH_REQUIRED, "CLI requires subscription sign-in", parsed)
                if not parsed.invalid_schema_output and (proc.returncode or parsed.failed):
                    return _result(RunOutcome.ERROR, f"CLI run failed (exit status {proc.returncode})", parsed)
            if parsed.output is None:
                return _result(RunOutcome.INVALID_OUTPUT, "CLI returned no structured JSON object", parsed)
            try:
                validate(parsed.output, dict(invocation.output_schema))
            except ValidationError:
                return _result(RunOutcome.INVALID_OUTPUT, "CLI output failed JSON Schema validation", parsed)
            return _result(RunOutcome.OK, parsed=parsed)
        except subprocess.TimeoutExpired:
            return _result(RunOutcome.TIMEOUT, "CLI run exceeded its timeout")
        except _InvocationCancelled:
            return _result(RunOutcome.ERROR, "CLI run interrupted")
        except Exception as exc:
            return _result(RunOutcome.ERROR, f"CLI run failed ({type(exc).__name__})")

    def _args(self, invocation: Invocation, files: Path, cwd: Path) -> list[str]:
        raise NotImplementedError

    def _parse(self, stdout: str) -> _Parsed:
        raise NotImplementedError


class ClaudeCodeProvider(_CliProvider):
    name = Provider.CLAUDE
    _auth_args = ("auth", "status", "--json")

    def __init__(self, binary: str = "claude") -> None:
        super().__init__(binary)

    def _subscription_signed_in(self, proc: subprocess.CompletedProcess[str]) -> bool:
        status = _object(proc.stdout)
        return bool(status and status.get("loggedIn") is True and status.get("authMethod") == "claude.ai")

    def _args(self, invocation: Invocation, files: Path, cwd: Path) -> list[str]:
        spec = invocation.mcp_server
        command, args = spec.command, list(spec.args)
        if spec.cwd is not None:
            # Claude's stdio config has command/args/env. Pass the cwd and command as shell positional arguments.
            command = "/bin/sh"
            args = ["-c", 'cd "$1" && shift && exec "$@"', "lat-mcp", str(spec.cwd.resolve()), spec.command, *args]
        config = files / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {
            spec.name: {"type": "stdio", "command": command, "args": args, "env": _mcp_env(spec)},
        }}), encoding="utf-8")
        allow = [f"mcp__{spec.name}__{tool}" for tool in invocation.allowed_tools]
        deny = [f"mcp__{spec.name}__{tool}" for tool in RESEARCH_TOOL_NAMES if tool not in invocation.allowed_tools]
        argv = [
            "-p", "--output-format", "json", "--json-schema", json.dumps(dict(invocation.output_schema)),
            "--mcp-config", str(config), "--strict-mcp-config", "--restricted", "--tools", "",
            "--disallowedTools", ",".join((*_CLAUDE_DENIED, *deny)), "--disable-slash-commands",
            "--permission-mode", "dontAsk", "--permission-prompts", "none", "--no-session-persistence",
            "--max-turns", str(invocation.max_turns),
        ]
        if allow:
            argv += ["--allowedTools", ",".join(allow)]
        if invocation.model:
            argv += ["--model", invocation.model]
        return argv

    def _parse(self, stdout: str) -> _Parsed:
        result = _object(stdout)
        if result is None:
            return _Parsed(error=stdout)
        usage = _usage(result.get("usage"))
        if type(result.get("total_cost_usd")) in (int, float):
            usage["cost_usd"] = result["total_cost_usd"]
        if type(result.get("duration_ms")) is int:
            usage["duration_s"] = result["duration_ms"] / 1000
        failed = bool(result.get("is_error")) or str(result.get("subtype", "")).startswith("error_")
        invalid = result.get("subtype") == "error_max_structured_output_retries"
        answer = _answer(result.get("structured_output") if "structured_output" in result else result.get("result"))
        # ``result`` is the agent's own answer unless Claude flags an error; exhausted structured-output retries leave
        # the agent's prose there. Only error channels may decide USAGE_LIMIT / AUTH_REQUIRED.
        channels = ("errors", "error", "api_error_status")
        if result.get("is_error") is True and not invalid:
            channels = ("result", *channels)
        errors = " ".join(str(result.get(key, "")) for key in channels)
        return _Parsed(answer, usage, _string(result.get("session_id")), errors, failed, invalid)


class CodexProvider(_CliProvider):
    name = Provider.CODEX
    _auth_args = ("login", "status")

    def __init__(self, binary: str = "codex") -> None:
        super().__init__(binary)

    def _subscription_signed_in(self, proc: subprocess.CompletedProcess[str]) -> bool:
        return "logged in using chatgpt" in (proc.stdout + proc.stderr).lower()

    def _args(self, invocation: Invocation, files: Path, cwd: Path) -> list[str]:
        output_schema = files / "output-schema.json"
        output_schema.write_text(json.dumps(dict(invocation.output_schema)), encoding="utf-8")
        spec = invocation.mcp_server
        mcp = f"mcp_servers.{spec.name}"
        argv = [
            "exec", "--json", "--output-schema", str(output_schema), "--sandbox", "read-only",
            "--skip-git-repo-check", "--cd", str(cwd), "--ephemeral", "--ignore-user-config", "--strict-config",
            "-c", 'approval_policy="never"', "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
            "-c", 'web_search="disabled"',
        ]
        # These 0.172.0 features are listed by `codex features list`. Disable both shell implementations.
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "computer_use"):
            argv += ["--disable", feature]
        # Trust this disposable invocation root in memory so the stdio MCP can start.
        quoted_cwd = json.dumps(str(cwd), ensure_ascii=False)
        argv += ["-c", f'projects={{{quoted_cwd}={{trust_level="trusted"}}}}', "-c", "project_root_markers=[]"]
        # `codex mcp get ... --json` confirms command, args, env, cwd and enabled_tools are parsed.
        for key, value in (
            ("command", spec.command),
            ("args", spec.args),
            ("cwd", str(spec.cwd.resolve() if spec.cwd else cwd)),
            ("enabled_tools", invocation.allowed_tools),
            ("default_tools_approval_mode", "prompt"),
            ("env", _mcp_env(spec)),
        ):
            literal = json.dumps(value, ensure_ascii=False)
            if isinstance(value, dict):
                literal = "{" + ", ".join(f"{key}={json.dumps(val, ensure_ascii=False)}" for key, val in value.items()) + "}"
            argv += ["-c", f"{mcp}.{key}={literal}"]
        for tool in invocation.allowed_tools:
            if tool in RESEARCH_TOOL_NAMES:
                argv += ["-c", f'{mcp}.tools.{tool}.approval_mode="approve"']
        if invocation.model:
            argv += ["--model", invocation.model]
        return [*argv, "-"]

    def _parse(self, stdout: str) -> _Parsed:
        parsed = _Parsed()
        diagnostics: list[str] = []
        for line in stdout.splitlines():
            event = _object(line)
            if event is None:
                diagnostics.append(line)
                continue
            kind = event.get("type")
            if kind == "thread.started":
                parsed.session_id = _string(event.get("thread_id"))
            elif kind == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    parsed.output = _answer(item.get("text"))
            elif kind == "turn.completed":
                parsed.usage = _usage(event.get("usage"))
            elif kind == "turn.failed" or kind == "error":
                parsed.failed |= kind == "turn.failed"
                diagnostics.append(str(event.get("error") or event.get("message") or ""))
        parsed.error = "\n".join(diagnostics)
        return parsed
