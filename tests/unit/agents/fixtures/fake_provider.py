#!/usr/bin/env python3
"""Recognize the adapter's CLI flags and emit controlled stdout/stderr fixtures."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("fixture 1.0")
    sys.exit(0)

is_codex = args[0] == "exec"
prompt = sys.stdin.read()
inspection = {"argv": args, "env_keys": sorted(os.environ), "cwd": os.getcwd()}
if is_codex:
    inspection["schema"] = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
else:
    inspection["mcp"] = json.loads(Path(args[args.index("--mcp-config") + 1]).read_text())
Path("inspection.json").write_text(json.dumps(inspection))

if prompt == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path("child.pid").write_text(str(child.pid))
    time.sleep(60)
    sys.exit(1)

fixture = Path(__file__).parent
if is_codex:
    events = [json.loads(line) for line in (fixture / "codex-success.jsonl").read_text().splitlines()]
    events[0]["thread_id"] = "fixture-thread-id"
    if prompt in ("invalid", "missing-output"):
        events[-2]["item"]["text"] = '{"ok":"yes"}' if prompt == "invalid" else "finished"
    elif prompt == "auth":
        events = [{"type": "turn.failed", "error": {"message": "HTTP 401: please log in"}}]
    elif prompt == "limit-time":
        events = [{"type": "error", "message": "usage_limit_reached. Try again at 2099-01-02T03:04:05Z."}]
    elif prompt == "limit-no-time":
        print("Rate limit reached. Try again later.", file=sys.stderr)
        sys.exit(1)
    elif prompt != "ok":
        print("Unexpected provider error with OPENAI_API_KEY=fake-test-value", file=sys.stderr)
        sys.exit(1)
    for event in events:
        print(json.dumps(event))
else:
    result = json.loads((fixture / "claude-success.json").read_text())
    result["session_id"] = "fixture-session-id"
    if prompt == "invalid":
        result["structured_output"] = {"ok": "yes"}
    elif prompt == "missing-output":
        del result["structured_output"]
        result["result"] = "finished"
    elif prompt == "auth":
        result.update(is_error=True, result="Not signed in. Please run /login")
    elif prompt == "limit-time":
        result.update(is_error=True, result="You've hit your 5-hour limit. Try again IN 1h 30m.")
    elif prompt == "limit-no-time":
        result.update(is_error=True, errors=["Usage limit reached. Resets sometime soon."])
    elif prompt != "ok":
        result.update(is_error=True, error="Unexpected provider error with ANTHROPIC_API_KEY=fake-test-value")
    print(json.dumps(result))
    if result["is_error"]:
        sys.exit(1)
