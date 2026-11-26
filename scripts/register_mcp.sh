#!/usr/bin/env bash
# Register the local, PAPER-only MCP server with Claude Code and Codex.
#
#   scripts/register_mcp.sh                         # print commands and registration state
#   scripts/register_mcp.sh --apply                 # run installed vendor CLIs
#   scripts/register_mcp.sh --write-project-configs # update checkout paths in both project files
#
# See docs/runbooks/ai-runtime-mcp.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="local-ai-trader"
SERVER_CMD=(uv run --frozen --directory "$ROOT" python -m services.mcp_server)
APPLY=0
WRITE_CONFIGS=0

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --write-project-configs) WRITE_CONFIGS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $WRITE_CONFIGS -eq 1 ]]; then
  if [[ $APPLY -eq 1 ]]; then
    echo "--write-project-configs and --apply are separate operations" >&2
    exit 2
  fi
  uv run --frozen --offline --no-project --python 3.11 python - "$ROOT" "$NAME" "${SERVER_CMD[@]:1}" <<'PYTHON'
import json
import re
import sys
import tomllib
from pathlib import Path

root, name, *args = sys.argv[1:]
claude_path = Path(root) / ".mcp.json"
codex_path = Path(root) / ".codex" / "config.toml"

claude = json.loads(claude_path.read_text(encoding="utf-8"))
claude["mcpServers"][name]["args"] = args

codex_text = codex_path.read_text(encoding="utf-8")
codex = tomllib.loads(codex_text)
codex["mcp_servers"][name].update(args=args, cwd=root)
header = re.escape(f"[mcp_servers.{name}]")
section = re.search(rf"(?m)^{header}[ \t]*\n[^\n]*(?:\n(?![ \t]*\[)[^\n]*)*", codex_text)
if section is None:
    sys.exit("Missing local-ai-trader server section in .codex/config.toml")

updated = section[0]
for key, value in (("args", args), ("cwd", root)):
    # Edit the project registration in place so comments and approval tables are retained.
    line = f"{key} = {json.dumps(value, ensure_ascii=False)}"
    updated, count = re.subn(rf"(?m)^[ \t]*{key}[ \t]*=.*$", lambda _: line, updated)
    if count != 1:
        sys.exit(f"Expected one {key} entry in the local-ai-trader server section")

codex_text = codex_text[:section.start()] + updated + codex_text[section.end():]
if tomllib.loads(codex_text) != codex:
    sys.exit("Could not safely update checkout paths in .codex/config.toml")

claude_path.write_text(json.dumps(claude, indent=2) + "\n", encoding="utf-8")
codex_path.write_text(codex_text, encoding="utf-8")
print(f"Updated .mcp.json and .codex/config.toml for {root}")
PYTHON
  exit 0
fi

quote() { printf '%q ' "$@"; }

echo "local-ai-trader MCP registration (project: $ROOT)"
echo "server command: $(quote "${SERVER_CMD[@]}")"
echo "After connecting, ask the client to start trading. The session tool starts the local API on demand."
echo

# ---------------------------------------------------------------- Claude Code (project scope -> .mcp.json)
echo "== Claude Code (project scope: $ROOT/.mcp.json) =="
CLAUDE_ADD=(claude mcp add --scope project -e LAT_AGENT_NAME=claude-code "$NAME" -- "${SERVER_CMD[@]}")
echo "command: $(quote "${CLAUDE_ADD[@]}")"
if command -v claude >/dev/null 2>&1; then
  if [[ -f "$ROOT/.mcp.json" ]] && grep -q "\"$NAME\"" "$ROOT/.mcp.json"; then
    echo "status: already registered in .mcp.json (project-scope file in this repo)."
  elif [[ $APPLY -eq 1 ]]; then
    (cd "$ROOT" && "${CLAUDE_ADD[@]}")
  else
    echo "status: not registered; re-run with --apply to execute the command above."
  fi
  echo "verify: (cd $(quote "$ROOT")&& claude mcp get $NAME)"
  echo "note: Claude Code asks you to approve project-scoped servers from .mcp.json the first time; approve"
  echo "      'local-ai-trader' in the prompt (or via /mcp). Claude Code uses your own Claude sign-in and limits."
  if [[ $APPLY -eq 1 ]]; then
    (cd "$ROOT" && claude mcp get "$NAME") || true
  fi
else
  echo "status: Claude Code CLI ('claude') not found on PATH. Nothing was installed."
  echo "        Install it yourself from https://docs.claude.com/en/docs/claude-code/setup, sign in with your"
  echo "        own Claude account, then re-run this script."
fi
echo

# ---------------------------------------------------------------- Codex (project file + optional user registration)
echo "== Codex =="
CODEX_ADD=(codex mcp add "$NAME" --env LAT_AGENT_NAME=codex -- "${SERVER_CMD[@]}")
echo "project config: $ROOT/.codex/config.toml (loaded by Codex for a trusted project; read-only tools"
echo "                auto-approved, mutating tools keep approval_mode = \"prompt\")"
echo "command (optional user-level registration): $(quote "${CODEX_ADD[@]}")"
if command -v codex >/dev/null 2>&1; then
  if [[ $APPLY -eq 1 ]]; then
    if codex mcp get "$NAME" >/dev/null 2>&1; then
      echo "status: already registered with Codex."
    else
      "${CODEX_ADD[@]}"
    fi
    codex mcp list || true
  else
    echo "status: Codex CLI found; re-run with --apply to execute the command above (or rely on the project file)."
  fi
else
  echo "status: Codex CLI ('codex') is NOT installed on this Mac. This script does not install anything."
  echo "        To use Codex yourself:"
  echo "          1. install it, e.g.  npm install -g @openai/codex   (or: brew install --cask codex)"
  echo "          2. run 'codex login' and choose 'Sign in with ChatGPT' (your plan's usage limits apply)"
  echo "          3. open Codex in $ROOT and trust the project so .codex/config.toml is loaded,"
  echo "             or run the 'codex mcp add' command above."
fi
