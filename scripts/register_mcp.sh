#!/usr/bin/env bash
# Print (default) or run (--apply) the OFFICIAL MCP registrations of the local, PAPER-only `local-ai-trader`
# server for Claude Code and Codex. It never installs anything, never signs in, and never edits ~/.claude.json
# or ~/.codex directly: it only prints/runs each vendor's own `mcp add` command.
#
#   scripts/register_mcp.sh            # dry run: print commands and current registration state
#   scripts/register_mcp.sh --apply    # run the vendor CLIs where installed (Claude Code: project scope)
#
# See docs/runbooks/ai-runtime-mcp.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="local-ai-trader"
SERVER_CMD=(uv run --frozen --directory "$ROOT" python -m services.mcp_server)
APPLY=0

usage() {
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

quote() { printf '%q ' "$@"; }

echo "local-ai-trader MCP registration (project: $ROOT)"
echo "server command: $(quote "${SERVER_CMD[@]}")"
echo "The server only forwards to the local API on 127.0.0.1; start the API with 'make dev' (or 'make paper')."
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
