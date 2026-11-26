#!/usr/bin/env bash
# Local validation: the complete check an agent or the operator runs before calling a change done.
# Runs against a temporary copy of the current source (tracked files plus unstaged edits and new, unignored
# files), with a clean environment, Keychain disabled, the simulated feed and a temporary HOME. No broker,
# paid model or GUI is touched. Every step is reported, and the exit status is nonzero if any step fails.
#
#   scripts/validate.sh               all steps (the dependency audit needs network access)
#   scripts/validate.sh --skip-audit  all steps except the dependency audit (offline)
set -euo pipefail

skip_audit=false
case "${1:-}" in
  "") ;;
  --skip-audit) skip_audit=true ;;
  *) echo "Usage: scripts/validate.sh [--skip-audit]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Usage: scripts/validate.sh [--skip-audit]" >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lat-validate.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/source" "$scratch/home"

# Copy current source, including unstaged edits and new files, without ignored runtime data or Git state.
git ls-files --cached --others --exclude-standard -z | while IFS= read -r -d '' file; do
  if [[ -f "$file" && ! -L "$file" ]]; then
    printf '%s\0' "$file"
  fi
done | tar -cf "$scratch/source.tar" --null -T -
tar -xf "$scratch/source.tar" -C "$scratch/source"

uv_cache=$(uv cache dir)
uv_python_dir=$(uv python dir)
cd "$scratch/source"

clean=(env -i "PATH=$PATH" "HOME=$scratch/home" "UV_CACHE_DIR=$uv_cache"
       "UV_PYTHON_INSTALL_DIR=$uv_python_dir" UV_NO_ENV_FILE=true
       LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED)
offline=("${clean[@]}" UV_NO_SYNC=true UV_OFFLINE=true)
test_marker="not live_models and not paper_broker and not gui"
# Match TORCH_TESTS in the Makefile. Both invocations are reported even if the first fails.
torch_tests=(tests/unit/research/test_kronos.py tests/unit/research/test_advanced_models.py)
status=0
summary=()

run_step() {
  local label=$1 result
  shift
  printf '\n== %s ==\n' "$label"
  if "$@"; then
    printf '%s: passed\n' "$label"
    summary+=("passed  $label")
  else
    result=$?
    printf '%s: FAILED (exit %s)\n' "$label" "$result" >&2
    summary+=("FAILED  $label")
    status=1
    return "$result"
  fi
}

audit() {
  "${clean[@]}" uv export --frozen --extra advanced-research --extra kronos \
    --no-hashes --no-emit-project --format requirements-txt --output-file "$scratch/requirements.txt" >/dev/null &&
  "${clean[@]}" uvx --python 3.11 pip-audit@2.10.0 --strict --no-deps --disable-pip \
    --requirement "$scratch/requirements.txt"
}

lean_tests() {
  # LEAN and the local .NET SDK are ignored build outputs, so the suite runs from the real checkout.
  make -s -C "$repo" test-lean
}

if run_step "Frozen Python 3.11 sync" "${clean[@]}" uv sync --frozen --python 3.11 \
    --extra advanced-research --extra kronos; then
  run_step "Ruff lint" "${offline[@]}" uv run --frozen ruff check . || :
  run_step "Pyright type check" "${offline[@]}" uv run --frozen pyright || :
  run_step "Tool, MCP and HTTP contract checks" "${offline[@]}" uv run --frozen pytest -q \
    tests/unit/tools/test_tools_registry.py tests/unit/mcp_server/test_mcp_registrations.py \
    tests/unit/api/test_api_routes_security.py || :
  run_step "Python test suite" "${offline[@]}" uv run --frozen pytest -q -m "$test_marker" \
    "${torch_tests[@]/#/--ignore=}" || :
  run_step "Torch test files" "${offline[@]}" uv run --frozen pytest -q -m "$test_marker" "${torch_tests[@]}" || :
  run_step "Isolated startup smoke (API + MCP + ledger audit chain)" "${offline[@]}" scripts/smoke_check.sh || :
fi
if [[ -f "$repo/trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj" && -x "$repo/.tools/dotnet/dotnet" ]]; then
  run_step "LEAN .NET tests" lean_tests || :
else
  printf '\n== LEAN .NET tests ==\nLEAN is not built in this checkout (make lean-build); suite not run\n'
  summary+=("skipped LEAN .NET tests (LEAN not built: run make lean-build)")
fi
if [[ "$skip_audit" == false ]]; then
  run_step "Locked dependency security audit" audit || :
else
  summary+=("skipped Locked dependency security audit (--skip-audit)")
fi

printf '\n== Summary ==\n'
printf '%s\n' "${summary[@]}"
exit "$status"
