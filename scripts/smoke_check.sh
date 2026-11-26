#!/usr/bin/env bash
# Run make smoke in a clean environment and fail unless the simulated session completed after the MCP client
# disconnected and the ledger audit chain verified. Called by scripts/validate.sh; can also run on its own.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lat-smoke.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT

# make smoke creates its own workspace and selects the simulated gateway.
# Limit inherited state to tool discovery and the temporary HOME.
env -i "PATH=$PATH" "HOME=$scratch" "UV_CACHE_DIR=$(uv cache dir)" \
  UV_NO_SYNC=true UV_OFFLINE=true UV_NO_ENV_FILE=true \
  LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED \
  make -s smoke > "$scratch/report.json"

uv run --frozen --no-sync --offline --no-env-file python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as report_file:
    report = json.load(report_file)
if (
    report["session"]["state"] != "COMPLETED"
    or not report["operator_disconnected_during_session"]
    or report["audit_verified_rows"] < 1
):
    sys.exit("Isolated startup smoke did not complete or verify the audit chain")
cycles = report["cycles"]
rows = report["audit_verified_rows"]
print(f"Simulated session completed {cycles} cycles and verified {rows} audit rows")
' "$scratch/report.json"
