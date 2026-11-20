#!/usr/bin/env bash
# Run REAL LEAN backtests (self-hosted Launcher from pinned source, no LEAN CLI) of LatBaselineAlgorithm with the
# LAT fee/slippage/fill models, then parse LEAN's own result JSON into experiments/lean/<run_id>/summary.json.
#
# Usage:
#   scripts/lean_backtest.sh                       # Daily SPY 2010-2020 + Minute SPY 2013-10-07..11 on LEAN's REAL sample data
#                                                  # (+ a run on trading/lean/data-local if that SYNTHETIC export exists)
#   scripts/lean_backtest.sh --resolution Minute   # one run
#   scripts/lean_backtest.sh --data-folder trading/lean/data-local --dataset SYNTHETIC [--symbol XYZ] [--resolution Daily]
#   extra LEAN parameters: --param fast=10 --param slow=50 ...
# Each run writes its own config to trading/lean/runs/<run_id>/config.json (the upstream config is never edited).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DOTNET_ROOT="${DOTNET_ROOT:-$ROOT/.tools/dotnet}"
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
PY=(uv run --frozen python)
TOOLS="$ROOT/trading/lean/tools/lat_lean.py"
LAUNCHER="$ROOT/trading/lean/Lean/Launcher/bin/Release/QuantConnect.Lean.Launcher.dll"
SAMPLE="$ROOT/trading/lean/Lean/Data"
DATA_LOCAL="$ROOT/trading/lean/data-local"

RESOLUTION="all"; DATA_FOLDER=""; DATASET=""; SYMBOL=""; START=""; END=""; RUN_ID=""; ALGORITHM="Lat.Lean.Algorithms.LatBaselineAlgorithm"; PARAMS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --data-folder) DATA_FOLDER="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --symbol) SYMBOL="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --param) PARAMS+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 64 ;;
  esac
done

[[ -f "$LAUNCHER" && -f "$ROOT/trading/lean/bin/plugins/LatStrategies.dll" ]] || {
  echo "LEAN not built: run scripts/lean_build.sh (make lean-build)" >&2; exit 3; }

run_one() { # run_one <resolution> <data-folder> <dataset> <symbol> <start> <end> <run-id>
  local res="$1" folder="$2" dataset="$3" sym="$4" start="$5" end="$6" rid="$7"
  local simulated=""
  [[ "$dataset" == "LEAN_SAMPLE" ]] || simulated="--simulated"
  local run_dir="$ROOT/trading/lean/runs/$rid"
  mkdir -p "$run_dir"
  local data="$folder"
  if [[ ! -d "$folder/market-hours" || ! -d "$folder/symbol-properties" ]]; then
    # overlay: the dataset plus LEAN's market-hours / symbol-properties databases (read-only symlinks)
    data="$run_dir/data-overlay"; mkdir -p "$data"
    for e in "$folder"/*; do ln -sfn "$e" "$data/$(basename "$e")"; done
    for e in market-hours symbol-properties; do [[ -e "$data/$e" ]] || ln -sfn "$SAMPLE/$e" "$data/$e"; done
  fi
  local args=(--algorithm "$ALGORITHM" --run-id "$rid" --data-folder "$data" --dataset "$dataset" --param "resolution=$res" --param "symbol=$sym")
  [[ -n "$start" ]] && args+=(--param "start=$start")
  [[ -n "$end" ]] && args+=(--param "end=$end")
  for p in ${PARAMS[@]+"${PARAMS[@]}"}; do args+=(--param "$p"); done
  local cfg
  cfg="$("${PY[@]}" "$TOOLS" backtest-config "${args[@]}")"
  local started
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== LEAN backtest $rid ($res $sym on $dataset) config $cfg"
  ( cd "$run_dir" && dotnet "$LAUNCHER" --config "$cfg" ) >"$run_dir/launcher-stdout.log" 2>&1 || {
    echo "LEAN Launcher exited non-zero; see $run_dir/launcher-stdout.log" >&2; tail -30 "$run_dir/launcher-stdout.log" >&2; return 1; }
  if grep -q "Runtime Error\|Algorithm.Initialize() Error\|During the algorithm initialization" "$run_dir/launcher-stdout.log"; then
    echo "LEAN reported an algorithm error; see $run_dir/launcher-stdout.log" >&2
    grep -m5 "Error" "$run_dir/launcher-stdout.log" >&2; return 1
  fi
  local notes="$ALGORITHM via self-hosted LEAN Launcher with LIMIT orders; "
  notes+="LatFeeModel (Alpaca \$0 commission + SEC Section 31 + FINRA TAF) and LatSpreadSlippageModel/LatFillModel. "
  if [[ "$dataset" == "LEAN_SAMPLE" ]]; then notes+="Real historical LEAN sample bars, simulated fills; no edge claimed."
  else notes+="SYNTHETIC data exported by services.market_data (not market history); simulated fills."; fi
  "${PY[@]}" "$TOOLS" summarize --run-id "$rid" --strategy "$ALGORITHM" --dataset "$dataset" \
    --started-at "$started" --notes "$notes" --data-folder "$folder" $simulated
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$DATA_FOLDER" ]]; then
  folder="$(cd "$DATA_FOLDER" && pwd)"
  dataset="${DATASET:-SYNTHETIC}"
  [[ "$RESOLUTION" == "all" ]] && RESOLUTION="Daily"
  res_lc="$(echo "$RESOLUTION" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$SYMBOL" ]]; then
    SYMBOL="$(ls "$folder/equity/usa/$res_lc" 2>/dev/null | head -1 | sed 's/\.zip$//' | tr '[:lower:]' '[:upper:]')"
    [[ -n "$SYMBOL" ]] || SYMBOL="$(ls "$folder/equity/usa/daily" 2>/dev/null | head -1 | sed 's/\.zip$//' | tr '[:lower:]' '[:upper:]')"
  fi
  if [[ -z "$START" || -z "$END" ]]; then
    read -r d0 d1 < <("${PY[@]}" "$TOOLS" data-range --data-folder "$folder" --symbol "${SYMBOL:-SPY}" --resolution "$RESOLUTION")
    START="${START:-$d0}"; END="${END:-$d1}"
  fi
  run_one "$RESOLUTION" "$folder" "$dataset" "${SYMBOL:-SPY}" "$START" "$END" "${RUN_ID:-lean-bt-$(echo "$dataset" | tr '[:upper:]_' '[:lower:]-')-$res_lc-$stamp}"
  exit 0
fi

sym="${SYMBOL:-SPY}"
if [[ "$RESOLUTION" == "all" || "$RESOLUTION" == "Daily" ]]; then
  run_one Daily "$SAMPLE" LEAN_SAMPLE "$sym" "${START:-2010-01-04}" "${END:-2020-12-31}" "${RUN_ID:-lean-bt-sample-daily-$stamp}"
fi
if [[ "$RESOLUTION" == "all" || "$RESOLUTION" == "Minute" ]]; then
  run_one Minute "$SAMPLE" LEAN_SAMPLE "$sym" "${START:-2013-10-07}" "${END:-2013-10-11}" "${RUN_ID:-lean-bt-sample-minute-$stamp}"
fi
if [[ "$RESOLUTION" == "all" && -d "$DATA_LOCAL/equity/usa" ]]; then
  echo "== trading/lean/data-local found: running the same strategy on the SYNTHETIC export (Minute SPY)"
  # Minute: the export holds only a few weeks of sessions, too short for the Daily 20/100 SMA to trade
  "$0" --data-folder "$DATA_LOCAL" --dataset SYNTHETIC --resolution Minute --symbol "$sym"
fi
