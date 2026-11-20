#!/usr/bin/env bash
# Build pinned LEAN and LAT research/backtest projects. No brokerage plugin or deployment license.
#
# Usage: scripts/lean_build.sh [--skip-tests]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DOTNET_ROOT="${DOTNET_ROOT:-$ROOT/.tools/dotnet}"
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

LOG_DIR="$ROOT/var/logs"
LOG="$LOG_DIR/lean-build.log"
mkdir -p "$LOG_DIR"
SKIP_TESTS=0
[[ "${1:-}" == "--skip-tests" ]] && SKIP_TESTS=1

LEAN_DIR="$ROOT/trading/lean/Lean"
BUNDLE="$ROOT/trading/lean/bin/plugins"
LAUNCHER_OUT="$LEAN_DIR/Launcher/bin/Release"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
die() { log "ERROR: $*"; exit 1; }

pin() { # pin <json-path...> from trading/PINS.json (no python deps needed)
  /usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2:]: d=d[k]
print(d)' "$ROOT/trading/PINS.json" "$@"
}

: >"$LOG"
log "lean_build: start (root $ROOT)"
command -v dotnet >/dev/null || die "dotnet not found at $DOTNET_ROOT (run: make setup)"
SDK="$(dotnet --version)"
WANT_SDK="$(pin dotnet sdk_version)"
log "dotnet SDK $SDK (pinned $WANT_SDK)"
[[ "$SDK" == "$WANT_SDK" ]] || die "dotnet SDK $SDK != pinned $WANT_SDK"

LEAN_SHA="$(pin lean commit)"


ensure_clone() { # ensure_clone <dir> <url> <sha>
  local dir="$1" url="$2" sha="$3"
  if [[ ! -d "$dir/.git" ]]; then
    log "cloning $url @ $sha"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
    git -C "$dir" fetch -q --depth 1 origin "$sha" 2>&1 | tee -a "$LOG"
    git -C "$dir" checkout -q FETCH_HEAD
  fi
  local head
  head="$(git -C "$dir" rev-parse HEAD)"
  [[ "$head" == "$sha" ]] || die "$dir is at $head, pinned $sha (refusing to build an unpinned tree)"
  if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
    die "$dir has local modifications to tracked upstream files; the build uses unmodified pinned sources"
  fi
  log "verified $dir @ $sha (clean)"
}
ensure_clone "$LEAN_DIR" "$(pin lean repo)" "$LEAN_SHA"
build() { # build <csproj> <label>
  log "dotnet build $2 (Release)"
  if ! dotnet build "$1" -c Release -nologo -v minimal -clp:ErrorsOnly >>"$LOG" 2>&1; then
    tail -40 "$LOG"
    die "build failed: $2"
  fi
}
build "$LEAN_DIR/Launcher/QuantConnect.Lean.Launcher.csproj" "LEAN Launcher"
build "$ROOT/trading/lean/LatStrategies/LatStrategies.csproj" "LatStrategies"
build "$ROOT/trading/lean/LatTestAlgorithms/LatTestAlgorithms.csproj" "LatTestAlgorithms"
build "$ROOT/trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj" "LatStrategies.Tests"

[[ -f "$LAUNCHER_OUT/QuantConnect.Lean.Launcher.dll" ]] || die "Launcher output missing"

# ---- plugin bundle: ONLY assemblies the Launcher output does not already provide (one QuantConnect.Common)
log "assembling plugin bundle $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"
cp "$ROOT/trading/lean/LatStrategies/bin/Release/LatStrategies.dll" "$BUNDLE/"
# no .pdb next to LatStrategies.dll: LEAN's Loader then uses Assembly.LoadFrom (same path the Composer used),
# so the algorithm and the brokerage factories share ONE loaded LatStrategies assembly
log "bundle contents: $(cd "$BUNDLE" && ls | tr '\n' ' ')"

# ---- record resolved versions for the pins document
{
  echo "{"
  echo "  \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"dotnet_sdk\": \"$SDK\","
  echo "  \"bundle\": ["
  first=1
  for f in "$BUNDLE"/*.dll; do
    sha="$(shasum -a 256 "$f" | cut -d' ' -f1)"
    [[ $first -eq 1 ]] || echo ","
    printf '    {"file": "%s", "sha256": "%s"}' "$(basename "$f")" "$sha"
    first=0
  done
  echo ""
  echo "  ]"
  echo "}"
} >"$ROOT/trading/lean/bin/build-manifest.json"
log "wrote trading/lean/bin/build-manifest.json"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  log "dotnet test LatStrategies.Tests"
  if ! dotnet test "$ROOT/trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj" -c Release --no-build -nologo -v q >>"$LOG" 2>&1; then
    tail -40 "$LOG"
    die "LatStrategies.Tests failed"
  fi
  grep -E "Passed!|Failed!|Total tests" "$LOG" | tail -2 | while read -r l; do log "$l"; done
fi
log "lean_build: OK"
