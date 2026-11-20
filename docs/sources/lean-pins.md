# LEAN + Alpaca plugin pins

Machine-readable source of truth: `trading/PINS.json` (checked by `scripts/lean_build.sh` and
`tests/unit/lean/test_lean_tools.py::test_pins_match_the_checked_out_sources_and_lock_files`).
Recorded 2026-11-07 on macOS 27 arm64 (M4 Pro).

## Upstream sources (unmodified)

| Component | Repository | Commit | Commit date | License |
|---|---|---|---|---|
| QuantConnect LEAN | https://github.com/QuantConnect/Lean | `d1c09284c438b70eb965b7dc40f44e7dca69e7b5` | 2026-11-05T10:16:35-03:00 | Apache-2.0 |
| LEAN Alpaca brokerage plugin | https://github.com/QuantConnect/Lean.Brokerages.Alpaca | `b92d6ecd391d5a79e91b060668d1ea4fafec9b89` | 2026-10-19T20:08:38+03:00 | Apache-2.0 |

* Both are shallow clones at exactly these commits (`trading/lean/Lean`, `trading/alpaca_plugin/Lean.Brokerages.Alpaca`).
  `scripts/lean_build.sh` refuses to build if `HEAD` differs from the pin or a tracked upstream file is modified.
  **No local patches** (`local_patches: []`): all integration is configuration plus new projects.
* LEAN targets `net10.0`; its assemblies carry version `2.5.0.0` (`Common/Properties/SharedAssemblyInfo.cs`).
* The plugin vendors `QuantConnect.AlpacaBrokerage/Alpaca.Markets.dll` (QuantConnect fork, 8.0.0-beta6, assembly
  8.0.0.6), sha256 `92125fd7e8d6f8e4949cf69b29a1acabb0a7dcde6195be6fdfaaf8fe0ecead6f`.
* `trading/alpaca_plugin/Lean -> ../lean/Lean` symlink exists only because the plugin's own test project expects LEAN
  as a sibling; our build does not use that test project.

## Toolchain

| Tool | Version | Location |
|---|---|---|
| .NET SDK | 10.0.403 (official `dotnet-install.sh`, channel 10.0) | `.tools/dotnet` (`scripts/lean_build.sh` refuses any other SDK) |
| NUnit / adapter / Test SDK | 4.2.2 / 4.6.0 / 17.12.0 (same as the pinned LEAN `Tests` project) | `trading/lean/LatStrategies.Tests` |

## How the plugin is compiled: against the pinned LEAN *source*, not NuGet

The upstream plugin csproj references the floating NuGet packages `QuantConnect.Brokerages 2.5.*` and
`QuantConnect.Lean.Engine 2.5.*`. Those would resolve to whatever QuantConnect last published, not to commit
`d1c09284`, and loading such a plugin next to a source-built Launcher puts two different builds of
`QuantConnect.Common` (same assembly version 2.5.0.0, different code) in play.

We therefore build the plugin with `trading/alpaca_plugin/source-build/QuantConnect.Brokerages.Alpaca.SourceBuild.csproj`:
it compiles the upstream `.cs` files in place (`Compile Include`, no copy, no edit) with `ProjectReference`s to the pinned
LEAN projects (`Private=false`), and the plugin's third-party packages at the exact upstream versions
(`Newtonsoft.Json 13.0.2`, `Polly 8.6.3`, `MessagePack 3.1.7`). The floating `2.5.*` range is never restored: the NuGet
cache on this Mac contains no `quantconnect.lean.engine` / `quantconnect.brokerages` package at all.

Runtime verification (every live run logs `LAT-ASSEMBLY-CHECK` lines; the bridge refuses to start unless exactly one
`QuantConnect.Common` is loaded). From `trading/lean/runs/lean-live-sim-bridge-*/live_sim_report.json`:

```
QuantConnect.Common copies=1 2.5.0.0 .../trading/lean/Lean/Launcher/bin/Release/QuantConnect.Common.dll
QuantConnect.Brokerages copies=1 2.5.0.0 .../trading/lean/Lean/Launcher/bin/Release/QuantConnect.Brokerages.dll
QuantConnect.Lean.Engine copies=1 2.5.0.0 .../trading/lean/Lean/Launcher/bin/Release/QuantConnect.Lean.Engine.dll
QuantConnect.Brokerages.Alpaca copies=1 2.5.0.0 .../trading/lean/bin/plugins/QuantConnect.Brokerages.Alpaca.dll
Alpaca.Markets copies=1 8.0.0.6 .../trading/lean/bin/plugins/Alpaca.Markets.dll
resolved typeof(Symbol) from QuantConnect.Common .../Lean/Launcher/bin/Release/QuantConnect.Common.dll
```

## Plugin discovery

LEAN's `Composer` loads every `*.dll` of the Launcher directory plus the directory named by the `plugin-directory`
config key (`Common/Util/Composer.cs`). `scripts/lean_build.sh` assembles `trading/lean/bin/plugins/` with only the
assemblies the Launcher output does not already provide (plugin, `Alpaca.Markets`, Polly, MessagePack,
`LatStrategies`, `LatAlpacaGate`); no `QuantConnect.*` engine assembly is ever copied, and the upstream Launcher output
is never modified. Hashes of the bundle: `trading/lean/bin/build-manifest.json` (regenerated on every build).

## NuGet lock files (our projects)

`RestorePackagesWithLockFile=true` in every LAT project; lock files are committed next to each csproj:

* `trading/lean/LatStrategies/packages.lock.json`
* `trading/lean/LatAlpacaGate/packages.lock.json`
* `trading/lean/LatTestAlgorithms/packages.lock.json`
* `trading/lean/LatStrategies.Tests/packages.lock.json`
* `trading/alpaca_plugin/source-build/packages.lock.json`

The resolved closure (direct + transitive, including LEAN's own pinned dependencies, no version conflicts across
projects) is recorded in `trading/PINS.json` → `resolved_nuget_packages` (40 packages), e.g. `NodaTime 3.0.5`,
`Newtonsoft.Json 13.0.2`, `QuantConnect.pythonnet 2.0.68`, `MathNet.Numerics 5.0.0`, `protobuf-net 3.1.33`,
`RestSharp 106.12.0`, `System.Drawing.Common 4.7.0`. The unit test fails if `PINS.json` drifts from the lock files.

Known advisories: NuGet reports NU1903/NU1904 for some packages of LEAN's pinned closure (e.g. `System.Drawing.Common
4.7.0`, `RestSharp 106.12.0`). They are suppressed in the LAT csproj files (not upgraded) so the engine runs the exact
dependency set of the pinned LEAN commit; the engine is a local-only process. Revisit when the LEAN pin moves.

## Updating the pins

1. Fetch the new commits into the clones (`git fetch --depth 1 origin <sha> && git checkout FETCH_HEAD`).
2. Update `trading/PINS.json` (commit, date), run `scripts/lean_build.sh` (rebuilds, regenerates lock files on restore,
   runs the NUnit suite), refresh `resolved_nuget_packages`, then `uv run --frozen pytest tests/unit/lean tests/integration/lean tests/failure_injection/lean`.
3. Re-check the plugin for `ValidateSubscription`, client-order-id handling and `CreateAlpacaOrder` changes
   (see `trading/README.md`).
