# LEAN research and backtesting

Current application execution uses `services/order_ledger/alpaca_gateway.py` and the official `alpaca-py` SDK.
`make lean-build` builds research/backtest assemblies without the Alpaca plugin or `LatAlpacaGate`.
`make backtest` remains available. The runtime refuses both legacy LEAN execution gateway selections.
`make lean-paper` and `make lean-live-sim` refuse to launch; use `make paper` or `make demo`.
No QuantConnect deployment license is required by the application execution path.

The material below is historical implementation evidence. Legacy source and isolated regression harnesses
are preserved for audit/research; their old execution instructions are retired.

---

# trading/ — self-hosted QuantConnect LEAN + official Alpaca plugin, behind a signed-intent gate

Status labels: **TESTED** (run here, with the command and counts), **SIMULATED** (real engine, simulated broker/data),
**IMPLEMENTED_NOT_INTEGRATION_TESTED**, **BLOCKED** (external prerequisite missing).

| Part | Status | Evidence |
|---|---|---|
| LEAN Launcher (Release) + Alpaca plugin + LAT projects built from pinned source | TESTED | `scripts/lean_build.sh` → `var/logs/lean-build.log`, 0 errors |
| Gate / fee / slippage / bridge unit tests (C#, NUnit) | TESTED | `dotnet test trading/lean/LatStrategies.Tests -c Release` → 84/84 |
| Cross-language ECDSA P-256 vectors (Python signer → C# verifier and back) | TESTED | NUnit `SignatureVectorTests` + `tests/unit/lean/test_lean_tools.py` |
| Real LEAN backtest of `LatBaselineAlgorithm` on LEAN's bundled real SPY data | TESTED (real historical data, simulated fills) | `scripts/lean_backtest.sh` → `experiments/lean/lean-bt-sample-*/summary.json` |
| Same runner on a SYNTHETIC export (`--data-folder`) | TESTED (SYNTHETIC) | `experiments/lean/lean-bt-synthetic-minute-*/summary.json` |
| LEAN live mode + GatedPaperBrokerage + simulated data queue + bridge to a 127.0.0.1 stub | SIMULATED, TESTED | `scripts/lean_live_sim.sh`, `pytest tests/integration/lean tests/failure_injection/lean` → 15 passed |
| Same real engine against the REAL order ledger (`services.order_ledger.bridge` + `ExecutionService` + `Reconciler`) | SIMULATED, TESTED | `pytest tests/e2e/lean` → 1 passed: BUY filled → ledger `FILLED`, oversell refused by the gate → `REJECTED`, closing SELL filled, reconcile `CLEAN` |
| Same engine for the REAL local API (`scripts/lean_live_sim.sh api`) | SIMULATED, smoke-run only | heartbeat/claim/gate/ledger path works; fills impossible while the API runs its replay `SimulationClock` (see Limitations) |
| Alpaca PAPER via LEAN (`GatedAlpacaBrokerage`) | BLOCKED (no paper keys; plugin needs a QuantConnect licence) | `scripts/lean_paper_alpaca.sh` exits 3 with the reason |
| Alpaca LIVE | not present by design | no config, no launcher; tests prove `alpaca-paper-trading=false` cannot be produced |

## Layout

```
trading/PINS.json                        pinned commits, SDK, resolved NuGet closure (see docs/sources/lean-pins.md)
trading/lean/Lean/                       upstream LEAN clone @ d1c09284 (unmodified, git-ignored)
trading/alpaca_plugin/Lean.Brokerages.Alpaca/   upstream plugin clone @ b92d6ecd (unmodified, git-ignored)
trading/alpaca_plugin/source-build/      csproj compiling the plugin sources against the pinned LEAN projects
trading/lean/LatStrategies/              gate, cost models, bridge, algorithms, gated PaperBrokerage, sim data queue
trading/lean/LatAlpacaGate/              GatedAlpacaBrokerage (official AlpacaBrokerage subclass, paper hard-coded)
trading/lean/LatTestAlgorithms/          TEST ONLY hostile strategy (rogue scenario)
trading/lean/LatStrategies.Tests/        NUnit tests + vectors/intent_vectors.json (from the Python signer)
trading/lean/config/live-alpaca-paper.template.json   PAPER template (no secrets)
trading/lean/tools/lat_lean.py           per-run config generation, result parsing, paper config (SecretStore)
trading/lean/tools/gen_intent_vectors.py regenerates the cross-language vectors
trading/lean/bin/plugins/                plugin bundle LEAN's Composer loads via "plugin-directory" (build output)
trading/lean/runs/<run_id>/              per-run config.json + LEAN outputs (git-ignored)
experiments/lean/<run_id>/summary.json   backtest summaries for the API/dashboard
```

## Build

```bash
scripts/lean_build.sh            # or: make lean-build   (idempotent; log var/logs/lean-build.log)
scripts/lean_build.sh --skip-tests
```

Verifies SDK `10.0.403` and that both clones are exactly at the pinned commits with no tracked modifications, then
builds (Release): LEAN `Launcher`, the plugin source build, `LatStrategies`, `LatAlpacaGate`, `LatTestAlgorithms`,
`LatStrategies.Tests`; assembles `trading/lean/bin/plugins` (only assemblies the Launcher output does not provide — never a
second `QuantConnect.*` engine assembly), writes `trading/lean/bin/build-manifest.json` (sha256 of the bundle) and runs
the NUnit suite. A clean first build of the Launcher takes a few minutes; incremental runs ~10 s.

Pins and the "plugin compiled against pinned LEAN source, not NuGet `2.5.*`" decision: `docs/sources/lean-pins.md`.
At runtime every live run logs `LAT-ASSEMBLY-CHECK` lines; the bridge refuses to start unless exactly one
`QuantConnect.Common` (the Launcher's) is loaded.

## Backtest (real LEAN engine, no LEAN CLI)

```bash
scripts/lean_backtest.sh                                   # Daily SPY 2010-01-04..2020-12-31 + Minute SPY 2013-10-07..11
scripts/lean_backtest.sh --resolution Minute --param fast=10 --param slow=40
scripts/lean_backtest.sh --data-folder trading/lean/data-local --dataset SYNTHETIC --resolution Minute --symbol SPY
```

Each run writes its own flat `trading/lean/runs/<run_id>/config.json` (the upstream `Launcher/config.json` is never
edited) and runs `dotnet QuantConnect.Lean.Launcher.dll --config <that file>`. LEAN's own result JSON is parsed into
`experiments/lean/<run_id>/summary.json` (`run_id, engine:"LEAN", lean_commit, strategy, dataset, is_simulated,
started_at, metrics{total_return, sharpe, max_drawdown, trades, fees, …}, equity_curve[{time,equity}], notes,
source_path, log_excerpt_path, data_provenance`). Money values are decimal strings. A `--data-folder` that carries the
`lat-export-manifest.json` of `services.market_data` export-lean gets its provenance copied and is always labelled
`is_simulated: true` when synthetic. When `trading/lean/data-local` exists, the default invocation also runs there.

Results recorded on 2026-11-08 UTC (real execution; LEAN logs in `trading/lean/runs/<run_id>/`, excerpts in
`experiments/lean/<run_id>/lean-log-excerpt.txt`). No edge is claimed:

| Run | Data | Orders | Fees (USD) | Total return | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| `lean-bt-sample-daily-20261108T034257Z` | LEAN sample SPY daily (real) | 43 | 49.49 | 0.701311 | 0.262 | 0.290 |
| `lean-bt-sample-minute-20261108T034257Z` | LEAN sample SPY minute Oct 2013 (real) | 30 | 14.57 | -0.003571 | -10.895 | 0.006 |
| `lean-bt-synthetic-minute-20261108T034357Z` | SYNTHETIC `ds_1c99c07583eae908` in `trading/lean/data-local` | 100 | 56.24 | -0.012983 | -3.745 | 0.016 |

Re-running gives identical numbers (deterministic engine + data). The `data-local` run uses Minute SPY: the export holds
only a few weeks of sessions, too short for the Daily 20/100 SMA to trade.

### Strategy and cost model (identical objects in backtest and live paths)

* `LatBaselineAlgorithm`: deterministic SMA trend (Daily 20/100, Minute 15/60), long-only, one position, LIMIT orders
  only, regular hours (minute runs flatten at 15:50 ET and skip the first/last 15 minutes). Backtest-only: it refuses
  to run in live mode because it originates orders itself.
* `LatBrokerageModel` (decorator over `AlpacaBrokerageModel` / `DefaultBrokerageModel`): cash account, leverage 1,
  `NoShortingProvider` (LEAN's default would allow unlimited shorting), delayed equity settlement — so intraday round
  trips are limited by settled cash, as on a cash account.
* `LatFeeModel`: Alpaca commission $0/order (configurable), SEC Section 31 fee on sell proceeds (default $27.80 per
  $1M; `lat-fee-sec-rate-per-million`), FINRA TAF $0.000166/share sold capped at $8.30 per trade, regulatory fees
  rounded up to the cent. Verify the current SEC/FINRA rates before relying on them. LEAN computes fees before the fill
  price is known, so the security's current price is the notional proxy.
* `LatSpreadSlippageModel` + `LatFillModel`: limit fills pay half the quoted spread (or half of
  `lat-assumed-spread-bps` without a quote) plus `lat-slippage-bps` impact, rounded adversely to the tick, never worse than
  the limit. Backtest fills are still LEAN's simulation: no queue position, latency or market impact.

## The gate (spec invariants 1–3)

Only orders that carry a valid **ECDSA P-256** signed intent from the Python risk engine can reach a brokerage.

* Token on the LEAN order tag: `LATv1.<base64url(UTF-8 signing string)>.<hex r||s>`. The signing string is
  `packages.contracts.models.intent_signing_string` (`lat-intent-v1|intent_id=…|client_order_id=…|…|signature_alg=…`);
  C# verifies with `ECDsa.ImportSubjectPublicKeyInfo` + `VerifyData(…, SHA256, IeeeP1363FixedFieldConcatenation)` over
  its UTF-8 bytes, then parses every order field **from the signed string** (strict: exact prefix, field order and
  count). LEAN holds only the public key (`LAT_INTENT_PUBKEY_PATH` or config `lat-intent-pubkey-path`); a missing or
  non-P-256 key aborts start-up.
* `IntentGate` (pure logic, shared) refuses — never degrades — anything not exactly supported: unsigned/forged/tampered
  tokens, wrong `key_id` or `signature_alg`, mode ≠ PAPER, wrong gateway, missing human approval id when required,
  expired or future-dated intents, non-LIMIT (signed or LEAN type), TIF ≠ DAY, extended hours / outside-RTH properties,
  unknown order-properties types, combo/contingent orders, non-equity, non-USA, closed exchange, fractional shares,
  any symbol/side/quantity/limit mismatch, an existing short, sells exceeding holdings net of other open sells (or when
  open orders cannot be inspected), and replays: one intent → one LEAN order per run, plus a durable fsync'ed journal
  (`lat-gate-journal-path`; PAPER: `var/paper/lean/used-intents.tsv`) that refuses intents consumed by any earlier
  engine run. Duplicate/replay refusals are reported **without** ids so they can never alter the (possibly live) original
  order in the ledger. `LatBuild.LiveExecutionCompiled = false`.
* Enforced at two LEAN extension points:
  1. `GatedBrokerageModel.CanSubmitOrder` (algorithm-level hook; `CanUpdateOrder` always refuses — intents are
     immutable);
  2. the **final engine boundary** `IBrokerage.PlaceOrder` of `GatedPaperBrokerage` (LEAN `PaperBrokerage` subclass)
     and `GatedAlpacaBrokerage` (official `AlpacaBrokerage` subclass) — `UpdateOrder` refused. This catches strategies
     that replace the brokerage model (`SetBrokerageModel` is public) or race it: LEAN signals a synchronous ticket
     *before* `BrokerageTransactionHandler` runs `CanSubmitOrder`.
  The brokerage factories also install the gated model as the default, and the bridge halts intake if the model is
  replaced, if any non-intent order produces activity, or if `live-mode-brokerage` is not the gated one.
* `LatIntentBridgeAlgorithm` never originates orders: it polls `GET /api/v1/engine/intents/next` (header
  `X-LAT-Engine-Token` from env `LAT_ENGINE_TOKEN`, loopback base URL only, `lat-bridge-url`), submits each claimed intent
  once as a DAY `LimitOrder` built from the signed string, posts every `OnOrderEvent` to `/api/v1/engine/events`
  (decimal strings, ISO-UTC `Z` timestamps, `lean_order_id`, `lean_event_id`, `broker_order_id`), heartbeats to
  `/api/v1/engine/heartbeat` every 2 s and reports every refusal to `/api/v1/engine/rejections`. Network I/O runs on two
  background threads; outbound records are sent strictly in order with capped back-off retries and also appended to a
  local JSONL outbox. Duplicate deliveries are not resubmitted; claimed-but-unsent intents are reported at stop. No retries
  of order submission, ever.

### Client order id mapping (reconciliation)

The pinned plugin's `CreateAlpacaOrder()` (`AlpacaBrokerageExtensions.cs`) never sets Alpaca's `client_order_id`, and
`PlaceOrder` stores Alpaca's order id in `order.BrokerId` (`AlpacaBrokerage.cs` ~L667). Alpaca therefore assigns its own
client id; **our `client_order_id` is not carried to Alpaca**. Reconciliation relies on:
`client_order_id` + `intent_id` (signed, in the LEAN tag) → LEAN order id (bridge map) → `broker_order_id`
(`order.BrokerId`, sent in every `/engine/events` payload once the broker acknowledged). An Alpaca order that has no
LEAN/LAT mapping is an unexpected order (reconciler must lock entries). With `GatedPaperBrokerage` the broker id is
LEAN's paper id.

## LOCAL SIMULATED live path (no credentials)

```bash
scripts/lean_live_sim.sh            # bridge + rogue scenarios; reports in trading/lean/runs/<run_id>/live_sim_report.json
LAT_MODE=PAPER LAT_GATEWAY=LEAN_PAPER_SIM scripts/lean_live_sim.sh api   # engine for a running local API (SIMULATED)
uv run --frozen pytest tests/integration/lean tests/failure_injection/lean -q   # marked lean+slow; skip if LEAN not built
```

The real Launcher runs in live mode with `GatedPaperBrokerage` and `LatSimulatedDataQueue` (fixed SIMULATED SPY
bid/ask/last ticks) against `tests/integration/lean/stub_bridge.py` (127.0.0.1, engine-token checked, atomic one-time
claims, rejects JSON floats, fault injection) holding intents signed by the real `IntentSigner` with a throw-away key.
Demonstrated: a signed BUY 10 SPY @ 450.10 is claimed, submitted, filled by PaperBrokerage (fill 10 @ 450.10:
LEAN fill 450.00 + modelled half-spread/impact, capped at the limit) and the `Submitted`/`Filled` events reach the stub
with the right `client_order_id`, quantity and price; tampered, forged (other key), expired, short-sale, signed MARKET
and wrong-gateway intents are refused by `GatedBrokerageModel` and reported; a duplicate delivery is not resubmitted;
a hostile strategy's unsigned limit order is refused by `GatedBrokerageModel`, and after it swaps in
`DefaultBrokerageModel` its unsigned market order and forged-token order are refused by `GatedPaperBrokerage.PlaceOrder`;
the bridge halts. Failure injection: event-post outage (retried, in order, outbox kept), engine restart with the API
re-delivering an executed intent (refused by the journal, no event under the original ids), wrong engine token (nothing
claimed), missing public key (engine does not start). `lsof` sampling shows the LEAN process only holds loopback sockets.

**Real order ledger (e2e, `tests/e2e/lean/test_lean_e2e_ledger.py`).** The same real engine talks to the production
`services.order_ledger.bridge` functions (`claim_next_intents`, `ingest_engine_event`, `record_heartbeat`,
`record_engine_rejection`) behind a 127.0.0.1 shim that mirrors the four engine routes of `services/api/app.py`
(engine-token check, `json.loads(…, parse_float=Decimal)`); intents are recorded and queued by the real `ExecutionService`
+ `LeanBridgeGateway`, reconciled by the real `Reconciler`. Result: startup reconciliation `CLEAN` only after a `RUNNING`
heartbeat; BUY 10 @ 450.10 → ledger `FILLED` (one fill, 10 @ ≤ 450.10, broker id = PaperBrokerage id); SELL 50 while
holding 10 → `GatedBrokerageModel` "short sale refused" → `/engine/rejections` → ledger `REJECTED`, no fill; SELL 10 →
`FILLED`, flat; every claimed intent produced exactly the attributed events, no duplicates; reconcile `CLEAN`; after the
engine's `STOPPED` heartbeat `engine_health()` is false (entries lock); audit chain verifies.

**Engine for the real local API (`scripts/lean_live_sim.sh api`).** Start the API with `LAT_MODE=PAPER
LAT_GATEWAY=LEAN_PAPER_SIM`, then run the script with the same `LAT_*` environment: it uses that namespace's public intent
key, its `engine.token` file (value only in LEAN's environment) and the durable journal `var/<ns>/lean/used-intents.tsv`,
and runs until Ctrl-C (or `--exit-after N`). Smoke run on 2026-11-08 against the real `services/api` (scratch
`LAT_PROJECT_ROOT`, port 18765): `/api/v1/status` showed the engine `UP (RUNNING)`; `/control/reconcile` → `CLEAN`; an
agent-tool `propose_paper_trade` (BUY 1 SPY) was risk-APPROVED, signed, queued and claimed by LEAN, then refused by
`GatedBrokerageModel`: `intent expired at 2026-11-02T15:45:30Z (now 2026-11-08T03:42:00Z)` → ledger `REJECTED`
(`trading/lean/runs/lean-live-sim-api-20261108T034146Z-82825/launcher-stdout.log`). Cause: without Alpaca keys the API
runs the SYNTHETIC replay feed on a `SimulationClock` (a past session at 30x), so its intent timestamps are not wall-clock
time; the engine correctly fails closed. See Limitations.

LEAN live-mode requirements and how they are met locally (no QuantConnect cloud, no paid CLI):
* job packet: `QuantConnect.Queues.JobQueue` builds a `LiveNodePacket` from the flat config (`live-mode: true`,
  `live-mode-brokerage`, brokerage factory `BrokerageData`); `job-user-id: 0`, empty `api-access-token`;
  `QuantConnect.Api.Api` is initialised but never contacted (`SetAlgorithmStatus` is a no-op locally);
* handlers as upstream `live-paper` (`BrokerageSetupHandler`, `LiveTradingResultHandler`, `LiveTradingDataFeed`,
  `LiveTradingRealTimeHandler`, `BacktestingTransactionHandler` — PaperBrokerage requires it); the cloud
  `LiveDataQueue` is replaced by `LatSimulatedDataQueue`; `live-cash-balance` seeds the paper account;
* `force-exchange-always-open: true` (SIMULATION ONLY, so the gate's open-exchange rule can be exercised outside US
  hours; refused by the PAPER config check);
* the run ends via `Quit()` after `lat-exit-after-seconds`; LEAN's Launcher then exits with code 1 because the final
  algorithm status is `Stopped`, not `Completed` (`Launcher/Program.cs`) — expected for a stopped live algorithm.

## Alpaca PAPER (BLOCKED)

```bash
scripts/lean_paper_alpaca.sh [--bridge-url http://127.0.0.1:8765] [--symbols SPY]
```

* refuses (exit 2) unless `LAT_MODE` is `PAPER`, refuses any other argument and non-loopback bridge URLs;
* reads `ALPACA_PAPER_KEY_ID` / `ALPACA_PAPER_SECRET_KEY` only through `packages.core.secrets.SecretStore` inside
  `uv run --frozen python` (never echoed; the ambiguous `APCA_*` names are ignored); refuses key ids starting with `AK`;
* merges `trading/lean/config/live-alpaca-paper.template.json` (`GatedAlpacaBrokerage` as brokerage and data queue,
  `BrokerageTransactionHandler`, `alpaca-paper-trading: true`, `force-exchange-always-open: false`,
  `ignore-unknown-asset-holdings: false`, no secrets) into a per-run config (0600, run dir 0700) under
  `trading/lean/runs/`, re-checks it (`assert_paper_only`) and deletes it on exit; the engine token goes only into
  LEAN's environment. `GatedAlpacaBrokerage` passes `isPaperTrading: true` unconditionally and refuses
  `alpaca-paper-trading` ≠ true and `AK…` keys. No live template, launcher or gateway exists
  (`tests/unit/lean/test_lean_paper_config.py` greps/parses every owned script, config and C# source to prove it).

Exact blocked items:
1. **No Alpaca PAPER credentials** on this Mac (Keychain / env / `.env`): the launcher exits 3 with
   `BLOCKED: Alpaca PAPER credentials not found …` and instructions.
2. **QuantConnect module licence**: the official plugin's `AlpacaBrokerage.Initialize()` calls
   `ValidateSubscription()` (`AlpacaBrokerage.cs` ~L1431): it authenticates `job-user-id` / `api-access-token` against
   `api.quantconnect.com` (`modules/license/read`, product 347) and calls `Environment.Exit(1)` on any failure. So even
   with paper keys the plugin needs a QuantConnect account holding that licence (`LAT_QC_USER_ID`, `LAT_QC_API_TOKEN`,
   optional `LAT_QC_ORGANIZATION_ID`; launcher exits 3 without them). The check sends machine name, user name, OS version
   and network-interface MAC/IP addresses; the launcher additionally requires `LAT_QC_LICENSE_ACK=1`. The plugin was not
   patched (an owner decision; it would also be a local patch of upstream code).
3. Not exercised at all: Alpaca streaming/order paths through `GatedAlpacaBrokerage` (IMPLEMENTED_NOT_INTEGRATION_TESTED:
   it compiles against the pinned plugin source and shares the gate code that is tested above).

## Limitations / residual risks

* The gate protects every LEAN order route of the configs these scripts generate. A hand-written LEAN config naming the
  ungated `AlpacaBrokerage` with keys would bypass it; the SecretStore hands PAPER keys only to `lean_paper_alpaca.sh`,
  whose config is structurally checked.
* A forged token whose (unverified) ids name a real order in `SUBMITTING` is reported with those ids; the ledger only
  rejects `SUBMITTING` orders whose intent id matches, and a later fill for such an order would surface as an anomaly
  (fail closed), not as a silent loss.
* **LEAN gateways need a wall-clock API.** The engine checks `created_at`/`expires_at` against the Mac's UTC clock (it
  never trusts an API-supplied clock). The API's keyless default (SYNTHETIC replay on a `SimulationClock`) therefore
  cannot get a LEAN fill: every intent is refused as expired (fail closed). The replay prices (e.g. SPY ≈ 722) also
  differ from `LatSimulatedDataQueue`'s fixed quotes (SPY 450.00). Fix belongs to `services/api` (requested): when
  `LAT_GATEWAY` is a LEAN gateway, stamp intents with the wall clock and use a wall-clock quote source, or refuse to start.
* `LatSimulatedDataQueue` prices are fixed; PaperBrokerage fills are LEAN's simulation. Nothing here predicts Alpaca
  paper or live fills.
* LEAN's dependency closure contains packages with NuGet advisories (see `docs/sources/lean-pins.md`).

## Tests

```bash
scripts/lean_build.sh                                       # includes dotnet test (84 NUnit tests)
uv run --frozen pytest tests/unit/lean -q                   # 54: paper-only proofs, pins/lock sync, vectors, stub, tooling
uv run --frozen pytest tests/integration/lean tests/failure_injection/lean -q   # 15: real LEAN live-mode runs (SIMULATED)
uv run --frozen pytest tests/e2e/lean -q                    # 1: real LEAN <-> real order ledger (SIMULATED, ~60 s)
```
