# trading/ progress (LEAN + Alpaca plugin component)

Attempt 3 started 2026-11-06. Inventory of prior (unverified) work:
- LEAN clone d1c09284 clean; Launcher/bin/Release built 2026-11-05 (log var/logs/lean-build-initial.log, 0 errors).
- alpaca_plugin/source-build csproj (plugin compiled against pinned LEAN source projects) + bin/Release present.
- LatStrategies/Gate/{SignedIntent,IntentSignatureVerifier,IntentToken,IntentGate}.cs present (reviewed: sound).
- Everything else (algorithms, models, bridge, tests, scripts, configs, docs) missing.

Requirement log:
- [3a-c code] LatStrategies (gate, journal, fee/slippage/fill models, LatBrokerageModel, GatedBrokerageModel, GatedPaperBrokerage(+Factory), bridge client, sim data queue, baseline + bridge algorithms), LatAlpacaGate (GatedAlpacaBrokerage, paper hard-coded), LatTestAlgorithms (rogue) all compile (dotnet build -c Release, 0 errors). Not yet run.
- FINDING: official Alpaca plugin AlpacaBrokerage.Initialize() calls ValidateSubscription() -> api.quantconnect.com module licence (product 347), Environment.Exit(1) on failure. Not patched (owner decision). Alpaca path BLOCKED on paper keys AND QC licence.
- [4] NUnit trading/lean/LatStrategies.Tests: vectors from Python IntentSigner (trading/lean/tools/gen_intent_vectors.py -> vectors/intent_vectors.json), gate refusals, GatedBrokerageModel w/ real LEAN Equity/LimitOrder, fee/slippage/fill arithmetic, bridge settings/payloads. `dotnet test ... -c Release` Passed 79/79.
- [1] trading/PINS.json written (LEAN/plugin SHAs, SDK 10.0.403, package versions, source-build rationale). docs/sources/lean-pins.md: TODO.
- [2] scripts/lean_build.sh: verifies pins/clean trees, builds Launcher+plugin(source)+Lat* projects, assembles trading/lean/bin/plugins (plugin-directory), runs dotnet test. Ran OK (var/logs/lean-build.log, 79/79 tests).
- [5] scripts/lean_backtest.sh: REAL LEAN Launcher runs: Daily SPY 2010-2020 (43 orders, ret 0.7013, fees 49.49) + Minute SPY 2013-10-07..11 (30 orders, fees 14.57) -> experiments/lean/<run_id>/summary.json + lean-log-excerpt.txt. data-local absent at this time. Fixed: LatBrokerageModel must derive DefaultBrokerageModel (LEAN GetBrokerageName threw on result save).

Attempt 4 (2026-11-07):
- [2] lean_build.sh re-run OK (incremental), dotnet test 79/79. [5] lean_backtest.sh re-run: daily 43 orders ret 0.701311 fees 49.49; minute 30 orders fees 14.57 (new run ids 20261107T224221Z).
- [3c/6] FIX: rogue algo raced LEAN (ticket OrderSet fires before CanSubmitOrder) -> now waits; GatedBrokerageModel refusal of unsigned strategy order observed live. FIX: duplicate/replay gate refusals no longer carry ids (could reject a live original in the ledger); bridge ignores duplicate deliveries (unkeyed report); unsent claimed intents reported at stop. Harness: +signed MARKET, wrong gateway, duplicate delivery; per-case reason checks. bridge + rogue live-sim runs completed=True.
- TODO: pytest lean tests, lean_live_sim.sh, paper template/launcher/test, lean-pins.md, README, quality gate.
- [6] scripts/lean_live_sim.sh + tests/integration/lean/test_lean_live_sim.py (pytest -m lean) + tests/failure_injection/lean/test_lean_engine_faults.py: 3/3 passed (60 s): event-post outage retried in order; restart replay refused via journal (unkeyed, no event under original ids); wrong token claims nothing; missing pubkey fails closed. FIX: outbox JSONL had UTF-8 BOM. +BridgeClientTests (NUnit 82/82).
- [7] trading/lean/config/live-alpaca-paper.template.json + scripts/lean_paper_alpaca.sh (REFUSED rc2 non-PAPER/overrides; BLOCKED rc3 no keys; then QC licence creds + LAT_QC_LICENSE_ACK; 0600 cfg deleted on exit). tests/unit/lean 49 passed (paper config, no-live grep, pins/lock sync, vectors in Python, stub). PINS.json + resolved_nuget_packages (40, no conflicts).
- ruff + pyright clean on owned python. TODO: lean-pins.md, README, data-local, cleanup old runs, full re-run.
- [5] backtest runner: data-range helper + provenance (lat-export-manifest.json -> data_provenance, synthetic forced is_simulated). Ran on SYNTHETIC export of ds_1c99c07583eae908 (scratch copy; data-local still absent): 100 orders, fees 56.24.
- [1][8] docs/sources/lean-pins.md, trading/README.md written. PINS.json enriched (mapping, licence check, discovery).
- FIX: journal now mandatory in LatGateFactory (NUnit 83/83). lsof sampling: LEAN live-sim holds loopback sockets only. Ledger contract test added.
- Attempt 5 (2026-11-07/08): lean_build.sh re-run found 1 failing NUnit test (asserted STOPPED heartbeat from a sink-only client; code correctly sends it only with the poller). Test fixed + new poller heartbeat test. dotnet test 84/84 (x4).
- [6] re-verified: pytest tests/integration/lean tests/failure_injection/lean 15 passed (115 s); tests/unit/lean 49 passed. NEXT: e2e real LEAN <-> real order_ledger bridge functions; lean_live_sim.sh api mode (real local API).
- [6+] NEW tests/e2e/lean/test_lean_e2e_ledger.py: REAL LEAN live (SIMULATED) <-> REAL services.order_ledger bridge fns via 127.0.0.1 shim + ExecutionService/LeanBridgeGateway/Reconciler: BUY filled->ledger FILLED, oversell refused by gate->REJECTED, close SELL filled, reconcile CLEAN, STOPPED hb->unhealthy. 1 passed (78 s).
- [6+] NEW scripts/lean_live_sim.sh api (+ lat_lean live-sim-api-config): SIMULATED LEAN for the REAL services/api. Smoke run vs real API (scratch LAT_PROJECT_ROOT, port 18765, LAT_GATEWAY=LEAN_PAPER_SIM): heartbeat RUNNING seen by /api/v1/status, reconcile CLEAN, agent propose_paper_trade -> risk APPROVED -> LEAN claimed -> GatedBrokerageModel refused 'intent expired' (API SimulationClock 2026-11-02 vs LEAN wall clock) -> ledger REJECTED. FINDING for integrator. API+LEAN stopped.
- [5] re-run scripts/lean_backtest.sh: daily 43 orders ret 0.701311 fees 49.49; minute 30 orders ret -0.003571 fees 14.57 (identical, deterministic). Pruned superseded run outputs. data-local still absent.
- [5] data-local now exists (exported via services.market_data.cli export-lean, ds_1c99c07583eae908); default runner's data-local run switched to Minute SPY; run lean-bt-synthetic-minute-20261108T034357Z: 100 orders fees 56.24 is_simulated=true.
- [8] README updated (84 NUnit, 54 unit, 15 integ+FI, 1 e2e; api mode; SimulationClock finding; new run ids). ruff + pyright clean on trading/lean/tools + tests/*/lean. harness _lean_env -> lean_env (public).
- FINAL (attempt 5): lean_build.sh OK (NUnit 84/84); pytest tests/unit/lean tests/integration/lean tests/failure_injection/lean tests/e2e/lean 70 passed (163 s); ruff/pyright clean; paper launcher rc=3 BLOCKED (no keys), rc=2 for LAT_MODE=LIVE_APPROVAL / unknown args. No processes left running.
