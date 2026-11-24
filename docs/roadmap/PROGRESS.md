# Roadmap progress

Checkpoint for the [roadmap](README.md). Each entry says what exists, how it was verified, and what is still open.
Simulated results are labelled; nothing here is evidence of profitability.

## Phase 1 — position and exit management: implemented, verified in simulation; real-paper test written, not run

Design: [position lifecycle](../architecture/position-lifecycle.md).

* `services/positions` (plan store, append-only action log, `PositionManager`), runtime task `position-manager`.
* Broker-held protective STOP_LIMIT SELLs through signed intents (`lat-intent-v2` signs the stop; LIMIT intents keep
  `lat-intent-v1` byte for byte), FakeBroker trigger simulation, Alpaca `stop_limit` submit/mapping, ledger anomaly on
  a broker stop that differs from the signed one.
* Exits: EOD flatten (10 min before the close), overnight liquidation at the next open, time exit (signal horizon),
  stop breach escalation (gap through the stop-limit), local stop when no broker stop rests; cancel-then-exit; re-price
  a tick lower without tripping the duplicate window.
* Partial fills (protect filled qty, top up, cancel entry remainder), cancel/fill races, conflicting external SELLs.
* Exit lock vs entry lock; ENTRIES-scope halts and position safety states lock entries only; ALL-scope kill switch
  (`POST /api/v1/control/kill-switch`, operator token) blocks exits too. No entries in the last 30 min.
* Alpaca SELL preflight against the broker's own position (no accidental short in a margin paper account).
* Explicit safety states (UNPROTECTED, EXITING, BREACHED, OVERNIGHT, UNVERIFIED) lock new entries; realized P&L per
  round trip; position risk-to-stop view (`GET /api/v1/positions`, `get_portfolio`, `get_risk_status`).

Verification (2026-11-20, SIMULATED unless stated): `tests/unit/positions` 22, `tests/integration/positions` 14,
`tests/failure_injection/positions` 6, `tests/e2e/api/test_e2e_position_lifecycle.py` 1 — all pass.
`tests/integration/positions/test_position_lifecycle_paper_broker.py` places real PAPER orders; it is double-gated
(`-m paper_broker` + `LAT_PAPER_ORDER_TESTS=1`) and has **not been run**: it needs market hours and a flat paper account,
and its orders would appear as UNKNOWN_ORDER to the main paper ledger's reconciliation (see its docstring).

Continuation (2026-11-20, SIMULATED): fixed unfilled entry cancellation at the configured liquidation cutoff,
including early closes, even before the entry timeout expires. Cancellation actions now distinguish `ENTRY_EOD`
from `ENTRY_TIMEOUT`. Flat symbols with overdue `CANCEL_PENDING` orders request broker-authoritative reconciliation
instead of waiting indefinitely for a lost cancellation event. Five new regression cases first failed on the old
implementation, then passed: normal/early close with and without a fill racing cancellation, plus a dropped cancel
event recovered without duplicate cancellation. Position unit/integration/failure-injection/API lifecycle suite:
**48 passed, 1 real-paper test skipped**. Broader ledger/risk unit and position integration/failure-injection suite:
**552 passed, 1 skipped**. Targeted Ruff and Pyright checks passed. Real-paper verification remains open.

## Phase 2 — historical data pipeline: implemented; first real run done

Design and evidence: [data pipeline and parity](../architecture/data-pipeline-and-parity.md).

* `HistoryPipeline` (plan/fetch/validate/store/snapshot/trigger), CLI `pipeline | snapshot | compare-feeds`, make
  targets, optional runtime task (`LAT_DATA_PIPELINE_ENABLED`).
* Immutable research snapshots (constituents + knowledge cutoff + content hash + lineage + coverage), resolved by the
  store wherever a dataset version is accepted; research picks the pipeline's *ready* snapshot.
* Real run (Alpaca IEX, free): 200/1,024 windows, 618,514 bars, 24 symbols, snapshot `rs_95135dab40bccd81`.
  Open: ~824 older-history windows remain (run `make data-pipeline` about five more times, or enable the runtime task).
  Corporate actions are not yet ingested by the pipeline for the new symbols (research `_adjust` reports RAW prices).
* IEX vs SIP comparison (real, read-only): see the report. No paid feed purchased or enabled.

## Phase 3 — prediction parity: core implemented

* `services/research/decision.py` shared contract (GROSS/NET basis, cost charged once, explicit uncertainty buffer and
  threshold). Fixed the autonomous paper model's full double charge and the suite adapter's double-charged impact.
* `QuantSignal.uncertainty_bps`; `net_edge_bps` is now gross − live cost − buffer.
* `tests/unit/research/test_decision_parity.py`: research-panel features + model == runtime `PredictionService` on PIT
  bars (fitted LightGBM 60-min model), engine net edge == research net forecast at the reference notional.
* Open: walk-forward selection still scores at a 0 bps threshold; serialized-artifact parity for autonomous models is
  covered by the existing `test_actual_training_and_portable_parity` (passes in its own process, see Phase 7).

## Multi-asset program (2026-11-23) — implemented; development research run on real data; nothing promoted

Design: [multi-asset research](../architecture/multi-asset-research.md); operations: [runbook](../runbooks/quantlab.md);
results: [verification report](../verification/multi-asset-research-2026-11-23.md).

1. **Alpaca capabilities verified** (`services/capabilities`, GET-only): per asset class API / data / account /
   policy layers. Research-eligible: equities long/short, margin, leveraged ETPs, crypto, options (incl. SPX/XSP/VIX
   European), extended hours. Execution-eligible: U.S. equity long only. OPRA and real-time SIP not entitled. PDT
   retired by FINRA (2026-06-04); Alpaca fields deprecated.
2. **Safety-critical lifecycle**: no new gap blocks today's execution path (intraday equity long; reconciliation,
   protective stops, recovery unchanged). Open, *evidence-gated*: a multi-day holding lifecycle (re-arming DAY
   stops each session, horizon in sessions) is required before any multi-day family could reach PAPER; a crypto
   execution adapter (fractional qty, GTC/IOC, 24/7, symbol format) before any crypto strategy could. Neither was
   built because no family passed validation. Real-paper lifecycle test still not run (needs market hours).
3. **Point-in-time datasets**: 20,273 equity symbols incl. delisted (corporate actions), 24.1M raw / 17.7M adjusted
   daily SIP bars, panel 2,738 × 10,174; crypto 33 pairs; stocks-in-play 5-minute bars + pre-open news; delta
   updates with adjusted-basis rescaling. Survivorship residual: failures without corporate-action records.
4. **Unified contracts**: `Opportunity` (gross/cost/net, std, CVaR, capital, BP, exposure multiple, capacity, fill
   probability, evidence, executability) and evidence-weighted mean-variance utility; cost models per asset class.
5–9. **Families A, A2, B, C1–C3, D1–D2, E, F, G** implemented and evaluated after costs with walk-forward, DSR (all
   trials), SPA, PBO, 2× cost stress, alpha t, BH-FDR. **None passed** (see report); the holdout stays unopened.
   Controlled comparisons: LightGBM > ridge (same features/folds) but not > SPY; IBIT > BTC/USD for the same trend
   rule; open vs close execution (reversion decays with a later fill).
10. **Reversible promotion**: registry (append-only transitions), shadow journal, PAPER checks with explicit
    blockers, automatic suspension on drawdown breach. Empty today (correct under the evidence).
11. **Interfaces/tests**: read-only `GET /api/v1/quantlab/*`, MCP tools (3 read + bounded experiment), dashboard
    panel; unit + integration tests for every module (synthetic data only); full suite run.

## Phase 4 — opportunity analysis: started

* `services/research/opportunity.py` + `GET /api/v1/research/no-trade`: every NO_TRADE cycle attributed to LOCKED,
  DATA, NO_FORECAST, PREDICTION, COSTS, UNCERTAINTY, THRESHOLD, RISK or NO_SIGNALS; near misses; cost/edge medians.
* Open: cross-sectional/sector ranking, regime-aware selection, calibration, 5/15/60 horizon experiments, symbolic
  regression / GP / Qlib / RD-Agent(Q) discovery, ensembles, mean-reversion / relative-value / event families.

## Phase 5 — full-lifecycle backtesting: not started

Depends on Phase 1–4 pieces now in place: replay the signal engine + `decide_edge` + risk engine + position manager
over a snapshot with a fill simulator; compare with paper orders.

## Phase 6 — autonomous research improvement: not started

## Phase 7 — reliability: started

* `.github/workflows/ci.yml` (ruff, pyright, contract freshness, tests, `pip-audit`) — written, **not yet run on
  GitHub**. pyright now excludes vendored Kronos code (as ruff already did).
* Fixed test-runner hang: torch tests (`test_kronos.py`, `test_advanced_models.py`) deadlocked at 0 % CPU when run
  late in one long pytest process (it already hung before any roadmap change); in their own process all 20 pass in
  ~7 s. `make test-py` and CI now run them in a separate pytest invocation. Root cause (likely duplicate OpenMP
  runtimes / thread pools between torch and the other native ML libraries) is not yet isolated.
* Dependency audit 2026-11-20: 111 locked Python packages (`pip-audit`) and the dashboard's npm lockfile
  (`pnpm audit`): no known vulnerabilities.
