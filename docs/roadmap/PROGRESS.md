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

## Stocks/ETF program (2026-11-24) — session timing, hypothesis memory, after-close automation

Scope narrowed by the operator to U.S. stocks and ETFs (no new crypto/options work; existing research-only modules
untouched). Results: [session-timing report](../verification/session-timing-research-2026-11-24.md).

**Verdict (development period 2016-01-04 … 2025-11-20, real Alpaca SIP data, runtime cost model): no family passed;
holdout not opened (0 vault looks); registry empty; selection stays `NO_TRADE`.** 376 trials recorded. Walk-forward
OOS Sharpe: H1 −0.48, H2 −1.10, H3 −0.96, I1 −0.08, I2 −0.18 (gross edges 0.2–5.9 bps per trade, not significant,
below ≈ 7 bps modelled cost; I2 21 bps gross but PBO 0.61). J_vol_managed (QQQ, 15 % target vol): Sharpe 1.14 vs
SPY 0.89 / QQQ 0.95, max DD −23.5 % vs −33.8 % / −35 %, DSR 0.991 — fails on edge vs benchmark (SPA p 0.23,
alpha t 1.81) and PBO 0.54; a risk-reducing overlay, not a validated edge. Open, evidence-gated: prospective
journal for near-miss candidates (J; the AAPL open→10:00 drift found descriptively); a strategy-horizon exit for
multi-day plans (only needed if an overnight rule ever validates); measured slippage needs live fills (not
authorized).

* **Multi-day lifecycle (already in the tree, now recorded here)**: commit `2abb1ce` added the opt-in multi-day
  equity lifecycle (`RiskPolicy.allow_overnight_positions`, `max_holding_sessions`, `overnight_gap_budget_bps`,
  `multi_day_strategy_allowlist`; `holding_period_authorized` check incl. final gate; session deadline fixed once,
  DAY stops re-armed at the open, gap-through-stop exit, `POLICY_REVOKED` exits). Defaults keep it disabled.
  Verified 2026-11-24 in the baseline run: `tests/unit/risk/test_holding_period.py`,
  `tests/integration/positions/test_multi_day_lifecycle.py` pass (backend baseline 2,179 passed, 1 skipped).
* **Session-timing research** (`services/quantlab/intraday.py`, `strategies/session.py`): 5-minute SIP panel for the
  24 allowlisted symbols (dev period only; the fetch end date is clamped before the holdout), a simulator that
  mirrors the runtime lifecycle (marketable LIMIT entries with non-fills, no entry in the last 30 minutes, flatten
  10 minutes before the close incl. early closes, protective stop on every position incl. overnight gap exits,
  runtime cost model once per leg, one position per symbol, concurrency cap), families H1–H3 (intraday) and I1/I2
  (overnight / turn-of-month), execution tier under today's policy, trade-level diagnostics, descriptive time-of-day
  profile, observed SIP NBBO spreads vs the cost model. CLI: `session-fetch`, `build-panel --what session`,
  `session-profile`, `research --asset session`, `session-diagnostics`, `spread-sample`, `report --session`;
  `make quantlab-session`.
* **Hypothesis memory**: `TrialLedger.prior_outcomes/evaluated_with`; agent factor specs already evaluated on the
  same development data are refused (`DUPLICATE_HYPOTHESIS`, not counted); family re-runs on identical data+code are
  skipped (`--force` to override). Development-content fingerprint (`EquityPanel.content_fingerprint`) so a delta
  that only adds post-holdout rows is not mistaken for new evidence.
* **After-close automation** (`services/quantlab/daily.py`, opt-in `LAT_QUANTLAB_DAILY_ENABLED=1`): the API schedules
  fetch-delta → build-panel → scan → shadow once per completed session in `nice`-d child processes with timeouts,
  only in the quiet window (close + 30 min … 90 min before the next open, re-checked before every step), at most
  60 data requests/minute, ≤ 3 attempts per session, wall-clock re-checks every ≤ 30 min (a sleeping Mac stops the
  monotonic clock), the child killed if the task is cancelled. Durable state prevents repeats after restarts; a
  missed period is caught up once. Default off.
* **Independent review before any trial was recorded** (2 rounds, 6 read-only reviewers; research was paused until
  it finished). Fixed: multi-session P&L was booked in one lump on the exit day (biased beta → 0 and inflated alpha t,
  hid drawdowns) → marked to market every session; stops filled at the stop even when a bar traded through the
  STOP_LIMIT's limit → `STOP_BREACH` at min(limit, bar close); I1's next-open exit is not something the multi-day
  lifecycle can do → added the executable `next_eod` form and `NOT_EXECUTABLE`/`NOT_PERMITTED` execution tiers;
  family re-run skip could hide an interrupted family and shrink BH-FDR → run-completion markers and BH across all
  recorded families; incomplete skip keys (ORB/news, crypto spreads, daily panel, cost/stats code); development
  fingerprint unstable under adjusted re-basing and new listings → raw levels + adjustment events per present
  symbol; whitespace-only factor canonicalization → AST canonical form with legacy-spelling matching; holdout path:
  separately back-adjusted downloads → overlap re-basing, look id validation, resumable looks with a process lease,
  atomic result write; all-skipped runs wiped the disclaimer. Every fix has a regression test (one verified by
  mutation).

## Ranking / cost-aware program (2026-11-25) — family K implemented and run on development data; nothing promoted

Prompted by an external research brief (ranking objectives, 1/5/20-session horizons, cost-aware selection,
calibration, originality-screened factor discovery). Report: [xs-rank research](../verification/xs-rank-research-2026-11-25.md).

**Verdict: K_xs_rank fails the development gate (DSR 0.195, WF OOS Sharpe 0.35 vs SPY 0.76, SPA p 1.00, alpha t
−1.73, WF max DD −45.5 %); holdout not opened (0 looks); registry empty; selection stays `NO_TRADE`.** 30 trials.
Ranking objectives did not beat LightGBM regression (LambdaRank/pairwise IC significantly lower at h=1; YetiRank
ties). Rank IC is small and persistent at h=5 (0.016–0.024, t ≈ 3, all years positive for ridge/YetiRank) but the
top decile earns 4–9 bps vs ≈ 13 bps round-trip cost; h=1 top-N loses 8–17 %/yr to costs; best top-N (LightGBM
reg h=20) Sharpe 0.66 vs SPY 0.76. The cost-aware rule bought no stock in 13 of 15 variants (held SPY).

* `services/quantlab/ranking.py`: executable open(t+1)→open(t+1+h) labels, group demeaning, relevance grades, rank
  IC, decile spreads, paired tests, top-set turnover.
* `services/quantlab/calibration.py`: Fama-MacBeth binned edge with date-level standard errors (overlap-adjusted),
  isotonic smoothing; Mondrian split-conformal bands and realized coverage.
* `services/quantlab/portfolio.py`: cost-aware selector (edge lower bound vs round-trip cost at intended size,
  held names exempt from re-entry cost, unused slots in SPY) and round-trip cost surface.
* `services/quantlab/strategies/xs_rank.py`: family K (ridge, LightGBM reg, LightGBM lambdarank, XGBoost
  rank:pairwise, CatBoost YetiRank × h 1/5/20 × top_n/cost_aware), 24 PIT features incl. inferred-sector-relative
  returns, purged walk-forward per year, diagnostics with SPY / equal-weight / sector-momentum / cash benchmarks.
  CLI `research --asset xsrank`, `xsrank-diagnostics`; `make quantlab-xsrank`; `holdout --family` now dispatches
  G and K (G previously failed with StopIteration).
* `services/quantlab/novelty.py`: agent factors refused as `REDUNDANT_HYPOTHESIS` (no trial) when |mean rank
  correlation| ≥ 0.9 with a reference library of already-searched signals or the 25 latest agent factors (≈ 5 s on
  the real panel); accepted experiments report their novelty.
* Interfaces: `GET /api/v1/quantlab/research?family=K_xs_rank` and `get_research_results(family=K_xs_rank)` add a
  compact `diagnostics` digest (optional field in the free-form `detail`; no schema change).
* Verification (SIMULATED unit/integration): 27 new tests; `make test-py` 2,257 passed, 1 skipped (+20 torch);
  `make lint`, `make typecheck` clean.
* Open, evidence-gated: new information sources with availability timestamps (SEC EDGAR filings / XBRL
  fundamentals — needs an operator SEC User-Agent and download authorization); newer foundation models (TimesFM
  3.0, Toto 2.0, TabPFN 3.5) only after licence review and only if a cheaper model shows an edge to improve.

## Phase 4 — opportunity analysis: started

* `services/research/opportunity.py` + `GET /api/v1/research/no-trade`: every NO_TRADE cycle attributed to LOCKED,
  DATA, NO_FORECAST, PREDICTION, COSTS, UNCERTAINTY, THRESHOLD, RISK or NO_SIGNALS; near misses; cost/edge medians.
* Cross-sectional ranking, calibration and multi-horizon experiments: done for daily stocks (family K, above).
* Open: regime-aware selection, symbolic regression / GP / Qlib / RD-Agent(Q) discovery, ensembles, relative-value /
  event families.

## Phase 5 — full-lifecycle backtesting: harness implemented (2026-11-25), verified in simulation

Design: [replay](../architecture/replay.md). `services/replay` runs the production DecisionCycle, `decide_edge`,
RiskEngine (evaluate, authorize and final_gate with real signatures, using a temp key), the OrderLedger claim,
ExecutionService, Reconciler and PositionManager over a bar snapshot on a simulated clock. It does this in a fresh,
empty temp workspace. The narrow adapters are a quote feed synthesized from bars and a next-open fill model with a
seeded non-fill/partial-fill RNG.

The harness writes an append-only `REPLAY` journal with provenance and supports resume. It checks parity against
`quantlab.intraday`. `python -m services.replay evaluate --policy default|active|FILE --equity N` (`make
replay-eval`) evaluates a strategy under an exact account and policy (audit P1), and a read-only paper-ledger
comparison hook is included.

Built by a Claude worker and reviewed by a Codex worker. The review fixed workspace aliasing, crash-resume
duplicates, the next-open fill race, a cost escape on reconciled fills, marks and exposure, snapshot validation,
and read-only URIs. Verification (SIMULATED: synthetic bars, FakeBroker, fixtures): 74 replay tests pass, and
synthetic parity matched 5/5 trades within about 1 bps. Open: no comparison against a real paper ledger, and
trained-model suites are not yet replayable.

## Phase 6 — autonomous research improvement: research crew implemented (2026-11-25)

Design: [research crew](../architecture/agent-crew.md). `services/agents` coordinates the operator's own Claude Code and
Codex CLIs on subscription sign-in. It covers hypothesis, cross-provider review, market analysis and improvement
tasks. It runs as a separate process with no execution authority (`make agents`, `make agents-once`,
`make agents-status`).

* Agents get only the read-only `research` MCP profile (`python -m services.mcp_server --profile research`):
  quantlab views plus `search_memory`, in-process, with no API or agent token. Completed holdout outcomes are
  withheld from agents. The crew validates agent specs with the closed DSL and runs them with
  `run_agent_experiment`, which applies the same budget, novelty refusal, gates and trial ledger as every other
  family. The crew stores the ledger's result, never the numbers an agent reports.
* Quota: one agent at a time, a daily per-provider cap, no run without new evidence, and a check for experiment
  headroom before dispatch. A usage limit, auth failure or missing CLI requeues the task without consuming an
  attempt. The provider then cools down until the stated reset time, or for an exponential backoff when no reset is
  stated, and the other provider takes the work. Tasks left `RUNNING` recover on restart, and in-flight CLI process
  groups are killed on shutdown.
* Memory: `experiments/agents/crew.sqlite3` holds append-only notes labelled `AGENT_NOTE` and shared across
  providers. Prompts receive them as sanitized digests.
* The work was built with Orca workers: Codex built the providers and the MCP profile, Claude built the crew core.
  Each then reviewed the other's code. The Claude review fixed false usage-limit classification of agent prose and
  "connection reset". The Codex review fixed QuantLab child roots, credential-like and multiline note injection,
  headroom accounting, stale claims, duplicate reviews, process-group cleanup and Codex MCP tool approval.

Verification (2026-11-25): `tests/unit/agents` + `tests/unit/mcp_server` + `tests/unit/tools` gave 460 passed, plus
the 2 baseline `test_mcp_registrations` failures. Those come from the tracked `.mcp.json` and `.codex/config.toml`
pointing at `/Users/victor/Developer/ai-trading`, not at this worktree. `make lint` passed, pyright on the changed
packages was clean, and `make smoke` passed with 29 audit rows. One real `run --once` per provider in a disposable
worktree with no data panel completed with `OK` and schema-valid output, a valid DSL spec, and a deterministic
experiment that returned `ERROR: no panel`. The Claude task chose momentum and the Codex task chose reversal. The
real trial ledger was not touched. Open: no crew campaign has run on the real development panel yet. Codex agents
can still read files that the user can read (the shell is disabled and writes are sandboxed), so holdout integrity
rests on the holdout ledger. The quantlab hourly admission check can race other concurrent callers.

## Phase 7 — reliability: started

* Local validation replaces remote CI (2026-11-25). The application is local and single-user, so the GitHub
  Actions workflow was removed. `make validate` (`scripts/validate.sh`, [runbook](../runbooks/validation.md)) runs
  ruff, pyright, contract checks, both pytest invocations, the isolated startup smoke, the LEAN tests and the
  `pip-audit` dependency audit locally. pyright excludes vendored Kronos code, as ruff already did.
* Fixed test-runner hang: torch tests (`test_kronos.py`, `test_advanced_models.py`) deadlocked at 0 % CPU when run
  late in one long pytest process (it already hung before any roadmap change); in their own process all 20 pass in
  ~7 s. `make test-py` and `make validate` run them in a separate pytest invocation. Root cause (likely duplicate OpenMP
  runtimes / thread pools between torch and the other native ML libraries) is not yet isolated.
* Dependency audit 2026-11-20: 111 locked Python packages (`pip-audit`) and the dashboard's npm lockfile
  (`pnpm audit`): no known vulnerabilities.

## Autonomy program (2026-11-25): autopilot, EDGAR point-in-time data, replay, local validation

The operator asked for a system that orchestrates research and paper trading by itself with no commands. Built
with Orca workers: Claude built the autopilot and replay, Codex built EDGAR and the validation runner, and each provider reviewed
the other's code.

* **Autopilot** (`services/autopilot`, [design](../architecture/autopilot.md), [runbook](../runbooks/autopilot.md),
  `make autopilot`). A deterministic XNYS daily supervisor. Before the open it checks health and makes one
  idempotent `start_paper_session`. Intraday it makes one bounded read-only check every 15 min. After the close
  it runs subprocess research in this order: the quantlab daily pipeline, research-crew ticks (Claude and Codex),
  shadow, and at most one vault-enforced holdout look. It writes a daily JSON/MD report.
  * It holds only the agent token and calls only allowlisted tools and argv.
  * It never selects Live, changes policy, clears halts or touches the kill switch. It stands down on any halt
    until the operator clears it.
  * Off by default (`LAT_AUTOPILOT_ENABLED`).
  * The Codex review fixed late or duplicate starts, a halt missed after the close, the subprocess boundary,
    linked token and state files, and holdout crash recovery.
  * 73 tests pass, and the integration test reached STARTED against a real API in a SIMULATED workspace.
  * Open: the API can activate an already-queued session after its grace period if the Mac sleeps across the
    open. That needs a session-manager contract change. Alerts are local only.
* **SEC EDGAR point-in-time data** (`services/market_data/edgar`, `services/quantlab/events.py`,
  [design](../architecture/edgar-pit-data.md)). Acceptance time is mapped to the first tradable open, and
  amendments are versioned. Covers XBRL fundamentals, 8-K 2.02 and 10-Q/10-K earnings events, and Form 4
  open-market P/S.
  * Features: SUE, PEAD age, accruals, asset growth, gross profitability, net issuance, insider net-buy ratio and
    cluster buys. They are in the closed DSL, and the crew's earnings, quality and insider mechanisms are now
    expressible once data exists.
  * Family `L_events` (`research --asset events`) compares price-only, price+fundamentals, price+insider and all
    features after costs.
  * The Claude review fixed Form 4 issuer mismatches, free-text symbols, and a planner evidence hash that ignored
    new EDGAR data. The coordinator made the submissions `acceptanceDateTime` fallback take the later of the UTC
    and Eastern readings.
  * Not yet run on real data. Downloads need `LAT_SEC_USER_AGENT` ("Name email", SEC fair-access policy), which
    only the operator can set.
* **Local validation** (`make validate`, `scripts/validate.sh`, [runbook](../runbooks/validation.md)). There is no
  remote CI. The two path-dependent MCP registration tests are fixed, and
  `scripts/register_mcp.sh --write-project-configs` regenerates the tracked configs for any checkout.
* **Paper risk profile** [`paper-autonomous`](../../ops/policy/README.md): 10 slots of 12 %, $600 per position,
  200 liquid names picked from development-period data only, $150 daily loss, a 15 bps spread cap, and no
  overnight holding. It is validated against the hard ceilings. Applying it is an operator action
  (`PolicyStore.apply_change`).
* **Verification (2026-11-25, SIMULATED):** `make test-py` gave 2,469 passed, 28 skipped, plus 20 torch tests.
  `make lint`, `make typecheck` and `make smoke` (29 audit rows) are clean.
