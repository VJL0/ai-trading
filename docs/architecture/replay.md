# Execution-equivalent replay (roadmap Phase 5)

Research backtests (`services/quantlab`) and the real order path are different code. `services/replay` replays the
**real** decision and execution stack over a point-in-time bar snapshot on a simulated clock. The replay runs in a
disposable workspace that has its own ledger, signing key, operator token and FakeBroker state. It then evaluates a
strategy under the exact account size and risk policy that would trade it.

Everything the harness produces is labelled `REPLAY` (journal, summary, CLI output). The runtime objects
underneath label their fills `SIMULATED`. Quotes are synthesized from bars. Replay output is not evidence of
profitability and not evidence of real execution quality.

## Components: real vs adapter

| Stage | What runs | Real or adapter |
|---|---|---|
| Signals and prediction | `QuantSignalEngine.generate` → `decide_edge` (costs charged once, threshold, uncertainty) | real; the strategy supplies the `SignalModel` |
| Decision cycle | `services.api.pipeline.DecisionCycle` (snapshot, evidence, `_quant_proposal` sizing/limit/stop, risk, intent, submit, reconcile) | real; `ReplayDecisionCycle` only sets the proposal's `strategy_id` and `holding_sessions`, then re-validates |
| Risk | `RiskEngine.evaluate` → `authorize` (ECDSA-signed intent; key generated in the workspace) → `final_gate` | real |
| Order path | `Runtime.execute_proposal`, `risk_context`, `_daily_position_pnl`, `prepare_gate`, `_final_gate`, `OrderLedger.record_intent` / `claim_for_submission`, `ExecutionService.submit` | real (inherited Runtime methods) |
| Policy | `PolicyStore` in the workspace; a profile is installed through its own audited `apply_change` | real |
| Locks and health | `EntriesLock` (entry and exit locks, startup guard), `FeedQualityMonitor`, `Runtime.health` | real |
| Reconciliation | `Reconciler.startup_recovery` / `reconcile` (after every cycle, every 15 min, and at the close) | real |
| Positions | `PositionManager` via the runtime's own `RuntimePositionPort` (protective STOP_LIMIT, partial fills, entry timeout, EOD and TIME exits, multi-day re-arm, stop breach, gaps) | real |
| Broker | `ReplayBroker(FakeBroker)`: entry fill model only | **adapter** (below) |
| Market data | `ReplayFeed`: `load_bars_df` (store surface), `latest_quote_asof` / `get_bars_asof` (provider surface for `build_market_snapshot`) | **adapter** |
| Composition | `ReplayRuntime(Runtime)` | **adapter** (below) |
| Event stream | broker events applied synchronously (`pump`) after every broker call, quote, manager tick and cycle | **adapter** (the runtime uses an asyncio consumer task) |

### Adapters and why each is needed

* **`ReplayRuntime`** subclasses `services.api.runtime.Runtime` and deliberately does not call `Runtime.__init__`.
  That constructor configures process logging, reads the keychain, and builds the AI pipeline, the session
  scheduler and the bundled synthetic fixture. Instead, `ReplayRuntime` builds the same classes with the same
  arguments on the workspace (mode `REPLAY`, gateway `FAKE`, client order ids `lat-replay-*`). It inherits the
  order-path methods unchanged. Its overrides are:
  * `current_signal_engine`: the strategy's model;
  * `_prediction_service`: returns `None`, because a workspace has no trained registry;
  * `run_cycle`: uses `ReplayDecisionCycle`;
  * `execute_proposal` / `reconcile`: call the inherited method, then pump broker events.
* **`ReplayFeed`**. A bar snapshot has no NBBO, so quotes are synthesized (feed `SYNTHETIC`):
  * The intrabar path is open (+1 s) → first extreme (+100 s) → second extreme (+200 s) → close (+299 s). The low
    comes first on an up bar and the high first on a down bar. The path is sampled every `quote_step_s`
    seconds (default 5).
  * The quoted spread is the runtime `CostModel` liquidity-bucket spread at the symbol's point-in-time ADV (prior
    completed sessions). Bid and ask are rounded outward to the tick.
  * The path jumps between bars, so a gap is never interpolated. Inside a bar the path is continuous: when it
    crosses a resting protective stop on the way down, an exact quote at the stop is inserted.
  * The runtime asks for `1Min` bars. The feed serves the snapshot's native 5-minute bars, the same grid as
    QuantLab's session panel.
* **`ReplayBroker`** is the unchanged FakeBroker except for entry matching (see the fill model below). It also
  writes its state file only when a quote changed broker state, instead of on every quote.

## Clock and loop

The clock is a `SimulationClock` (the runtime's) and only moves forward, following the XNYS calendar (holidays
and early closes come from `exchange_calendars`). Each session runs as follows:

1. For each 5-minute bar, the engine goes through the quote samples of every symbol in time order, with stop
   crossings inserted. At each sample it publishes the quote, lets the broker match resting orders and pumps the
   resulting events into the ledger.
2. `PositionManager.tick()` runs on every sample while the manager has work. That covers a working
   non-protective order, an unprotected or partly protected position, an exit in progress, an open plan with no
   position, or a bid within 0.2 % of a plan's stop. Otherwise it runs once a minute; EOD, TIME and holding
   deadlines all fall on minute boundaries.
3. At the bar end, the bar becomes available. If the strategy decides at that time, a decision cycle runs. The
   engine then tails the ledger into the journal and writes a `MARK`.
4. At the close, DAY orders expire, the manager marks overnight holdings, and a reconciliation runs, followed by
   `SESSION_END`.

The first `warmup_sessions` sessions (default 1) are history only: they supply lookback bars and ADV.

## Fill model

* **Entries** are marketable LIMIT BUYs against the next bar. An entry never fills at acceptance. It is matched
  once, against the first quote of its symbol after submission (the next bar's open):
  * `limit < ask` (the market gapped past the limit): `NON_FILL_LIMIT`. There is no second chance, and the
    manager cancels the order after `entry_fill_timeout_s`.
  * Otherwise the order does not fill with probability `entry_nonfill_prob` (`NON_FILL_RANDOM`). Failing that, it
    partly fills with probability `entry_partial_prob`, for `U(partial_fraction)` of the quantity (at least one
    share, at most qty − 1). Otherwise it fills in full. The price is `min(limit, ask × (1 + slippage))`. A
    partial remainder only completes through a cancel/fill race.
* **Exits and stops** use unchanged FakeBroker semantics. A SELL limit fills against the current or a later bid.
  A STOP_LIMIT triggers when the bid is at or below the stop, then rests as a limit. A gap below that limit
  leaves it unfilled, and the position manager escalates (`STOP_BREACH`).
* **Randomness** comes from a per-order generator seeded by `(seed, symbol, submitted_at, qty, limit)`. A run, a
  re-run and a resumed run draw identical outcomes. Matched entries are persisted beside the broker state.
* **Costs** come from the runtime `CostModel`, applied once per filled order leg (`LEG_COST`, when the order is
  terminal): fees plus fixed slippage plus square-root impact at point-in-time ADV. The half spread is not
  charged again, because it is already in the bid/ask fill price.

## Journal schema (`journal.jsonl`, append-only)

Every record has `seq`, `kind`, `at` (UTC ISO), `execution_label: "REPLAY"`, `runtime_label: "SIMULATED"`,
`is_simulated: true` and `provenance`. The provenance fields are the dataset version, the bar feed, the quote feed
(`SYNTHETIC`), whether the bars are synthetic, the content SHA-256, the source, and the quote method.

| kind | content |
|---|---|
| `RUN_START` | config (equity, policy version/source, fill model, position policy version, window), strategy, full policy, sessions, symbols |
| `RESUMED`, `STARTUP_RECONCILE` | restart point; reconciliation status/detail |
| `SESSION_START` / `SESSION_END` | session, close, early close; end equity, cash, positions |
| `DECISION` | causal id, outcome (`SUBMITTED`/`REJECTED`/`NO_TRADE`/`BLOCKED`/`ERROR`), reason, candidate, client order id, entry locks |
| `RISK_DECISION` / `REJECTION` | decision id, verdict, **failed check names**, reasons, policy version, symbol, side, type, qty, limit, stop, proposer, strategy |
| `INTENT` | signed intent recorded (intent id, key id, algorithm, expiry, policy version, mode, gateway) |
| `ORDER` | every ledger state transition (from, to, reason) |
| `ENTRY_MATCH` | fill-model outcome per entry (`FILLED`, `PARTIAL`, `NON_FILL_LIMIT`, `NON_FILL_RANDOM`) |
| `FILL` | fill id, order, qty, price, order type, protective-stop fill flag |
| `LEG_COST` | cost model charge per filled leg (fees, slippage, impact, spread already in fill) |
| `STOP` / `EXIT` / `CANCEL` / `POSITION_ACTION` | position-manager actions (protect/top-up, exit trigger/exit order, cancels, other) |
| `ROUND_TRIP` | closed position plan (entry/exit qty and averages, exit and close reason, realized P&L, holding sessions) |
| `MARK` | bar-end broker equity, cash, position value, positions, and the journal's resume cursors |
| `RUN_END` | completion |

`summary.json` holds the metrics below plus every round trip, the config, the strategy and the provenance.

## Restart

`ReplayEngine.run(resume=True)` on the same workspace and journal resumes an interrupted replay:

1. It reopens the ledger and the FakeBroker state.
2. It runs the runtime's startup recovery: SUBMITTING becomes UNKNOWN, and a broker-authoritative reconciliation
   resolves orders by client order id. Nothing is resubmitted.
3. It takes the tail cursors from the last `MARK`, so ledger rows written after it are journaled exactly once.
4. It continues at the first bar that starts at or after the last recorded activity. A crash in the middle of a
   bar skips the rest of that bar, as if the process were down for it.

A journal line cut short by a crash is skipped by readers and terminated, never rewritten.

## Executable-portfolio evaluation (P1)

```
uv run --frozen python -m services.replay evaluate --policy <default|active|FILE> --equity 5000 \
    --strategy <id> (--bars FILE | --synthetic [--symbols SPY --sessions 6 --seed 7]) [--parity]
    [--fill-seed N --nonfill P --partial P] [--out DIR] [--workspace DIR] [--resume]
uv run --frozen python -m services.replay parity ...          # replay vs services.quantlab.intraday
uv run --frozen python -m services.replay compare-paper --paper-db PATH --period-start D --period-end D ...
uv run --frozen python -m services.replay strategies
```

* `--policy active` reads the operator's active PAPER policy through `PolicyReader` (SQLite `mode=ro`,
  integrity-checked). A JSON file contains `RiskPolicy` fields and must pass `validate_policy`. Either way the
  profile is installed only in the replay workspace. The active policy is never changed.
* The report includes capital exposure (mean, max, share of bars invested), net CAGR, Sharpe (session returns,
  √252), max drawdown (bar marks net of costs), turnover, cost drag (USD, % of equity, annualized, bps of
  traded notional), rejection counts by risk check (split into entries and exits), final-gate and broker
  rejections, decisions, fill-model outcomes, and round trips.
* Strategies:
  * `runtime-baseline`: the runtime's engine without a trained registry, deciding on every bar.
  * `tod-<SYM>-<HHMM>-<M>m`: a deterministic time-of-day entry held for `M` minutes.
  * `tod-<SYM>-<HHMM>-<N>d`: the same entry held for `N` sessions. This needs a policy with
    `allow_overnight_positions` and the id in `multi_day_strategy_allowlist`; otherwise the risk engine rejects
    it on `holding_period_authorized`.
* Example (synthetic data): a 10 % concentration profile at $5,000 equity rejects every runtime-sized ($1,000)
  entry on `max_concentration` / `max_allowable_quantity`. The research backtest would have traded all of them.

## Parity with the research simulator

`parity.compare` runs the unchanged `services.quantlab.intraday.simulate` on an `IntradayPanel` of the same bars.
The `DailyContext` is built from the snapshot's own point-in-time aggregates (an adapter). The comparison uses
the runtime's sizing, `max_positions` as the concurrency cap, and the decision close as the limit. The report
lists:

* entries missed on either side;
* research non-fills;
* entry and exit fill price deltas (bps), exit reason categories (TIME / EOD incl. HOLDING_LIMIT / STOP incl.
  STOP_GAP and STOP_BREACH), exit time deltas, and gross and net bps per trade;
* totals and a tolerance verdict.

On synthetic data, `tod-SPY-1000-60m` matched 5 of 5 trades. Entry and exit deltas were about 1 bps, which is the
synthesized half spread. Net deltas were under 0.5 bps, and exit reasons agreed in every case.

## Paper comparison hook

`paper.export_paper_ledger(db_path, start, end, mode="PAPER")` reads only the `orders` and `fills` tables. It
opens the database read-only (`mode=ro` + `query_only`) and reads no account data, keys or credentials.
`paper.compare_paper(export, replay_records)` matches each order leg by symbol and side to the nearest replay leg
within a time window. It reports legs present on only one side, price, time and quantity deltas, and order-state
counts. It is tested with fixtures only: a hand-built minimal database and a real-schema PAPER ledger produced
by the position-manager test rig.

## Known limits

* Quotes are synthesized from bars. There is no real NBBO, queue position, depth, halts or latency, and the
  intrabar path is an assumption: low first on up bars, linear, continuous inside a bar, a jump between bars.
* The entry fill model is one-shot against the next bar's open. A real resting order could fill later within the
  entry timeout.
* Exits fill at the bid of the current or next sample. The cost model's slippage and impact are charged on top
  and are not calibrated against live fills.
* Signals and horizons are computed on 5-minute bars, while the runtime asks for 1-minute bars.
* There is no trained model registry in a workspace. Replaying a trained suite would need a read-only copy of a
  registry.
* The FakeBroker reports no intraday P&L for positions carried overnight. The runtime's daily-P&L check therefore
  locks new entries while a multi-day position is held, which matches the current FakeBroker paper behavior.
* Between checkpoints, broker state is written on submit, cancel and state-changing quotes. A crash in the middle
  of a bar resumes at the next bar.
* The protected QuantLab holdout (2025-11-21 … 2026-11-20) is refused for real data. Synthetic data is exempt.
  Paper periods after the holdout can be compared.
* `ExecutionService`'s wall-clock suspension watchdog and the LEAN heartbeat are not exercised; neither applies
  to a replay.

## Tests

```
LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q \
  tests/unit/replay tests/integration/replay tests/failure_injection/replay
```

* **Unit:** snapshot contract and holdout refusal, point-in-time daily facts, feed point-in-time bars/quotes and
  stop crossings, fill model (no fill at acceptance, gaps, random non-fill, partial + race, determinism, restart),
  metrics, journal append-only and resume, strategies, policy profiles (the active policy is read-only), paper
  export and compare.
* **Integration:**
  * synthetic sessions end to end through the real RiskEngine (every required check), signed intents, the
    ledger claim, the final gate, the FakeBroker and the PositionManager;
  * determinism with a seeded fill model;
  * research parity within tolerance, including the EOD flatten;
  * a multi-day hold with DAY stop re-arm and the HOLDING_LIMIT exit, and its rejection under the default policy;
  * CLI evaluation with a policy file and with the active policy;
  * the paper hook on a real-schema PAPER ledger.
* **Failure injection:**
  * a gap between the stop and its limit (the broker stop fills) and a gap through the limit (STOP_BREACH
    escalation);
  * a partial fill followed by a cancel/fill race;
  * a restart in the middle of a replay, which resumes from the journal with identical trades;
  * an ambiguous submission (accepted, response lost, lookup failed) resolved by client order id and never
    resubmitted.

## Review changes (2026-11-25, Codex review)

The behaviour described above was tightened after review:

- **Workspace.** A fresh run requires an empty workspace. Resume rejects symlinked or aliased state and uses
  explicit settings, never the operator's.
- **Resume.** Resume deduplicates records by journaled source key. It finishes an interrupted close and can
  regenerate a missing summary without executing again.
- **Fills.** Entries fill only on the next opening print. Broker state and the match outcome are saved in one
  atomic write.
- **Stops.** An intra-bar stop crossing inserts a quote at the triggering bid.
- **Costs.** A reconciled catch-up fill on a canceled entry is charged exactly once.
- **Marks.** Valuation uses the last observed mid, exposure is sampled once per bar, and final equity is the last
  mark.
- **Journal.** Invalid journal lines are skipped, never rewritten. The writer owns labels and provenance.
