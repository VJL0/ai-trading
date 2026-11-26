# services/replay — progress checkpoint (Phase 5)

Task: execution-equivalent replay harness + executable-portfolio evaluation. Owner paths: `services/replay/**`,
`tests/{unit,integration,failure_injection}/replay/**`, `docs/architecture/replay.md`.

## Design (fixed)

* `ReplayRuntime(Runtime)`: a narrow subclass of the real `services.api.runtime.Runtime` that does NOT call
  `Runtime.__init__` (which would build the AI pipeline, sessions, keychain access and the synthetic fixture). It wires
  the same objects on a disposable workspace (own `ledger.sqlite`, keys, FakeBroker state) and inherits the real
  `execute_proposal`, `risk_context`, `_daily_position_pnl`, `_final_gate`, `prepare_gate`, `health`,
  `entries_status`, `reconcile`, `run_cycle`; the real `DecisionCycle`, `RuntimePositionPort`, `PositionManager`,
  `RiskEngine`, `OrderLedger`, `ExecutionService`, `Reconciler`, `EntriesLock`, `PolicyStore`, `FeedQualityMonitor`,
  `ActivityLog`, `QuantSignalEngine` + `decide_edge` run unchanged.
* Adapters (documented in docs/architecture/replay.md): `ReplayFeed` (store + provider over a bar snapshot,
  synthesized quotes on an intrabar path), `ReplayBroker(FakeBroker)` (entry fill model), `ReplayDecisionCycle`
  (strategy id + holding sessions on the runtime's quant proposal), `current_signal_engine` override (strategy model).
* Journal: append-only JSONL tailing the ledger tables + summary JSON, label REPLAY.

## Status

- [x] snapshot.py, feed.py, broker.py, runtime.py, strategies.py, journal.py, engine.py, metrics.py (smoke: 2
      synthetic sessions end to end, 2 TIME round trips, ~3 s)
- [x] parity.py (smoke: 5/5 matched for tod-SPY-1000-60m; entry/exit delta ~1 bps = half spread; net < 0.5 bps)
- [x] paper.py, policy.py (default | active read-only | JSON file), __main__.py CLI (evaluate/parity/compare-paper)
- [x] tests/unit/replay: 18 pass (found+fixed: broker state not saved on a non-fill; journal resume after a
      truncated line merged the next record into it)
- [x] tests integration (9) / failure_injection (5)
- [x] docs/architecture/replay.md
- [x] ruff + pyright clean on services/replay and all replay test dirs

## Verified (2026-11-25, SIMULATED: synthetic bars, in-process FakeBroker, no network, no broker)

* `LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q tests/unit/replay
  tests/integration/replay tests/failure_injection/replay` -> 32 passed (49 s).
* `uv run --frozen ruff check services/replay tests/{unit,integration,failure_injection}/replay` -> clean;
  `uv run --frozen pyright` on the same paths -> 0 errors.
* Parity (synthetic, tod-SPY-1000-60m, 5 sessions): 5/5 matched, entry/exit ~1 bps (half spread), net < 0.5 bps.
* Not run: replay of real (non-synthetic) bars; comparison against a real paper ledger (fixtures only).

## Open

* No real-data replay yet (needs an exported 5-minute bar file outside the holdout; nothing downloaded).
* Trained model suites are not replayable (no registry in the workspace; would need a read-only registry copy).
* Entry fill model is one-shot at the next open; exit slippage uncalibrated (needs real paper fills).
* FakeBroker has no intraday P&L for overnight positions -> entries locked while a multi-day position is held.

## Integration requests

1. `Makefile`: add `test-replay` (and include it in `test-py`):
   `LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q tests/unit/replay
   tests/integration/replay tests/failure_injection/replay`; optional `replay-eval` target wrapping
   `uv run --frozen python -m services.replay evaluate --policy active --equity 5000 --strategy $(STRATEGY) ...`.
   Reason: discoverability; `make test-py` and `make validate` already collect `tests/`.
2. `docs/roadmap/PROGRESS.md` Phase 5: replace "not started" with "harness implemented (services/replay), verified in
   simulation; see docs/architecture/replay.md", listing the verification line above.
3. No API/MCP/registry change requested (replay results are files under experiments/replay/, gitignored).
