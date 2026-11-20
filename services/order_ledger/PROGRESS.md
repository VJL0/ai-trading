# services/order_ledger PROGRESS

Attempt 3 (2026-11-06) inventory: ledger, state_machine, entries, execution, fake_broker, alpaca_reader,
reconcile, bridge, gateway_factory, live_gate, intervals, sessions, timeutil, errors + migration 0400 existed
from interrupted attempts (UNVERIFIED, no tests except tests/unit/order_ledger/ol_support.py helper).
Initial gate: ruff 27 errors, pyright 2 errors. Known defect found on read: AlpacaPaperStateReader compared
str(BaseURL enum) ('BaseURL.TRADING_PAPER') to the URL => constructor always raised.
- Fixes: str(BaseURL) bug (alpaca_reader); process-start guard in EntriesLock (restarted process locked until its own
  CLEAN reconciliation); reconcile() runs the startup path until a CLEAN run in this process; ExecutionService
  watchdog (wall-clock sleep detection + LEAN heartbeat) inside run_event_consumer; CancelNotSupported + local
  cancel revert; completion override no longer hides stale ACKs; human approval freshness/verifier; ruff+pyright clean.
- R1 ledger: tests/unit/order_ledger/{test_order_ledger_state_machine,test_order_ledger_ledger,test_order_ledger_properties}.py
  `pytest tests/unit/order_ledger -q` 75 passed (property tests 200 examples each; mutation-checked: disabling
  dedupe or terminal stickiness makes them fail).
- R2 entries: tests/unit/order_ledger/test_order_ledger_entries.py 21 passed. R3 execution: test_order_ledger_execution.py
  21 passed. R4 fake broker: test_order_ledger_fake_broker.py 15 passed. Left: R5-R9, README, gate.
Attempt 4 (2026-11-06): inventory found R5-R8 tests + tests/failure_injection/order_ledger/test_order_ledger_fi_submission.py
from attempt 3 (not logged). Re-ran: `pytest tests/{unit,integration,failure_injection}/order_ledger -q` 216 passed;
`pytest tests/integration/risk -q` 51 passed (uses this service). Fixed 21 ruff findings in tests (scoped). Next: review R5-R8,
R9 remaining scenarios, paper_broker test, integration test, README, gate.
- Review fixes: FakeBroker filled resting orders on a future/stale quote pushed via advance() (look-ahead) -> fixed + test;
  EntriesLock.clear_halt used finished_at ordering (ties under Manual/Sim clock let a halt clear after a MISMATCH) -> rowid;
  handle_event crashed on malformed stream objects -> fail-closed; engine health now also checks reported gateway/mode.
- R9: tests/failure_injection/order_ledger/test_order_ledger_fi_state.py (empty account, malformed payloads, read 429s,
  cross-day expiry+carry, full event-log replay, Mac sleep x2, unexpected position, cancel/fill race).
  `pytest tests/failure_injection/order_ledger -q` 26 passed. Left: integration tests (bridge round trip, paper_broker), README, gate.
Attempt 5 (2026-11-06): inventory: R1-R9 code + 235 tests present. Re-ran `pytest tests/{unit,integration,failure_injection}/order_ledger -q`
235 passed; ruff + pyright clean. Left: integration tests (factory/restart, LEAN bridge round trip, paper_broker), README, review.
- R-int: tests/integration/order_ledger/test_order_ledger_integration_fake_path.py (factory FakeBroker, async consumer, restart, DAY expiry) 1 passed. Fixes: FAKE factory fails closed w/o verification key; LEAN consumer no longer re-applies pre-applied engine events.
- R-int2: test_order_ledger_integration_lean_bridge.py (scripted engine stand-in round trip) + test_order_ledger_paper_broker.py (skips w/o keys) + unit property test_order_ledger_properties_reconcile.py (mutation-checked). integration: 2 passed 1 skipped.
- Review fixes: reconcile never resolves an in-doubt/working order through another gateway's reader (GATEWAY_MISMATCH, stays locked); factory FakeBroker executes only ledger-claimed (SUBMITTING) intents (claim_check). Suites: 292 passed 1 skipped (incl. tests/integration/risk).
- Review fix: claim_next_intents fails closed while entries are locked (withdraws queued intents -> REJECTED, never handed out) + unit test; FakeBroker rejects float max_fill_per_match.
- Final gate (attempt 5): ruff clean, pyright 0 errors; unit 214 passed, integration 2 passed 1 skipped (paper_broker, no keys), failure_injection 26 passed; tests/integration/risk 51 passed. Runtime smoke (scratch, AI stubbed): startup CLEAN, cycle NO_TRADE, reconcile CLEAN. README written. Remaining: real Alpaca paper + real LEAN engine runs (need keys / C# bridge client).
