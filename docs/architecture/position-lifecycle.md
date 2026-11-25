# Position lifecycle (entry → protection → exit → flat)

Phase 1 of the [development roadmap](../roadmap/README.md). Code: [`services/positions`](../../services/positions),
exit-aware locking in [`services/order_ledger/entries.py`](../../services/order_ledger/entries.py), exit and
protective-stop rules in [`services/risk/engine.py`](../../services/risk/engine.py).

## What runs

`PositionManager` is a background task in the API process (`Runtime.start`, task `position-manager`, 1 s ticks;
0.5 s in replay). It manages every symbol that has a ledger position, an open plan or a working order, in PAPER
and REPLAY only. It **never sends an order itself**: each protective stop, exit and re-price is a `TradeProposal`
from `Proposer.POSITION_MANAGER` through the unchanged single order path

```
PositionManager -> Runtime.execute_proposal -> RiskEngine.evaluate/authorize (signed intent)
  -> OrderLedger.record_intent -> [submit_lock] prepare_gate + ExecutionService.submit (final_gate) -> gateway
```

and each cancel goes through `ExecutionService.cancel`. `Runtime.submit_lock` serializes the single-use final-gate
context between decision cycles, agent proposals and the manager (held only around `prepare_gate` + `submit`).

Everything the manager decides comes from durable state: booked fills (`ledger_positions`), ledger order states and
the `position_plans` row, plus one fresh quote. A restart therefore resumes on the next tick after the new process's
own CLEAN broker reconciliation. Every decision that changed something (or was refused) is appended to
`position_actions`; plan open/close and safety-state changes are also in the hash-chained audit log.

## Per-symbol decision (one order action per symbol per tick)

| State | Action |
|---|---|
| Exits locked (state uncertain / kill switch) | No order action. Broker-held stops keep working at the broker. Protection is still evaluated (UNPROTECTED after the grace period) and a reconciliation is requested (rate-limited; never for a kill switch). |
| Order in doubt (SUBMITTING / UNKNOWN) | Wait. UNKNOWN orders also lock exits until reconciliation resolves them by client order id. |
| Outside regular hours with a position | Safety state OVERNIGHT (broker DAY stops expired at the close). |
| Flat | Cancel stale protective/exit SELLs; cancel a working entry older than `entry_fill_timeout_s`; close the plan with realized P&L from booked fills once nothing is working. |
| Open, holding | Cover the position (net of other working SELLs) with broker-held STOP_LIMIT SELLs at the plan stop: protect, top up a delta, or cancel and re-place over-coverage. Cancel an entry remainder after `entry_fill_timeout_s`. |
| Exit trigger | Persist the reason on the plan (sticky across restarts) → cancel protection → wait for broker-confirmed terminal state → marketable LIMIT SELL → cancel and re-price after `exit_reprice_s`. |

Exit triggers in priority order: **EOD** (`flatten_before_close_s`, default 10 min, before the XNYS close, early closes
included), **OVERNIGHT** (opened on an earlier trading day), **STOP_BREACH** (the broker stop triggered but the bid
gapped below its limit, or it is still unfilled after `breach_grace_s`), **STOP** (bid at the stop and no
acknowledged broker stop), **TIME** (plan horizon: the entry signal's `horizon_minutes`, default 60).

### Prices

* Plan stop: the entry proposal's `stop_price` (what the risk engine sized the planned loss with) when it is below
  the average cost; otherwise `default_stop_loss_bps` below the average cost, floored to the tick.
* Protective order: `STOP_LIMIT` SELL, stop = plan stop, limit = stop − `stop_limit_gap_bps` (default 50), DAY,
  regular hours. The risk engine requires the stop ≥ 10 bps below the bid (it would otherwise trigger at once), within
  2000 bps of the mid, and the limit within 300 bps below the stop.
* Exit: LIMIT SELL at the bid, one tick lower per attempt, never more than 10 bps through the bid or outside the
  policy band around the mid. A price whose fingerprint was authorized inside the duplicate window, or that equals a
  working order, is skipped (a re-price is never refused as a duplicate of a canceled order).

## Locks: entries vs exits

`EntriesLock.is_locked()` (entries) is unchanged except that it now also includes every active `position_safety`
row. `EntriesLock.exits_locked()` is the narrower lock for risk-reducing SELLs:

| Reason | Entries | Exits |
|---|---|---|
| Process-start guard, mode, any entries flag (`broker_disconnected`, `reconciliation_mismatch`, `process_suspended`, `ledger_anomaly`, ...), UNKNOWN orders, no CLEAN reconciliation | locked | locked |
| Halt, scope `ENTRIES` (monitor entry halt, `request_halt`, reconciler halts) | locked | **open** |
| Position safety state (UNPROTECTED, EXITING, BREACHED, OVERNIGHT, UNVERIFIED) | locked | **open** |
| Halt, scope `ALL` (`POST /api/v1/control/kill-switch`, operator token) | locked | locked |

`SystemHealth.exits_locked/exit_lock_reasons` carry the exit lock into the risk engine; when a caller leaves them
unset, exits are locked exactly like entries (fail closed). In a long-only system every SELL is risk-reducing, so the
risk engine, `ExecutionService` and `AlpacaPaperGateway` apply the exit lock to SELLs and the entry lock to BUYs.

## Safety states (explicit, never silent)

A position without broker-confirmed protection for longer than `protect_grace_s` (default 30 s, and at least two
ticks) becomes **UNPROTECTED**. Any active safety state locks new entries (the risk engine's
`health_entries_unlocked` check rejects BUYs) but never blocks the exits that resolve it. A state clears only when
ledger state shows the position protected or flat. States are visible in `/api/v1/status`, `/api/v1/positions`,
`get_risk_status` and `get_portfolio`.

## Risk-engine rules added in Phase 1

* `STOP_LIMIT` is accepted only as a protective long-only SELL (contracts, risk engine, final gate, both
  gateways, Alpaca mapping). The stop price is signed: STOP_LIMIT intents use `lat-intent-v2` (v1 fields +
  `stop_price`); LIMIT intents keep the `lat-intent-v1` string byte for byte, so the LEAN verifier is unchanged and
  refuses v2 (fail closed).
* `entry_window`: no new BUY in the last 30 minutes of the session (`ENTRY_CUTOFF_BEFORE_CLOSE_SECONDS`), also at the
  final gate. Exits are never cut off.
* Pending orders: working protective stops do not consume `max_pending_orders` entry slots (they cannot add
  exposure); SELLs instead have a structural cap of 2 working SELLs per symbol, and their total remaining quantity can
  never exceed the long (`long_only`, unchanged).
* `AlpacaPaperGateway` preflights every SELL against the **broker's** position net of the broker's other open SELLs
  (paper accounts may allow shorting; a SELL sized from a stale ledger must never open a short).

Everything else (rate limits, duplicate window, quote age, spread, session, signature, policy version) applies to
exits unchanged. Consequence: with the default 3 orders/minute, an entry plus its protective stop use two slots; a
refused protective attempt is retried after the rate window.

## What is *not* guaranteed

* A protective stop is a trigger, not a guaranteed exit price. Gaps, halts and thin books can fill far below the stop
  or not at all (a gapped STOP_LIMIT rests unfilled — the manager escalates that to a marketable exit, which can
  still fill lower). `risk_to_stop` is the loss *at* the stop, not a maximum loss.
* Nothing protects a position while its broker order is absent: between entry fill and stop acknowledgement, during
  the cancel-then-exit sequence, and overnight (DAY orders expire at the close). These windows are reported as
  safety states, not hidden.
* Simulated fills (FakeBroker) trigger stops on the bid and have no queue position or latency. Real-paper behaviour is
  verified only by the opt-in `tests/integration/positions/test_position_lifecycle_paper_broker.py`.
* The LEAN bridge (historical, unsupported runtime) still blocks all orders while entries are locked and cannot
  verify v2 intents.

## Tests

```
uv run --frozen pytest tests/unit/positions tests/integration/positions tests/failure_injection/positions \
  tests/e2e/api/test_e2e_position_lifecycle.py -q
# real Alpaca PAPER orders (opt-in; read the module docstring first):
LAT_PAPER_ORDER_TESTS=1 uv run --frozen pytest -m paper_broker tests/integration/positions -q
```
