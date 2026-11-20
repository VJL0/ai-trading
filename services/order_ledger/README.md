# Current execution adapter

`AlpacaPaperGateway` in `alpaca_gateway.py` implements the existing execution interface using official
`alpaca-py` clients. `build_gateway` selects it for `ALPACA_PAPER`; legacy LEAN selections are refused by
the application factory. FAKE remains available for explicit simulation. See the
[paper onboarding runbook](../../docs/runbooks/broker-paper-onboarding.md).

Signed intents, atomic ledger claims, recovery, cumulative-fill accounting and hash-chained audit remain
in the existing services. Broker polling feeds those same services; it never originates trades.
The LEAN bridge material below is retained as historical/research documentation, not a supported runtime path.

---

# services/order_ledger

Durable order ledger, the single execution path, broker reconciliation/startup recovery, the entries lock, the
deterministic fake broker, the read-only Alpaca PAPER state reader, the LEAN engine bridge (server side) and the
gateway factory with live-activation denial. PAPER/SIMULATED only; nothing here can route a real-money order.

## Order path (the only one)

```
RiskEngine.authorize -> OrderLedger.record_intent (RISK_APPROVED | awaits HUMAN_APPROVED)
  -> ExecutionService.submit: entries lock + final_gate (any violation => REJECTED locally, nothing sent)
  -> OrderLedger.claim_for_submission (atomic CAS -> SUBMITTING; never twice; expired => EXPIRED)
  -> gateway.submit under a timeout
       GatewayRejected / RateLimited (429)          => REJECTED, never retried
       timeout / SubmitTimeout / connection error   => UNKNOWN + entries locked, ONE lookup by client order id;
                                                       found => broker state applied, else stays UNKNOWN
                                                       (NEVER resubmitted automatically)
  -> run_event_consumer: gateway.order_events() -> OrderLedger.apply_event (idempotent, order tolerant)
  -> Reconciler (startup + periodic): broker-authoritative; resolves UNKNOWN by client order id; never invents fills
```

## Public API (import paths used by services/api)

| Name | Signature / notes |
|---|---|
| `services.order_ledger.ledger.OrderLedger` | `OrderLedger(db, clock, mode, *, verifier=None, approval_verifier=None)` |
| | `record_proposal(p)`, `record_decision(d)`, `record_market_snapshot(s)` -> bool (insert-once; different content => `LedgerConflict`) |
| | `record_intent(intent) -> OrderRecord`; `record_human_approval(a) -> OrderRecord` |
| | `claim_for_submission(coid, claimer) -> ApprovedOrderIntent \| None`; `mark_submitted(coid, BrokerOrder)`; `mark_unknown(coid, reason)`; `mark_rejected`; `reject_locally`; `cancel_local`; `request_cancel`; `expire_stale_intents()` |
| | `apply_event(OrderEvent) -> ApplyResult(applied, state, duplicate, note, booked_quantity, absorbed_quantity, anomalies)`; `apply_broker_order(BrokerOrder, EventSource)` |
| | queries: `get_order`, `get_record`, `open_orders() -> list[BrokerOrder]`, `positions() -> list[PositionSnapshot]`, `recent_submission_times(since)`, `recent_fingerprints(since)`, `daily_realized_pl(day)`, `events(coid)`, `transitions(coid)`, `fills(coid)`, `event_log()`, `duplicate_count()` |
| `services.order_ledger.state_machine` | `ALLOWED_TRANSITIONS` (all 14 states), `decide(current, target, source=...)` |
| `services.order_ledger.entries.EntriesLock` | `EntriesLock(db, mode, *, clock=None)`: `is_locked() -> (bool, reasons)`, `lock(reason, source) -> halt_id`, `set_flag(flag, reason, source)`, `clear_flag(flag, report=CLEAN report)`, `clear_halt(halt_id, by=, reason=)` (operator; requires latest reconciliation CLEAN), `flags()`, `halts()` |
| `services.order_ledger.execution.ExecutionService` | `ExecutionService(ledger, gateway, final_gate, clock, *, submit_timeout_s=10.0, entries_lock=None, ...)`: `await submit(intent) -> BrokerOrder \| None`, `await cancel(coid) -> OrderState \| None`, `await run_event_consumer(stop_event)`, `handle_event(ev)`, `watchdog_tick()` |
| `services.order_ledger.fake_broker.FakeBroker` | `FakeBroker(clock, quote_source, *, starting_cash=Decimal, slippage_bps=Decimal, max_fill_per_match=None, verifier=None, session_bounds_fn=None, state_path=None, ...)`; `advance(quote)`, `expire_day_orders()`, `injector: FailureInjector`, `submission_count(coid)` |
| `services.order_ledger.alpaca_reader.AlpacaPaperStateReader` | `AlpacaPaperStateReader(secret_store, *, http_session=None, clock=None, ...)`: `get_account`, `get_positions`, `get_order_by_client_id`, `list_orders(status=, after=)` — GET to the paper endpoint only |
| `services.order_ledger.reconcile.Reconciler` | `Reconciler(db, ledger, reader, entries_lock, clock, *, unknown_grace_s=60)`: `await reconcile() -> ReconciliationReport`, `await startup_recovery()` |
| `services.order_ledger.bridge` | `EngineTokenAuth(keys_dir)`, `claim_next_intents(ledger, engine_id, max_n)`, `ingest_engine_event(ledger, payload, *, gateway_kind)`, `record_heartbeat(db, payload)`, `record_engine_rejection(db, payload)`, `LeanBridgeGateway(ledger, kind, clock, *, state_reader=None)` |
| `services.order_ledger.gateway_factory.build_gateway` | `build_gateway(settings, secret_store, *, ledger, clock, quote_source, verifier=None, **fake_broker_kwargs) -> ExecutionGateway` |
| `services.order_ledger.live_gate.LiveReleaseGate` | `evaluate()` lists unmet live requirements; `activate()` / `request_from()` always raise `LiveActivationDenied`; CLI `python -m services.order_ledger.live_gate --activate` exits 3 |

## Ledger semantics

* One `orders` row per client order id; every change is a compare-and-set in one `BEGIN IMMEDIATE` transaction,
  written to `order_state_transitions` + a hash-chained audit event in the same transaction.
* Every received event is stored append-only in `order_events` (`applied`, `apply_note`). Dedupe key = broker
  execution id (fills), else event id; re-deliveries are counted in `order_event_duplicates`, never re-applied.
* Out-of-order: progress is rank-monotonic (an ACK after a fill is recorded, not applied); terminal states are
  sticky except (a) fill evidence completing the order (fill racing a cancel => FILLED) and (b) reconciliation
  correcting `RECONCILED`. Fills are always booked, whatever the state; cumulative filled quantity never exceeds the
  order quantity (excess => `ledger_anomaly`, entries locked).
* `ledger_positions` change only from recorded `fills` (exact Decimal cost basis, realized P&L per fill and per
  America/New_York trading day). Reconciliation books catch-up fills only for quantity ranges no execution covered;
  late execution reports for those ranges are absorbed (`fill_absorptions`), never double-booked.
* Migration `migrations/0400_order_ledger.sql` adds columns/tables (engine queue, duplicates, absorptions,
  reconciliation epochs) and append-only triggers on fills/transitions/runs/approvals.

## Entries lock

Locked when any holds: active halt; a flag (`startup_unreconciled`, `broker_disconnected`, `unknown_orders_present`,
`reconciliation_mismatch`, `engine_heartbeat_stale`, `ledger_anomaly`, `process_suspended`); any UNKNOWN order; no
CLEAN reconciliation ever; no CLEAN reconciliation *in this process* (a restart is locked from the moment its
`Database` opens); RESEARCH_ONLY or a live mode. Flags clear only via a persisted CLEAN run that started after
the flag was last raised (epoch rule). Halts clear only by the operator (`clear_halt`) after a CLEAN run.
While entries are locked the LEAN bridge hands out nothing and withdraws every queued intent (order REJECTED,
never sent), so an approval granted before a halt is never released to the engine afterwards.

## Gateways and labels

| Gateway | Built as | Label | Broker state for reconciliation |
|---|---|---|---|
| FAKE | `FakeBroker` (state in `var/<ns>/fake_broker_state.json`) | SIMULATED | the fake broker |
| LEAN_PAPER_SIM | `LeanBridgeGateway` | SIMULATED | ledger-derived (LEAN PaperBrokerage keeps no external state) |
| LEAN_ALPACA_PAPER | `LeanBridgeGateway` + `AlpacaPaperStateReader` | PAPER | Alpaca PAPER REST (requires paper keys) |
| LEAN_ALPACA_LIVE / LIVE_APPROVAL / BOUNDED_LIVE | — | — | `LiveActivationDenied` listing every unmet requirement |

Every decision (accepted or denied) is written to `mode_transitions` and the audit chain. The FAKE gateway refuses
to build without the risk engine's public key (it verifies every intent signature) and, as built by the factory,
executes only intents the ledger has claimed for submission (state SUBMITTING), so a signed intent handed to the
broker object by any other route is refused. Reconciliation never resolves an order through another gateway's
reader (`GATEWAY_MISMATCH`, entries stay locked): use one ledger namespace per gateway.

## Tests

```
uv run --frozen pytest tests/unit/order_ledger tests/integration/order_ledger tests/failure_injection/order_ledger -q
```

All of these run offline against the deterministic `FakeBroker`, a scripted LEAN stand-in, or mocked alpaca-py
HTTP sessions. `tests/integration/order_ledger/test_order_ledger_paper_broker.py` (`-m paper_broker`) is the only
test that would touch the real Alpaca PAPER endpoint (read-only); it skips when paper keys are absent.

## Limitations (honest)

* No real broker has been exercised: no Alpaca paper keys exist on this machine. The Alpaca reader is tested with
  the SDK's HTTP session mocked; the real read-only test skips.
* The LEAN bridge is tested against a Python stand-in for the C# engine, not the real LEAN process. LEAN must
  submit our `client_order_id` as the Alpaca client order id for lookups by client id to resolve UNKNOWN orders.
* Bridge contract v1 has no engine cancel channel: a cancel after engine pickup raises `CancelNotSupported` and the
  local CANCEL_PENDING is reverted (state then follows engine/broker events).
* FakeBroker fills are simulated (no queue position, market impact, fees or latency); it is not a paper broker.
* Alpaca `replaced` orders, non-DAY time in force, stop/notional orders and non-equity positions are refused
  (`BrokerPayloadError`) rather than degraded; reconciliation then fails closed.
