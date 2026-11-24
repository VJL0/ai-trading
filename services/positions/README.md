# services/positions

Autonomous position manager: entry fill → broker-held protective stop → exit (time, end of day, overnight, stop,
stop breach) → flat with realized P&L. PAPER/REPLAY only; every order goes through the risk engine. Design, locks,
safety states and non-guarantees: [position lifecycle](../../docs/architecture/position-lifecycle.md).

| Module | Contents |
|---|---|
| `policy.py` | `PositionPolicy` (immutable operating parameters, versioned and audited at startup) |
| `store.py` | `PositionStore`: `position_plans` (one row per flat-to-flat round trip) and append-only `position_actions` |
| `manager.py` | `PositionPort` (runtime surface), `PositionManager.tick()/run()/view()` |
| `migrations/0800_positions.sql` | plans + actions tables, append-only triggers, one OPEN plan per symbol |
| `migrations/0810_multi_day_plans.sql` | fixed holding-session count and originating risk policy; existing plans remain intraday |

Multi-day equity plans require the entry's persisted `holding_period_authorized` approval, an intact originating
risk policy, an allowed strategy and an explicit `holding_sessions > 1` request. The manager stores the final
exchange-session deadline once, including holidays and early closes. Restarting does not extend it. Revoking
overnight permission, removing the strategy, or reducing the holding limit triggers a `POLICY_REVOKED` exit.

DAY protective orders expire at the close. Overnight holdings are explicitly marked unprotected and lock new
entries; at the next open the manager reconciles missed expiry updates, then re-arms protection or exits a gap
through the stop. The ordinary intraday lifecycle stays the default. Crypto execution remains unsupported.

Read-only views: `GET /api/v1/positions`, `get_portfolio.managed_positions/recent_round_trips`,
`get_risk_status.exits_locked/exit_lock_reasons/position_safety`. Operator kill switch (blocks exits too):
`POST /api/v1/control/kill-switch` (operator token).

```
uv run --frozen pytest tests/unit/positions tests/integration/positions tests/failure_injection/positions \
  tests/e2e/api/test_e2e_position_lifecycle.py -q
```
