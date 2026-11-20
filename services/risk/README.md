# services/risk: independent deterministic Risk & Policy Gate

Every order needs this gate's final authorization (spec invariants 1-4, sections 5, 6, 8 and 12, acceptance
criteria 5 and 12). Models and the GUI agent never call it with write power. They only produce a
`TradeProposal`. The gate recomputes everything from market and account state and signs an
`ApprovedOrderIntent` (ECDSA P-256), which is the only object an execution gateway accepts. It checks that
intent again at the order boundary.

```
TradeProposal ─► RiskEngine.evaluate(proposal, ctx) ─► RiskDecision (APPROVED | REJECTED + every check)
                 RiskEngine.authorize(proposal, decision, ctx) ─► signed ApprovedOrderIntent (TTL 60 s)
   ledger.record_intent ─► RiskEngine.final_gate(intent, fresh_ctx, verifier, active_version) ─► [] => submit
```

## Public API

| Import | Signature / use |
|---|---|
| `services.risk.engine.RiskEngine` | `RiskEngine(policy: RiskPolicy, signer: IntentSigner, clock: Clock, *, id_factory=None)` |
| | `.evaluate(proposal, ctx) -> RiskDecision`. Pure, deterministic and never raises. An internal error produces REJECTED. |
| | `.authorize(proposal, decision, ctx) -> ApprovedOrderIntent`. Raises `RiskAuthorizationError` (or `ModeViolation`) instead of signing when in doubt. |
| | `.final_gate(intent, ctx, verifier, active_policy_version, *, human_approval=None, approval_verifier=None, active_halts=(), self_recorded=True, strategy_id=None) -> list[str]`. An empty list means pass. Never raises. |
| | `.recompute(proposal, ctx) -> Exposure`, `.policy`, `.policy_version`, `.key_id`, `.policy_violations` |
| `services.risk.engine.fingerprint` | `fingerprint(proposal_or_intent, strategy_id=None)`, re-exported from `context` |
| `services.risk.engine.max_allowable_quantity` | `(decision) -> Decimal \| None`. The largest qty every limit allows. It is reported, never applied. |
| `services.risk.policy_store.PolicyStore` | `PolicyStore(db: Database, mode: TradingMode, *, operator_auth=None, clock=None)` |
| | `.ensure_default() -> RiskPolicy`, `.active() -> RiskPolicy`, `.active_version()`, `.get(v)`, `.history()`, `.changes()`, `.read_only_view() -> dict`, `.reader() -> PolicyReader` |
| | `.apply_change(new_policy, *, operator_token, actor, reason) -> str` (the new version). This is the **only** way to change limits. |
| `services.risk.policy_store.PolicyReader` | Read-only access. It holds only the DB path and opens SQLite in `mode=ro` with `query_only`. |
| `services.risk.operator_auth.OperatorAuth` | `OperatorAuth(keys_dir)`: `.ensure_token() -> Path`, `.verify(token) -> bool`, `.require(token)`, `.rotate() -> Path`, `.create_approval(token, intent, *, approver, now, ...)`, `.verify_approval(approval, intent)` |
| `services.risk.context` | `build_risk_context(...)`, `build_system_health(...)`, `unknown_health(reason)`, `order_fingerprint`, `proposal_fingerprint`, `intent_fingerprint`, `fingerprint_variants`, `recent_fingerprints`, `FingerprintEntry` |
| `services.risk.limits` | `validate_policy(policy) -> list[str]`, `HARD_*` ceilings, `hard_ceilings()`, `STRUCTURAL_INVARIANTS`, engine constants |
| `services.risk.market_hours` | `is_regular_session_open(now)`, `market_session_at(now)`, `session_bounds(now)`, `ny_date(now)` (XNYS via `exchange_calendars`) |

## What `evaluate` checks

Every check is recorded as a `RiskCheck` with `observed` and `limit`. APPROVED requires that every check in
`REQUIRED_TRADE_CHECKS` ran and passed. A check that raises counts as failed. Missing or unknown inputs
always fail.

* **Proposal and inputs:**
  * `action_trade`: NO_TRADE produces REJECTED with "no trade requested".
  * `input_contracts_valid`: re-validates the proposal and context from their raw values. **Every later check
    reads the validated copies**, and an input whose validated copy differs from the original fails. This
    defeats `model_construct`/`model_copy` smuggling, including truthy strings such as `feed_ok="no"` or
    `is_paper="false"` that a raw `bool()` would have accepted.
  * `proposal_complete`, `evidence_present`, `whole_shares`, `limit_price_tick`.
  * `order_type_limit_only`, `time_in_force_day`.
* **Mode:**
  * `mode_permits_trading`: RESEARCH_ONLY rejects, `assert_gateway_allowed` applies, and live modes and
    gateways are always denied.
  * `policy_valid`: the policy must stay within the hard ceilings and re-validate.
* **Symbol:** `symbol_format` (listed U.S. symbol), `symbol_allowlist`, `symbol_denylist` (the policy list
  plus the hard leveraged/inverse list), `min_price`.
* **Market data:**
  * `market_snapshot_present`, `quote_present`, `data_provenance`: synthetic data cannot drive
    `LEAN_ALPACA_PAPER` orders.
  * `max_spread`.
  * `max_quote_age`: both `exchange_ts` and `received_ts`; a future or not-yet-available quote is rejected.
  * `data_asof_fresh`: proposal and snapshot.
  * `regular_session`: the snapshot session must be REGULAR **and** the XNYS calendar must say open.
    Holidays and early closes count as closed, and a calendar error counts as closed.
* **Price bands:**
  * `limit_vs_mid`: within `max_limit_deviation_bps`.
  * `limit_vs_touch`: a BUY may sit at most 10 bps above the ask, a SELL at most 10 bps below the bid.
* **Account:**
  * `account_present`, `account_fresh` (60 s or less).
  * `account_status`: ACTIVE, not trading- or account-blocked, a paper account, USD.
  * `positions_consistent`: no shorts, no duplicate rows, snapshots not stale.
  * `context_fresh`: `ctx.now` is within 10 s of the engine clock.
* **Long-only and no margin:**
  * `long_only`: a SELL may only reduce an existing long, net of pending sells.
  * `no_margin`: `notional * min_buying_power_buffer <= min(cash, buying_power,
    non_marginable_buying_power) - open BUY commitments`, and cash must be greater than 0. Margin buying
    power is never used as headroom.
* **Exposure:** every value below is recomputed. `notional = qty * limit`.
  * `max_order_notional`.
  * `max_position_notional`: `(held + pending buys + qty) * max(limit, ask)`.
  * `max_concentration`: the post-trade position divided by equity.
  * `max_positions`: one position at a time.
* **Loss:**
  * `max_planned_loss`: `qty * (limit - stop)`, using the proposal stop if it is valid. Otherwise
    `default_stop_loss_bps`, floored to the tick.
  * `proposal_arithmetic`: the proposal's `max_planned_loss` is recorded and cross-checked. It is never
    used. Understating the loss by more than max($0.05, 5%) rejects.
  * `max_daily_loss`: the worse of (realized + unrealized) and (equity - start-of-day equity, falling back
    to `last_equity`). A value at or below `-max_daily_loss` rejects, and an unknown baseline rejects.
  * `daily_loss_budget`.
  * `account_equity_floor`.
* **Activity and health:**
  * `max_pending_orders`, `order_rate_per_minute`, `order_rate_per_day` (New York date).
  * `health_*`: feed, broker, reconciled, engine, entries unlocked, strategy state, permissions.
  * `duplicate_order`: the fingerprint of (symbol, side, qty, limit, strategy) is checked in both its
    strategy-specific and strategy-agnostic forms against `ctx.recent_intent_fingerprints`, and against
    identical open orders. The caller supplies only the fingerprints authorized within
    `policy.duplicate_window_seconds`, for example through `OrderLedger.recent_fingerprints(now - window)`
    or `context.recent_fingerprints`.
* **Size, never resized:** `max_allowable_quantity`. An oversized order is REJECTED and the largest
  allowable whole-share qty is reported, along with the binding limit.

Exits are deliberately exempt from entry-only limits: order and position notional, concentration, daily
loss, equity floor, cash and max positions. These are SELLs that reduce a long. Exits still need
everything else, including health, session, quote, spread and band.

## `authorize` and `final_gate`

`authorize` signs only an APPROVED decision that this engine instance issued. A digest registry covers
both the decision and the evaluated proposal content, so a forged or edited decision or proposal is
refused. It also refuses a mismatched proposal, a different mode or policy, and any re-evaluation of the
same context that no longer approves. It is **idempotent**: calling it again for the same decision returns
the same intent, so a retry never mints a second order. Once that intent expires, a new evaluation is
required. The resulting intent is built
as follows:

* `client_order_id = client_order_id(client_order_tag(mode, gateway))`.
* TTL is `policy.intent_ttl_seconds`.
* `requires_human_approval = policy.require_human_approval or mode == LIVE_APPROVAL`.
* It is signed over `signing_string()` with `ECDSA_P256_SHA256` and the signer's `key_id`, then verified
  once more.

`final_gate` must run immediately before gateway submission, using a freshly captured context. It returns
every failure among the following:

* **Signature:** algorithm, key id, and the supplied verifier must all equal the engine's key, and the
  signature must verify.
* **Time:** the intent must not be expired, must not have been created in the future, and its TTL must not
  exceed the policy TTL. The context must be fresh.
* **Mode and gateway:** they must equal the runtime's and be allowed. Live is always denied. The client
  order id must carry the right tag.
* **Policy:** the intent's policy version must still be active, and the engine's policy must be current.
* **Operational state:** the health gates, `active_halts`, the entries lock, account state and blocks, and
  that XNYS is open now.
* **Inputs:** the intent and context are re-validated, and every check reads the validated copies, as in
  `evaluate`.
* **Order shape:** LIMIT and DAY only, no extended hours, whole shares, valid tick, allowlisted and not
  denylisted, BUY notional within the limit, SELL at most the position.
* **Money, re-checked for a BUY:** no-margin cash cover (same formula as `evaluate`, other open BUYs
  committed), and the daily-loss stop. An unknown start-of-day equity fails. Both can change between
  `authorize` and submission.
* **Duplicates:** with `self_recorded=True` (the default), the ledger already holds this intent, so its
  own fingerprint may appear once. The real `OrderLedger.recent_fingerprints` behaves this way, as an
  integration test confirms.
* **Human approval:** when required, a `HumanApproval` bound to this `intent_id` must be present, must
  not be expired, and must verify through `approval_verifier` (for example `OperatorAuth.verify_approval`).

## Policy store and the operator settings flow

* Policies are immutable versions. `version` is the SHA-256 of the canonical JSON. 0001_core triggers
  block edits to the content columns and any delete. `migrations/0300_risk_policy_guards.sql` also
  freezes `approved_by` and `supersedes`, and adds the append-only `risk_policy_changes` table.
* `active()` re-parses the stored JSON and requires it to be canonical and to re-hash to its version id.
  It also requires the newest hash-chained activation audit event (`POLICY_BOOTSTRAPPED` /
  `POLICY_CHANGED`, always written in the same transaction as the flip) to name exactly the active
  version. `is_active` is the one mutable column, so a flip made by any other code path is detected. Both
  reads happen in one read-only snapshot transaction. On any failure it raises `PolicyIntegrityError`, so
  the runtime cannot build an engine and fails closed. `ensure_default()` applies the same check on
  restart. The recovery path is the audited operator `apply_change`.
* `apply_change` follows these steps:
  1. Verify the operator token.
  2. Require a non-reserved actor and a reason.
  3. Re-validate the raw values. Floats and `model_construct` are rejected.
  4. Run `limits.validate_policy`. It rejects margin, shorting, extended hours, non-LIMIT orders, removal
     of leveraged/inverse names, and values beyond the hard ceilings, for example
     `max_order_notional <= 25000`.
  5. In **one** writer transaction: insert the new version, flip `is_active`, append a hash-chained
     `POLICY_CHANGED` audit event with a field-level diff, and add a `risk_policy_changes` row.

  Rejected attempts, including a wrong token, are audited as `POLICY_CHANGE_REJECTED`. The token value is
  never stored or audited. `actor` and `reason` are passed through `OperatorAuth.redact` first, so a
  token pasted into the wrong form field is stored as `[REDACTED]`.
* `read_only_view()` returns a detached JSON dict for models and the dashboard. `PolicyReader` has no
  mutators and no handle on the single writer.

## Operator token

`var/<namespace>/keys/operator.token` is created with `secrets.token_urlsafe(32)` and permissions `0600`
in a `0700` directory. The token is compared with `hmac.compare_digest`, replaced atomically by
`rotate()`, and never logged or included in a repr. `redact(text, *also)` removes the stored token (and
any extra values) from free text before it is audited. A symlinked token file is refused, and a weak or empty
one is replaced. The token is for the **human operator only**, who sends it as `X-LAT-Operator-Token`.
Models, prompts, the Peekaboo/Playwright GUI agent, logs, screenshots and the dashboard bundle never
receive it. `PolicyStore` defaults to `OperatorAuth(db.path.parent / "keys")`, which is the same file that
`services/api/security.ensure_token` creates.

## Limitations

* The local API process can read the operator token, so a fully compromised API could forge a change or
  an approval. This is a documented residual risk. Live execution stays compiled off.
* The intent TTL is 60 s by default. An intent that needs human approval must be approved and submitted
  within that window, or it expires. This fails closed.
* `MAX_CONTEXT_AGE_SECONDS = 10` and the quote age of 15 s or less are measured on the injected clock. A
  sped-up simulation clock should not advance between building the context and evaluating it.
* After a long AI run, build the `RiskContext` with a freshly fetched `MarketSnapshot`. Otherwise
  `max_quote_age` correctly rejects.
* The XNYS calendar covers roughly 20 years back to 1 year ahead of today. Timestamps outside that range
  count as closed.
* Protective stops are not guaranteed execution prices. Planned loss is an estimate.
* `no_margin` subtracts open BUY commitments from `min(cash, buying_power, non_marginable_buying_power)`.
  Brokers such as Alpaca and the FakeBroker already reserve open orders in `buying_power`, so open orders
  may be counted twice. This is deliberately conservative; it can only reject, never approve too much.
* `ctx.recent_intent_fingerprints` carries no timestamps. The caller must pre-filter it to
  `policy.duplicate_window_seconds`, as the runtime does through `OrderLedger.recent_fingerprints`.
* `services/api` currently lets `PolicyChangeRejected` (a `ValueError`) and `OperatorAuthError` (a
  `PermissionError`) surface as HTTP 500. They should map to 422 and 401 (integrator).
* The API caches the operator token at startup. After `OperatorAuth.rotate()` the API must be restarted.

## Tests

```
uv run --frozen pytest tests/unit/risk tests/integration/risk tests/failure_injection/risk -q
uv run --frozen ruff check services/risk tests/unit/risk tests/integration/risk tests/failure_injection/risk
uv run --frozen pyright services/risk        # strict mode
```

* **Unit:** a pass and a fail case for each check, `authorize` and `final_gate`, the hypothesis
  properties, the mode matrix including the `LAT_MODE`/`LAT_GATEWAY` switch, limits, context,
  `OperatorAuth` and the XNYS calendar. The properties use an independent policy oracle, check that a
  reported max qty is itself approvable, reject floats in every money field, check that NO_TRADE never
  yields an intent, allow approvals only in allowed modes, and fail any mutated signed field.
* **Integration:**
  * The policy store on real SQLite: immutability, the activation-audit cross-check and recovery,
    snapshot-consistent reads under concurrent changes, and redaction.
  * The gate flow with the keys-directory verifier and the real `OrderLedger` fingerprint contract.
  * `test_risk_execution_boundary.py`: the real `OrderLedger`, `EntriesLock`, `Reconciler`,
    `FakeBroker` and `ExecutionService`, with `RiskEngine.final_gate` as the boundary callback. A valid
    intent reaches the broker exactly once. A policy change, a halt, expiry or a forged signature (even in
    a ledger without a verifier) never reaches the broker. This path is SIMULATED.
* **Failure injection:** the acceptance-5 set of risky orders that must be stopped, and the fail-closed
  cases (broken or naive clock, broken calendar, signer failures, audit failure rollback, missing token).
