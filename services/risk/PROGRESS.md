# services/risk PROGRESS

Attempt 3 (2026-11-06) inventory: engine.py, limits.py, context.py, market_hours.py, operator_auth.py,
policy_store.py existed from interrupted attempts (UNVERIFIED). No migrations dir, no tests, no README.
policy_store referenced a missing risk_policy_changes table (migration 0300 absent).
- R1-R4 code review pass: fixed ruff (12x E501, S105) + pyright strict (missing stubs, payload typing) errors;
  engine: max allowable qty floored, daily-loss now fails closed without start-of-day/last equity, added
  max_allowable_quantity(decision); policy_store: PolicyReader now holds only the DB path and opens SQLite
  read-only; added missing migrations/0300_risk_policy_guards.sql (risk_policy_changes + provenance trigger).
  `ruff check services/risk` + `pyright services/risk` clean. Tests: none yet (next).
- R5 (partial): tests/unit/risk/{risk_factories.py,test_risk_engine_checks.py,test_risk_engine_authorize.py,
  test_risk_engine_properties.py} -> `uv run --frozen pytest tests/unit/risk -q` 159 passed. market_hours now
  prebuilds calendar caches (a bad warm-up timestamp made every call fail closed - caught by tests, fixed).
  Left: modes/limits/context/operator_auth/market_hours unit tests, policy store integration, acceptance set, README.
- R5 (cont): +test_risk_modes, test_risk_limits_and_context, test_risk_operator_auth (unit);
  tests/integration/risk/{test_risk_policy_store,test_risk_gate_flow} (real SQLite + real OrderLedger contract);
  tests/failure_injection/risk/{test_risk_acceptance_stops,test_risk_fail_closed}.
  Defects found by tests and fixed: float smuggled via model_construct passed apply_change (JSON dump laundered
  it) -> python-mode re-validation; engine now also re-validates its policy; stale position snapshots rejected.
  `pytest tests/unit/risk tests/integration/risk tests/failure_injection/risk -q` 350 passed, 0 skipped.
  Left: README, final adversarial review, quality gate.
- Self-review fixes: input_contracts_valid check (re-validates proposal+ctx); authorize bound to the evaluated
  proposal digest and idempotent per decision (retry returns the same intent, never a 2nd order). README.md written.
  354 passed (unit+integration+failure_injection risk).
- FINAL (all 5 requirements done): unit 270 passed, integration 41 passed, failure_injection 46 passed (357 total,
  0 skipped); ruff clean on services/risk + risk tests; pyright strict services/risk 0 errors. Extra fail-closed
  fix: any non-empty lock_reasons locks entries. Integrator e2e (tests/e2e/api) currently blocked by missing
  services.order_ledger.gateway_factory (not risk); Runtime got past PolicyStore.ensure_default + RiskEngine build.
- Attempt 4 (2026-11-06) re-verification: all R1-R5 files present; re-ran gate: risk unit+integration+failure_injection
  357 passed; ruff clean; pyright strict services/risk 0 errors. Starting adversarial self-review + integrator compat.
- Attempt 4 fixes: (a) FAIL-OPEN found+fixed: string booleans smuggled via model_copy/model_construct (health.feed_ok="no",
  account.is_paper="false") were truthy => APPROVED + final_gate pass. evaluate/final_gate now run every check on
  re-validated copies and fail input_contracts_valid when a copy != original. (b) evaluate raised AttributeError for
  ctx=None (violated never-raises) => _error_decision uses getattr. (c) +tests/integration/risk/test_risk_execution_boundary.py
  (real OrderLedger+EntriesLock+Reconciler+FakeBroker+ExecutionService with final_gate as callback): 6 passed.
- Attempt 4 (cont): (d) policy_store.active()/ensure_default now also prove the active version matches the newest
  hash-chained activation audit event (unaudited is_active flips fail closed; operator apply_change recovers), both
  reads in one snapshot txn. (e) final_gate re-checks BUY no-margin cash cover + daily-loss stop. Tests pyright-clean.
  `pytest tests/{unit,integration,failure_injection}/risk -q` 386 passed.
- Attempt 4 FINAL: (f) apply_change redacts the operator token from actor/reason before audit (OperatorAuth.redact).
  Self-introduced defect caught by tests and fixed: test factories briefly used Decimal(value) (laundered floats).
  Gate: ruff clean (services/risk + risk tests); pyright strict services/risk 0 errors (risk tests also 0);
  pytest unit 287 + integration 51 + failure_injection 50 = 388 passed, 0 skipped.
  Scratch-only check: real services.api Runtime (AI module stubbed in sys.modules, SIMULATED replay + FakeBroker):
  approve->submit->FILLED, tampered refused, oversized/far-limit/TQQQ/duplicate/2nd-position rejected, operator change
  ok, unsafe/wrong-token refused. tests/e2e/api blocked: services.ai.pipeline missing + e2e helper builds qty=0.
