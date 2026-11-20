# apps/dashboard progress (checkpoint log)

Inventory at resume (2026-11-07, attempt 3): package.json/lockfile, vite/vitest/tsconfig, src/api (client, stream,
types), src/lib (decimal, mode, time, operatorToken), src/state, components, 8 views, fixtures + builder
(tests/e2e/dashboard/build_fixtures.py), scripts/gen_ts_types.py existed UNVERIFIED. Missing: main.tsx, App.tsx,
styles, test setup, 7 views, check-bundle/lint scripts, e2e, all tests, README. Generated types stale.

- R1 pnpm project: `pnpm install --frozen-lockfile` OK (pnpm 12.11.2); `pnpm typecheck` passes on existing code.
- R2 type generation: `uv run --frozen python scripts/gen_ts_types.py` (24 models, 27 enums) + `pnpm gen:types` (54 types) regenerated incl. PredictionSet/HorizonForecast/AiRuntimeMode; fixed inherited BaseModel docstring leak; ruff+pyright clean.

Resume (2026-11-08, attempt 4): re-verified R1 (`pnpm install --frozen-lockfile`, `pnpm typecheck` pass) and R2
(schemas.json was stale: ECDSA docstring; regenerated, `pnpm gen:types --check` clean). Existing src reviewed: api/lib/state/
components + 13 views sound. Still missing: main.tsx, App.tsx, styles, test setup, Predictions + AI Runtime views,
fixtures for models/predictions/ai_runtime, check-bundle/lint scripts, all tests, e2e, README.
- R3 app (2026-11-08): main.tsx, App.tsx (15 tabs incl. new Predictions + AI Runtime views), styles.css, lint.mjs,
  check-bundle.mjs written; `pnpm build` OK (check-bundle clean). Fixtures +models/predictions/ai_runtime;
  `uv run --frozen pytest tests/e2e/dashboard -q` 7 passed.
- R4a unit tests: `pnpm test` 13 files / 96 tests passed. Left: Playwright e2e (mock-api.mjs), README, self-review.
- R4b e2e (2026-11-08): playwright.config.ts + e2e/mock-api.mjs (127.0.0.1:18765, real SSE, POST/Origin/operator guards)
  + e2e/dashboard.spec.ts; `pnpm e2e` 7 passed (vite preview :4179). Servers stopped by Playwright.
- Real-API smoke (not a committed test): built app via vite preview :4180 -> real `python -m services.api` on :18766 with
  LAT_PROJECT_ROOT=<scratchpad> (isolated var/, FAKE gateway, SIMULATED): all 15 tabs, 0 "invalid" decimals, 0 error
  notes, 0 console errors; halt + operator resume + run-cycle (risk REJECTED stale quote) rendered. Servers stopped.
- R5 README.md written. Self-review fixes: client-clock negative ages, drawer closes on hash change, resume result
  persisted, failed checks first, status accepted in server_time order, single in-flight cancel, phone layout.
- Final gate: pnpm install --frozen-lockfile / gen:types --check / typecheck / lint / test (13 files, 99 tests) /
  build (check-bundle clean) / e2e (7) all pass; ruff+pyright clean on scripts/gen_ts_types.py + tests/e2e/dashboard;
  `uv run --frozen pytest tests/e2e/dashboard -q` 7 passed. Nothing left open.

## Phase 2: read-only live operations UI (owner update 2026-11-08, attempt 1 of this phase)
Inventory: previous 15-tab operator dashboard (halt/resume/run-cycle/token modal) existed and passed its own gates.
Backup of the old sources: scratchpad backup/dashboard-src-before.tgz. Removed: views/*, HaltControls, OperatorTokenDialog,
RunCycle, charts (Lightweight Charts), operatorToken, metrics. lightweight-charts removed (`pnpm remove --offline`;
`pnpm install --frozen-lockfile` OK). Kept: lib/decimal, lib/time, lib/mode, api/stream, state/useResource, generated types.
Backend facts found: services/api/app.py registers GET /api/v1/health twice (first = simple health wins), so the UI must
accept both shapes; services/sessions not present yet (performance/sessions may 500); activity ts uses the simulated clock.
- P2-R1..R7 sources written: api/client.ts (GET-only), api/types.ts (live-ops types), ops/{reducer,lifecycle,timeline,
  performance,health,labels}.ts, state/{AppContext,useLiveOpsSync}.tsx, components/{Pipeline,AgentActivity,DecisionCard,
  OrderLifecycle,EventTimeline,PerformanceStrip,HealthPanel,ModeBanner,StreamIndicator,Icon,ui}.tsx, styles.css (CSS
  animations + reduced-motion). `pnpm typecheck` clean for app code; `node scripts/lint.mjs` clean (new read-only rules).
  check-bundle: attribution only required if Lightweight Charts is bundled; forbids state-changing routes/headers.
  Left: fixture builder (real ActivityLog recording), unit tests, mock API + e2e, real-API run, README.

## Codex continuation (2026-11-08)
Read-only helpers/client tests and SSE mock/e2e updated; reducer resync/dedupe covered. Fixture IDs made
deterministic. 42 Vitest + 3 Playwright tests pass; lint/typecheck/build/bundle/contracts pass. Real API browser
run observed completed MCP session and two filled verification orders, no page errors, GET-only API requests.
See docs/verification/continuation.md.
