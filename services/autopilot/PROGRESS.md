# services/autopilot — progress checkpoint

Task: deterministic autonomous orchestrator for research + PAPER sessions (task_af74cb6fb7f4). Status: **done**
(2026-11-25), pending the coordinator's integration requests below.

## Done
- [x] config.py (LAT_AUTOPILOT_*), daycal.py (XNYS phases, entry cutoff), store.py (SQLite, append-only events,
      jobs, kv, flock single-instance lock)
- [x] backend.py (GuardedBackend allowlist, agent token only, Paper-only runtime guard)
- [x] checks.py (pre-open / intraday verdicts, stand-down on any halt, kill switch = scope=ALL)
- [x] supervisor.py (pure decide + execute), research.py (subprocess allowlist), report.py
- [x] cli.py (`run`, `run --once`, `status`, `plan [--at]`); SIGTERM stops between actions, cancels a long child
      after 20 s (child killed, job resumes)
- [x] unit tests (tests/unit/autopilot), authority tests, integration test (tests/integration/autopilot)
- [x] docs/architecture/autopilot.md (mission, boundaries, why), docs/runbooks/autopilot.md, ops/launchd template

## Verified
- `LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q tests/unit/autopilot
  tests/integration/autopilot tests/unit/agents/test_import_boundaries.py` -> 45 passed (42 unit + 1 integration +
  2 crew boundary tests).
- `uv run --frozen ruff check services/autopilot tests/unit/autopilot tests/integration/autopilot` clean;
  `uv run --frozen pyright` on the same paths: 0 errors.
- Integration test: a real `python -m services.api` in a temp SIMULATED workspace (FAKE gateway, synthetic fixture);
  `run --once` at a Monday pre-open produced `outcome=STARTED`, `requested_by=agent:autopilot`, and a report
  labelled SIMULATED; a second tick sent no second request.
- `plutil -lint` on the launchd template (placeholders substituted): OK.
- Simulated vs real: all session/health evidence is SIMULATED (fake broker, synthetic data, API simulated clock);
  research steps are tested with a fake subprocess runner and a fake DailyJob. No broker, Alpaca, Claude/Codex or
  network call was made. Real Alpaca-paper behaviour and real crew/quantlab children are NOT verified.

## Open / notes for follow-up
- Alerts are local (event log, `status`, report, WARNING log). No push notification channel yet.
- Holdout candidates come from `get_research_results` (API must be up in the evening); with the API down the step
  retries hourly (max 3) and then FAILS for that day.
- The pre-open accepts a STALE feed before the open (warning only); DISCONNECTED/BACKFILLING block. The risk gate
  still checks the feed every cycle.

## Integration requests (coordinator applies; I did not edit these files)

1. **`.env.example`**: append this block (names from `services.autopilot.config.env_names()`):
   ```
   # --- autopilot (services/autopilot; docs/runbooks/autopilot.md). Paper only; off by default.
   LAT_AUTOPILOT_ENABLED=false
   LAT_AUTOPILOT_TICK_S=60
   LAT_AUTOPILOT_PREOPEN_LEAD_MIN=60
   LAT_AUTOPILOT_START_GRACE_MIN=5
   LAT_AUTOPILOT_PREOPEN_RECHECK_MIN=5
   LAT_AUTOPILOT_INTRADAY_INTERVAL_MIN=15
   LAT_AUTOPILOT_SETTLE_MIN=30
   # empty = full regular session minus the 30-min entry cutoff
   LAT_AUTOPILOT_SESSION_MINUTES=
   LAT_AUTOPILOT_CADENCE_MINUTES=
   LAT_AUTOPILOT_SYMBOLS=
   LAT_AUTOPILOT_USE_AI=false
   LAT_AUTOPILOT_START_API=true
   LAT_AUTOPILOT_RESEARCH=true
   LAT_AUTOPILOT_CREW_TICKS=2
   LAT_AUTOPILOT_CREW_TIMEOUT_S=3600
   LAT_AUTOPILOT_HOLDOUT=true
   LAT_AUTOPILOT_STEP_TIMEOUT_S=7200
   LAT_AUTOPILOT_STATE_DIR=experiments/autopilot
   LAT_AUTOPILOT_REPORT_DIR=experiments/autopilot/reports
   ```
   Reason: configuration names belong in `.env.example` (AGENTS.md).
2. **`Makefile`**: add
   ```make
   .PHONY: autopilot autopilot-once autopilot-status autopilot-plan
   autopilot: ## autonomous daily orchestrator: pre-open checks + PAPER session, after-close research (LAT_AUTOPILOT_ENABLED=true)
   	caffeinate -i $(PY) -m services.autopilot run

   autopilot-once: ## one autopilot tick in the foreground, then exit
   	$(PY) -m services.autopilot run --once

   autopilot-status: ## autopilot phase, jobs, stand-down, alerts and next actions (read-only, no API calls)
   	$(PY) -m services.autopilot status

   autopilot-plan: ## dry run: what the next autopilot tick would do and why (AT=ISO to evaluate another time)
   	$(PY) -m services.autopilot plan $(if $(AT),--at $(AT),)
   ```
   Reason: requested entrypoints `make autopilot`, `make autopilot-once`, `make autopilot-status`.
3. **`scripts/supervisor.py`** (optional): add `"autopilot": [str(ROOT / ".venv" / "bin" / "python"), "-m",
   "services.autopilot", "run"]` to `SERVICES` so `scripts/supervisor.py start autopilot` detaches it like the crew.
4. **`docs/roadmap/PROGRESS.md`**: add an entry "2026-11-25 autopilot: deterministic daily orchestrator
   (services/autopilot), paper only, off by default; docs/architecture/autopilot.md".
5. **`docs/runbooks/README.md`** / **`README.md`**: link `docs/runbooks/autopilot.md` and
   `docs/architecture/autopilot.md`.
6. **`.gitignore`**: no change needed (`experiments/` is already ignored).
7. No change requested to `services/api/app.py`, `services/tools/registry.py` or `services/mcp_server/*`: the
   autopilot uses the existing agent tool interface unchanged.
