# Autopilot runbook

The autopilot runs research and the daily PAPER session without operator commands. Its design and hard
boundaries are in [the architecture note](../architecture/autopilot.md). It is **off by default**.

## Quick start

```sh
# 1. Dry run: what would the next tick do, and why? (no API calls, no writes)
uv run --frozen python -m services.autopilot plan
uv run --frozen python -m services.autopilot plan --at 2026-11-27T08:45:00-05:00   # Friday early close

# 2. Enable it in .env (names: see "Configuration")
LAT_AUTOPILOT_ENABLED=true

# 3. One tick in the foreground, then inspect
uv run --frozen python -m services.autopilot run --once
uv run --frozen python -m services.autopilot status          # or: status --json

# 4. The loop (foreground; Ctrl-C stops it between actions)
caffeinate -i uv run --frozen python -m services.autopilot run
```

Make targets are integration requests in `services/autopilot/PROGRESS.md` (`make autopilot`,
`make autopilot-once`, `make autopilot-status`). Until those are added, use the commands above.

When `LAT_AUTOPILOT_START_API=true` (the default), the autopilot starts the local API if it is not running. It
uses the same launcher as the MCP `start_paper_session` path. The launcher reuses a running API for this workspace
and never restarts or reconfigures it. A running API with a different mode or gateway is refused. Start it the
way you normally do (`make dev`, or `make paper` for Alpaca paper) if you want to choose the gateway yourself.

## What happens each trading day

| New York time (full day) | Step |
| --- | --- |
| 08:30–09:35 | Pre-open checks, then one `start_paper_session` request (09:30–15:30). |
| 09:30–16:00 | A status check every 15 min. Observation only. |
| from 16:30 | QuantLab daily pipeline, crew ticks, shadow journal, at most one holdout look, then the final report. |
| until 08:00 next session | Research must finish by 5 min before the next pre-open. |

On early closes the times follow the real close. On 2026-11-27 the session runs 09:30–12:30 and research starts
at 13:30. Holidays (2026-11-26) have no pre-open.

The daily report is written to `experiments/autopilot/reports/YYYY-MM-DD.json` and `.md`. A partial report is
written after the pre-open and after each check, and the final report after research.

## Operator controls (the autopilot obeys, never overrides)

* **Halt new entries**: `make halt REASON="..."`, the monitor, or any agent's `request_halt`. The autopilot stands
  down: no session request and no research. It resumes by itself only after you clear the halt
  (`make resume HALT_ID=...`, which needs the operator token and a CLEAN reconciliation). The autopilot never
  clears halts.
* **Kill switch (ALL scope, also blocks exits)**: same stand-down. The report and `status` say `kill_switch: true`.
* **Stop today's session**: use `stop_session` or the monitor as usual. The autopilot does not restart it. The
  day's pre-open job is already DONE, so there is no second request.
* **Stop the autopilot**: Ctrl-C or SIGTERM in the foreground, or `launchctl bootout` (see below). Stopping it
  leaves any running API session alone.
* **Your own schedule**: if an operator `schedule_session` covers today, the autopilot adopts it
  (`OPERATOR_SCHEDULE`) and does not request a second session.

## Alerts and failures

`status` lists recent alerts. Each alert is also an `ALERT` row in the append-only event log and a warning in the
log.

| Alert | Meaning | Action |
| --- | --- | --- |
| `no session start yet: health reconciliation DEGRADED ...` | A pre-open check failed. The autopilot re-checks every 5 min inside the window. | Check `make status` and `make reconcile`. Nothing starts late. |
| `no autopilot session on D: pre-open checks never passed` | The window closed while still blocked. | Same. The next trading day starts fresh. |
| `... the pre-open window passed without a session request` | The autopilot was not running, or the Mac was asleep. | See "Sleep and wake". |
| `stand-down: ...` | A halt or the kill switch is active. | Clear it yourself when it is safe. |
| `quantlab_daily failed after 3 attempts` | A pipeline step failed (data or network). | `make quantlab-daily` by hand. The next evening catches up once. |
| `holdout look for F exited N` | The holdout child failed. | The autopilot never retries a look. Inspect `experiments/quantlab/holdout_results` and resume by hand (`python -m services.quantlab.cli holdout --family F`, which resumes the same unresolved look for session families). |

If the start outcome is UNKNOWN (a timeout), the next tick inspects `get_session_status` and adopts the session if
it exists. It never resubmits blindly.

## Sleep and wake

* A sleeping Mac pauses the monotonic clock. The autopilot sleeps at most `LAT_AUTOPILOT_TICK_S` (60 s), re-reads
  the wall clock after waking and logs a `SLEEP_GAP` event. Windows missed while asleep are MISSED. Research
  catches up once, in the next quiet window.
* `caffeinate -i` keeps the Mac from idle-sleeping while the autopilot runs. It does not prevent sleep when the
  lid is closed (unless on power with an external display) or when you choose Sleep.
* To wake for the pre-open, schedule a wake yourself. This is a system setting and the autopilot never changes it:
  `sudo pmset repeat wakeorpoweron MTWRF 08:15:00` (check with `pmset -g sched`; undo with `sudo pmset repeat cancel`).

## Run it under launchd (optional, manual install)

The repository ships a **template only**: `ops/launchd/com.local-ai-trader.autopilot.plist.template`. Nothing
installs it. To use it:

```sh
REPO=$(pwd)
mkdir -p experiments/autopilot/logs
sed -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME#g" ops/launchd/com.local-ai-trader.autopilot.plist.template \
  > ~/Library/LaunchAgents/com.local-ai-trader.autopilot.plist
plutil -lint ~/Library/LaunchAgents/com.local-ai-trader.autopilot.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local-ai-trader.autopilot.plist
launchctl print gui/$(id -u)/com.local-ai-trader.autopilot | head -20     # state, pid, last exit
# stop and remove:
launchctl bootout gui/$(id -u)/com.local-ai-trader.autopilot
```

* Log: `experiments/autopilot/logs/autopilot.log` (every event at INFO, alerts at WARNING). Research children's
  output is captured, and its tail is stored with the job in the store, not in this log. The API keeps its own
  log under `var/<ns>/logs/api.log`.
* launchd restarts the process after a crash (`KeepAlive.SuccessfulExit=false`, at most every 300 s). The store
  resumes each job idempotently. If `LAT_AUTOPILOT_ENABLED` is not true, the process exits 3 and launchd retries
  every 300 s, so enable it first or do not install the template.
* The crew needs your own `claude` and `codex` CLIs on `PATH`, signed in. Adjust `PATH` in the plist if they are
  installed elsewhere.

## State and files

| Path | What |
| --- | --- |
| `experiments/autopilot/autopilot.sqlite3` | Append-only `events`, per-day `jobs`, `kv`. Safe to read with `sqlite3 -readonly`. |
| `experiments/autopilot/autopilot.lock` | Single-instance lock. Holds the PID while `run` is active. |
| `experiments/autopilot/reports/` | Daily reports. |
| `experiments/quantlab/daily_state.json` | Shared with the API's daily job, so neither repeats the other's completed day. |

Exit codes: `0` ok, `2` configuration error, `3` disabled, `4` another instance holds the lock, `5` refused
(Live or a non-paper runtime).

## Configuration

All variables are optional, and every value is bounded.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LAT_AUTOPILOT_ENABLED` | `false` | Master switch for `run`. `status` and `plan` always work. |
| `LAT_AUTOPILOT_TICK_S` | `60` | Longest single sleep. The wall clock is re-read after it. |
| `LAT_AUTOPILOT_PREOPEN_LEAD_MIN` | `60` | Pre-open checks start this long before the open. |
| `LAT_AUTOPILOT_START_GRACE_MIN` | `5` | A session may still be requested this long after the open, never later. |
| `LAT_AUTOPILOT_PREOPEN_RECHECK_MIN` | `5` | Minimum gap between re-runs of failed pre-open checks. |
| `LAT_AUTOPILOT_INTRADAY_INTERVAL_MIN` | `15` | One status check per interval. |
| `LAT_AUTOPILOT_SETTLE_MIN` | `30` | Research starts this long after the close. |
| `LAT_AUTOPILOT_SESSION_MINUTES` | empty | Caps the session length. Empty means the full regular session minus the 30-min entry cutoff. |
| `LAT_AUTOPILOT_CADENCE_MINUTES` | empty | Minutes between session cycles. Empty means the session manager default (5). |
| `LAT_AUTOPILOT_SYMBOLS` | empty | Comma-separated tickers, at most 10. Empty means the eligible universe. |
| `LAT_AUTOPILOT_USE_AI` | `false` | In-app model review inside session cycles. |
| `LAT_AUTOPILOT_START_API` | `true` | Start the local API if it is not running. It never restarts a running API. |
| `LAT_AUTOPILOT_RESEARCH` | `true` | Run the after-close research steps. |
| `LAT_AUTOPILOT_CREW_TICKS` | `2` | Crew ticks per evening (`python -m services.agents run --once`). |
| `LAT_AUTOPILOT_CREW_TIMEOUT_S` | `3600` | Timeout per crew tick. |
| `LAT_AUTOPILOT_HOLDOUT` | `true` | Let a `DEV_PASSED` family take its single protected-holdout look. |
| `LAT_AUTOPILOT_STEP_TIMEOUT_S` | `7200` | Timeout per QuantLab child. |
| `LAT_AUTOPILOT_STATE_DIR` | `experiments/autopilot` | State directory. |
| `LAT_AUTOPILOT_REPORT_DIR` | `experiments/autopilot/reports` | Report directory. |

The autopilot also honours the existing `LAT_QUANTLAB_DAILY_ENABLED`. When that is true, the API runs the daily
pipeline and the shadow journal, and the autopilot skips both. With a SIMULATED feed it skips the data downloads
and the holdout. It still runs the crew, which works on development data.

## Verification (offline)

```sh
LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q tests/unit/autopilot tests/integration/autopilot
uv run --frozen ruff check services/autopilot tests/unit/autopilot tests/integration/autopilot
uv run --frozen pyright services/autopilot tests/unit/autopilot tests/integration/autopilot
```

The integration test starts a real API process in a temporary SIMULATED workspace, as `make smoke` does. It runs
`run --once` at a pre-open instant and checks that a PAPER session was started or queued and that a report was
written. A second tick must not request another session. These tests use only the fake broker and synthetic
data. They do not verify the broker.
