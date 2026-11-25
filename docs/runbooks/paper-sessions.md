# Autonomous paper sessions

The API process owns the session manager, SQLite state and scheduler. Claude Code and Codex are operators over
MCP; their chat or subscription usage limit does not stop a running quant session. Keep the API process running.

Tell the connected AI “start trading.” Use `null` for unspecified `start_paper_session` arguments. The service
uses eligible symbols from the configured universe and active policy, a five-minute cadence, and up to 60
minutes without in-app model review. During regular hours the session starts immediately. Otherwise it queues
once for the next XNYS open. Sessions end at the planned cutoff or the regular close, including early closes.

Every order passes the deterministic risk engine, signed intent, final gate and ledger gateway. Only PAPER/REPLAY
modes are accepted. SQLite allows one queued, running, or stopping session per ledger.

`stop_session` completes an in-flight cycle before reaching STOPPED. It does not cancel orders or flatten holdings.
`get_session_status` returns current and recent sessions; `get_session_report` returns cycle outcome counts,
NO_TRADE reasons, failed risk checks, trades, ledger performance, API cost and a deterministic narrative.
`explain_decisions` explains recent decisions without calling an LLM.

Schedules use `schedule_session(weekdays=["MON", ...], start_time_et="09:45", duration_minutes=60, ...)`.
Times are America/New_York, including DST. XNYS holidays and early closes are respected. Each occurrence is
persisted at most once. Missed windows become MISSED; missed cadence slots never cause catch-up bursts.
`cancel_schedule` disables future starts; an already-started session remains active until stopped or complete.

After restart, sessions resume only with the same execution label, a remaining window, and CLEAN
broker-authoritative reconciliation. An elapsed window, unclean recovery, or changed execution label produces
INTERRUPTED. STOPPING at restart becomes STOPPED. For unknown broker submissions, inspect reconciliation and
session status before making another tool call.

An in-app AI review failure can rerun that slot quant-only and suspend AI review for 30 market-clock minutes.
Three consecutive cycle failures end a session as FAILED. This fallback does not override a model's valid
NO_TRADE decision. External Claude Code/Codex modes do not invoke in-app inference.

Synthetic mode uses an accelerated historical clock. Restarting the synthetic provider restarts its replay;
existing positions may cause reconciliation to lock entries because the fake broker is not persistent.
Use the isolated `make smoke` command for repeatable runs. Alpaca paper execution uses the official SDK and
requires the paper gateway to be selected and paper credentials to be available.

Performance is ledger-based Decimal/FIFO accounting. Session reports scope realized activity to their window;
account equity and unrealized holdings reflect the whole account at the cutoff, not a separate session portfolio.
Synthetic performance is not evidence of trading profitability.
