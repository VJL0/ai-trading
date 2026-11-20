# Autonomous paper sessions

The API process owns the session manager, SQLite state and scheduler. Claude Code and Codex are operators over
MCP; their chat or subscription usage limit does not stop a running quant session. Keep the API process running.

Use `start_paper_session` with `duration_minutes` (1–480), `cadence_minutes` (1–60), configured/allowed `symbols`,
and `use_ai`. Set `use_ai=false` for no inference. Every actual order still passes the existing deterministic
risk engine, signed intent, final gate and ledger gateway. Only PAPER/REPLAY modes are accepted; only one
RUNNING/STOPPING session is allowed per ledger, enforced transactionally.

`stop_session` completes an in-flight cycle before reaching STOPPED. It does not cancel orders or flatten holdings.
`get_session_status` returns current and recent sessions; `get_session_report` returns cycle outcome counts,
NO_TRADE reasons, failed risk checks, trades, ledger performance, API cost and a deterministic narrative.
`explain_decisions` explains recent decisions without calling an LLM.

Schedules use `schedule_session(weekdays=["MON", ...], start_time_et="09:45", duration_minutes=60, ...)`.
Times are America/New_York, including DST. XNYS holidays and early closes are respected. Each occurrence is
persisted at most once. Missed windows become MISSED; missed cadence slots never cause catch-up bursts.
`cancel_schedule` disables future starts; an already-started session remains active until stopped or complete.

After restart, sessions resume only within their window and after CLEAN broker-authoritative reconciliation.
Unclean recovery or an elapsed window produces INTERRUPTED. STOPPING at restart becomes STOPPED. Unknown broker
submission outcomes retain the existing reconciliation behavior; never blindly resubmit a failed tool call.

An in-app AI review failure can rerun that slot quant-only and suspend AI review for 30 market-clock minutes.
Three consecutive cycle failures end a session as FAILED. This fallback does not override a model's valid
NO_TRADE decision. External Claude Code/Codex modes do not invoke in-app inference.

Synthetic mode uses an accelerated historical clock. Restarting the synthetic provider restarts its replay;
existing positions may cause reconciliation to lock entries because the fake broker is not persistent.
Use the isolated `make smoke` command for repeatable runs. Real scheduled paper execution still requires
market data, Alpaca paper credentials, LEAN licensing, and broker integration verification.

Performance is ledger-based Decimal/FIFO accounting. Session reports scope realized activity to their window;
account equity and unrealized holdings reflect the whole account at the cutoff, not a separate session portfolio.
Synthetic performance is not evidence of trading profitability.
