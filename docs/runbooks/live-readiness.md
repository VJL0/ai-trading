# Live-readiness checklist (NOT enabled in this build)

Real-money execution is impossible in this build by construction:
`packages/core/modes.py` sets `LIVE_EXECUTION_COMPILED = False` and `BOUNDED_LIVE_COMPILED = False`;
`assert_gateway_allowed` refuses `LEAN_ALPACA_LIVE` in every mode; the live secret names are unreadable by
`SecretStore.get`; no LEAN live-Alpaca config or launcher exists (only a paper template with
`alpaca-paper-trading: true`). Tests in `tests/` assert each of these.

Before anyone could consider `LIVE_APPROVAL`, ALL of the following would have to be done by the owner,
outside any model or GUI-agent session:

- [ ] Months of PAPER operation with clean reconciliation, documented incidents and fixes.
- [ ] Independent review of the risk policy limits, the intent signing scheme and the LEAN `GatedBrokerageModel`.
- [ ] Comparison of backtest vs Alpaca paper fill semantics; explicit acknowledgment that paper P&L ≠ live.
- [ ] A code change (reviewed, committed) flipping `LIVE_EXECUTION_COMPILED` — never a config/env change.
- [ ] Separate live credentials provisioned only in the Keychain under `ALPACA_LIVE_*` names, never in `.env`.
- [ ] A user-enabled, operator-signed release lock file, with the verified live broker account id recorded.
- [ ] A separate live namespace (`var/live/`) with its own signing key, ledger and tokens.
- [ ] Per-order fresh signed human confirmation (`HumanApproval`) plus independent risk approval.
- [ ] Broker identity verification at start-up (account number matches the release lock).
- [ ] `BOUNDED_LIVE` remains compiled off; autonomous live execution requires a separate future decision.

Screen clicking is never a live order route; the GUI agent denies broker order controls.
