@AGENTS.md

## Completion requirement for Claude

AGENTS.md is the single source of project rules, and everything in it applies. In particular, before you report
any implementation as complete:

1. Run the affected tests, then `make validate`. Use `make validate SKIP_AUDIT=1` only when offline, and say so.
2. Resolve every failure your change causes. Never weaken tests, checks, risk limits, or trading safeguards to get
   a pass.
3. Verify the changed functionality directly: tests that fail without the change, plus `make smoke`, failure
   injection, replay, or a disposable simulated run where applicable.
4. Report the checks run with their results, what was simulated, and what was not run.
