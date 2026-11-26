# Local validation

This application is local and single-user. There is no remote CI. `make validate` is the complete check, and a
change is not done until it passes.

```sh
make validate               # everything, including the dependency security audit (needs network access)
make validate SKIP_AUDIT=1  # everything except the audit (offline)
```

`make validate` runs `scripts/validate.sh`. The script copies the current source to a temporary directory: tracked
files plus unstaged edits and new files that are not ignored. It then runs every step in a clean environment with
Keychain disabled, `LAT_FEED_MODE=SIMULATED` and a temporary `HOME`. Each step is reported and a summary is
printed. The exit status is nonzero if any step fails.

| Step | What it checks |
|---|---|
| Frozen Python 3.11 sync | `uv.lock` installs as is, including the `advanced-research` and `kronos` extras. |
| Ruff lint | `ruff check .` |
| Pyright type check | `pyright` (strict for contracts, core and risk). |
| Tool, MCP and HTTP contract checks | The canonical tool registry, MCP registrations and schema budget, and HTTP route security. |
| Python test suite | All Python tests except the `live_models`, `paper_broker` and `gui` markers. This includes the order-ledger and position failure-injection suites. |
| Torch test files | Torch tests in their own pytest process, matching `TORCH_TESTS` in the Makefile. |
| Isolated startup smoke | `scripts/smoke_check.sh` runs `make smoke`: fresh API and MCP processes on loopback with a fake gateway. It passes only if the session completes after the MCP client disconnects and the ledger audit chain verifies. |
| LEAN .NET tests | `make test-lean`, run from the real checkout because LEAN builds are ignored files. The summary marks it `skipped`, not passed, until `make lean-build` has run. |
| Locked dependency security audit | `pip-audit` (pinned 2.10.0, via `uvx`) over the locked requirements for this platform, strict mode. |

## Requirements

You need uv (0.13 here), Git, Make, Node.js for Pyright, and `brew install libomp` for LightGBM. uv supplies
Python 3.11. The first run populates the shared uv cache, and later runs reuse it.

## What validation does not cover

These need explicit operator authorization and run separately:

- broker verification (`-m paper_broker`, which places real paper orders)
- paid model calls (`-m live_models`)
- GUI automation (`-m gui`, `make gui-verify`)

Report them as not run unless they were.

## Faster loops while working

Run the affected tests first, for example
`LAT_USE_KEYCHAIN=false LAT_FEED_MODE=SIMULATED uv run --frozen pytest -q tests/unit/<area>`, then `make lint`
and `make typecheck`. Finish with `make validate`.

## MCP project registration

Run `scripts/register_mcp.sh --write-project-configs` in each checkout you connect to Claude Code or Codex. It
refreshes the absolute paths in `.mcp.json` and `.codex/config.toml`, keeps the Codex approval settings, and does
not call either client. Keep the generated paths local to that checkout.
