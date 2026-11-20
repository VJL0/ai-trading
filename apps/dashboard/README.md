# Read-only trading operations dashboard

React/Vite application using only GET requests and the local API's SSE stream. It shows the decision pipeline,
current decision, order lifecycle, activity timeline, health, session state and performance. No order controls,
operator tokens, AI credentials or manual-trading flows are included.

Run from the repository root with `make demo`, then open http://127.0.0.1:5173.
The proxy defaults to API port 8765; `LAT_DASHBOARD_API_TARGET` must be an HTTP loopback origin.

From this directory: `pnpm test`, `pnpm lint`, `pnpm build`, `pnpm gen:types --check`, `pnpm e2e`.
Playwright runs the built app and a local SSE fixture API on dedicated ports 4179/18765. Fixtures are recorded
from backend activity/ledger components with deterministic IDs; regenerate with
`uv run --frozen python tests/e2e/dashboard/build_fixtures.py` from the repository root.

Tests cover GET-only transport, errors, stream retry/deduplication/resync, mode warnings, Decimal formatting,
streamed filled-order rendering, reconnect, phone layout and reduced motion. A real-API browser smoke and real
MCP session are documented in `docs/verification/continuation.md`; the browser fixture suite is not broker proof.

## Market Mycelium

The WebGPU observatory above the operations panels maps recorded activity into procedural neural colonies, risk membranes, confidence fans, and memory vessels. Camera controls, event inspection, and historical replay are local and read-only. See [architecture, data semantics, limits, and GPU verification](src/mycelium/README.md).
