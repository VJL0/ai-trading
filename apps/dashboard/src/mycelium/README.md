# Market Mycelium

The dashboard's read-only intelligence observatory. It consumes the existing `AppContext` operations reducer: stage/activity/decision SSE, order events, audit records, and REST resynchronization. It opens no connection and cannot issue trading commands. Production has no synthetic event generator.

## Evidence and visual grammar

- A colony is a reported decision proposer, actor, or service source. The UI does not assert that every source is a competing AI strategy. Associated activity follows the proposer when its causal decision is available.
- Causal IDs anchor common roots. Each observed record creates four procedural dendrites, sampled at 64 points each. Recorded event density builds the organism; time alone does not generate activity.
- Teal: market processing. Violet: model/decision activity. Amber: risk membranes. Blue: order activity. Pale green: outcomes/reconciliation (not a claim of profit). Coral: failure, rejection, or unapplied order events. Source records remain inspectable, including skipped stages.
- Prediction fan width is `1 − reported confidence`. Missing or invalid confidence produces no fan. Calibrated versus uncalibrated confidence is identified in the inspector. A confidence score is **not** a predictive price distribution; the current API does not expose one. Likewise, the current dashboard stream exposes market-processing events rather than raw quote ticks. Nothing invents ticks, strategy competition, probability paths, or returns.
- Curl-like displacement and branching are procedural geometry evaluated in WGSL, not a physical fluid solver or an inference model. Pulses decay with event age, leaving persistent vessels. Selecting an event highlights its own geometry.

## Architecture and limits

`model.ts` normalizes validated reducer records, deterministically hashes identities, deduplicates corrections, and retains the newest 512 records. It packs 48 bytes per event. Decimal-to-float conversion is restricted to display confidence; accounting stays in the existing decimal helpers.

`MarketMycelium.tsx` owns local camera/inspection/replay state and worker lifecycle. Drag to pan, use zoom/recenter buttons, select a colony to filter the archive, or click a colony endpoint to select its most recent visible signal. Keyboard-accessible archive buttons expose exact source JSON and causal IDs. Reduced motion disables flow effects. Hidden/offscreen scenes stop GPU submission. A fresh canvas is created on retry and StrictMode remount because offscreen transfer is single-use.

`worker.ts` receives packed buffers and scene settings. It has no API client or execution import. Latest data is retained while initialization is pending. GPU commands and animation run off the main thread. The scene uses the server/simulation clock, advancing at the measured simulation rate; newly observed event timestamps can advance the view beyond a delayed heartbeat.

`renderer.ts` owns all GPU resources. Compute writes a fixed 4 MiB particle buffer; event storage is 24 KiB and uniforms are 64 bytes. Two instanced draws accumulate continuous vascular paths and soft emissive splats over a procedural vignette into a half-float HDR texture. A final tone-mapping pass preserves source colors under dense overlap. The HDR texture adds at most about 24 MB, excluding the browser swapchain. Raster size is capped near 3 million pixels and device texture limits. No per-frame particle allocations or CPU readback of particle state. One outstanding submission prevents GPU queue growth. Optional timestamp queries use one 16-byte resolve/readback pair, sampled once per second. Device loss, initialization timeouts, shader errors, unavailable WebGPU, and worker errors fall back to the same event inspector; Retry WebGPU reconstructs the renderer without losing memory.

Replay freezes a copy of the currently retained 512-record window; live ingestion continues separately. Scrubbing and replay are deterministic because geometry is a function of source identity, timestamp, and view time, not accumulated simulation steps. Only records at/before the cursor are shown. Replay speed is 1×, 10×, or 60× source time. The retained window is bounded, not a durable archive: reload restores the API's retained history. Raw source data never goes to browser persistence.

## Verification

From `apps/dashboard`:

```sh
pnpm build
pnpm lint
pnpm test
pnpm e2e
```

Hardware GPU measurements use installed Chrome; Playwright's default headless Chromium may expose no GPU adapter. Run the isolated fixture server and preview in separate terminals:

```sh
LAT_MOCK_API_PORT=18766 node e2e/mock-api.mjs
LAT_DASHBOARD_API_TARGET=http://127.0.0.1:18766 pnpm exec vite preview --port 4180
```

Then:

```sh
node scripts/benchmark-mycelium.mjs http://127.0.0.1:4180 /tmp/mycelium-benchmark.json
node scripts/verify-mycelium-gpu.mjs http://127.0.0.1:4180 /tmp/mycelium-recovery.json
```

The benchmark injects 540 **synthetic test-only** decisions through the fixture server's SSE stream, verifies the 512-record/131,072-sample capacity, warms up, then samples 12 one-second windows at DPR 2. It writes screenshots alongside JSON. Never point it at production; its stress endpoint exists only in `e2e/mock-api.mjs`. The recovery script destroys the first real WebGPU device and verifies a successful retry with one canvas and unchanged memory. FPS measures completed worker render cadence, not a guarantee for every display refresh rate, browser, or concurrent GPU workload. GPU time excludes browser compositing and CPU work.
