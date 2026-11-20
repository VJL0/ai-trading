# Market Mycelium verification — November 18, 2026

Implemented in the existing React dashboard, using its read-only SSE/REST state. Compute, vascular geometry, emissive lighting, and HDR tone mapping run in a dedicated WebGPU worker. Replay, inspection, pan/zoom, reduced motion, bounded memory, visibility suspension, and recoverable rendering failure are included.

## Hardware measurements

Apple M4 Pro, 48 GB RAM, Apple Metal 3 adapter; installed Chrome 157.0.8103.79, headless on macOS arm64. Viewport 1512 × 1100, DPR 2; actual GPU raster 2286 × 1313 (3 MP cap). Each case warmed up before twelve one-second observation windows. Stress data is explicitly synthetic and delivered through the isolated test server's actual SSE stream. No brokerage service was contacted.

| Retained records | GPU samples | FPS in every sampled window | Render interval p95 | Median sampled GPU time | Sampled GPU range | Main-thread long tasks |
| --- | --- | --- | --- | --- | --- | --- |
| 27 | 6,912 | 60 | 16.7–16.8 ms | 0.59 ms | 0.52–0.92 ms | 0 |
| 512 | 131,072 | 60 | 16.7–16.8 ms | 6.33 ms | 6.09–7.08 ms | 0 |

Main-thread animation-frame p95 was 16.8 ms at baseline and 16.7 ms at capacity. No page errors occurred. Frame cadence is a measured worker submission/completion cadence, not a promise of 120 Hz or of sustained performance under all workloads. GPU timestamps include compute, HDR accumulation, and tone mapping; browser compositing is excluded. Short benchmark windows do not establish long-term thermal behavior.

## Checks

- Production TypeScript/Vite build and bundle secret/read-only-route checks passed.
- Lint passed. The existing ResearchPanel Sharpe formatter was moved to the project's decimal formatting helper to clear its pre-existing lint violation.
- 49 unit tests passed, including bounded/idempotent memory, invalid confidence and timestamps, unapplied fills, replay snapshot isolation, and existing stream/reducer checks.
- Five Playwright tests passed, including the existing live order lifecycle, reconnect, mobile layout, reduced motion, unsupported GPU inspection/replay, and independent live ingestion during frozen replay.
- Actual GPU device destruction was injected into the rendering worker. The fallback retained all 27 records, Retry restored WebGPU, and exactly one canvas remained.
- Dense visual review corrected similar-name hash clustering, separated colony labels, and introduced HDR tone mapping to preserve source colors under overlapping vessels.

## Deliberate data limits

The existing dashboard stream supplies market-processing events and reported confidence, not raw tick streams or predictive price distributions. Fan spread uses 1 − confidence, with calibration status exposed. Motion is procedural curl-like geometry rather than a physical fluid solver. Colonies identify reported proposers/actors/sources; they do not fabricate competing strategies. Replay retains 512 session records and restores only API-retained history on reload. The trading execution path was not changed.

See [raw measurements](market-mycelium-2026-11-18.json) and [architecture / reproduction commands](../../apps/dashboard/src/mycelium/README.md).
