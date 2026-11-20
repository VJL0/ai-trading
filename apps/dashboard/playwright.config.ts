/**
 * Playwright e2e: the BUILT dashboard (`vite preview`) against e2e/mock-api.mjs. Both servers bind
 * 127.0.0.1 on dedicated ports (so a running dev API on :8765 or preview on :4173 is never touched)
 * and are stopped by Playwright when the run ends. Chromium only, headless.
 */
import { defineConfig, devices } from "@playwright/test";

const MOCK_API_PORT = 18765;
const PREVIEW_PORT = 4179;
process.env.LAT_E2E_BASE_URL ??= `http://127.0.0.1:${PREVIEW_PORT}`;
process.env.LAT_E2E_MOCK_URL ??= `http://127.0.0.1:${MOCK_API_PORT}`;

export default defineConfig({
  testDir: "./e2e",
  testMatch: /.*\.spec\.ts$/,
  fullyParallel: false,
  workers: 1, // the fixture API holds shared stream playback state
  retries: 0,
  timeout: 30_000,
  expect: { timeout: 7_500 },
  reporter: [["list"]],
  outputDir: "./test-results",
  use: {
    baseURL: process.env.LAT_E2E_BASE_URL,
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ...devices["Desktop Chrome"],
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: [
    {
      command: "node e2e/mock-api.mjs",
      url: `http://127.0.0.1:${MOCK_API_PORT}/api/v1/health`,
      reuseExistingServer: false,
      timeout: 20_000,
      env: {
        LAT_MOCK_API_PORT: String(MOCK_API_PORT),
      },
    },
    {
      command: `pnpm exec vite preview --host 127.0.0.1 --port ${PREVIEW_PORT} --strictPort`,
      url: `http://127.0.0.1:${PREVIEW_PORT}/`,
      reuseExistingServer: false,
      timeout: 30_000,
      env: { LAT_DASHBOARD_API_TARGET: `http://127.0.0.1:${MOCK_API_PORT}` },
    },
  ],
});
