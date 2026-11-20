// Headless browser end-to-end (needs the Playwright Chromium build; no macOS permissions, no network).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { chromium } from 'playwright';
import { runVerification } from '../src/verify-fixture.mjs';
import { runMcpSmoke } from '../src/mcp-smoke.mjs';

const haveBrowser = existsSync(chromium.executablePath());

test('guarded Playwright fixture verification', { skip: !haveBrowser && 'playwright chromium not installed', timeout: 120000 }, async () => {
  const res = await runVerification();
  assert.equal(res.error, undefined, res.error);
  for (const [k, v] of Object.entries(res.checks)) assert.equal(v, true, k);
  assert.equal(res.ok, true);
});

test('Playwright MCP through the guarded proxy', { skip: !haveBrowser && 'playwright chromium not installed', timeout: 180000 }, async () => {
  const res = await runMcpSmoke();
  assert.equal(res.error, undefined, res.error);
  for (const [k, v] of Object.entries(res.checks)) assert.equal(v, true, k);
  assert.equal(res.ok, true);
});
