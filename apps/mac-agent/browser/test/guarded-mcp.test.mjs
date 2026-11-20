import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ALLOWED_TOOLS, buildConfig, McpGuard, parseSnapshotText } from '../src/guarded-mcp.mjs';
import { GuiPolicy } from '../src/rules.mjs';

const PAGE = `### Page
- Page URL: http://127.0.0.1:41999/fake_paper_dashboard.html
- Page Title: FAKE PAPER DASHBOARD — SIMULATED
### Snapshot
\`\`\`yaml
- generic [active] [ref=e1]:
  - heading "FAKE PAPER DASHBOARD — SIMULATED" [level=1] [ref=e3]
  - definition [ref=e10]: PAPER (SIMULATED)
  - textbox "Paper note" [ref=e18]:
  - button "Save note" [ref=e19] [cursor=pointer]
  - button "Place LIVE order" [ref=e25] [cursor=pointer]
\`\`\``;

function guard(text = PAGE) {
  const p = new GuiPolicy();
  p.registerFixturePort(41999);
  const g = new McpGuard(p);
  g.absorb([{ type: 'text', text }]);
  return g;
}

test('parseSnapshotText extracts url, title, refs and texts', () => {
  const s = parseSnapshotText(PAGE);
  assert.equal(s.url, 'http://127.0.0.1:41999/fake_paper_dashboard.html');
  assert.equal(s.title, 'FAKE PAPER DASHBOARD — SIMULATED');
  assert.deepEqual(s.refs.get('e19'), { role: 'button', label: 'Save note', help: '' });
  assert.equal(s.refs.get('e25').label, 'Place LIVE order');
  assert.ok(s.texts.includes('PAPER (SIMULATED)'));
});

test('tool allowlist: dangerous tools denied', () => {
  const g = guard();
  for (const t of ['browser_evaluate', 'browser_run_code_unsafe', 'browser_file_upload', 'browser_route', 'browser_mouse_click_xy', 'browser_cookie_set', 'browser_fill_form']) {
    assert.equal(ALLOWED_TOOLS.has(t), false);
    assert.equal(g.check(t, {}).allowed, false, t);
  }
});

test('click: benign ref allowed, live-order ref denied, unknown/selector targets denied', () => {
  const g = guard();
  assert.equal(g.check('browser_click', { target: 'e19', element: 'Save note' }).allowed, true);
  assert.equal(g.check('browser_click', { target: 'e25', element: 'Save note' }).reason, 'LIVE_ORDER_ELEMENT');
  assert.equal(g.check('browser_click', { target: 'e19', element: 'Place LIVE order button' }).reason, 'LIVE_ORDER_ELEMENT');
  assert.equal(g.check('browser_click', { target: 'e999' }).reason, 'ELEMENT_UNKNOWN');
  assert.equal(g.check('browser_click', { target: '#place-live-order' }).reason, 'ELEMENT_UNKNOWN');
  assert.equal(g.check('browser_click', { target: 'text=Place LIVE order' }).reason, 'ELEMENT_UNKNOWN');
  assert.equal(g.check('browser_click', { target: 'e19', button: 'right' }).allowed, false);
  assert.equal(g.check('browser_click', { target: 'e19', modifiers: ['Meta'] }).allowed, false);
});

test('type: text policy, secrets refused, submit needs an allowed key', () => {
  const g = guard();
  assert.equal(g.check('browser_type', { target: 'e18', text: 'hello' }).allowed, true);
  assert.equal(g.check('browser_type', { target: 'e18', text: 'sk-ant-api03-abcdefghijklmnop' }).reason, 'SECRET_TEXT');
  assert.equal(g.check('browser_type', { target: 'e18', text: 'hello', submit: true }).allowed, true);
});

test('navigate: brokerage + non-loopback denied; LIVE page denies interaction', () => {
  const g = guard();
  assert.equal(g.check('browser_navigate', { url: 'https://app.alpaca.markets' }).reason, 'BROKERAGE_DENIED');
  assert.equal(g.check('browser_navigate', { url: 'https://example.com' }).reason, 'URL_NOT_ALLOWLISTED');
  assert.equal(g.check('browser_navigate', { url: 'http://127.0.0.1:41999/fake_paper_dashboard.html' }).allowed, true);
  const live = guard(PAGE.replace('PAPER (SIMULATED)', 'Mode: LIVE'));
  assert.equal(live.check('browser_click', { target: 'e19' }).reason, 'LIVE_SURFACE');
});

test('screenshot/snapshot file names cannot escape artifacts', () => {
  const g = guard();
  assert.equal(g.check('browser_take_screenshot', { filename: '../x.png' }).reason, 'PATH_OUTSIDE_ARTIFACTS');
  assert.equal(g.check('browser_take_screenshot', { filename: '/tmp/x.png' }).reason, 'PATH_OUTSIDE_ARTIFACTS');
  assert.equal(g.check('browser_take_screenshot', {}).allowed, true);
  assert.equal(g.check('browser_snapshot', { filename: '../../x.yml' }).allowed, false);
  assert.equal(g.check('browser_tabs', { action: 'new', url: 'https://app.alpaca.markets' }).allowed, false);
});

test('generated MCP config is hardened', () => {
  const cfg = buildConfig();
  assert.equal(cfg.browser.browserName, 'chromium');
  assert.equal(cfg.browser.isolated, true);
  assert.ok(cfg.browser.launchOptions.args.some((a) => a.startsWith('--host-resolver-rules=MAP * ~NOTFOUND')));
  assert.ok(cfg.browser.initPage[0].endsWith('/src/mcp-init-page.cjs'));
  assert.ok(cfg.outputDir.endsWith('/apps/mac-agent/artifacts'));
  assert.deepEqual(cfg.capabilities, ['core']);
  assert.equal(cfg.webmcp, false);
});
