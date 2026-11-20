// Headless Playwright verification of the FAKE PAPER DASHBOARD fixture (no network, no broker).
//
//   node src/verify-fixture.mjs [--headed]
//
// Prints ONE JSON object on stdout. Exit code 0 only when every check passed.
// Steps: start a 127.0.0.1 fixture server -> open the fixture -> type a note (role-based locator) ->
// click "Save note" -> reload -> confirm the persisted text -> attempt "Place LIVE order" (must be
// DENIED by policy and never clicked) -> attempt https://app.alpaca.markets (DENIED by the pre-check)
// -> force a raw page.goto to the broker (must be aborted by the route layer) -> a page-script POST
// to the paper broker API (aborted) -> a 302 redirect to the broker (never committed) -> file://
// fixture allowed / file:// outside fixtures denied.

import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { AuditLog } from './audit.mjs';
import { GuardedBrowser } from './guard.mjs';
import { DEFAULT_FIXTURES_DIR, GuiPolicy, MAC_AGENT_ROOT, PolicyDenied, SurfaceKind } from './rules.mjs';
import { startFixtureServer, TEST_REDIRECT_PATH } from './static-server.mjs';

const FIXTURE = 'fake_paper_dashboard.html';

export async function runVerification({ headed = false, auditPath } = {}) {
  const audit = new AuditLog(auditPath, { writer: 'node-verify' });
  const policy = new GuiPolicy({ audit });
  const server = await startFixtureServer({ testRoutes: true });
  policy.registerFixturePort(server.port);
  const checks = {};
  const result = { tool: 'playwright', playwright_driven: true, fixture_server: server.origin, checks, artifacts: [] };
  let g;
  try {
    g = await GuardedBrowser.launch({ headless: !headed, policy, audit });
    const page = g.page;
    const url = server.url(FIXTURE);
    await g.goto(url);
    const surface = await g.surface();
    checks.surface_is_fixture = surface.kind === SurfaceKind.BROWSER_FIXTURE;

    const note = `local-ai-trader browser test ${new Date().toISOString()}`;
    const textbox = page.getByRole('textbox', { name: 'Paper note', exact: true });
    await g.type(textbox, note);
    await g.click(page.getByRole('button', { name: 'Save note', exact: true }));
    await g.reload();
    const persisted = (await page.getByTestId('saved-note').innerText()).trim();
    checks.note_persisted_after_reload = persisted === note;
    result.note = note;

    // 1. element-level denial of the fake live-order button (pre-check; the click never happens)
    const live = page.getByRole('button', { name: 'Place LIVE order', exact: true });
    try {
      await g.click(live);
      checks.live_order_click_denied = false;
    } catch (e) {
      checks.live_order_click_denied = e instanceof PolicyDenied && e.decision.reason === 'LIVE_ORDER_ELEMENT';
      result.live_order_denial = e instanceof PolicyDenied ? e.decision.reason : String(e);
    }
    checks.live_order_never_activated = (await page.locator('html').getAttribute('data-live-order-clicked')) === null;

    // 2. navigation to the brokerage (pre-check)
    try {
      await g.goto('https://app.alpaca.markets');
      checks.broker_navigation_denied = false;
    } catch (e) {
      checks.broker_navigation_denied = e instanceof PolicyDenied && e.decision.reason === 'BROKERAGE_DENIED';
      result.broker_navigation_denial = e instanceof PolicyDenied ? e.decision.reason : String(e);
    }

    // 3. route layer: a raw page.goto that bypasses the pre-check is aborted before any connection
    let rawErr = '';
    try { await page.goto('https://app.alpaca.markets/', { timeout: 10000 }); } catch (e) { rawErr = String(e.message || e); }
    checks.raw_broker_goto_aborted = /ERR_BLOCKED_BY_CLIENT|ERR_NAME_NOT_RESOLVED|net::ERR/i.test(rawErr)
      && !page.url().includes('alpaca');
    result.raw_broker_goto_error = rawErr.split('\n')[0].slice(0, 160);

    // 4. a page script trying to POST an order to the paper API is aborted by the route handler.
    //    (Run on about:blank: the fixture's own CSP would already block it, which would not prove the route layer.)
    await page.waitForLoadState('load').catch(() => {});
    await g.goto('about:blank').catch(async () => { await page.waitForTimeout(300); await g.goto('about:blank'); });
    const fetchOutcome = await page.evaluate(async () => {
      try { await fetch('https://paper-api.alpaca.markets/v2/orders', { method: 'POST', body: '{}' }); return 'sent'; }
      catch (e) { return `blocked:${e.name}`; }
    });
    checks.page_script_broker_request_blocked = fetchOutcome.startsWith('blocked')
      && g.blocked.some((b) => b.url.includes('paper-api.alpaca.markets') && b.method === 'POST');

    // 5. redirect to the broker: never commits to a broker URL (stopped by the route layer, the navigation
    //    watcher or, last, the host-resolver rules)
    let redirectErr = '';
    const blockedBefore = g.blocked.length;
    try { await page.goto(server.url(TEST_REDIRECT_PATH.slice(1)), { timeout: 10000 }); } catch (e) { redirectErr = String(e.message || e); }
    await page.waitForTimeout(300);
    const routed = g.blocked.slice(blockedBefore).some((b) => b.url.includes('alpaca'));
    const watched = g.navViolations.some((b) => String(b.url).includes('alpaca'));
    const unresolvable = /ERR_NAME_NOT_RESOLVED/.test(redirectErr);
    result.redirect_stopped_by = routed ? 'route_handler' : watched ? 'navigation_watcher' : unresolvable ? 'host_resolver_rules' : 'none';
    checks.redirect_to_broker_not_committed = !page.url().includes('alpaca') && (routed || watched || unresolvable);

    // 6. LIVE surface: the same benign button is denied when the page declares LIVE mode
    await g.goto(url);
    await page.evaluate(() => document.documentElement.setAttribute('data-trading-mode', 'LIVE'));
    try {
      await g.click(page.getByRole('button', { name: 'Save note', exact: true }));
      checks.live_surface_denies_benign_click = false;
    } catch (e) {
      checks.live_surface_denies_benign_click = e instanceof PolicyDenied && e.decision.reason === 'LIVE_SURFACE';
    }
    await g.reload();

    // 7. file:// fixture allowed; file:// outside fixtures denied
    const fileUrl = pathToFileURL(resolve(DEFAULT_FIXTURES_DIR, FIXTURE)).href;
    checks.file_fixture_allowed = policy.checkUrl(fileUrl).allowed;
    const outside = pathToFileURL(resolve(MAC_AGENT_ROOT, '..', '..', '.env')).href;
    checks.file_outside_fixtures_denied = !policy.checkUrl(outside).allowed;
    await g.goto(fileUrl);
    checks.file_fixture_surface_is_fixture = (await g.surface()).kind === SurfaceKind.BROWSER_FIXTURE;

    await g.goto(url);
    result.artifacts.push(await g.screenshot('browser-fixture'));
    result.blocked_requests = g.blocked.map((b) => ({ url: b.url, method: b.method, reason: b.reason }));
  } catch (e) {
    result.error = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e);
  } finally {
    if (g) await g.close();
    await server.close();
  }
  result.ok = !result.error && Object.values(checks).length > 0 && Object.values(checks).every((v) => v === true);
  return result;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const headed = process.argv.includes('--headed');
  const res = await runVerification({ headed });
  process.stdout.write(`${JSON.stringify(res)}\n`);
  process.exit(res.ok ? 0 : 1);
}
