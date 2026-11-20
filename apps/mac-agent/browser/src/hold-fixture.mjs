// Headed Chromium session holding the FAKE PAPER DASHBOARD open for the Peekaboo (native) browser pass.
//
//   node src/hold-fixture.mjs
//
// Line protocol on stdio (one JSON object per line on stdout; commands on stdin):
//   -> {"event":"ready","url":...,"title":...,"trading_mode":...,"texts":[...],"browser_app":"Google Chrome for Testing"}
//   <- state   -> {"event":"state","saved_note":...,"textarea_len":N,"live_order_clicked":bool,"url":...}
//   <- reload  -> {"event":"reloaded","url":...}   (guarded Playwright reload of the same fixture URL)
//   <- close   -> {"event":"closed"} and exit 0
// The session closes itself after MAC_AGENT_HOLD_SECONDS (default 180) as a safety net.
// The browser is the guarded one (route allowlist + host-resolver rules); Peekaboo drives its window.

import { createInterface } from 'node:readline';
import { AuditLog } from './audit.mjs';
import { GuardedBrowser } from './guard.mjs';
import { GuiPolicy } from './rules.mjs';
import { startFixtureServer } from './static-server.mjs';

const out = (obj) => process.stdout.write(`${JSON.stringify(obj)}\n`);
const audit = new AuditLog(undefined, { writer: 'node-hold' });
const policy = new GuiPolicy({ audit });
const server = await startFixtureServer();
policy.registerFixturePort(server.port);
const g = await GuardedBrowser.launch({ headless: false, policy, audit, nativeAccessibility: true });
const url = server.url('fake_paper_dashboard.html');
let closed = false;

async function shutdown(code = 0) {
  if (closed) return;
  closed = true;
  await g.close();
  await server.close();
  out({ event: 'closed' });
  process.exit(code);
}

try {
  await g.goto(url);
  await g.page.bringToFront();
  const s = await g.surface();
  out({ event: 'ready', url, title: s.windowTitle, kind: s.kind, trading_mode: s.tradingModeAttr, texts: s.texts.slice(0, 50),
    browser_app: 'Google Chrome for Testing' });
} catch (e) {
  out({ event: 'error', error: String(e.message || e).split('\n')[0] });
  await shutdown(1);
}

const timer = setTimeout(() => shutdown(2), Number(process.env.MAC_AGENT_HOLD_SECONDS || 180) * 1000);
timer.unref();

const rl = createInterface({ input: process.stdin });
rl.on('line', async (line) => {
  const cmd = line.trim();
  if (cmd === 'state') {
    const page = g.page;
    const st = await page.evaluate(() => ({
      saved: document.getElementById('saved-note')?.textContent ?? null,
      len: document.getElementById('paper-note')?.value.length ?? -1,
      live: document.documentElement.getAttribute('data-live-order-clicked') === 'true',
      stored: (() => { try { return window.localStorage.getItem('lat.fakePaperDashboard.note'); } catch { return null; } })(),
    }));
    out({ event: 'state', saved_note: st.saved, stored_note: st.stored, textarea_len: st.len, live_order_clicked: st.live, url: page.url() });
  } else if (cmd === 'reload') {
    try { await g.reload(); out({ event: 'reloaded', url: g.page.url() }); }
    catch (e) { out({ event: 'reloaded', error: String(e.message || e).split('\n')[0] }); }
  } else if (cmd === 'close') {
    await shutdown(0);
  }
});
rl.on('close', () => shutdown(0));
