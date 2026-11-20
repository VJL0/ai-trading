// End-to-end check that a browser is controlled THROUGH Playwright MCP (pinned local @playwright/mcp,
// behind src/guarded-mcp.mjs), speaking the MCP JSON-RPC protocol over stdio exactly like a Claude
// session would. Prints ONE JSON object; exit 0 only if every check passed.
//
//   node src/mcp-smoke.mjs

import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseSnapshotText } from './guarded-mcp.mjs';
import { startFixtureServer, TEST_REDIRECT_PATH } from './static-server.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

class McpClient {
  constructor(proc) {
    this.proc = proc;
    this.nextId = 1;
    this.waiters = new Map();
    createInterface({ input: proc.stdout }).on('line', (line) => {
      let msg;
      try { msg = JSON.parse(line); } catch { return; }
      if (msg.id !== undefined && msg.method === undefined && this.waiters.has(msg.id)) {
        const w = this.waiters.get(msg.id);
        this.waiters.delete(msg.id);
        w(msg);
      } else if (msg.method && msg.id !== undefined) {
        // server -> client request (e.g. roots/list): we advertise no capabilities, answer empty/err
        proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: msg.id, error: { code: -32601, message: 'not supported' } })}\n`);
      }
    });
  }

  request(method, params, timeoutMs = 60000) {
    const id = this.nextId++;
    return new Promise((ok, fail) => {
      const t = setTimeout(() => { this.waiters.delete(id); fail(new Error(`${method} timed out`)); }, timeoutMs);
      this.waiters.set(id, (msg) => { clearTimeout(t); ok(msg); });
      this.proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }

  notify(method, params = {}) {
    this.proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`);
  }

  async call(name, args = {}) {
    const msg = await this.request('tools/call', { name, arguments: args });
    if (msg.error) return { isError: true, text: `${msg.error.code}: ${msg.error.message}` };
    const text = (msg.result?.content || []).filter((c) => c.type === 'text').map((c) => c.text).join('\n');
    return { isError: msg.result?.isError === true, text };
  }
}

function findRef(text, role, name) {
  const snap = parseSnapshotText(text);
  for (const [ref, el] of snap.refs) if (el.role === role && el.label === name) return ref;
  return null;
}

export async function runMcpSmoke() {
  const server = await startFixtureServer({ testRoutes: true });
  const proxy = spawn(process.execPath, [resolve(HERE, 'guarded-mcp.mjs'), '--fixture-port', String(server.port)], {
    stdio: ['pipe', 'pipe', 'pipe'], env: process.env,
  });
  let stderr = '';
  proxy.stderr.on('data', (d) => { stderr = (stderr + d.toString()).slice(-2000); });
  const c = new McpClient(proxy);
  const checks = {};
  const result = { tool: 'playwright-mcp', via: 'src/guarded-mcp.mjs -> @playwright/mcp (pinned local)', checks, artifacts: [] };
  try {
    const init = await c.request('initialize', {
      protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'lat-mcp-smoke', version: '0.1.0' },
    });
    result.server_info = init.result?.serverInfo;
    c.notify('notifications/initialized');
    const tools = (await c.request('tools/list', {})).result?.tools?.map((t) => t.name) || [];
    result.tools_exposed = tools;
    checks.tools_listed = tools.includes('browser_navigate') && tools.includes('browser_click');
    checks.dangerous_tools_hidden = !tools.includes('browser_evaluate') && !tools.includes('browser_run_code_unsafe')
      && !tools.includes('browser_file_upload') && !tools.includes('browser_route');

    const url = server.url('fake_paper_dashboard.html');
    let r = await c.call('browser_navigate', { url });
    checks.navigate_fixture_ok = !r.isError && r.text.includes('FAKE PAPER DASHBOARD');
    if (!r.text.includes('[ref=')) r = await c.call('browser_snapshot');
    const boxRef = findRef(r.text, 'textbox', 'Paper note');
    const saveRef = findRef(r.text, 'button', 'Save note');
    const liveRef = findRef(r.text, 'button', 'Place LIVE order');
    checks.refs_found = Boolean(boxRef && saveRef && liveRef);

    const note = `local-ai-trader MCP test ${new Date().toISOString()}`;
    result.note = note;
    r = await c.call('browser_type', { element: 'Paper note textbox', target: boxRef, text: note });
    checks.type_ok = !r.isError;
    r = await c.call('browser_click', { element: 'Save note button', target: saveRef });
    checks.save_click_ok = !r.isError;
    r = await c.call('browser_navigate', { url }); // reload through a fresh navigation
    if (!r.text.includes(note)) r = await c.call('browser_snapshot');
    checks.note_persisted_after_reload = r.text.includes(note);

    const live2 = findRef(r.text, 'button', 'Place LIVE order') || liveRef;
    r = await c.call('browser_click', { element: 'Place LIVE order button', target: live2 });
    checks.live_order_click_denied = r.isError && r.text.startsWith('DENIED') && r.text.includes('LIVE_ORDER_ELEMENT');
    r = await c.call('browser_click', { element: 'Save note button', target: live2 }); // honest label, malicious ref
    checks.mislabelled_live_ref_denied = r.isError && r.text.includes('LIVE_ORDER_ELEMENT');
    r = await c.call('browser_click', { element: 'Save note button', target: '#place-live-order' }); // raw selector
    checks.raw_selector_target_denied = r.isError && r.text.includes('ELEMENT_UNKNOWN');
    r = await c.call('browser_evaluate', { function: "() => document.getElementById('place-live-order').click()" });
    checks.evaluate_tool_denied = r.isError && r.text.includes('CAPABILITY_NOT_ALLOWED');
    r = await c.call('browser_snapshot');
    checks.live_order_never_activated = !r.text.includes('must never be clicked');

    r = await c.call('browser_navigate', { url: 'https://app.alpaca.markets' });
    checks.broker_navigation_denied = r.isError && r.text.includes('BROKERAGE_DENIED');
    r = await c.call('browser_navigate', { url: 'file:///etc/hosts' });
    checks.file_outside_fixtures_denied = r.isError && r.text.startsWith('DENIED');
    r = await c.call('browser_navigate', { url: server.url(TEST_REDIRECT_PATH.slice(1)) });
    const after = parseSnapshotText(r.text).url || '';
    result.redirect_result = r.isError ? r.text.split('\n').slice(0, 3).join(' ').slice(0, 240) : `page url: ${after}`;
    checks.redirect_to_broker_not_committed = !after.includes('alpaca');
    r = await c.call('browser_navigate', { url });
    r = await c.call('browser_take_screenshot', { filename: '../escape.png' });
    checks.screenshot_path_escape_denied = r.isError && r.text.includes('PATH_OUTSIDE_ARTIFACTS');
    await c.call('browser_close');
  } catch (e) {
    result.error = String(e.message || e);
    result.stderr_tail = stderr.split('\n').slice(-5).join(' | ');
  } finally {
    proxy.stdin.end();
    await new Promise((ok) => { const t = setTimeout(() => { proxy.kill('SIGKILL'); ok(); }, 8000); proxy.on('exit', () => { clearTimeout(t); ok(); }); });
    await server.close();
  }
  result.ok = !result.error && Object.values(checks).length > 0 && Object.values(checks).every((v) => v === true);
  return result;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const res = await runMcpSmoke();
  process.stdout.write(`${JSON.stringify(res)}\n`);
  process.exit(res.ok ? 0 : 1);
}
