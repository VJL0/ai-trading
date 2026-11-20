#!/usr/bin/env node
// Guarded Playwright MCP: a stdio MCP proxy in front of the PINNED local @playwright/mcp server.
//
//   node src/guarded-mcp.mjs [--fixture-port N]... [--headed]
//
// Why a proxy: @playwright/mcp's --allowed-origins is documented as "not a security boundary" and the
// server has no notion of "never click a live-order button". The proxy enforces mac_agent/rules.json on
// every tools/call before it reaches the browser:
//   * tool allowlist (no evaluate/run_code/file upload/route/cookie/storage/coordinate-mouse tools);
//     tools/list responses are filtered so clients never see the denied tools;
//   * browser_navigate URL allowlist (loopback dashboard + fixtures only; brokerage hosts denied);
//   * element policy on click/hover/type: the target `ref` must come from the latest page snapshot,
//     and its accessible role/name (plus the caller's element description) must not match the
//     live-order / destructive patterns; any LIVE-mode indication on the page denies interaction;
//   * typed text: no secrets/control characters; key presses: allowlisted keys only.
// Underneath, the MCP server runs with: isolated profile, chromium, the init-page request guard
// (src/mcp-init-page.cjs), --allowed-origins loopback, blocked brokerage origins, service workers
// blocked, host-resolver rules that make every non-loopback host unresolvable, minimal child env.
//
// Denied calls get an MCP tool error result {isError:true, content:[{type:'text', text:'DENIED ...'}]}
// and are audited to apps/mac-agent/artifacts/audit.jsonl (no typed text, no page content).

import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { AuditLog } from './audit.mjs';
import { normalizeKey } from './guard.mjs';
import { Action, GuiPolicy, SurfaceKind } from './rules.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_DIR = resolve(HERE, '..');
export const MCP_CLI = resolve(BROWSER_DIR, 'node_modules', '@playwright', 'mcp', 'cli.js');
export const MCP_CONFIG_TEMPLATE = resolve(BROWSER_DIR, 'mcp', 'playwright-mcp.config.json');

export const ALLOWED_TOOLS = new Set([
  'browser_navigate', 'browser_navigate_back', 'browser_reload', 'browser_snapshot', 'browser_click',
  'browser_hover', 'browser_type', 'browser_press_key', 'browser_wait_for', 'browser_take_screenshot',
  'browser_close', 'browser_tabs', 'browser_verify_text_visible', 'browser_verify_element_visible',
]);

// "- button "Save note" [ref=e16]" / "- textbox "Paper note" [ref=e15]: text" / "- generic [ref=e3]: Hello"
const REF_LINE = /^\s*-\s+([A-Za-z]+)(?:\s+"((?:[^"\\]|\\.)*)")?((?:\s+\[[^\]]*\])*)(?::\s*(.*))?$/;

export function parseSnapshotText(text) {
  const refs = new Map();
  const texts = [];
  let url = null;
  let title = null;
  for (const raw of String(text).split('\n')) {
    const u = /^\s*-\s+Page URL:\s*(\S+)/.exec(raw);
    if (u) { url = u[1]; continue; }
    const t = /^\s*-\s+Page Title:\s*(.*)$/.exec(raw);
    if (t) { title = t[1].trim(); continue; }
    const m = REF_LINE.exec(raw);
    if (!m) continue;
    const [, role, nameRaw, attrs = '', tail] = m;
    const name = nameRaw ? nameRaw.replace(/\\(.)/g, '$1') : '';
    if (name) texts.push(name);
    if (tail && !/^\s*$/.test(tail)) texts.push(tail.replace(/^"|"$/g, ''));
    const ref = /\[ref=([^\]\s]+)\]/.exec(attrs);
    if (ref) refs.set(ref[1], { role, label: name, help: tail ? tail.slice(0, 200) : '' });
  }
  return { refs, texts, url, title };
}

export class McpGuard {
  constructor(policy, { outputDir = null } = {}) {
    this.policy = policy;
    this.outputDir = outputDir;
    this.url = 'about:blank';
    this.title = '';
    this.refs = new Map();
    this.texts = [];
  }

  surface() {
    if (this.url === 'about:blank' || !this.url) return { kind: SurfaceKind.UNKNOWN, url: this.url, windowTitle: this.title, texts: [] };
    return this.policy.browserSurface({ url: this.url, title: this.title, texts: this.texts });
  }

  absorb(resultContent) {
    let text = (resultContent || []).filter((c) => c && c.type === 'text').map((c) => c.text).join('\n');
    // Large snapshots are written to <outputDir>/page-<ts>.yml and only linked; read them (inside outputDir only).
    for (const m of text.matchAll(/\[Snapshot\]\(([^)]+\.yml)\)/g)) {
      const file = resolve(this.outputDir || '.', m[1]);
      if (this.outputDir && file.startsWith(`${this.outputDir}/`)) {
        try { text += `\n${readFileSync(file, 'utf8')}`; } catch { /* unreadable -> refs stay unknown (fail closed) */ }
      }
    }
    const snap = parseSnapshotText(text);
    if (snap.url && snap.url !== this.url) { this.refs = new Map(); this.texts = []; }
    if (snap.url) this.url = snap.url;
    if (snap.title !== null) this.title = snap.title;
    if (snap.refs.size) {
      for (const [k, v] of snap.refs) this.refs.set(k, v);
      this.texts = snap.texts;
    }
  }

  deny(action, reason, detail, target = {}) {
    return this.policy.decide(false, action, reason, detail, target);
  }

  // Returns a decision for one tools/call (audited).
  check(name, args = {}) {
    const p = this.policy;
    if (!ALLOWED_TOOLS.has(name)) {
      return this.deny(Action.REQUEST, 'CAPABILITY_NOT_ALLOWED', `MCP tool ${name} is not allowlisted for the GUI agent`, { tool: String(name).slice(0, 64) });
    }
    switch (name) {
      case 'browser_navigate':
        return p.checkUrl(String(args.url ?? ''), Action.NAVIGATE);
      case 'browser_click':
      case 'browser_hover': {
        if ((args.button && args.button !== 'left') || (Array.isArray(args.modifiers) && args.modifiers.length)) {
          return this.deny(Action.CLICK, 'CAPABILITY_NOT_ALLOWED', 'only plain left clicks are allowed', { tool: name });
        }
        return this.elementDecision(args, Action.CLICK);
      }
      case 'browser_type': {
        const el = this.refElement(args);
        const d = p.checkText(this.surface(), el, String(args.text ?? ''), p.browserRule());
        if (!d.allowed) return d;
        const desc = this.descriptionDecision(args, Action.TYPE);
        if (desc) return desc;
        if (args.submit) return p.checkKeys(p.browserRule(), ['return'], this.surface());
        return d;
      }
      case 'browser_press_key':
        return p.checkKeys(p.browserRule(), [normalizeKey(args.key ?? '')], this.surface());
      case 'browser_snapshot':
        if (args.filename !== undefined) {
          return this.deny(Action.INSPECT, 'PATH_OUTSIDE_ARTIFACTS', 'snapshot files are only written by the server itself', { tool: name });
        }
        return p.decide(true, Action.INSPECT, 'ALLOWED', 'page snapshot', { tool: name });
      case 'browser_take_screenshot': {
        if (args.target !== undefined && !this.refElement(args)) {
          return this.deny(Action.SCREENSHOT, 'ELEMENT_UNKNOWN', 'screenshot target must be a ref from the latest snapshot', { tool: name });
        }
        if (args.filename !== undefined) {
          const d = p.checkArtifactPath(resolve(p.artifactsDir, String(args.filename)));
          if (!d.allowed) return d;
        }
        return p.checkSurface(this.surface(), Action.SCREENSHOT);
      }
      case 'browser_tabs':
        if (!['list', 'select', 'close'].includes(String(args.action))) {
          return this.deny(Action.NAVIGATE, 'CAPABILITY_NOT_ALLOWED', 'only list/select/close tab actions are allowed', { tool: name });
        }
        return p.decide(true, Action.LIST, 'ALLOWED', 'tab management', { tool: name });
      default:
        // read-only / history / close tools: allowed; the post-response URL check keeps the surface honest.
        return p.decide(true, Action.INSPECT, 'ALLOWED', `read-only MCP tool ${name}`, { tool: name });
    }
  }

  descriptionDecision(args, action) {
    if (typeof args.element !== 'string' || !args.element) return null;
    const hit = this.policy.elementDenial({ role: '', label: args.element }, action);
    if (!hit) return null;
    return this.deny(action, hit.reason, 'caller element description matches a denied pattern', { tool: 'description', rule: hit.rule });
  }

  // `target` (0.0.84) may be a snapshot ref OR an arbitrary selector; only refs from the latest snapshot are
  // accepted, so the element we evaluated is exactly the element the server acts on.
  refElement(args) {
    const t = String(args.target ?? args.ref ?? '');
    return /^[A-Za-z0-9_-]{1,32}$/.test(t) ? (this.refs.get(t) || null) : null;
  }

  elementDecision(args, action) {
    const el = this.refElement(args);
    const d = this.policy.checkElement(this.surface(), el, action);
    if (!d.allowed) return d;
    return this.descriptionDecision(args, action) || d;
  }
}

function minimalEnv() {
  const dirs = [dirname(process.execPath), '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin'];
  return { PATH: [...new Set(dirs)].join(':'), HOME: process.env.HOME || '', LANG: process.env.LANG || 'en_US.UTF-8' };
}

export function buildConfig({ headed = false } = {}) {
  const cfg = JSON.parse(readFileSync(MCP_CONFIG_TEMPLATE, 'utf8'));
  delete cfg.$comment;
  cfg.browser.initPage = cfg.browser.initPage.map((p) => resolve(dirname(MCP_CONFIG_TEMPLATE), p));
  cfg.outputDir = resolve(dirname(MCP_CONFIG_TEMPLATE), cfg.outputDir);
  cfg.browser.launchOptions.headless = !headed;
  return cfg;
}

export async function main(argv = process.argv.slice(2)) {
  const fixturePorts = [];
  let headed = false;
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--fixture-port') fixturePorts.push(Number(argv[++i]));
    else if (argv[i] === '--headed') headed = true;
  }
  const audit = new AuditLog(undefined, { writer: 'node-mcp-proxy' });
  const policy = new GuiPolicy({ audit, actor: 'mac-agent-mcp' });
  for (const port of fixturePorts) if (Number.isInteger(port) && port > 0) policy.registerFixturePort(port);
  const cfg = buildConfig({ headed });
  const guard = new McpGuard(policy, { outputDir: cfg.outputDir });

  const tmp = mkdtempSync(resolve(tmpdir(), 'lat-mcp-'));
  const cfgPath = resolve(tmp, 'config.json');
  writeFileSync(cfgPath, JSON.stringify(cfg), { mode: 0o600 });
  const child = spawn(process.execPath, [MCP_CLI, '--config', cfgPath], {
    cwd: cfg.outputDir, env: minimalEnv(), stdio: ['pipe', 'pipe', 'inherit'],
  });
  audit.recordEvent('mcp_proxy_started', { mcp_cli: MCP_CLI, fixture_ports: fixturePorts, headed });

  const pending = new Map(); // request id -> tool name | '__list__'
  const toClient = (msg) => process.stdout.write(`${JSON.stringify(msg)}\n`);
  const toServer = (msg) => child.stdin.write(`${JSON.stringify(msg)}\n`);

  createInterface({ input: process.stdin }).on('line', (line) => {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg && msg.method === 'tools/call' && msg.id !== undefined) {
      const name = msg.params?.name;
      const args = msg.params?.arguments || {};
      const d = guard.check(name, args);
      if (!d.allowed) {
        toClient({ jsonrpc: '2.0', id: msg.id, result: { isError: true, content: [{ type: 'text', text: `DENIED by mac-agent policy [${d.reason}]: ${d.detail}` }] } });
        return;
      }
      pending.set(msg.id, name);
    } else if (msg && msg.method === 'tools/list' && msg.id !== undefined) {
      pending.set(msg.id, '__list__');
    }
    toServer(msg);
  }).on('close', () => { child.stdin.end(); });

  createInterface({ input: child.stdout }).on('line', (line) => {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg && msg.id !== undefined && pending.has(msg.id) && msg.method === undefined) {
      const kind = pending.get(msg.id);
      pending.delete(msg.id);
      if (kind === '__list__' && Array.isArray(msg.result?.tools)) {
        msg.result.tools = msg.result.tools.filter((t) => ALLOWED_TOOLS.has(t.name));
      } else if (msg.result) {
        guard.absorb(msg.result.content);
        const s = guard.surface();
        if (s.url && s.url !== 'about:blank' && !policy.classifyUrl(s.url).allowed) {
          audit.recordEvent('mcp_surface_violation', { tool: kind });
        }
      }
    }
    toClient(msg);
  });

  const cleanup = () => { try { rmSync(tmp, { recursive: true, force: true }); } catch { /* ignore */ } };
  child.on('exit', (code) => { cleanup(); process.exit(code ?? 0); });
  process.on('SIGTERM', () => { child.kill('SIGTERM'); });
  process.on('SIGINT', () => { child.kill('SIGTERM'); });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
