// Policy-enforcing wrapper around a Playwright Chromium context.
//
// Layers (each independently sufficient for the brokerage-denial requirement):
//   1. Pre-checks: goto()/click()/type()/press() evaluate mac_agent rules BEFORE acting
//      (URL allowlist, LIVE-surface detection, element deny patterns on the accessible name/role).
//   2. Route handler: every request of the context (navigations, sub-resources, fetch/XHR, redirects
//      that Playwright routes) is checked; non-allowlisted ones are aborted ('blockedbyclient').
//   3. Navigation watcher: if a frame ever commits to a non-allowlisted URL (e.g. a redirect that was
//      not routed), the page is sent to about:blank and the surface becomes UNKNOWN (all actions denied).
//   4. Chromium host-resolver rules: every host except localhost/127.0.0.1/[::1] resolves to NOTFOUND, so a
//      brokerage host cannot be reached even if layers 1-3 failed. Background networking is disabled.
//
// Element targeting is accessibility/role based (getByRole / getByLabel); coordinates are never used.

import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { Action, GuiPolicy, PolicyDenied, SurfaceKind } from './rules.mjs';

export const HARDENED_CHROMIUM_ARGS = [
  '--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost , EXCLUDE 127.0.0.1 , EXCLUDE [::1]',
  '--disable-background-networking',
  '--disable-component-update',
  '--disable-sync',
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-features=MediaRouter,OptimizationHints,Translate',
];

const KEY_ALIASES = { enter: 'return', arrowup: 'up', arrowdown: 'down', arrowleft: 'left', arrowright: 'right', esc: 'escape' };

export function normalizeKey(key) {
  const k = String(key).trim().toLowerCase();
  return KEY_ALIASES[k] || k;
}

// "- button "Place LIVE order" [ref=e3]" / "- textbox "Paper note"" -> { role, name }
export function parseAriaLine(line) {
  const m = /^\s*-\s+([A-Za-z]+)(?:\s+"((?:[^"\\]|\\.)*)")?/.exec(line || '');
  if (!m) return null;
  return { role: m[1], name: m[2] ? m[2].replace(/\\(.)/g, '$1') : '' };
}

export class GuardedBrowser {
  constructor(browser, context, page, policy, audit) {
    this.browser = browser;
    this.context = context;
    this.page = page;
    this.policy = policy;
    this.audit = audit;
    this.blocked = [];
    this.navViolations = [];
  }

  static async launch({ headless = true, policy = new GuiPolicy(), audit = policy.audit, slowMo = 0, nativeAccessibility = false } = {}) {
    // nativeAccessibility: expose the web-content accessibility tree to macOS AX clients (Peekaboo pass).
    const args = nativeAccessibility ? [...HARDENED_CHROMIUM_ARGS, '--force-renderer-accessibility'] : HARDENED_CHROMIUM_ARGS;
    const browser = await chromium.launch({ headless, args, slowMo });
    const context = await browser.newContext({ serviceWorkers: 'block', acceptDownloads: false, viewport: { width: 1100, height: 900 } });
    const g = new GuardedBrowser(browser, context, null, policy, audit);
    await context.route('**/*', (route, request) => g.onRoute(route, request));
    context.on('page', (p) => g.watch(p));
    const page = await context.newPage();
    g.page = page;
    return g;
  }

  async onRoute(route, request) {
    const d = this.policy.checkRequest(request.url(), request.method(), request.resourceType());
    if (!d.allowed) {
      this.blocked.push({ url: d.target.url, method: d.target.method, reason: d.reason });
      return route.abort('blockedbyclient');
    }
    return route.continue();
  }

  watch(page) {
    page.on('dialog', (dialog) => dialog.dismiss().catch(() => {}));
    page.on('download', (dl) => dl.cancel().catch(() => {}));
    page.on('framenavigated', (frame) => {
      const url = frame.url();
      if (url === 'about:blank' || url.startsWith('chrome-error://')) return;
      const c = this.policy.classifyUrl(url);
      if (!c.allowed) {
        this.navViolations.push({ url, reason: c.reason });
        this.audit.recordEvent('navigation_violation', { reason: c.reason });
        if (frame === page.mainFrame()) page.goto('about:blank').catch(() => {});
      }
    });
    if (this.page && page !== this.page) {
      // Popups are only kept when they open an allowlisted URL.
      const c = this.policy.classifyUrl(page.url());
      if (!c.allowed && page.url() !== 'about:blank') page.close().catch(() => {});
    }
  }

  async goto(url) {
    PolicyGuard.enforce(this.policy.checkUrl(url, Action.NAVIGATE));
    const resp = await this.page.goto(url, { waitUntil: 'domcontentloaded' });
    return resp ? resp.status() : null;
  }

  async reload() {
    PolicyGuard.enforce(this.policy.checkUrl(this.page.url(), Action.NAVIGATE));
    await this.page.reload({ waitUntil: 'domcontentloaded' });
  }

  async surface() {
    const url = this.page.url();
    let title = '';
    let texts = [];
    let mode = '';
    try {
      title = await this.page.title();
      ({ texts, mode } = await this.page.evaluate(() => ({
        texts: (document.body ? document.body.innerText : '').split('\n').map((t) => t.trim()).filter(Boolean).slice(0, 400),
        mode: document.documentElement.getAttribute('data-trading-mode') || '',
      })));
    } catch {
      return { kind: SurfaceKind.UNKNOWN, url, windowTitle: title, texts: [], tradingModeAttr: '' };
    }
    if (url.startsWith('chrome-error://')) return { kind: SurfaceKind.UNKNOWN, url, windowTitle: title, texts, tradingModeAttr: mode };
    return this.policy.browserSurface({ url, title, texts, tradingModeAttr: mode });
  }

  // Policy view of exactly one element: accessible role/name from Playwright's ARIA snapshot plus the
  // DOM attributes that could carry an action caption (aria-label, title, value, id, placeholder...).
  async describe(locator) {
    const n = await locator.count();
    if (n !== 1) return null; // ambiguous or missing -> fail closed
    const aria = parseAriaLine((await locator.ariaSnapshot({ timeout: 3000 })).split('\n')[0]);
    const dom = await locator.evaluate((el) => ({
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type') || '',
      ariaLabel: el.getAttribute('aria-label') || '',
      ariaDescription: el.getAttribute('aria-description') || '',
      roleDescription: el.getAttribute('aria-roledescription') || '',
      title: el.getAttribute('title') || '',
      id: el.id || '',
      name: el.getAttribute('name') || '',
      placeholder: el.getAttribute('placeholder') || '',
      // Only button-like elements carry their caption in .value; text inputs carry user content there.
      value: (el.tagName === 'BUTTON' || (el.tagName === 'INPUT' && ['submit', 'button', 'reset'].includes((el.getAttribute('type') || '').toLowerCase())))
        && typeof el.value === 'string' ? el.value : '',
      text: (el.innerText || el.textContent || '').trim().slice(0, 200),
    }));
    const role = aria?.role || '';
    const isPassword = dom.type.toLowerCase() === 'password';
    return {
      role,
      ax_role: isPassword ? 'secureTextField' : dom.tag,
      label: aria?.name || dom.ariaLabel || '',
      title: dom.title,
      description: dom.ariaDescription,
      identifier: [dom.id, dom.name].filter(Boolean).join(' '),
      help: role === 'textbox' ? '' : dom.text,
      role_description: dom.roleDescription,
      value: dom.value,
      placeholder: dom.placeholder,
      subrole: isPassword ? 'secure' : '',
    };
  }

  async check(locator, action) {
    const surface = await this.surface();
    const el = await this.describe(locator);
    return { surface, el, decision: this.policy.checkElement(surface, el, action) };
  }

  async click(locator) {
    const { decision } = await this.check(locator, Action.CLICK);
    PolicyGuard.enforce(decision);
    await locator.click({ timeout: 5000 });
    return decision;
  }

  async type(locator, text) {
    const surface = await this.surface();
    const el = await this.describe(locator);
    const decision = this.policy.checkText(surface, el, text, this.policy.browserRule());
    PolicyGuard.enforce(decision);
    await locator.fill(text, { timeout: 5000 });
    return decision;
  }

  async press(locator, key) {
    const surface = await this.surface();
    const el = await this.describe(locator);
    PolicyGuard.enforce(this.policy.checkElement(surface, el, Action.PRESS));
    PolicyGuard.enforce(this.policy.checkKeys(this.policy.browserRule(), [normalizeKey(key)], surface));
    await locator.press(key, { timeout: 5000 });
  }

  async screenshot(name) {
    const ts = new Date().toISOString().replace(/[-:.]/g, '');
    const path = resolve(this.policy.artifactsDir, `${name.replace(/[^A-Za-z0-9_.-]/g, '-').slice(0, 60)}-${ts}.png`);
    mkdirSync(this.policy.artifactsDir, { recursive: true });
    PolicyGuard.enforce(this.policy.checkArtifactPath(path));
    PolicyGuard.enforce(this.policy.checkSurface(await this.surface(), Action.SCREENSHOT));
    await this.page.screenshot({ path });
    this.audit.recordEvent('artifact_written', { kind: 'screenshot', path, app: 'playwright-chromium' });
    return path;
  }

  async close() {
    await this.context.close().catch(() => {});
    await this.browser.close().catch(() => {});
  }
}

export const PolicyGuard = {
  enforce(decision) {
    if (!decision.allowed) throw new PolicyDenied(decision);
    return decision;
  },
};
