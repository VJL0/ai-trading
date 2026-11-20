// JavaScript port of mac_agent/policy.py for the Playwright guard and the guarded MCP proxy.
//
// Both languages load the SAME rules file (apps/mac-agent/mac_agent/rules.json) and are checked
// against the same vectors (test/policy_vectors.json) from `node --test` and from pytest, so a
// rule change cannot silently diverge between the native (Peekaboo) and browser (Playwright) paths.
//
// Default deny: an action is allowed only when an allow rule matches and no deny rule matches.

import { readFileSync, realpathSync } from 'node:fs';
import { dirname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const MAC_AGENT_ROOT = resolve(HERE, '..', '..');
export const RULES_PATH = resolve(MAC_AGENT_ROOT, 'mac_agent', 'rules.json');
export const DEFAULT_FIXTURES_DIR = resolve(MAC_AGENT_ROOT, 'fixtures');
export const DEFAULT_ARTIFACTS_DIR = resolve(MAC_AGENT_ROOT, 'artifacts');

// Same characters as policy._FORMAT_CHARS (soft hyphen, zero-width, bidi controls, word joiners, BOM).
const FORMAT_CHARS = /[­​-‏‪-‮⁠-⁤⁦-⁩﻿]/g;
// Python's str.split() whitespace set == JS \s plus U+001C..U+001F and U+0085.
const WHITESPACE = /[\s\x1c-\x1f\x85]+/;
const ARTIFACT_NAME = /^[A-Za-z0-9_.-]{1,96}\.(png|jpg)$/;
const PAPER_MODES = new Set(['paper', 'simulated', 'paper_simulated', 'replay', 'research_only']);

export const Action = Object.freeze({
  LIST: 'list', INSPECT: 'inspect', SCREENSHOT: 'screenshot', CLICK: 'click', TYPE: 'type', PRESS: 'press',
  LAUNCH: 'launch', FOCUS: 'focus', NAVIGATE: 'navigate', REQUEST: 'request',
});

export const SurfaceKind = Object.freeze({
  NATIVE_APP: 'NATIVE_APP', BROWSER_FIXTURE: 'BROWSER_FIXTURE', BROWSER_DASHBOARD: 'BROWSER_DASHBOARD',
  UNKNOWN: 'UNKNOWN',
});

export class PolicyDenied extends Error {
  constructor(decision) {
    super(`${decision.action} denied [${decision.reason}]: ${decision.detail}`);
    this.name = 'PolicyDenied';
    this.decision = decision;
  }
}

export function normalizeText(text) {
  if (typeof text !== 'string') return '';
  return text.normalize('NFKC').replace(FORMAT_CHARS, '').split(WHITESPACE).filter(Boolean).join(' ');
}

function compile(patterns) {
  // Python compiles with re.IGNORECASE | re.ASCII; for these ASCII-only patterns JS 'i' (no 'u') is equivalent.
  return patterns.map((p) => new RegExp(p, 'i'));
}

function firstMatch(patterns, texts) {
  for (const text of texts) {
    const norm = normalizeText(text);
    if (!norm) continue;
    for (const pat of patterns) {
      if (pat.test(norm)) return { rule: pat.source, text: norm };
    }
  }
  return null;
}

function isTextInputRole(role) {
  const r = String(role || '').toLowerCase().replace(/[ _]/g, '');
  return ['textfield', 'textarea', 'textbox', 'searchfield', 'combobox', 'searchbox', 'textinput', 'editable']
    .some((k) => r.includes(k));
}

export function elementActionTexts(el) {
  const isInput = isTextInputRole(el.role) || isTextInputRole(el.ax_role);
  const texts = [el.label, el.title, el.description, el.identifier, el.help, el.placeholder, el.role_description];
  if (!isInput) texts.push(el.value);
  return texts.filter((t) => typeof t === 'string' && t.length > 0);
}

function elementIsSecure(el) {
  return [el.role, el.ax_role, el.subrole, el.role_description]
    .some((r) => typeof r === 'string' && r.toLowerCase().includes('secure'));
}

export function stripQuery(url) {
  let u;
  try { u = new URL(url); } catch { return '<malformed>'; }
  if (u.protocol === 'file:') return `file://${u.pathname}`;
  if (u.protocol === 'http:' || u.protocol === 'https:') return `${u.protocol}//${u.hostname}${u.port ? `:${u.port}` : ''}${u.pathname}`;
  return `${u.protocol}<opaque>`;
}

// Mirror of Python urlsplit's netloc: the text between "scheme://" and the first of "/?#".
function rawAuthority(raw) {
  const m = /^[a-z][a-z0-9+.-]*:\/\/([^/?#]*)/i.exec(raw);
  return m ? m[1] : '';
}

function realOrResolved(p) {
  try { return realpathSync(p); } catch { return resolve(p); }
}

export class Rules {
  constructor(raw) {
    this.raw = raw;
    this.version = raw.version;
    this.apps = [
      ...raw.native_apps.map((a) => Rules.appRule(a, false)),
      ...raw.browser_apps.map((a) => Rules.appRule(a, true)),
    ];
    this.allowedSurfaceTitles = compile(raw.allowed_surface_titles);
    this.fixtureSurfaceTitles = compile(raw.fixture_surface_titles);
    this.denyApp = compile(raw.deny_app_patterns);
    this.denyHost = compile(raw.deny_host_patterns);
    this.loopbackHosts = new Set(raw.loopback_hosts.map((h) => h.toLowerCase()));
    this.loopbackAllowedMethods = new Set(raw.loopback_allowed_methods.map((m) => m.toUpperCase()));
    this.loopbackPathDeny = compile(raw.loopback_path_deny_patterns);
    this.elementDeny = compile(raw.element_deny_patterns);
    this.typeTargetDeny = compile(raw.type_target_deny_patterns);
    this.secureRoles = new Set(raw.secure_roles.map((r) => r.toLowerCase()));
    this.liveSurface = compile(raw.live_surface_patterns);
    this.sensitiveTitle = compile(raw.sensitive_title_patterns);
    this.secretText = compile(raw.secret_text_patterns);
    this.auditRedact = raw.audit_redact_patterns.map((p) => new RegExp(p, 'g'));
    this.maxTextLength = Number(raw.max_text_length);
  }

  static appRule(a, isBrowser) {
    return {
      key: a.key,
      names: a.names,
      bundleIds: a.bundle_ids,
      capabilities: new Set(a.capabilities),
      keys: new Set((a.keys || []).map((k) => k.toLowerCase())),
      typeCharset: a.type_charset ? new RegExp(a.type_charset) : null,
      isBrowser,
      displayName: a.names[0],
    };
  }

  static load(path = RULES_PATH) {
    return new Rules(JSON.parse(readFileSync(path, 'utf8')));
  }
}

export class NullAudit {
  recordDecision() {}
  recordEvent() {}
}

export class GuiPolicy {
  constructor({ rules = Rules.load(), audit = new NullAudit(), fixturesDir = DEFAULT_FIXTURES_DIR,
    artifactsDir = DEFAULT_ARTIFACTS_DIR, allowedLoopbackPorts = null, actor = 'mac-agent-browser' } = {}) {
    this.rules = rules;
    this.audit = audit;
    this.fixturesDir = realOrResolved(fixturesDir);
    this.artifactsDir = realOrResolved(artifactsDir);
    this.ports = allowedLoopbackPorts === null ? null : new Set(allowedLoopbackPorts);
    this.fixturePorts = new Set();
    this.actor = actor;
  }

  get version() { return this.rules.version; }

  allowLoopbackPort(port) { if (this.ports !== null) this.ports.add(port); }

  registerFixturePort(port) { this.fixturePorts.add(port); this.allowLoopbackPort(port); }

  decide(allowed, action, reason, detail, target = {}, rule = null) {
    const decision = { allowed, action, reason, detail, target, rule };
    this.audit.recordDecision(decision, { actor: this.actor, policyVersion: this.version });
    return decision;
  }

  static enforce(decision) {
    if (!decision.allowed) throw new PolicyDenied(decision);
    return decision;
  }

  appDenylisted(...names) {
    const hit = firstMatch(this.rules.denyApp, names.filter(Boolean));
    return hit ? hit.rule : null;
  }

  resolveApp(app) {
    const n = normalizeText(app).toLowerCase();
    return this.rules.apps.find((r) => [...r.names, ...r.bundleIds].some((x) => x.toLowerCase() === n)) || null;
  }

  // ---------------------------------------------------------------- URLs
  classifyUrl(url) {
    const raw = String(url ?? '').trim();
    if (raw === 'about:blank') return { allowed: true, reason: 'ALLOWED', detail: 'blank page', rule: null };
    if ([...raw].some((c) => c.codePointAt(0) < 0x20)) return deny('URL_MALFORMED', 'control characters in URL');
    const schemeMatch = /^([a-z][a-z0-9+.-]*):/i.exec(raw);
    const scheme = schemeMatch ? schemeMatch[1].toLowerCase() : '';
    if (['http', 'https', 'ws', 'wss'].includes(scheme)) {
      const authority = rawAuthority(raw);
      if (authority.includes('\\')) return deny('URL_MALFORMED', 'backslash in URL authority');
      if (authority.includes('@')) return deny('URL_MALFORMED', 'credentials in URL are refused');
      let u;
      try { u = new URL(raw); } catch { return deny('URL_MALFORMED', 'URL could not be parsed'); }
      const host = u.hostname.toLowerCase();
      if (!host) return deny('URL_MALFORMED', 'URL has no host');
      const hit = firstMatch(this.rules.denyHost, [host]);
      if (hit) return deny('BROKERAGE_DENIED', `host ${host} matches the brokerage/finance denylist`, hit.rule);
      if (!this.rules.loopbackHosts.has(host)) return deny('URL_NOT_ALLOWLISTED', `host ${host} is not loopback`);
      const port = u.port ? Number(u.port) : (scheme === 'https' || scheme === 'wss' ? 443 : 80);
      if (this.ports !== null && !this.ports.has(port)) {
        return deny('URL_NOT_ALLOWLISTED', `loopback port ${port} is not allowlisted`);
      }
      let path = u.pathname || '/';
      try { path = decodeURIComponent(path); } catch { /* keep raw path */ }
      const phit = firstMatch(this.rules.loopbackPathDeny, [path]);
      if (phit) return deny('LOOPBACK_PATH_DENIED', 'operator/engine/control paths are denied', phit.rule);
      if (this.fixturePorts.has(port) && path.includes('/..')) return deny('URL_MALFORMED', 'path traversal');
      return { allowed: true, reason: 'ALLOWED', detail: 'loopback dashboard/fixture origin', rule: null };
    }
    if (scheme === 'file') {
      const authority = rawAuthority(raw);
      if (authority !== '' && authority !== 'localhost') return deny('FILE_OUTSIDE_FIXTURES', 'remote file URL');
      if (this.fileUrlInFixtures(raw)) return { allowed: true, reason: 'ALLOWED', detail: 'file fixture', rule: null };
      return deny('FILE_OUTSIDE_FIXTURES', 'file:// is limited to apps/mac-agent/fixtures');
    }
    return deny('URL_NOT_ALLOWLISTED', `scheme ${scheme || '<none>'} is not allowlisted`);
  }

  fileUrlInFixtures(url) {
    let p;
    try {
      const u = new URL(url);
      if (u.protocol !== 'file:') return false;
      p = realOrResolved(fileURLToPath(u));
    } catch { return false; }
    return p.startsWith(this.fixturesDir + sep);
  }

  isFixtureUrl(url) {
    if (this.fileUrlInFixtures(url)) return true;
    try {
      const u = new URL(url);
      const port = u.port ? Number(u.port) : (u.protocol === 'https:' ? 443 : 80);
      return (u.protocol === 'http:' || u.protocol === 'https:') && this.rules.loopbackHosts.has(u.hostname.toLowerCase())
        && this.fixturePorts.has(port);
    } catch { return false; }
  }

  checkUrl(url, action = Action.NAVIGATE) {
    const c = this.classifyUrl(url);
    return this.decide(c.allowed, action, c.reason, c.detail, { url: stripQuery(url) }, c.rule);
  }

  checkRequest(url, method = 'GET', resourceType = '') {
    const c = this.classifyUrl(url);
    const m = String(method || 'GET').toUpperCase();
    const target = { url: stripQuery(url), method: m.slice(0, 12), resource_type: String(resourceType).slice(0, 24) };
    const scheme = (/^([a-z][a-z0-9+.-]*):/i.exec(String(url).trim()) || [])[1]?.toLowerCase();
    if (c.allowed && ['http', 'https', 'ws', 'wss'].includes(scheme) && !this.rules.loopbackAllowedMethods.has(m)) {
      return this.decide(false, Action.REQUEST, 'METHOD_NOT_ALLOWED',
        `${m} requests are refused for the GUI agent (read-only)`, target);
    }
    return this.decide(c.allowed, Action.REQUEST, c.reason, c.detail, target, c.rule);
  }

  // ---------------------------------------------------------------- surfaces
  surfaceIsLive(surface) {
    const attr = surface.tradingModeAttr || '';
    if (attr && !PAPER_MODES.has(normalizeText(attr).toLowerCase())) {
      return `trading-mode attribute=${JSON.stringify(attr.slice(0, 40))}`;
    }
    const hit = firstMatch(this.rules.liveSurface, [surface.windowTitle || '', ...(surface.texts || [])]);
    return hit ? hit.rule : null;
  }

  classifyBrowserTitle(title) {
    const norm = normalizeText(title);
    if (this.rules.fixtureSurfaceTitles.some((p) => p.test(norm))) return SurfaceKind.BROWSER_FIXTURE;
    if (this.rules.allowedSurfaceTitles.some((p) => p.test(norm))) return SurfaceKind.BROWSER_DASHBOARD;
    return SurfaceKind.UNKNOWN;
  }

  browserSurface({ url, title, texts = [], tradingModeAttr = '', appName = '', bundleId = '' }) {
    let kind = this.classifyBrowserTitle(title);
    const ok = this.classifyUrl(url).allowed;
    if (!ok) kind = SurfaceKind.UNKNOWN;
    else if (kind === SurfaceKind.BROWSER_FIXTURE && !this.isFixtureUrl(url)) {
      kind = this.rules.allowedSurfaceTitles.some((p) => p.test(normalizeText(title)))
        ? SurfaceKind.BROWSER_DASHBOARD : SurfaceKind.UNKNOWN;
    }
    return { kind, appName, bundleId, windowTitle: title, url, texts, tradingModeAttr };
  }

  surfaceSummary(s) {
    const out = { kind: s.kind };
    if (s.appName) out.app = s.appName;
    if (s.bundleId) out.bundle_id = s.bundleId;
    if (s.windowTitle) out.window_title = s.windowTitle.slice(0, 160);
    if (s.url) out.url = stripQuery(s.url);
    return out;
  }

  checkSurface(surface, action) {
    const target = this.surfaceSummary(surface);
    if (surface.kind === SurfaceKind.UNKNOWN) {
      return this.decide(false, action, 'SURFACE_UNIDENTIFIED',
        'surface could not be identified as an allowlisted app/page', target);
    }
    if (surface.appName || surface.bundleId) {
      const by = this.appDenylisted(surface.appName, surface.bundleId);
      if (by) return this.decide(false, action, 'APP_DENYLISTED', 'application is on the GUI denylist', target, by);
    }
    if (surface.url) {
      const c = this.classifyUrl(surface.url);
      if (!c.allowed) return this.decide(false, action, c.reason, c.detail, target, c.rule);
    }
    const live = this.surfaceIsLive(surface);
    if (live && action !== Action.LIST) {
      return this.decide(false, action, 'LIVE_SURFACE', 'surface indicates LIVE trading mode', target, live);
    }
    if (surface.windowTitle && [Action.SCREENSHOT, Action.INSPECT, Action.CLICK, Action.TYPE, Action.PRESS].includes(action)) {
      const hit = firstMatch(this.rules.sensitiveTitle, [surface.windowTitle]);
      if (hit) return this.decide(false, action, 'SENSITIVE_WINDOW', 'window title looks sensitive', target, hit.rule);
    }
    return this.decide(true, action, 'ALLOWED', 'surface allowlisted', target);
  }

  // ---------------------------------------------------------------- elements
  elementDenial(el, action) {
    const hit = firstMatch(this.rules.elementDeny, elementActionTexts(el));
    if (hit) return { reason: 'LIVE_ORDER_ELEMENT', rule: hit.rule };
    if (action === Action.TYPE) {
      const roles = [el.role, el.ax_role, el.subrole].filter(Boolean).map((r) => r.toLowerCase());
      if (elementIsSecure(el) || roles.some((r) => this.rules.secureRoles.has(r))) {
        return { reason: 'SENSITIVE_TARGET', rule: 'secure text field' };
      }
      const t = firstMatch(this.rules.typeTargetDeny, elementActionTexts(el));
      if (t) return { reason: 'SENSITIVE_TARGET', rule: t.rule };
    }
    return null;
  }

  elementSummary(el) {
    return {
      role: String(el.role || el.ax_role || '').slice(0, 64),
      label: String(el.label || el.title || el.description || el.identifier || '').slice(0, 120),
    };
  }

  checkElement(surface, el, action) {
    const surf = this.checkSurface(surface, action);
    if (!surf.allowed) return surf;
    const target = { ...this.surfaceSummary(surface), element: el ? this.elementSummary(el) : null };
    if (!el) return this.decide(false, action, 'ELEMENT_UNKNOWN', 'element is not in the current inspected snapshot', target);
    const d = this.elementDenial(el, action);
    if (d) {
      const detail = d.reason === 'LIVE_ORDER_ELEMENT'
        ? 'element matches a live-order / destructive-action pattern' : 'element is a sensitive input';
      return this.decide(false, action, d.reason, detail, target, d.rule);
    }
    return this.decide(true, action, 'ALLOWED', 'element allowed', target);
  }

  // ---------------------------------------------------------------- text / keys / paths
  textProblem(text, appRule = null) {
    if (typeof text !== 'string') return { reason: 'TEXT_INVALID', detail: 'text must be a string' };
    if ([...text].length > this.rules.maxTextLength) {
      return { reason: 'TEXT_INVALID', detail: `text longer than ${this.rules.maxTextLength} chars` };
    }
    for (const ch of text) {
      if (ch === '\n' || ch === '\t') continue;
      if (/\p{C}/u.test(ch)) {
        return { reason: 'TEXT_INVALID', detail: `control/format character U+${ch.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')} refused` };
      }
    }
    if (firstMatch(this.rules.secretText, [text])) {
      return { reason: 'SECRET_TEXT', detail: 'text looks like a credential/secret and is never typed' };
    }
    if (appRule && appRule.typeCharset && !appRule.typeCharset.test(text)) {
      return { reason: 'TEXT_INVALID', detail: `text not permitted for ${appRule.displayName}` };
    }
    return null;
  }

  checkText(surface, el, text, appRule = null) {
    const d = this.checkElement(surface, el, Action.TYPE);
    if (!d.allowed) return d;
    const target = { ...this.surfaceSummary(surface), text_len: [...String(text)].length };
    const p = this.textProblem(text, appRule);
    if (p) return this.decide(false, Action.TYPE, p.reason, p.detail, target);
    return this.decide(true, Action.TYPE, 'ALLOWED', 'text allowed', target);
  }

  checkKeys(appRule, keys, surface) {
    const surf = this.checkSurface(surface, Action.PRESS);
    if (!surf.allowed) return surf;
    const target = { ...this.surfaceSummary(surface), keys: keys.map((k) => String(k).slice(0, 24)) };
    if (!keys.length) return this.decide(false, Action.PRESS, 'KEY_NOT_ALLOWED', 'no keys', target);
    for (const k of keys) {
      if (!appRule.keys.has(String(k).trim().toLowerCase())) {
        return this.decide(false, Action.PRESS, 'KEY_NOT_ALLOWED', `key ${JSON.stringify(String(k).slice(0, 24))} is not allowlisted for ${appRule.displayName}`, target);
      }
    }
    return this.decide(true, Action.PRESS, 'ALLOWED', 'keys allowed', target);
  }

  checkArtifactPath(p, action = Action.SCREENSHOT) {
    const resolved = resolve(p);
    const target = { path: resolved };
    if (dirname(resolved) !== this.artifactsDir) {
      return this.decide(false, action, 'PATH_OUTSIDE_ARTIFACTS',
        'output files may only be written directly under apps/mac-agent/artifacts', target);
    }
    if (!ARTIFACT_NAME.test(resolved.slice(this.artifactsDir.length + 1))) {
      return this.decide(false, action, 'PATH_OUTSIDE_ARTIFACTS', 'bad artifact file name', target);
    }
    return this.decide(true, action, 'ALLOWED', 'artifact path allowed', target);
  }

  browserRule() {
    const r = this.rules.apps.find((a) => a.isBrowser);
    if (!r) throw new Error('rules.json has no browser app rule');
    return r;
  }
}

function deny(reason, detail, rule = null) {
  return { allowed: false, reason, detail, rule };
}
