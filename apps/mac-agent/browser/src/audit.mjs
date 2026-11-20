// Append-only JSON-lines audit writer, byte-compatible with mac_agent/audit.py.
//
// * One record per line, written with a single O_APPEND write (no interleaving with the Python writer).
// * Per-writer hash chain: hash = sha256(prev + canonical(body)), canonical = sorted keys, no spaces,
//   non-ASCII kept as-is (== Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)).
// * Records never contain typed text (only its length), secrets or pixels; every string is scrubbed
//   before hashing so stored lines always verify (mac_agent.audit.verify_chain checks Node writers too).

import { createHash, randomBytes } from 'node:crypto';
import { closeSync, mkdirSync, openSync, writeSync, constants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { DEFAULT_ARTIFACTS_DIR, Rules } from './rules.mjs';

export const DEFAULT_AUDIT_PATH = resolve(DEFAULT_ARTIFACTS_DIR, 'audit.jsonl');
export const GENESIS = 'GENESIS';

// Port of packages.core.logging._REDACT_PATTERNS (applied first), then rules.json audit_redact_patterns.
const CORE_REDACT = [
  [/sk-ant-[A-Za-z0-9_-]{8,}/g, null],
  [/sk-[A-Za-z0-9_-]{16,}/g, null],
  [/\bPK[A-Z0-9]{14,}\b/g, null],
  [/(api[_-]?key|secret|token|authorization|password)(["'=:\s]+)([^\s"',}]{6,})/gi, 'kv'],
];

export function makeScrubber(rules = Rules.load()) {
  const extra = rules.auditRedact;
  return function scrub(text) {
    let out = String(text);
    for (const [pat, mode] of CORE_REDACT) {
      out = mode === 'kv' ? out.replace(pat, (_m, k, sepr) => `${k}${sepr}[REDACTED]`) : out.replace(pat, '[REDACTED]');
    }
    for (const pat of extra) out = out.replace(pat, '[REDACTED]');
    return out;
  };
}

export function scrubValue(value, scrub) {
  if (typeof value === 'string') return scrub(value);
  if (Array.isArray(value)) return value.map((v) => scrubValue(v, scrub));
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isInteger(value) ? value : scrub(String(value));
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      if (v === undefined) continue;
      out[scrub(k)] = scrubValue(v, scrub);
    }
    return out;
  }
  return scrub(String(value));
}

function sortDeep(value) {
  if (Array.isArray(value)) return value.map(sortDeep);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value).sort(comparePython)) out[k] = sortDeep(value[k]);
    return out;
  }
  return value;
}

// Python sorts str keys by code point; JS default sort compares UTF-16 code units. Compare code points.
function comparePython(a, b) {
  const A = [...a].map((c) => c.codePointAt(0));
  const B = [...b].map((c) => c.codePointAt(0));
  for (let i = 0; i < Math.min(A.length, B.length); i += 1) {
    if (A[i] !== B[i]) return A[i] - B[i];
  }
  return A.length - B.length;
}

export function canonicalJson(body) {
  return JSON.stringify(sortDeep(body));
}

export function chainHash(prev, body) {
  return createHash('sha256').update(prev + canonicalJson(body), 'utf8').digest('hex');
}

function nowIso() {
  return new Date().toISOString(); // 2026-11-08T18:00:00.123Z (millisecond precision, same as Python writer)
}

export class AuditLog {
  constructor(path = DEFAULT_AUDIT_PATH, { writer = 'node', rules = Rules.load() } = {}) {
    this.path = path;
    mkdirSync(dirname(path), { recursive: true });
    this.writerId = `${writer}-${randomBytes(6).toString('hex')}`;
    this.prev = GENESIS;
    this.seq = 0;
    this.scrub = makeScrubber(rules);
  }

  append(record) {
    this.seq += 1;
    const body = scrubValue({ ts: nowIso(), writer: this.writerId, seq: this.seq, ...record }, this.scrub);
    const digest = chainHash(this.prev, body);
    const stored = { ...body, prev: this.prev, hash: digest };
    const line = `${canonicalJson(stored)}\n`;
    const fd = openSync(this.path, constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT, 0o600);
    try { writeSync(fd, line); } finally { closeSync(fd); }
    this.prev = digest;
    return stored;
  }

  recordDecision(decision, { actor, policyVersion }) {
    this.append({
      kind: 'policy_decision', actor, policy_version: policyVersion, allowed: decision.allowed,
      action: decision.action, reason: decision.reason, detail: decision.detail, rule: decision.rule ?? null,
      target: decision.target ?? {},
    });
  }

  recordEvent(event, fields = {}) {
    this.append({ kind: 'event', event, ...fields });
  }
}
