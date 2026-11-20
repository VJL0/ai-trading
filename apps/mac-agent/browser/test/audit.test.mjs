import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AuditLog, canonicalJson, chainHash, makeScrubber } from '../src/audit.mjs';

test('canonical JSON + hash are byte-identical to the Python writer', () => {
  // Reference values computed with mac_agent.audit.canonical_json / chain_hash (Python).
  const body = { ts: '2026-11-08T00:00:00.000Z', writer: 'py-x', seq: 1, kind: 'event', event: 'probe', z: ['é', 1, null, true], a: { b: 'x\n"y' } };
  assert.equal(canonicalJson(body), '{"a":{"b":"x\\n\\"y"},"event":"probe","kind":"event","seq":1,"ts":"2026-11-08T00:00:00.000Z","writer":"py-x","z":["é",1,null,true]}');
  assert.equal(chainHash('GENESIS', body), 'cc936955ba9a8dea533b95005a7e63c99d4004b13cbfbae61572fe3220597c69');
});

test('records are chained, scrubbed and never contain secrets', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lat-audit-'));
  const path = join(dir, 'audit.jsonl');
  const log = new AuditLog(path, { writer: 'node-test' });
  log.recordEvent('probe', { note: 'key sk-ant-api03-SECRETSECRETSECRET and PKABCDEFGHIJKLMNOPQ', token: 'token=abcdef123456' });
  log.recordEvent('probe2', { n: 2 });
  const lines = readFileSync(path, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  assert.equal(lines.length, 2);
  assert.equal(lines[0].prev, 'GENESIS');
  assert.equal(lines[1].prev, lines[0].hash);
  const raw = readFileSync(path, 'utf8');
  assert.ok(!raw.includes('SECRETSECRET') && !raw.includes('PKABCDEFGHIJKLMNOPQ') && !raw.includes('abcdef123456'));
  for (const rec of lines) {
    const { prev, hash, ...body } = rec;
    assert.equal(chainHash(prev, body), hash);
  }
  // tampering breaks the chain
  const tampered = { ...lines[1], event: 'edited' };
  const { prev, hash, ...body } = tampered;
  assert.notEqual(chainHash(prev, body), hash);
  writeFileSync(join(dir, 'x'), '');
});

test('scrubber redacts core and rules patterns', () => {
  const s = makeScrubber();
  assert.equal(s('password: hunter2hunter2'), 'password: [REDACTED]');
  assert.ok(!s('ghp_abcdefghijklmnopqrstuvwxyz0123').includes('ghp_abc'));
});
