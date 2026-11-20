// Policy parity: the same vectors are asserted by tests/unit/mac_agent/test_mac_agent_policy.py.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_FIXTURES_DIR, GuiPolicy, normalizeText, Rules } from '../src/rules.mjs';

const V = JSON.parse(readFileSync(new URL('./policy_vectors.json', import.meta.url), 'utf8'));
const sub = (u) => u.replace('{FIXTURES}', DEFAULT_FIXTURES_DIR);
const policy = () => {
  const p = new GuiPolicy();
  p.registerFixturePort(V.fixture_port);
  return p;
};

test('rules.json loads with a version', () => {
  assert.match(Rules.load().version, /^mac-agent-policy-v\d+$/);
});

test('url vectors', () => {
  const p = policy();
  for (const v of V.urls) {
    const c = p.classifyUrl(sub(v.url));
    assert.equal(c.allowed, v.allowed, `allowed mismatch for ${v.url}`);
    assert.equal(c.reason, v.reason, `reason mismatch for ${v.url}`);
  }
});

test('request vectors (loopback is read-only)', () => {
  const p = policy();
  for (const v of V.requests) {
    const d = p.checkRequest(v.url, v.method);
    assert.equal(d.allowed, v.allowed, `${v.method} ${v.url}`);
    assert.equal(d.reason, v.reason, `${v.method} ${v.url}`);
  }
});

test('element vectors on the fixture surface', () => {
  const p = policy();
  const surface = p.browserSurface({ url: `http://127.0.0.1:${V.fixture_port}/fake_paper_dashboard.html`, title: 'FAKE PAPER DASHBOARD — SIMULATED', tradingModeAttr: 'PAPER_SIMULATED' });
  for (const v of V.elements) {
    const d = v.action === 'type' ? p.checkText(surface, v.element, 'hello', p.browserRule()) : p.checkElement(surface, v.element, v.action);
    if (v.denied === null) assert.equal(d.allowed, true, `expected allowed: ${JSON.stringify(v.element)} -> ${d.reason}`);
    else assert.equal(d.reason, v.denied, `expected ${v.denied}: ${JSON.stringify(v.element)}`);
  }
});

test('text vectors', () => {
  const p = policy();
  for (const v of V.texts) {
    const prob = p.textProblem(v.text);
    assert.equal(prob ? prob.reason : null, v.problem, JSON.stringify(v.text));
  }
});

test('surface vectors (LIVE indications deny every interaction)', () => {
  const p = policy();
  for (const v of V.surfaces) {
    const s = p.browserSurface(v.surface);
    assert.equal(s.kind, v.kind, `kind for ${JSON.stringify(v.surface)}`);
    const d = p.checkSurface(s, v.action);
    assert.equal(d.allowed, v.allowed, JSON.stringify(v.surface));
    assert.equal(d.reason, v.reason, JSON.stringify(v.surface));
  }
});

test('normalizeText strips zero-width/bidi characters and folds width', () => {
  assert.equal(normalizeText('P​lace  ‮LIVE order'), 'Place LIVE order');
  assert.equal(normalizeText('ＢＵＹ'), 'BUY');
  assert.equal(normalizeText('a\x1cb\x85c'), 'a b c'); // Python str.split() whitespace parity
});

test('artifact paths are confined to artifacts/', () => {
  const p = policy();
  assert.equal(p.checkArtifactPath(`${p.artifactsDir}/shot-1.png`).allowed, true);
  assert.equal(p.checkArtifactPath(`${p.artifactsDir}/../escape.png`).allowed, false);
  assert.equal(p.checkArtifactPath(`${p.artifactsDir}/sub/x.png`).allowed, false);
  assert.equal(p.checkArtifactPath(`${p.artifactsDir}/x.sh`).allowed, false);
});

test('keys: only allowlisted browser keys', () => {
  const p = policy();
  const s = p.browserSurface({ url: `http://127.0.0.1:${V.fixture_port}/fake_paper_dashboard.html`, title: 'FAKE PAPER DASHBOARD — SIMULATED' });
  assert.equal(p.checkKeys(p.browserRule(), ['return'], s).allowed, true);
  assert.equal(p.checkKeys(p.browserRule(), ['cmd+l'], s).allowed, false);
  assert.equal(p.checkKeys(p.browserRule(), [], s).allowed, false);
});
