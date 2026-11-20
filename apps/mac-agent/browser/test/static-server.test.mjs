import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startFixtureServer, TEST_REDIRECT_PATH } from '../src/static-server.mjs';

test('fixture server: loopback only, GET/HEAD only, no traversal or listings', async () => {
  const srv = await startFixtureServer({ testRoutes: true });
  try {
    assert.match(srv.origin, /^http:\/\/127\.0\.0\.1:\d+$/);
    const ok = await fetch(srv.url('fake_paper_dashboard.html'));
    assert.equal(ok.status, 200);
    assert.match(await ok.text(), /FAKE PAPER DASHBOARD/);
    for (const bad of ['', '..%2fmac_agent%2frules.json', '%2e%2e/%2e%2e/.env', '.hidden.html', 'sub/x.html', 'x.sh']) {
      const r = await fetch(`${srv.origin}/${bad}`);
      assert.equal(r.status, 404, bad);
    }
    const post = await fetch(srv.url('fake_paper_dashboard.html'), { method: 'POST', body: 'x' });
    assert.equal(post.status, 405);
    const redir = await fetch(`${srv.origin}${TEST_REDIRECT_PATH}`, { redirect: 'manual' });
    assert.equal(redir.status, 302);
  } finally {
    await srv.close();
  }
});

test('fixture server without testRoutes has no redirect route', async () => {
  const srv = await startFixtureServer();
  try {
    const r = await fetch(`${srv.origin}${TEST_REDIRECT_PATH}`, { redirect: 'manual' });
    assert.equal(r.status, 404);
  } finally {
    await srv.close();
  }
});
