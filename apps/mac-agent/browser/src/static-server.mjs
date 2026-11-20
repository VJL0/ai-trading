// Minimal read-only static server for apps/mac-agent/fixtures, bound to 127.0.0.1 only.
//
// * GET/HEAD only; no directory listings; only regular files directly inside the fixtures directory
//   (no sub-paths, no traversal, no symlinks escaping the directory, no dotfiles).
// * `testRoutes: true` adds ONE test-only route, /__test/redirect-to-broker, which answers 302 to a
//   brokerage origin. It exists solely to prove that redirects are stopped by the browser guard; the
//   browser under test is launched with host-resolver rules that make every non-loopback host
//   unresolvable, so even a guard failure could not reach the network.

import { createServer } from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import { basename, extname, resolve, sep } from 'node:path';
import { DEFAULT_FIXTURES_DIR } from './rules.mjs';

const TYPES = { '.html': 'text/html; charset=utf-8', '.json': 'application/json', '.css': 'text/css', '.png': 'image/png' };
export const TEST_REDIRECT_PATH = '/__test/redirect-to-broker';
export const TEST_REDIRECT_TARGET = 'https://app.alpaca.markets/';

export async function startFixtureServer({ root = DEFAULT_FIXTURES_DIR, port = 0, testRoutes = false } = {}) {
  const rootReal = await realpath(root);
  const server = createServer(async (req, res) => {
    const send = (code, body = '', headers = {}) => {
      res.writeHead(code, { 'Content-Type': 'text/plain; charset=utf-8', 'X-Content-Type-Options': 'nosniff',
        'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer', ...headers });
      res.end(req.method === 'HEAD' ? undefined : body);
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') return send(405, 'method not allowed', { Allow: 'GET, HEAD' });
    let pathname;
    try { pathname = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname); } catch { return send(400, 'bad request'); }
    if (testRoutes && pathname === TEST_REDIRECT_PATH) return send(302, '', { Location: TEST_REDIRECT_TARGET });
    const name = pathname.replace(/^\/+/, '');
    if (!name || name !== basename(name) || name.startsWith('.') || name.includes('\\') || name.includes('\0')) {
      return send(404, 'not found');
    }
    const type = TYPES[extname(name).toLowerCase()];
    if (!type) return send(404, 'not found');
    try {
      const full = await realpath(resolve(rootReal, name));
      if (!full.startsWith(rootReal + sep)) return send(404, 'not found');
      if (!(await stat(full)).isFile()) return send(404, 'not found');
      const body = await readFile(full);
      return send(200, body, { 'Content-Type': type });
    } catch {
      return send(404, 'not found');
    }
  });
  await new Promise((ok, fail) => {
    server.once('error', fail);
    server.listen(port, '127.0.0.1', ok);
  });
  const addr = server.address();
  return {
    port: addr.port,
    origin: `http://127.0.0.1:${addr.port}`,
    url: (file) => `http://127.0.0.1:${addr.port}/${file}`,
    close: () => new Promise((ok) => { server.closeAllConnections?.(); server.close(() => ok()); }),
  };
}
