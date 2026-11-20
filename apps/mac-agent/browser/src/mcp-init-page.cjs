// Playwright MCP `browser.initPage` hook (loaded by @playwright/mcp with require(); default export).
//
// Installs the mac-agent request guard on every MCP-driven page's context BEFORE any tool runs:
//   * routes registered last run first in Playwright, so this handler sees each request before the
//     MCP server's own allowed/blocked-origin routes; non-allowlisted requests are aborted, allowed
//     ones fall back to the MCP routes (which then apply --allowed-origins as well);
//   * a navigation watcher sends the page to about:blank if a frame ever commits to a denied URL
//     (redirect hops are not routed by Playwright; the host-resolver rules in the launch args are the
//     final layer for those).
// The policy is the same rules.json used by the Python controller.

'use strict';

const guarded = new WeakSet();

module.exports = {
  default: async ({ page }) => {
    const { GuiPolicy, stripQuery } = require('./rules.mjs');
    const { AuditLog } = require('./audit.mjs');
    const audit = new AuditLog(undefined, { writer: 'node-mcp-page' });
    const policy = new GuiPolicy({ audit, actor: 'mac-agent-mcp-route' });
    const ctx = page.context();
    if (!guarded.has(ctx)) {
      guarded.add(ctx);
      await ctx.route('**/*', (route, request) => {
        const d = policy.checkRequest(request.url(), request.method(), request.resourceType());
        return d.allowed ? route.fallback() : route.abort('blockedbyclient');
      });
    }
    page.on('framenavigated', (frame) => {
      const url = frame.url();
      if (frame !== page.mainFrame() || url === 'about:blank' || url.startsWith('chrome-error://')) return;
      if (!policy.classifyUrl(url).allowed) {
        audit.recordEvent('navigation_violation', { url: stripQuery(url), source: 'mcp-init-page' });
        page.goto('about:blank').catch(() => {});
      }
    });
  },
};
