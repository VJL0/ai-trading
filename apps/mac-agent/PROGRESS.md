# mac-agent PROGRESS (attempt 3, 2026-11-08)

Inventory at start of attempt 3 (all UNVERIFIED from earlier attempts):
- mac_agent/{policy,audit,env,peekaboo}.py + rules.json exist; no verify.py, no tests, no fixture, no README.
- browser/: package.json (playwright 1.64.1, @playwright/mcp 0.0.84), pnpm-lock.yaml, node_modules; src/ and test/ empty.
- Host: peekaboo 4.9.7 at /opt/homebrew/bin/peekaboo (brew openclaw/tap 4.9.7); permissions status: Screen Recording, Accessibility, Event Synthesizing = Granted.
- claude mcp (local scope): playwright -> npx -y @playwright/mcp@0.0.84 (Connected). npm latest @playwright/mcp = 0.0.84.

Log:
- R1 policy: policy.py/rules.json reviewed; added record()/title_is_sensitive(), native surfaces must be allowlisted, loopback deny for /api/v1/agent, /api/v1/ai/chatgpt, /auth, callback (rules v3). Tests pending.
- R2 peekaboo.py rewritten for real 4.9.7 CLI: single-use snapshots (SNAPSHOT_STALE observed), tree-only inspect (--tree --no-screenshot), clean --snapshot. Manual smoke: Calculator 7+8= read back "15" via AX (StandardInputView;value:15). Tests pending.
- R4 fixture written: fixtures/fake_paper_dashboard.html.
- R3 browser/: rules.mjs (JS port of policy), audit.mjs (chain-compatible; py verify_chain verified node writers), guard.mjs (route guard + nav watcher + host-resolver lockdown), static-server.mjs, verify-fixture.mjs (12/12 checks pass headless), hold-fixture.mjs (headed, --force-renderer-accessibility), guarded-mcp.mjs proxy + mcp-init-page.cjs + mcp/playwright-mcp.config.json, mcp-smoke.mjs (16/16 checks pass via pinned local @playwright/mcp 0.0.84). Finding: Playwright route handler does NOT see redirect hops; host-resolver rules stop them. MCP 0.0.84 uses `target` (ref OR selector) -> proxy only accepts known refs.
- R5 verify.py written (calculator/notes/browser/all). Running next.
- R5 GUI runs (real, permissions granted): verify notes PASSED (note "local-ai-trader GUI test 2026-11-08T18:52:20Z" left in place; needed app focus + `type --foreground`, background typing refused INVALID_INPUT); verify browser: playwright 12/12, mcp 16/16, peekaboo pass PASSED after --depth 60 (default AX depth hid nested page text). Leftover from failed attempts: a Notes note "probe" (+ possibly empty "New Note").
