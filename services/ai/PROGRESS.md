# services/ai progress (append-only checkpoints)

- 2026-11-06 attempt 3 start: inventory = config.py, versions.py, injection.py (unverified), prompts/*.md. No tests existed.
  Verified facts: anthropic 2.6 + openai 3.46 SDKs use httpx2 (fake transports must be httpx2.MockTransport);
  pydantic_ai 2.66 sends output_config.format (NativeOutput) + thinking adaptive + output_config.effort for claude-opus-5-5
  and reasoning.effort + tool_choice=required for gpt-6.1-sol Responses (scratch experiment).
  Pricing: Opus 5.5 from claude-api skill; Sol verified from developers.openai.com model page (WebFetch 2026-11-06).
- Req 1-7 code written: adapters.py, fake_transport.py, schemas.py, evidence.py, injection.py (fixed: invisible chars
  re-decoded from HTML entities, hidden-text extras now stripped before scan, generic "tokens" FP), tools.py, gating.py
  (+ migrations/0200_ai_invocations.sql), accounting.py, pipeline.py. ruff+pyright clean. Fake TRADE path smoke-run OK
  (TRADE qty 1 SPY @500.04, both handlers 0 violations). Found: anthropic 2.6 SDK sends `authorization: Bearer`, not
  x-api-key. Next: req 8 tests.
- Req 8 tests written: tests/unit/ai (pipeline paths, adapters wire, tool surface, repeatability 5x, schemas/conversion,
  injection corpus, evidence, gating, accounting), tests/integration/ai (cross-service real fixture+signals+snapshot,
  gate restart, live_models smoke x3 skip), tests/failure_injection/ai. Corpus found 3 injection gaps (markup delimiter
  forging stripped before scan, ".env" period breaking SECRET regex, letter-spaced text losing word boundaries) -> fixed.
  `uv run --frozen pytest tests/unit/ai tests/integration/ai tests/failure_injection/ai -q` = 141 passed, 3 skipped.
  ruff + pyright clean. Left: README, self-verification pass, API-call compatibility check.
- 2026-11-08 attempt 4 (project update): re-ran baseline 141 passed/3 skipped. Added jws.py (RS256/ES256 + OIDC claims),
  token_store.py (Keychain via `security -i` stdin + read-back verify; Memory store), chatgpt_oauth.py (discovery,
  PKCE loopback listener, code exchange, id_token verify, scope gate, refresh lock/rotation, revoke+logout),
  chatgpt_responses.py (SSE Responses client, read-only shared function tools, bounded loop, error->pause/block),
  chatgpt_fake.py (MockTransport auth+models+SSE), runtime_modes.py (AiRuntime status/login/logout/research),
  prompts/chatgpt_research.md; pipeline: mode dispatch (NONE/CLAUDE_CODE/CODEX -> NO_TRADE no model, CHATGPT_OAUTH ->
  AiRuntime), optional shared tool_backend (read-only PydanticAI tools), persist_pipeline_result extracted;
  schemas.ProposalContext.require_deep_review. Smoke: fake sign-in -> research TRADE -> logout OK, 0 violations.
  ruff+pyright clean. Next: tests for the new parts.
- 2026-11-08 attempt 5 start: baseline `pytest tests/unit/ai tests/integration/ai tests/failure_injection/ai -q` = 282 passed, 3 skipped (tests for jws/token_store/oauth/responses/runtime_modes/shared tools exist). Missing: chat.py (REPL), README; then self-review.
- chat (update 2): services/ai/chat.py (REPL, live catalog via HttpToolBackend, confirm-before-action, plan SSE + API-key PydanticAI paths, mode explanations), chat_fake.py (ChatScript, FakeChatOpenAIHandler), prompts/chat_operator.md; chatgpt_oauth: cross-process flock + fresh store re-read on refresh/logout; researcher public hooks. tests/unit/ai/test_ai_chat.py 18 passed. Full ai suite 300 passed, 3 skipped; ruff+pyright clean. Left: refresh-race test, live chat test, README, self-review.
- Added: tests/integration/ai/test_ai_chat_real_api.py (chat vs REAL FastAPI app+Runtime over ASGITransport, confirmed run_quant_cycle once), test_ai_chatgpt_live.py (2 live_models, skip), failure_injection/ai/test_ai_chatgpt_shared_session.py (4; mutation-checked: removing fresh re-read or flock fails them); httpx loggers silenced in chat log. Suite: 305 passed, 5 skipped; ruff+pyright (src+tests) clean. Left: README, self-review.
