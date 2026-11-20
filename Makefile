# local-ai-trader — every target runs locally on this Mac. PAPER/SIMULATED by default; live is compiled off.
SHELL := /bin/bash
.DEFAULT_GOAL := help
ROOT := $(abspath .)
UV := uv run --frozen
PY := $(UV) python
export DOTNET_ROOT := $(ROOT)/.tools/dotnet
export PATH := $(DOTNET_ROOT):$(PATH)
export DOTNET_CLI_TELEMETRY_OPTOUT := 1
export DOTNET_NOLOGO := 1

.PHONY: help setup preflight dev test test-py test-dashboard test-lean lint typecheck backtest replay paper \
        status reconcile stop audit lean-build fixture gui-verify sbom clean-run models mcp register-mcp lean-live-sim lean-paper halt resume chat

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## install runtimes + deps (uv, .NET 10 local, pnpm), build LEAN, create fixtures and keys (SKIP_LEAN=1 to skip LEAN)
	uv sync --frozen --python 3.11
	@if [ ! -x "$(DOTNET_ROOT)/dotnet" ]; then \
	  mkdir -p .tools && curl -sSL https://dot.net/v1/dotnet-install.sh -o .tools/dotnet-install.sh && \
	  bash .tools/dotnet-install.sh --channel 10.0 --install-dir "$(DOTNET_ROOT)"; fi
	cd apps/dashboard && pnpm install --frozen-lockfile
	@if [ -f apps/mac-agent/browser/package.json ]; then cd apps/mac-agent/browser && pnpm install --frozen-lockfile; fi
	$(PY) -m services.api.cli init
	$(PY) -m services.market_data.cli fixture
	@if [ -d trading/lean/Lean/Data ]; then $(PY) -m services.market_data.cli import-lean-sample; fi
	@if [ "$(SKIP_LEAN)" != "1" ]; then $(MAKE) lean-build; fi
	@echo "setup complete — run: make preflight && make test && make dev"

preflight: ## introspect this Mac and report actionable requirement status
	$(PY) scripts/preflight.py

lean-build: ## build pinned LEAN + LatStrategies for research/backtesting
	bash scripts/lean_build.sh

fixture: ## (re)generate the deterministic SYNTHETIC fixture and LEAN-format export
	$(PY) -m services.market_data.cli fixture
	$(PY) -m services.market_data.cli export-lean --out trading/lean/data-local

dev: ## start API (127.0.0.1:8765) + dashboard (127.0.0.1:5173) detached
	$(PY) scripts/supervisor.py start api dashboard

stop: ## stop local services (broker-held orders are NOT cancelled; reconcile on restart)
	$(PY) scripts/supervisor.py stop

status: ## process + mode/entries-lock/feed/broker status
	$(PY) scripts/supervisor.py status

test: test-py test-dashboard test-lean ## run all automated tests (no credentials required)

test-py: ## python suites excluding GUI, broker and paid-model opt-ins
	$(UV) pytest -q -m "not live_models and not paper_broker and not gui"

test-dashboard: ## vitest unit tests for the dashboard
	cd apps/dashboard && pnpm test

test-lean: ## .NET tests for the LEAN gate/fee models (skips if LEAN not built)
	@if [ -f trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj ] && [ -x "$(DOTNET_ROOT)/dotnet" ]; then \
	  dotnet test trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj -c Release --nologo -v q; \
	else echo "LEAN tests skipped: build LEAN first (make lean-build)"; fi

lint: ## ruff + dashboard type-aware lint
	$(UV) ruff check .
	cd apps/dashboard && pnpm lint

typecheck: ## pyright (strict for contracts/core/risk) + tsc
	$(UV) pyright
	cd apps/dashboard && pnpm typecheck

backtest: ## python no-LLM baselines on stored data + real LEAN backtest on LEAN sample data
	$(PY) -m services.research.cli baselines
	bash scripts/lean_backtest.sh

replay: ## deterministic end-to-end replay (SIMULATED clock, fake broker, fake model transport) into var/replay
	LAT_MODE=REPLAY $(PY) -m services.api.cli replay

paper: paper-sdk ## run official Alpaca SDK PAPER execution (requires paper credentials)

reconcile: ## broker-authoritative reconciliation now (uses the running API if up, else direct)
	$(PY) -m services.api.cli reconcile

halt: ## KILL SWITCH: stop new orders now (REASON="..."); works with or without the API running
	$(PY) -m services.api.cli halt --reason "$${REASON:-operator stop via make halt}"

resume: ## clear a halt (HALT_ID=...; needs the operator token and a CLEAN reconciliation)
	$(PY) -m services.api.cli resume "$(HALT_ID)"

chat: ## natural-language chat (Sign in with ChatGPT or API-key modes); Claude Code/Codex users talk to their own client
	$(PY) -m services.ai.chat

audit: ## verify audit hash chain, print recent audit events and the SBOM/licence summary
	$(PY) -m services.api.cli audit

lean-live-sim: ## retired execution command; use demo for simulation
	bash scripts/lean_live_sim.sh

lean-paper: ## retired execution command; use paper-sdk
	bash scripts/lean_paper_alpaca.sh

models: ## train + walk-forward-select the local quant model suite on the stored dataset (no AI needed)
	$(PY) -m services.research.cli train-models

mcp: ## run the local MCP server on stdio (normally launched by Claude Code / Codex, not by hand)
	$(PY) -m services.mcp_server

register-mcp: ## show/perform official MCP registration for Claude Code (project scope) and Codex
	bash scripts/register_mcp.sh

gui-verify: ## Peekaboo (Calculator/Notes) + Playwright (fake paper dashboard) verification
	$(PY) -m mac_agent.verify all

sbom: ## software bill of materials for python, node and .NET dependencies
	$(PY) scripts/sbom.py

clean-run: ## remove generated per-run LEAN configs (never secrets in repo)
	rm -rf trading/lean/runs/*

.PHONY: demo smoke test-dashboard-e2e

demo: ## start an explicitly SIMULATED local app, no credentials or model calls
	LAT_MODE=PAPER LAT_GATEWAY=FAKE LAT_FEED_MODE=SIMULATED LAT_AI_RUNTIME=NONE LAT_AI_TRANSPORT=fake LAT_USE_KEYCHAIN=false $(PY) scripts/supervisor.py start api dashboard

smoke: ## real-process MCP/API session smoke in an isolated temporary workspace, no credentials
	$(PY) -m scripts.smoke_paper

test-dashboard-e2e: ## built dashboard with real SSE fixture server in headless Chromium
	cd apps/dashboard && pnpm e2e

.PHONY: paper-sdk
paper-sdk: ## start official Alpaca SDK paper execution (requires paper credentials; no QuantConnect license)
	LAT_MODE=PAPER LAT_GATEWAY=ALPACA_PAPER $(PY) -m services.api.cli paper-plan
	LAT_MODE=PAPER LAT_GATEWAY=ALPACA_PAPER LAT_FEED_MODE=IEX_FREE $(PY) scripts/supervisor.py start api dashboard

research: ## start the autonomous local research service (no execution authority)
	$(PY) scripts/supervisor.py start research

research-status: ## inspect recorded research, validation screens and worker failures
	$(PY) -m services.research.autonomous --status
