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

.PHONY: help setup preflight dev test test-py test-lean lint typecheck backtest replay paper \
        data-pipeline data-snapshots compare-feeds \
        status reconcile stop audit lean-build fixture gui-verify sbom clean-run models mcp register-mcp lean-live-sim lean-paper halt resume chat

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## install runtimes and deps (uv, .NET 10 local), build LEAN, create fixtures and keys (SKIP_LEAN=1 to skip LEAN)
	uv sync --frozen --python 3.11
	@if [ ! -x "$(DOTNET_ROOT)/dotnet" ]; then \
	  mkdir -p .tools && curl -sSL https://dot.net/v1/dotnet-install.sh -o .tools/dotnet-install.sh && \
	  bash .tools/dotnet-install.sh --channel 10.0 --install-dir "$(DOTNET_ROOT)"; fi
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

dev: ## start the local API and monitor using .env settings
	$(PY) scripts/supervisor.py start api

stop: ## stop local services (broker-held orders are NOT cancelled; reconcile on restart)
	$(PY) scripts/supervisor.py stop

status: ## process + mode/entries-lock/feed/broker status
	$(PY) scripts/supervisor.py status

test: test-py test-lean ## run all automated tests (no credentials required)

TORCH_TESTS := tests/unit/research/test_kronos.py tests/unit/research/test_advanced_models.py

test-py: ## python suites excluding GUI, broker and paid-model opt-ins (torch tests in their own process)
	$(UV) pytest -q -m "not live_models and not paper_broker and not gui" $(addprefix --ignore=,$(TORCH_TESTS))
	$(UV) pytest -q -m "not live_models and not paper_broker and not gui" $(TORCH_TESTS)

test-lean: ## .NET tests for the LEAN gate/fee models (skips if LEAN not built)
	@if [ -f trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj ] && [ -x "$(DOTNET_ROOT)/dotnet" ]; then \
	  dotnet test trading/lean/LatStrategies.Tests/LatStrategies.Tests.csproj -c Release --nologo -v q; \
	else echo "LEAN tests skipped: build LEAN first (make lean-build)"; fi

lint: ## check Python lint
	$(UV) ruff check .

typecheck: ## pyright (strict for contracts/core/risk)
	$(UV) pyright

backtest: ## python no-LLM baselines on stored data + real LEAN backtest on LEAN sample data
	$(PY) -m services.research.cli baselines
	bash scripts/lean_backtest.sh

replay: ## deterministic end-to-end replay (SIMULATED clock, fake broker, fake model transport) into var/replay
	LAT_TRADING_MODE=PAPER LAT_MODE=REPLAY $(PY) -m services.api.cli replay

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

gui-verify: ## optional guarded mac-agent verification against local fixtures
	$(PY) -m mac_agent.verify all

sbom: ## software bill of materials for python, node and .NET dependencies
	$(PY) scripts/sbom.py

clean-run: ## remove generated per-run LEAN configs (never secrets in repo)
	rm -rf trading/lean/runs/*

.PHONY: demo smoke

demo: ## start an explicitly SIMULATED local app, no credentials or model calls
	LAT_TRADING_MODE=PAPER LAT_MODE=PAPER LAT_GATEWAY=FAKE LAT_FEED_MODE=SIMULATED LAT_AI_RUNTIME=NONE LAT_AI_TRANSPORT=fake LAT_USE_KEYCHAIN=false $(PY) scripts/supervisor.py start api

smoke: ## real-process MCP/API session smoke in an isolated temporary workspace, no credentials
	$(PY) -m scripts.smoke_paper

.PHONY: paper-sdk
paper-sdk: ## start official Alpaca SDK paper execution (requires paper credentials; no QuantConnect license)
	LAT_TRADING_MODE=PAPER LAT_MODE=PAPER LAT_GATEWAY=ALPACA_PAPER $(PY) -m services.api.cli paper-plan
	LAT_TRADING_MODE=PAPER LAT_MODE=PAPER LAT_GATEWAY=ALPACA_PAPER $(PY) scripts/supervisor.py start api

data-pipeline: ## backfill/increment real Alpaca history for the eligible universe, then build the research snapshot
	$(PY) -m services.market_data.cli pipeline

data-snapshots: ## list immutable research snapshots (verify one: ... snapshot verify --id rs_...)
	$(PY) -m services.market_data.cli snapshot list

compare-feeds: ## IEX vs historical SIP minute bars for the last 5 sessions (read-only, free plan)
	$(PY) -m services.market_data.cli compare-feeds --out var/reports/feed-compare.json

research: ## start the autonomous local research service (no execution authority)
	$(PY) scripts/supervisor.py start research

research-status: ## inspect recorded research, validation screens and worker failures
	$(PY) -m services.research.autonomous --status

.PHONY: agents agents-once agents-status
agents: ## start the research crew (Claude Code + Codex on your own sign-in; research only, no execution authority)
	$(PY) scripts/supervisor.py start agents

agents-once: ## plan and run at most one research-crew task in the foreground, then exit
	$(PY) -m services.agents run --once

agents-status: ## research crew queue, provider cooldowns, today's runs and recent notes
	$(PY) -m services.agents status

.PHONY: capabilities quantlab-data quantlab-research quantlab-daily quantlab-session quantlab-xsrank
QL := $(PY) -m services.quantlab.cli

capabilities: ## read-only Alpaca capability snapshot (asset classes, permissions, data entitlements)
	$(PY) -m services.capabilities probe

quantlab-data: ## multi-asset research data: daily SIP equities (+delisted), crypto, spreads, panels, ORB bars
	$(QL) fetch-equities && $(QL) fetch-crypto && $(QL) crypto-spreads && $(QL) build-panel && $(QL) orb-fetch

quantlab-research: ## development-stage research for every family (walk-forward, DSR, SPA, PBO; holdout untouched)
	$(QL) research --asset all

quantlab-session: ## session-timing research: 5-min SIP bars (dev period only), panel, families H1-H3/I1-I2
	$(QL) session-fetch && $(QL) build-panel --what session && $(QL) session-profile && $(QL) research --asset session

quantlab-xsrank: ## family K: learning-to-rank vs regression x horizons 1/5/20 x cost-aware construction (dev only)
	$(QL) research --asset xsrank

quantlab-daily: ## after the close: incremental bars, panel, scan + cross-asset selection, shadow journal
	$(QL) fetch-delta && $(QL) build-panel --what equity && $(QL) scan && $(QL) crypto-scan && $(QL) shadow
