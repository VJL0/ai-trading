"""Shared contracts of the research crew (``services.agents``). Data only: no I/O, no imports of execution code.

The crew coordinates the operator's own Claude Code and Codex CLIs (subscription sign-in) for research roles.
It has no execution authority: agents get read-only research tools, the crew itself runs agent factor specs
through ``services.quantlab.views.run_agent_experiment`` (development data only, same gates and trial ledger as
every family) and stores agent output as ``AGENT_NOTE`` evidence that never becomes a fact or a promotion.
Nothing in the trading runtime imports this package; trading runs identically with the crew stopped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol


class Provider(StrEnum):
    CLAUDE = "claude"  # Claude Code CLI, `claude -p`, the operator's Claude subscription sign-in
    CODEX = "codex"    # Codex CLI, `codex exec`, the operator's ChatGPT sign-in


class TaskKind(StrEnum):
    HYPOTHESIS = "hypothesis"              # economic mechanism -> declarative factor specs (crew runs them)
    REVIEW = "review"                      # independent critique of another provider's hypothesis + results
    MARKET_ANALYSIS = "market_analysis"    # read-only scan/research digest -> observations + hypothesis seeds
    IMPROVEMENT = "improvement"            # failures/rejections -> proposals for the operator (never applied)


class TaskState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunOutcome(StrEnum):
    OK = "OK"                        # structured output parsed and schema-valid
    INVALID_OUTPUT = "INVALID_OUTPUT"  # process finished but output missing / not schema-valid
    USAGE_LIMIT = "USAGE_LIMIT"      # subscription usage / rate limit: provider cools down, task is requeued
    AUTH_REQUIRED = "AUTH_REQUIRED"  # not signed in: provider disabled until the operator signs in
    UNAVAILABLE = "UNAVAILABLE"      # CLI missing or not runnable
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


# Outcomes that say nothing about the task itself: the task goes back to QUEUED without consuming an attempt.
REQUEUE_OUTCOMES: Final = frozenset({RunOutcome.USAGE_LIMIT, RunOutcome.AUTH_REQUIRED, RunOutcome.UNAVAILABLE})

# Read-only research tools offered to crew agents (subset of services.tools.registry; never a mutating tool).
RESEARCH_MCP_SERVER_NAME: Final = "lat-research"
RESEARCH_TOOL_NAMES: Final[tuple[str, ...]] = (
    "get_capabilities", "get_research_results", "get_opportunities", "search_memory",
)


@dataclass(frozen=True)
class McpServerSpec:
    """How a provider CLI launches the read-only research MCP server (stdio)."""

    name: str
    command: str
    args: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None


@dataclass(frozen=True)
class Invocation:
    """One headless agent turn. ``output_schema`` is a JSON Schema the final answer must satisfy."""

    provider: Provider
    prompt: str
    output_schema: Mapping[str, Any]
    mcp_server: McpServerSpec
    allowed_tools: tuple[str, ...] = RESEARCH_TOOL_NAMES
    workdir: Path | None = None    # disposable scratch dir, never the repository
    timeout_s: int = 900
    max_turns: int = 12
    model: str | None = None       # None = the CLI's configured default


@dataclass(frozen=True)
class RunResult:
    outcome: RunOutcome
    output: dict[str, Any] | None = None          # parsed structured output (OK only)
    usage: Mapping[str, Any] = field(default_factory=dict)  # input_tokens, output_tokens, cost_usd, duration_s, ...
    retry_at: datetime | None = None              # USAGE_LIMIT: parsed reset time when the CLI states one
    detail: str = ""                              # short, sanitized diagnostic (never credentials)
    session_id: str | None = None


class AgentProvider(Protocol):
    @property
    def name(self) -> Provider: ...

    def availability(self) -> tuple[bool, str]:
        """(usable, reason) without spending quota: binary present and runnable; never prompts for sign-in."""
        ...

    def run(self, invocation: Invocation) -> RunResult:
        """Run one turn synchronously. Never raises for provider failures: map them to a ``RunOutcome``."""
        ...
