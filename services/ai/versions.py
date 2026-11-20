"""Adapter and prompt versioning. Every ModelRunRecord carries ADAPTER_VERSION and the prompt version.

Prompt versions are content hashes of ``services/ai/prompts/<name>.md`` so any edit to a prompt
produces a new version string without manual bookkeeping.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pydantic_ai

AI_PACKAGE_VERSION = "1.0.0"
ADAPTER_VERSION = f"lat-ai-adapters/{AI_PACKAGE_VERSION} pydantic-ai/{pydantic_ai.__version__}"

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SOL_PROMPT = "sol_orchestrator"
CLAUDE_PROMPT = "claude_research"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    if not name.isidentifier():
        raise ValueError(f"bad prompt name {name!r}")
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def prompt_sha256(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()


def prompt_version(name: str) -> str:
    """e.g. ``claude_research@3f2a...`` (first 16 hex chars of the prompt file's SHA-256)."""
    return f"{name}@{prompt_sha256(name)[:16]}"


def prompt_bundle_version() -> str:
    """Hash over every prompt file (name + content), for manifests and the README."""
    h = hashlib.sha256()
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return f"prompts@{h.hexdigest()[:16]}"
