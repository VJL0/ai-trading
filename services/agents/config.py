"""Crew configuration from the environment (prefix ``LAT_AGENTS_``). Research crew only: nothing here reaches the
trading runtime. Budgets are deliberately small; quota is the scarce resource."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

from .contracts import Provider

ENV_PREFIX: Final = "LAT_AGENTS_"
COOLDOWN_CAP: Final = timedelta(hours=12)
MAX_ATTEMPTS: Final = 2
MAX_RETRY_AT: Final = timedelta(days=8)  # a CLI-stated reset further out than this is treated as a parse error

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]

# (suffix, default shown in .env.example)
_ENV: Final[tuple[tuple[str, str], ...]] = (
    ("PROVIDERS", "claude,codex"),
    ("CLAUDE_BIN", "claude"),
    ("CODEX_BIN", "codex"),
    ("MAX_CONCURRENT", "1"),
    ("DAILY_RUNS_PER_PROVIDER", "12"),
    ("RUN_TIMEOUT_S", "900"),
    ("MAX_TURNS", "12"),
    ("SPECS_PER_TASK", "3"),
    ("COOLDOWN_MINUTES", "60"),
    ("REVIEW_WAIT_HOURS", "24"),
    ("IDLE_SLEEP_S", "600"),
    ("STORE", ""),
    ("MODEL_CLAUDE", ""),
    ("MODEL_CODEX", ""),
)


def env_names() -> list[str]:
    """Every environment variable the crew reads (for ``.env.example``)."""
    return [ENV_PREFIX + s for s, _ in _ENV]


def env_defaults() -> dict[str, str]:
    return {ENV_PREFIX + s: d for s, d in _ENV}


def default_project_root(env: Mapping[str, str] | None = None) -> Path:
    """``LAT_PROJECT_ROOT`` when set (same override as ``packages.core.config``), else the repository root.
    Constructing ``Settings()`` would read credentials configuration, which the crew has no use for."""
    env = os.environ if env is None else env
    root = env.get("LAT_PROJECT_ROOT")
    return Path(root).expanduser().resolve() if root else _REPO_ROOT


class ConfigError(ValueError):
    pass


def _int(env: Mapping[str, str], suffix: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(ENV_PREFIX + suffix, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{suffix} must be an integer, got {raw!r}") from None
    if not lo <= val <= hi:
        raise ConfigError(f"{ENV_PREFIX}{suffix} must be in [{lo}, {hi}], got {val}")
    return val


def _providers(raw: str) -> tuple[Provider, ...]:
    out: list[Provider] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        try:
            p = Provider(name)
        except ValueError:
            raise ConfigError(f"{ENV_PREFIX}PROVIDERS: unknown provider {name!r} "
                              f"(known: {', '.join(x.value for x in Provider)})") from None
        if p not in out:
            out.append(p)
    return tuple(out)


@dataclass(frozen=True)
class CrewConfig:
    project_root: Path
    store_path: Path
    providers: tuple[Provider, ...] = (Provider.CLAUDE, Provider.CODEX)
    claude_bin: str = "claude"
    codex_bin: str = "codex"
    max_concurrent: int = 1
    daily_runs_per_provider: int = 12
    run_timeout_s: int = 900
    max_turns: int = 12
    specs_per_task: int = 3
    cooldown_minutes: int = 60
    review_wait_hours: int = 24
    idle_sleep_s: int = 600
    model_claude: str | None = None
    model_codex: str | None = None
    max_attempts: int = MAX_ATTEMPTS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, project_root: Path | None = None) -> CrewConfig:
        env = os.environ if env is None else env
        root = project_root or default_project_root(env)
        store_raw = env.get(ENV_PREFIX + "STORE", "").strip()
        store = Path(store_raw).expanduser() if store_raw else root / "experiments" / "agents" / "crew.sqlite3"
        if not store.is_absolute():
            store = root / store
        return cls(
            project_root=root,
            store_path=store,
            providers=_providers(env.get(ENV_PREFIX + "PROVIDERS", "claude,codex")),
            claude_bin=env.get(ENV_PREFIX + "CLAUDE_BIN", "").strip() or "claude",
            codex_bin=env.get(ENV_PREFIX + "CODEX_BIN", "").strip() or "codex",
            max_concurrent=_int(env, "MAX_CONCURRENT", 1, 1, 4),
            daily_runs_per_provider=_int(env, "DAILY_RUNS_PER_PROVIDER", 12, 0, 500),
            run_timeout_s=_int(env, "RUN_TIMEOUT_S", 900, 30, 7200),
            max_turns=_int(env, "MAX_TURNS", 12, 1, 100),
            specs_per_task=_int(env, "SPECS_PER_TASK", 3, 1, 10),
            cooldown_minutes=_int(env, "COOLDOWN_MINUTES", 60, 1, 720),
            review_wait_hours=_int(env, "REVIEW_WAIT_HOURS", 24, 1, 24 * 14),
            idle_sleep_s=_int(env, "IDLE_SLEEP_S", 600, 1, 86400),
            model_claude=env.get(ENV_PREFIX + "MODEL_CLAUDE", "").strip() or None,
            model_codex=env.get(ENV_PREFIX + "MODEL_CODEX", "").strip() or None,
        )

    def model_for(self, provider: Provider) -> str | None:
        return self.model_claude if provider is Provider.CLAUDE else self.model_codex

    def binary_for(self, provider: Provider) -> str:
        return self.claude_bin if provider is Provider.CLAUDE else self.codex_bin

    def backoff(self, consecutive_limits: int) -> timedelta:
        """Cooldown after the n-th consecutive limit: COOLDOWN_MINUTES * 2**(n-1), capped at 12 h."""
        n = max(1, consecutive_limits)
        minutes = self.cooldown_minutes * (2 ** min(n - 1, 20))
        return min(timedelta(minutes=minutes), COOLDOWN_CAP)

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "experiments" / "agents" / "runs"

    @property
    def lock_path(self) -> Path:
        return self.store_path.with_suffix(".lock")
