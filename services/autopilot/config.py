"""Autopilot configuration from the environment (prefix ``LAT_AUTOPILOT_``).

Every value is bounded. Nothing here can select a trading mode, a gateway, a risk limit or a credential: the
autopilot reads the operator's own ``Settings`` for the API address and the agent token location only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

ENV_PREFIX: Final = "LAT_AUTOPILOT_"
_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_SYMBOL_RE: Final = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off", ""})

# (suffix, default shown in .env.example, description)
ENV_VARS: Final[tuple[tuple[str, str, str], ...]] = (
    ("ENABLED", "false", "master switch for `run`; status/plan always work"),
    ("TICK_S", "60", "longest single sleep; the wall clock is re-read after every sleep"),
    ("PREOPEN_LEAD_MIN", "60", "pre-open checks start this many minutes before the XNYS open"),
    ("START_GRACE_MIN", "5", "a session may still be requested this many minutes after the open, never later"),
    ("PREOPEN_RECHECK_MIN", "5", "re-run failed pre-open health checks at most this often inside the window"),
    ("INTRADAY_INTERVAL_MIN", "15", "one bounded status check per interval during regular hours"),
    ("SETTLE_MIN", "30", "after-close research starts this many minutes after the close"),
    ("SESSION_MINUTES", "", "cap on the session length; empty = full regular session minus the entry cutoff"),
    ("CADENCE_MINUTES", "", "minutes between session cycles; empty = the session manager default"),
    ("SYMBOLS", "", "comma-separated tickers; empty = the eligible universe"),
    ("USE_AI", "false", "in-app model review inside session cycles"),
    ("START_API", "true", "start the local API if it is not running (never restarts a running one)"),
    ("RESEARCH", "true", "run after-close research steps (quantlab daily, crew, shadow, holdout)"),
    ("CREW_TICKS", "2", "research-crew ticks per evening when the lead agent is off (`services.agents run --once`)"),
    ("CREW_TIMEOUT_S", "3600", "timeout per crew tick subprocess"),
    ("HOLDOUT", "true", "let a DEV_PASSED family take its single protected-holdout look"),
    ("STEP_TIMEOUT_S", "7200", "timeout per quantlab subprocess step"),
    ("LEAD", "true", "launch the lead agent (`services.agents lead`) through the trading day and the evening; it "
                     "decides and delegates the AI work (replaces the fixed crew ticks)"),
    ("LEAD_MAX_CYCLES", "10", "lead turns per trading day (each turn plus its delegations is one cycle)"),
    ("LEAD_MIN_INTERVAL_MIN", "20", "minimum minutes between two lead cycles, even on new evidence"),
    ("LEAD_MAX_INTERVAL_MIN", "180", "longest gap between lead cycles inside the lead's window"),
    ("LEAD_TIMEOUT_S", "2700", "wall-clock budget of one lead cycle (lead turn plus parallel subagents)"),
    ("LEAD_SESSION_ACTIONS", "true", "honour the lead's fail-safe requests (stand down today, request a halt)"),
    ("LEAD_FAKE", "false", "simulated lead and subagents (disposable workspaces and demos only)"),
    ("STATE_DIR", "experiments/autopilot", "state directory (SQLite store, lock, logs)"),
    ("REPORT_DIR", "experiments/autopilot/reports", "daily reports YYYY-MM-DD.{json,md}"),
)


class ConfigError(ValueError):
    pass


def env_names() -> list[str]:
    """Every environment variable the autopilot reads (for ``.env.example``)."""
    return [ENV_PREFIX + s for s, _, _ in ENV_VARS]


def project_root(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    root = env.get("LAT_PROJECT_ROOT")
    return Path(root).expanduser().resolve() if root else _REPO_ROOT


def _raw(env: Mapping[str, str], suffix: str) -> str:
    return env.get(ENV_PREFIX + suffix, "").strip()


def _int(env: Mapping[str, str], suffix: str, default: int, lo: int, hi: int) -> int:
    raw = _raw(env, suffix)
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ConfigError(f"{ENV_PREFIX}{suffix} must be an integer, got {raw!r}") from None
    if not lo <= val <= hi:
        raise ConfigError(f"{ENV_PREFIX}{suffix} must be in [{lo}, {hi}], got {val}")
    return val


def _opt_int(env: Mapping[str, str], suffix: str, lo: int, hi: int) -> int | None:
    return None if not _raw(env, suffix) else _int(env, suffix, lo, lo, hi)


def _bool(env: Mapping[str, str], suffix: str, default: bool) -> bool:
    raw = _raw(env, suffix).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{ENV_PREFIX}{suffix} must be true or false, got {raw!r}")


def _symbols(raw: str) -> tuple[str, ...] | None:
    if not raw:
        return None
    out: list[str] = []
    for part in raw.split(","):
        sym = part.strip().upper()
        if not sym:
            continue
        if not _SYMBOL_RE.fullmatch(sym):
            raise ConfigError(f"{ENV_PREFIX}SYMBOLS: {sym!r} is not a ticker")
        if sym not in out:
            out.append(sym)
    if len(out) > 10:
        raise ConfigError(f"{ENV_PREFIX}SYMBOLS: at most 10 tickers (the session manager cap)")
    return tuple(out) or None


def _path(root: Path, raw: str, default: str) -> Path:
    p = Path(raw or default).expanduser()
    return p if p.is_absolute() else root / p


@dataclass(frozen=True)
class AutopilotConfig:
    project_root: Path
    state_dir: Path
    report_dir: Path
    enabled: bool = False
    tick_s: int = 60
    preopen_lead_min: int = 60
    start_grace_min: int = 5
    preopen_recheck_min: int = 5
    intraday_interval_min: int = 15
    settle_min: int = 30
    session_minutes: int | None = None
    cadence_minutes: int | None = None
    symbols: tuple[str, ...] | None = None
    use_ai: bool = False
    start_api: bool = True
    research: bool = True
    crew_ticks: int = 2
    crew_timeout_s: int = 3600
    holdout: bool = True
    step_timeout_s: int = 7200
    lead: bool = True
    lead_max_cycles: int = 10
    lead_min_interval_min: int = 20
    lead_max_interval_min: int = 180
    lead_timeout_s: int = 2700
    lead_session_actions: bool = True
    lead_fake: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, root: Path | None = None) -> AutopilotConfig:
        base = root or project_root(env)
        if env is None:
            file_env = {key: value for key, value in dotenv_values(base / ".env", interpolate=False).items()
                        if key.startswith(ENV_PREFIX) and value is not None}
            env = {**file_env, **os.environ}
        return cls(
            project_root=base,
            state_dir=_path(base, _raw(env, "STATE_DIR"), "experiments/autopilot"),
            report_dir=_path(base, _raw(env, "REPORT_DIR"), "experiments/autopilot/reports"),
            enabled=_bool(env, "ENABLED", False),
            tick_s=_int(env, "TICK_S", 60, 5, 900),
            preopen_lead_min=_int(env, "PREOPEN_LEAD_MIN", 60, 5, 240),
            start_grace_min=_int(env, "START_GRACE_MIN", 5, 0, 30),
            preopen_recheck_min=_int(env, "PREOPEN_RECHECK_MIN", 5, 1, 60),
            intraday_interval_min=_int(env, "INTRADAY_INTERVAL_MIN", 15, 5, 120),
            settle_min=_int(env, "SETTLE_MIN", 30, 5, 180),
            session_minutes=_opt_int(env, "SESSION_MINUTES", 15, 390),
            cadence_minutes=_opt_int(env, "CADENCE_MINUTES", 1, 60),
            symbols=_symbols(_raw(env, "SYMBOLS")),
            use_ai=_bool(env, "USE_AI", False),
            start_api=_bool(env, "START_API", True),
            research=_bool(env, "RESEARCH", True),
            crew_ticks=_int(env, "CREW_TICKS", 2, 0, 12),
            crew_timeout_s=_int(env, "CREW_TIMEOUT_S", 3600, 60, 4 * 3600),
            holdout=_bool(env, "HOLDOUT", True),
            step_timeout_s=_int(env, "STEP_TIMEOUT_S", 7200, 60, 6 * 3600),
            lead=_bool(env, "LEAD", True),
            lead_max_cycles=_int(env, "LEAD_MAX_CYCLES", 10, 0, 48),
            lead_min_interval_min=_int(env, "LEAD_MIN_INTERVAL_MIN", 20, 5, 240),
            lead_max_interval_min=_int(env, "LEAD_MAX_INTERVAL_MIN", 180, 15, 24 * 60),
            lead_timeout_s=_int(env, "LEAD_TIMEOUT_S", 2700, 300, 4 * 3600),
            lead_session_actions=_bool(env, "LEAD_SESSION_ACTIONS", True),
            lead_fake=_bool(env, "LEAD_FAKE", False),
        )

    @property
    def store_path(self) -> Path:
        return self.state_dir / "autopilot.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.project_root / "experiments" / "autopilot" / "autopilot.lock"

    @property
    def lead_briefing_path(self) -> Path:
        return self.state_dir / "lead" / "briefing.json"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    def public(self) -> dict[str, object]:
        return {
            "enabled": self.enabled, "tick_s": self.tick_s, "preopen_lead_min": self.preopen_lead_min,
            "start_grace_min": self.start_grace_min, "preopen_recheck_min": self.preopen_recheck_min,
            "intraday_interval_min": self.intraday_interval_min, "settle_min": self.settle_min,
            "session_minutes": self.session_minutes, "cadence_minutes": self.cadence_minutes,
            "symbols": list(self.symbols) if self.symbols else None, "use_ai": self.use_ai,
            "start_api": self.start_api, "research": self.research, "crew_ticks": self.crew_ticks,
            "crew_timeout_s": self.crew_timeout_s, "holdout": self.holdout, "step_timeout_s": self.step_timeout_s,
            "lead": self.lead, "lead_max_cycles": self.lead_max_cycles,
            "lead_min_interval_min": self.lead_min_interval_min, "lead_max_interval_min": self.lead_max_interval_min,
            "lead_timeout_s": self.lead_timeout_s, "lead_session_actions": self.lead_session_actions,
            "lead_fake": self.lead_fake,
            "state_dir": str(self.state_dir), "report_dir": str(self.report_dir),
        }
