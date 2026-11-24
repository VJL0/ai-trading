"""Runtime configuration (env prefix ``LAT_``). Secrets are *not* here — see ``packages.core.secrets``.

Isolation: every mode namespace has its own ``var/<namespace>/`` directory (SQLite ledger, keys,
tokens, logs). Switching ``LAT_MODE`` therefore never reuses paper state or credentials for live.
"""

from __future__ import annotations

import ipaddress
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from packages.contracts.enums import AiRuntimeMode, FeedMode, GatewayKind, TradingMode

from .modes import DEFAULT_MODE, namespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAT_", env_file=None, extra="ignore", frozen=True)

    mode: TradingMode = DEFAULT_MODE
    gateway: GatewayKind = Field(
        default=GatewayKind.FAKE,
        description="FAKE until LEAN + Alpaca paper are available and explicitly selected",
    )
    feed_mode: FeedMode | None = Field(
        default=None, description="None = IEX_FREE when Alpaca paper keys exist, else SIMULATED"
    )
    project_root: Path = PROJECT_ROOT
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    dashboard_port: int = 5173
    lean_bridge_poll_ms: int = 500
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT")
    ai_runtime: AiRuntimeMode = Field(
        default=AiRuntimeMode.API_KEYS,
        description="NONE | API_KEYS | CLAUDE_CODE | CODEX | CHATGPT_OAUTH; the quant engine never depends on it",
    )
    chatgpt_model: str = "gpt-6.1-sol"
    chatgpt_oauth_port: int = Field(default=1455, description="loopback callback http://127.0.0.1:<port>/auth/callback")
    agent_max_proposals_per_hour: int = 6
    ai_transport: Literal["fake", "real"] = Field(
        default="fake",
        description="'real' calls paid model APIs and must be set explicitly; never inferred from keys",
    )
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    sol_model: str = "gpt-6.1-sol"
    sol_effort: Literal["low", "medium", "high"] = "medium"
    model_timeout_s: float = 120.0
    reconcile_interval_s: float = 30.0
    data_pipeline_enabled: bool = Field(
        default=False, description="run the history pipeline in the API process after each XNYS close (real feeds only)")
    data_pipeline_history_days: int = 730
    data_pipeline_max_requests: int = 200
    use_keychain: bool = True
    log_level: str = "INFO"

    @field_validator("api_host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        if v == "localhost":
            return "127.0.0.1"
        if not ipaddress.ip_address(v).is_loopback:
            raise ValueError("API must bind to a loopback address only (localhost-only product boundary)")
        return v

    @cached_property
    def namespace(self) -> str:
        return namespace(self.mode)

    @property
    def var_dir(self) -> Path:
        return self.project_root / "var" / self.namespace

    @property
    def db_path(self) -> Path:
        return self.var_dir / "ledger.sqlite"

    @property
    def keys_dir(self) -> Path:
        return self.var_dir / "keys"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "var" / "logs"

    @property
    def run_dir(self) -> Path:
        return self.project_root / "var" / "run"

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def analytics_db_path(self) -> Path:
        return self.datasets_dir / "analytics.duckdb"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments"

    @property
    def lean_dir(self) -> Path:
        return self.project_root / "trading" / "lean"

    def ensure_dirs(self) -> None:
        for d in (self.var_dir, self.keys_dir, self.log_dir, self.run_dir, self.datasets_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.keys_dir.chmod(0o700)


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
