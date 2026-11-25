"""Runtime configuration from project .env and process overrides (prefix ``LAT_``).

Isolation: every mode namespace has its own ``var/<namespace>/`` directory (SQLite ledger, keys,
tokens, logs). Switching ``LAT_MODE`` therefore never reuses paper state or credentials for live.
"""

from __future__ import annotations

import ipaddress
import os
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

from packages.contracts.enums import AiRuntimeMode, FeedMode, GatewayKind, TradingMode

from .modes import DEFAULT_MODE, OperatorTradingMode, namespace
from .secrets import SecretStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


class ProjectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ENV_FILE, extra="ignore", frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        explicit_root: object = init_settings().get("project_root")
        root = explicit_root if isinstance(explicit_root, (str, Path)) else os.environ.get("LAT_PROJECT_ROOT")
        if root and isinstance(dotenv_settings, DotEnvSettingsSource) and dotenv_settings.env_file == PROJECT_ENV_FILE:
            dotenv_settings = DotEnvSettingsSource(settings_cls, env_file=Path(root) / ".env")
        return init_settings, env_settings, dotenv_settings, file_secret_settings


class Settings(ProjectSettings):
    model_config = SettingsConfigDict(env_prefix="LAT_")

    trading_mode: OperatorTradingMode | None = None
    mode: TradingMode = DEFAULT_MODE
    gateway: GatewayKind = Field(
        default=GatewayKind.FAKE,
        description="FAKE for simulation, ALPACA_PAPER for Alpaca paper execution",
    )
    feed_mode: FeedMode | None = Field(
        default=None, description="None = IEX_FREE when Alpaca paper keys exist, else SIMULATED"
    )
    project_root: Path = PROJECT_ROOT
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    agent_name: str = "mcp"
    mcp_log_level: str = "WARNING"
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
    quantlab_daily_enabled: bool = Field(
        default=False, description="after each XNYS close run fetch-delta, build-panel, scan and shadow in an isolated "
                                   "low-priority subprocess (free Alpaca data; research artifacts only)")
    quantlab_per_minute: int = Field(default=150, description="lower the QuantLab data request rate")
    use_keychain: bool = True
    log_level: str = "INFO"

    @field_validator("trading_mode", mode="before")
    @classmethod
    def normalize_operator_mode(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def resolve_operator_mode(self) -> Settings:
        if self.trading_mode == OperatorTradingMode.LIVE:
            object.__setattr__(self, "mode", TradingMode.LIVE_APPROVAL)
            object.__setattr__(self, "gateway", GatewayKind.LEAN_ALPACA_LIVE)
        elif self.trading_mode == OperatorTradingMode.PAPER:
            if self.mode in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE):
                raise ValueError("Paper conflicts with the internal live mode override")
            if self.mode == TradingMode.PAPER and "gateway" not in self.model_fields_set:
                secrets = SecretStore(self.project_root, use_keychain=self.use_keychain)
                if secrets.alpaca_paper_available():
                    object.__setattr__(self, "gateway", GatewayKind.ALPACA_PAPER)
        return self

    @field_validator("api_host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        if v == "localhost":
            return "127.0.0.1"
        if not ipaddress.ip_address(v).is_loopback:
            raise ValueError("API must bind to a loopback address only (localhost-only product boundary)")
        return v

    @property
    def api_url(self) -> str:
        host = f"[{self.api_host}]" if ":" in self.api_host else self.api_host
        return f"http://{host}:{self.api_port}"

    def market_data_feed(self, secrets: SecretStore) -> FeedMode:
        return self.feed_mode or (FeedMode.IEX_FREE if secrets.alpaca_paper_available() else FeedMode.SIMULATED)

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
