"""AI-pipeline tunables (env prefix ``LAT_AI_``). Transport/model ids/efforts live in core ``Settings``.

``LAT_AI_TRANSPORT`` is read by ``packages.core.config.Settings.ai_transport`` (not here); nothing in
this module can switch the pipeline to the paid/real transport.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from packages.core.config import ProjectSettings


class AiConfig(ProjectSettings):
    model_config = SettingsConfigDict(env_prefix="LAT_AI_")

    # ---- invocation gating (never per tick) -------------------------------------------------
    min_interval_per_symbol_s: float = Field(default=900.0, ge=0, description="decision-clock seconds")
    min_wall_interval_per_symbol_s: float = Field(default=30.0, ge=0, description="wall-clock seconds")
    max_runs_per_day: int = Field(default=20, ge=0)
    daily_budget_usd: Decimal = Field(default=Decimal("10.00"), ge=0)
    est_max_cost_per_run_usd: Decimal = Field(
        default=Decimal("1.50"), ge=0, description="pessimistic reservation charged until the run's real cost is known"
    )
    budget_timezone: str = "America/New_York"

    # ---- request shaping ----------------------------------------------------------------------
    claude_max_tokens: int = Field(default=16000, gt=0, le=128000)
    sol_max_tokens: int = Field(default=16000, gt=0, le=128000)
    max_deep_reviews_per_run: int = Field(default=1, ge=0)
    sol_request_limit: int = Field(default=8, gt=0)
    claude_request_limit: int = Field(default=6, gt=0)
    tool_retries: int = Field(default=1, ge=0)
    output_retries: int = Field(default=0, ge=0, description="0 = schema/semantic-invalid output fails closed")
    run_timeout_s: float = Field(default=600.0, gt=0, description="wall-clock cap on one whole pipeline run")

    # ---- evidence freshness (seconds before data_asof) ----------------------------------------
    max_signal_age_s: float = Field(default=900.0, gt=0)
    max_news_age_s: float = Field(default=3 * 86400.0, gt=0)
    max_filing_age_s: float = Field(default=120 * 86400.0, gt=0)
    max_clock_skew_s: float = Field(default=2.0, ge=0)

    # ---- fake transport ----------------------------------------------------------------------
    fake_scenario: str = Field(default="TRADE", description="scripted FakeScenario used when transport=fake")


def load_ai_config(**overrides: object) -> AiConfig:
    return AiConfig(**overrides)  # type: ignore[arg-type]
