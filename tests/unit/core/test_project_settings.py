from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.contracts.enums import ExecutionLabel, GatewayKind, TradingMode
from packages.core.config import Settings
from packages.core.modes import OperatorTradingMode, operator_trading_mode
from packages.core.secrets import SecretName, SecretStore
from services.ai.chat import ChatConfig
from services.ai.chatgpt_responses import ChatGptConfig
from services.ai.config import AiConfig


def test_workspace_dotenv_configures_runtime_ai_and_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path))
    for name in ("LAT_API_HOST", "LAT_SYMBOLS", "LAT_RECONCILE_INTERVAL_S", "LAT_AI_MAX_RUNS_PER_DAY",
                 "LAT_CHAT_MAX_HISTORY_TURNS", "LAT_CHATGPT_MAX_TURNS", "ALPACA_PAPER_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        'LAT_API_HOST=::1\nLAT_API_PORT=18766\nLAT_RECONCILE_INTERVAL_S=18\nLAT_SYMBOLS=["SPY"]\n'
        'LAT_AI_MAX_RUNS_PER_DAY=8\nLAT_CHAT_MAX_HISTORY_TURNS=4\nLAT_CHATGPT_MAX_TURNS=3\n'
        'ALPACA_PAPER_KEY_ID=PKfixture\n'
    )
    monkeypatch.setenv("LAT_API_PORT", "18767")

    settings = Settings()
    assert settings.api_url == "http://[::1]:18767"
    assert settings.reconcile_interval_s == 18
    assert settings.symbols == ("SPY",)
    assert AiConfig().max_runs_per_day == 8
    assert ChatConfig().max_history_turns == 4
    assert ChatGptConfig().max_turns == 3
    assert "PKfixture" not in settings.model_dump_json()
    assert SecretStore(settings.project_root, use_keychain=False).has(SecretName.ALPACA_PAPER_KEY_ID)


def test_explicit_workspace_selects_its_own_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / ".env").write_text("LAT_RECONCILE_INTERVAL_S=18\n")
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(other))
    monkeypatch.delenv("LAT_RECONCILE_INTERVAL_S", raising=False)
    assert Settings(project_root=tmp_path).reconcile_interval_s == 30
    assert Settings(_env_file=None).reconcile_interval_s == 30


def test_paper_is_the_only_enabled_operator_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LAT_GATEWAY", "LAT_MODE", "LAT_TRADING_MODE", "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("LAT_TRADING_MODE=paper\n")
    settings = Settings(project_root=tmp_path, use_keychain=False)
    assert settings.trading_mode == OperatorTradingMode.PAPER and settings.gateway == GatewayKind.FAKE
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "PKfixture")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "fixture")
    settings = Settings(project_root=tmp_path, use_keychain=False)
    assert settings.gateway == GatewayKind.ALPACA_PAPER
    assert Settings(project_root=tmp_path, gateway=GatewayKind.FAKE).gateway == GatewayKind.FAKE
    assert Settings(project_root=tmp_path, mode=TradingMode.REPLAY).gateway == GatewayKind.FAKE
    with pytest.raises(ValidationError, match=r"PAPER.*LIVE"):
        Settings(project_root=tmp_path, trading_mode="replay")


@pytest.mark.parametrize(("mode", "label", "public"), [
    (TradingMode.PAPER, ExecutionLabel.PAPER, OperatorTradingMode.PAPER),
    (TradingMode.PAPER, ExecutionLabel.SIMULATED, OperatorTradingMode.PAPER),
    (TradingMode.REPLAY, ExecutionLabel.SIMULATED, OperatorTradingMode.PAPER),
    (TradingMode.RESEARCH_ONLY, ExecutionLabel.SIMULATED, OperatorTradingMode.PAPER),
    (TradingMode.LIVE_APPROVAL, ExecutionLabel.LIVE, OperatorTradingMode.LIVE),
    (TradingMode.BOUNDED_LIVE, ExecutionLabel.LIVE, OperatorTradingMode.LIVE),
])
def test_public_mode_labels(mode: TradingMode, label: ExecutionLabel, public: OperatorTradingMode) -> None:
    assert operator_trading_mode(mode, label) == public
