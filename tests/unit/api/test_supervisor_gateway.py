"""A launch command must not silently reuse a different execution gateway."""
from pathlib import Path

import pytest

from packages.core.modes import OperatorTradingMode
from scripts import supervisor
from services.api import launcher


@pytest.mark.parametrize('from_dotenv', [False, True])
def test_existing_other_gateway_is_not_reported_as_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_dotenv: bool,
) -> None:
    monkeypatch.setattr(supervisor, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(supervisor, 'LOGS', tmp_path / 'logs')
    monkeypatch.setattr(launcher, '_health', lambda _: {
        'gateway': 'FAKE', 'mode': 'PAPER', 'workspace_id': launcher.workspace_id(launcher.Settings(project_root=tmp_path)),
    })
    monkeypatch.setenv('LAT_PROJECT_ROOT', str(tmp_path))
    if from_dotenv:
        (tmp_path / '.env').write_text('LAT_GATEWAY=ALPACA_PAPER\n')
        monkeypatch.delenv('LAT_GATEWAY', raising=False)
    else:
        monkeypatch.setenv('LAT_GATEWAY', 'ALPACA_PAPER')
    assert supervisor.start(['api']) == 2


def test_live_selection_fails_before_starting_a_process(tmp_path: Path) -> None:
    settings = launcher.Settings(project_root=tmp_path, trading_mode=OperatorTradingMode.LIVE, use_keychain=False)
    with pytest.raises(launcher.ApiStartupError, match="Live trading is unavailable"):
        launcher.ensure_api(settings)
    assert not settings.run_dir.exists()
