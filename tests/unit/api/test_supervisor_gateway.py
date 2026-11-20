"""A launch command must not silently reuse a different execution gateway."""
from pathlib import Path

import pytest

from scripts import supervisor


def test_existing_other_gateway_is_not_reported_as_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, 'RUN', tmp_path / 'run')
    monkeypatch.setattr(supervisor, 'LOGS', tmp_path / 'logs')
    monkeypatch.setattr(supervisor, '_read_pid', lambda _: 123)
    monkeypatch.setattr(supervisor, '_http_json', lambda _: {'gateway': 'FAKE'})
    monkeypatch.setenv('LAT_GATEWAY', 'ALPACA_PAPER')
    assert supervisor.start(['api']) == 2
