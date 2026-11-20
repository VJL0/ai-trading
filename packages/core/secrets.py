"""Secret lookup: macOS Keychain first, then process environment, then a git-ignored ``.env``.

Isolation rules (spec §8, §12):
* Paper and live Alpaca credentials have *different names*. The ambiguous upstream names
  ``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` are deliberately ignored so a paper key can never be
  picked up as a live key (or vice versa) by changing one variable.
* Live credentials are not returned by ``get``; only the live release gate may read them, and it is
  compiled off in this build.
* Values are wrapped in ``SecretStr`` and never logged.

Keychain entries: service ``local-ai-trader``, account = secret name, e.g.
``security add-generic-password -s local-ai-trader -a ALPACA_PAPER_KEY_ID -w`` (prompts for the value).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import SecretStr

KEYCHAIN_SERVICE = "local-ai-trader"


class SecretName(StrEnum):  # names only, never values
    ALPACA_PAPER_KEY_ID = "ALPACA_PAPER_KEY_ID"
    ALPACA_PAPER_SECRET_KEY = "ALPACA_PAPER_SECRET_KEY"  # noqa: S105
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ALPACA_LIVE_KEY_ID = "ALPACA_LIVE_KEY_ID"
    ALPACA_LIVE_SECRET_KEY = "ALPACA_LIVE_SECRET_KEY"  # noqa: S105


LIVE_SECRETS = frozenset({SecretName.ALPACA_LIVE_KEY_ID, SecretName.ALPACA_LIVE_SECRET_KEY})
IGNORED_AMBIGUOUS_NAMES = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY")


class LiveSecretAccessDenied(PermissionError):
    pass


def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value:
            out[key] = value
    return out


class SecretStore:
    def __init__(
        self,
        project_root: Path,
        *,
        use_keychain: bool = True,
        env: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
    ) -> None:
        self._use_keychain = use_keychain and shutil.which("security") is not None
        self._env = os.environ if env is None else env
        self._dotenv = _parse_dotenv(dotenv_path if dotenv_path is not None else project_root / ".env")
        self._cache: dict[SecretName, SecretStr | None] = {}

    def _keychain(self, name: SecretName) -> str | None:
        if not self._use_keychain:
            return None
        try:
            proc = subprocess.run(
                ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name.value, "-w"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        value = proc.stdout.strip()
        return value or None

    def _lookup(self, name: SecretName) -> SecretStr | None:
        if name in self._cache:
            return self._cache[name]
        value = self._keychain(name) or self._env.get(name.value) or self._dotenv.get(name.value)
        result = SecretStr(value) if value else None
        self._cache[name] = result
        return result

    def get(self, name: SecretName) -> SecretStr | None:
        if name in LIVE_SECRETS:
            raise LiveSecretAccessDenied(
                f"{name} may only be read by the live release gate, which is compiled off in this build"
            )
        return self._lookup(name)

    def has(self, name: SecretName) -> bool:
        """Presence check only (safe for live names: never returns the value)."""
        return self._lookup(name) is not None

    def presence(self) -> dict[str, bool]:
        return {n.value: self.has(n) for n in SecretName}

    def ignored_ambiguous_present(self) -> list[str]:
        return [n for n in IGNORED_AMBIGUOUS_NAMES if self._env.get(n) or self._dotenv.get(n)]

    def alpaca_paper_available(self) -> bool:
        return self.has(SecretName.ALPACA_PAPER_KEY_ID) and self.has(SecretName.ALPACA_PAPER_SECRET_KEY)
