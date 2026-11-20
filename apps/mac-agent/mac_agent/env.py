"""Minimal subprocess environment for every GUI-agent child process (Peekaboo, Node/Playwright).

The GUI agent must never receive broker credentials, model API keys, or the operator/engine tokens
(spec §9, invariant 2). Rather than deny-listing known secret names, the child environment is built
from an explicit allowlist: ``PATH`` (a fixed system path, never the caller's), ``HOME`` and ``LANG``.
Anything else in the parent environment — ``ALPACA_*``, ``APCA_*``, ``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ``LAT_*`` tokens, ``SSH_AUTH_SOCK``... — is dropped.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

ALLOWED_ENV_KEYS: tuple[str, ...] = ("PATH", "HOME", "LANG")
SAFE_SYSTEM_PATH: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")
DEFAULT_LANG = "en_US.UTF-8"


def minimal_env(
    source: Mapping[str, str] | None = None,
    *,
    extra_path_dirs: Iterable[Path | str] = (),
) -> dict[str, str]:
    """Build the child environment. ``extra_path_dirs`` lets callers add the directory of a resolved
    binary (e.g. the Node install) without inheriting the caller's whole ``PATH``."""
    src = os.environ if source is None else source
    path_dirs: list[str] = []
    for d in (*[str(p) for p in extra_path_dirs], *SAFE_SYSTEM_PATH):
        if d and ":" not in d and d not in path_dirs:
            path_dirs.append(d)
    home = src.get("HOME") or str(Path.home())
    lang = src.get("LANG") or DEFAULT_LANG
    env = {"PATH": ":".join(path_dirs), "HOME": home, "LANG": lang}
    assert set(env) == set(ALLOWED_ENV_KEYS)
    return env
