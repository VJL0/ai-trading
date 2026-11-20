"""Where the Sign-in-with-ChatGPT credentials live.

* ``KeychainTokenStore``: the macOS login Keychain via the ``security`` CLI (service ``local-ai-trader``,
  accounts ``CHATGPT_REFRESH_TOKEN`` / ``CHATGPT_CLIENT_ID``). Commands are argv lists (never a shell). A value
  is written through ``security -i`` on **stdin**, so it never appears in a process listing; values outside a
  conservative token charset fall back to ``-w <value>`` as an argv element. Nothing is ever logged.
* ``MemoryTokenStore``: in-process only (tests, and the FAKE transport so simulated sign-ins never touch the
  user's Keychain).

The access token is never stored here: it lives only in memory (``ChatGptAuth``).
"""

from __future__ import annotations

import hmac
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import Any, Final, Protocol, runtime_checkable

from packages.core.secrets import KEYCHAIN_SERVICE

ACCOUNT_REFRESH_TOKEN: Final = "CHATGPT_REFRESH_TOKEN"  # noqa: S105 - an account name, not a secret
ACCOUNT_CLIENT_ID: Final = "CHATGPT_CLIENT_ID"
ACCOUNTS: Final[frozenset[str]] = frozenset({ACCOUNT_REFRESH_TOKEN, ACCOUNT_CLIENT_ID})

_SAFE_VALUE_RE: Final = re.compile(r"^[A-Za-z0-9._~+/=:-]{1,8192}$")
_NOT_FOUND_RC: Final = 44  # security: "The specified item could not be found in the keychain."


class TokenStoreError(RuntimeError):
    """The credential store could not be read or written (the caller fails closed)."""


@runtime_checkable
class TokenStore(Protocol):
    kind: str

    def get(self, account: str) -> str | None: ...

    def set(self, account: str, value: str) -> None: ...

    def delete(self, account: str) -> None: ...


def _check_account(account: str) -> None:
    if account not in ACCOUNTS:
        raise TokenStoreError(f"unknown credential account {account!r}")


class MemoryTokenStore:
    kind = "memory"

    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, account: str) -> str | None:
        _check_account(account)
        with self._lock:
            return self._values.get(account)

    def set(self, account: str, value: str) -> None:
        _check_account(account)
        if not value:
            raise TokenStoreError("refusing to store an empty credential")
        with self._lock:
            self._values[account] = value

    def delete(self, account: str) -> None:
        _check_account(account)
        with self._lock:
            self._values.pop(account, None)

    def __repr__(self) -> str:  # never show values
        return f"MemoryTokenStore(accounts={sorted(self._values)})"


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class KeychainTokenStore:
    kind = "keychain"

    def __init__(self, service: str = KEYCHAIN_SERVICE, *, runner: Runner | None = None, timeout_s: float = 10.0,
                 security_path: str | None = None) -> None:
        self.service = service
        self._runner: Runner = runner or subprocess.run
        self._timeout = timeout_s
        self._security = security_path or shutil.which("security") or "/usr/bin/security"

    def _run(self, argv: Sequence[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": self._timeout, "check": False}
        if stdin is not None:
            kwargs["input"] = stdin
        try:
            return self._runner(list(argv), **kwargs)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TokenStoreError(f"macOS Keychain unavailable ({type(exc).__name__})") from None

    def get(self, account: str) -> str | None:
        _check_account(account)
        proc = self._run([self._security, "find-generic-password", "-s", self.service, "-a", account, "-w"])
        if proc.returncode == _NOT_FOUND_RC:
            return None
        if proc.returncode != 0:
            raise TokenStoreError(f"Keychain read failed for {account} (security exit {proc.returncode})")
        value = (proc.stdout or "").strip()
        return value or None

    def set(self, account: str, value: str) -> None:
        _check_account(account)
        if not value or "\n" in value or "\r" in value or "\x00" in value:
            raise TokenStoreError("refusing to store an empty or multi-line credential")
        if _SAFE_VALUE_RE.match(value) and _SAFE_VALUE_RE.match(self.service):
            # Interactive mode reads the command from stdin: the value is not visible in `ps` output.
            command = f'add-generic-password -U -s "{self.service}" -a "{account}" -w "{value}"\n'
            proc = self._run([self._security, "-i"], stdin=command)
            failed = proc.returncode != 0 or "error" in (proc.stderr or "").lower()
        else:
            proc = self._run([self._security, "add-generic-password", "-U", "-s", self.service, "-a", account,
                              "-w", value])
            failed = proc.returncode != 0
        if failed:
            raise TokenStoreError(f"Keychain write failed for {account} (security exit {proc.returncode})")
        # `security -i` may report an inner failure only on stderr; verify by reading the item back.
        stored = self.get(account)
        if stored is None or not hmac.compare_digest(stored, value):
            raise TokenStoreError(f"Keychain write for {account} could not be verified")

    def delete(self, account: str) -> None:
        _check_account(account)
        proc = self._run([self._security, "delete-generic-password", "-s", self.service, "-a", account])
        if proc.returncode not in (0, _NOT_FOUND_RC):
            raise TokenStoreError(f"Keychain delete failed for {account} (security exit {proc.returncode})")

    def __repr__(self) -> str:
        return f"KeychainTokenStore(service={self.service!r})"


__all__ = [
    "ACCOUNTS",
    "ACCOUNT_CLIENT_ID",
    "ACCOUNT_REFRESH_TOKEN",
    "KeychainTokenStore",
    "MemoryTokenStore",
    "TokenStore",
    "TokenStoreError",
]
