"""Keychain credential store driven through a recording fake ``security`` runner (the real Keychain is never
touched by tests): argv lists only, the secret value travels on stdin, writes are verified by reading back."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from services.ai.token_store import (
    ACCOUNT_CLIENT_ID,
    ACCOUNT_REFRESH_TOKEN,
    KeychainTokenStore,
    MemoryTokenStore,
    TokenStoreError,
)

SECRET = "fake-rt-0123456789abcdef.XYZ_-~"


class FakeSecurity:
    """Emulates the subset of /usr/bin/security the store uses (including `-i` stdin commands)."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail_writes = False
        self.silent_write_failure = False

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)  # never a shell string
        assert kw.get("check") is False and kw.get("capture_output") is True
        stdin = kw.get("input")
        self.calls.append((argv, stdin))
        if argv[1:] == ["-i"]:
            assert stdin is not None
            parts = stdin.strip().split(" ")
            assert parts[0] == "add-generic-password"
            if self.fail_writes:
                return subprocess.CompletedProcess(argv, 1, "", "security: write failed")
            if self.silent_write_failure:
                return subprocess.CompletedProcess(argv, 0, "", "")
            svc = parts[parts.index("-s") + 1].strip('"')
            acct = parts[parts.index("-a") + 1].strip('"')
            self.items[(svc, acct)] = parts[parts.index("-w") + 1].strip('"')
            return subprocess.CompletedProcess(argv, 0, "", "")
        cmd = argv[1]
        svc, acct = argv[argv.index("-s") + 1], argv[argv.index("-a") + 1]
        if cmd == "find-generic-password":
            if (svc, acct) not in self.items:
                return subprocess.CompletedProcess(argv, 44, "", "not found")
            return subprocess.CompletedProcess(argv, 0, self.items[(svc, acct)] + "\n", "")
        if cmd == "add-generic-password":
            if self.fail_writes:
                return subprocess.CompletedProcess(argv, 1, "", "")
            self.items[(svc, acct)] = argv[argv.index("-w") + 1]
            return subprocess.CompletedProcess(argv, 0, "", "")
        if cmd == "delete-generic-password":
            if self.items.pop((svc, acct), None) is None:
                return subprocess.CompletedProcess(argv, 44, "", "")
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(argv)


def _store(sec: FakeSecurity) -> KeychainTokenStore:
    return KeychainTokenStore(runner=sec, security_path="/usr/bin/security")


def test_roundtrip_keeps_the_secret_out_of_argv() -> None:
    sec = FakeSecurity()
    store = _store(sec)
    assert store.get(ACCOUNT_REFRESH_TOKEN) is None  # rc 44 = not found
    store.set(ACCOUNT_REFRESH_TOKEN, SECRET)
    assert store.get(ACCOUNT_REFRESH_TOKEN) == SECRET
    assert ("local-ai-trader", ACCOUNT_REFRESH_TOKEN) in sec.items
    write = [c for c in sec.calls if c[0][1:] == ["-i"]]
    assert len(write) == 1 and SECRET in (write[0][1] or "")
    assert all(SECRET not in " ".join(argv) for argv, _ in sec.calls)  # never visible in a process listing
    store.delete(ACCOUNT_REFRESH_TOKEN)
    store.delete(ACCOUNT_REFRESH_TOKEN)  # idempotent (rc 44 tolerated)
    assert store.get(ACCOUNT_REFRESH_TOKEN) is None


def test_unusual_charset_falls_back_to_argv_value() -> None:
    sec = FakeSecurity()
    store = _store(sec)
    store.set(ACCOUNT_CLIENT_ID, "oaiapp with space")
    assert store.get(ACCOUNT_CLIENT_ID) == "oaiapp with space"
    assert any("-w" in argv and argv[1] == "add-generic-password" for argv, _ in sec.calls)


def test_failed_or_unverifiable_writes_raise() -> None:
    sec = FakeSecurity()
    sec.fail_writes = True
    with pytest.raises(TokenStoreError):
        _store(sec).set(ACCOUNT_REFRESH_TOKEN, SECRET)
    sec2 = FakeSecurity()
    sec2.silent_write_failure = True  # `security -i` exit 0 but nothing stored
    with pytest.raises(TokenStoreError, match="verified"):
        _store(sec2).set(ACCOUNT_REFRESH_TOKEN, SECRET)


@pytest.mark.parametrize("bad", ["", "a\nb", "a\rb", "a\x00b"])
def test_rejects_empty_or_multiline_values(bad: str) -> None:
    with pytest.raises(TokenStoreError):
        _store(FakeSecurity()).set(ACCOUNT_REFRESH_TOKEN, bad)


def test_unknown_accounts_are_refused() -> None:
    for store in (_store(FakeSecurity()), MemoryTokenStore()):
        with pytest.raises(TokenStoreError):
            store.get("OPENAI_API_KEY")
        with pytest.raises(TokenStoreError):
            store.set("ALPACA_LIVE_SECRET_KEY", "x")


def test_errors_and_repr_never_leak_values() -> None:
    def boom(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 10)

    store = KeychainTokenStore(runner=boom)
    with pytest.raises(TokenStoreError, match="unavailable") as exc:
        store.set(ACCOUNT_REFRESH_TOKEN, SECRET)
    assert SECRET not in str(exc.value)
    mem = MemoryTokenStore()
    mem.set(ACCOUNT_REFRESH_TOKEN, SECRET)
    assert SECRET not in repr(mem) and SECRET not in repr(store)
