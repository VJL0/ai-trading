"""OperatorAuth: 0600 token file, constant-time verification, rotation, no token leakage, and HMAC human
approvals bound to exactly one intent."""

from __future__ import annotations

import logging
import os
import stat
from datetime import timedelta
from pathlib import Path

import pytest

from packages.contracts.policy import RiskPolicy
from services.risk import operator_auth as oa
from services.risk.operator_auth import OperatorAuth, OperatorAuthError
from tests.unit.risk.risk_factories import NOW, make_ctx, make_engine, make_proposal


def _token(auth: OperatorAuth) -> str:
    return auth.token_path.read_text().strip()


def test_ensure_token_creates_0600_file_in_0700_dir(tmp_path: Path) -> None:
    keys = tmp_path / "var" / "paper" / "keys"
    auth = OperatorAuth(keys)
    path = auth.ensure_token()
    assert path == keys / "operator.token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(keys.stat().st_mode) == 0o700
    assert len(_token(auth)) >= 43  # token_urlsafe(32)


def test_ensure_token_is_idempotent_and_tightens_permissions(tmp_path: Path) -> None:
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    first = _token(auth)
    os.chmod(auth.token_path, 0o644)
    auth.ensure_token()
    assert _token(auth) == first
    assert stat.S_IMODE(auth.token_path.stat().st_mode) == 0o600


def test_weak_token_file_is_replaced(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "operator.token").write_text("short")
    auth = OperatorAuth(keys)
    auth.ensure_token()
    assert _token(auth) != "short" and len(_token(auth)) >= 32
    assert not auth.verify("short")


def test_symlinked_token_is_refused(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    keys.mkdir()
    target = tmp_path / "elsewhere.token"
    target.write_text("x" * 50)
    (keys / "operator.token").symlink_to(target)
    with pytest.raises(OperatorAuthError):
        OperatorAuth(keys).ensure_token()
    assert not OperatorAuth(keys).verify("x" * 50)  # O_NOFOLLOW read refuses the symlink


def test_verify(tmp_path: Path) -> None:
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    tok = _token(auth)
    assert auth.verify(tok)
    assert not auth.verify(tok + "x")
    assert not auth.verify(tok[:-1])
    assert not auth.verify("")
    assert not auth.verify(None)
    assert not auth.verify(123)  # type: ignore[arg-type]


def test_verify_without_token_file_fails_closed(tmp_path: Path) -> None:
    auth = OperatorAuth(tmp_path / "missing")
    assert not auth.verify("anything-at-all-anything-at-all-anything")
    with pytest.raises(OperatorAuthError):
        auth.require("anything")


def test_rotate_invalidates_old_token(tmp_path: Path) -> None:
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    old = _token(auth)
    auth.rotate()
    new = _token(auth)
    assert old != new and auth.verify(new) and not auth.verify(old)
    assert stat.S_IMODE(auth.token_path.stat().st_mode) == 0o600
    assert not list((tmp_path / "keys").glob("*.tmp"))


def test_token_never_appears_in_repr_errors_or_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    tok = _token(auth)
    assert tok not in repr(auth)
    with pytest.raises(OperatorAuthError) as exc:
        auth.require(tok + "nope")
    assert tok not in str(exc.value)
    auth.rotate()
    assert tok not in caplog.text and _token(auth) not in caplog.text
    assert not hasattr(auth, "__dict__")  # __slots__: nowhere to stash the token on the instance


def test_approval_binds_one_intent_and_detects_tampering(tmp_path: Path) -> None:
    engine = make_engine(RiskPolicy(require_human_approval=True))
    ctx = make_ctx()
    p1 = make_proposal()
    intent = engine.authorize(p1, engine.evaluate(p1, ctx), ctx)
    p2 = make_proposal(proposal_id="prop_2")
    other = engine.authorize(p2, engine.evaluate(p2, ctx), ctx)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    approval = auth.create_approval(_token(auth), intent, approver="victor", now=NOW)
    assert approval.intent_id == intent.intent_id
    assert approval.expires_at - approval.confirmed_at == timedelta(seconds=oa.DEFAULT_APPROVAL_TTL_SECONDS)
    assert "SPY" in approval.statement and intent.client_order_id in approval.statement
    assert auth.verify_approval(approval, intent)
    assert not auth.verify_approval(approval, other)
    assert not auth.verify_approval(approval.model_copy(update={"approver": "mallory"}), intent)
    assert not auth.verify_approval(approval, intent.model_copy(update={"quantity": intent.quantity + 1}))


def test_approval_ttl_bounds_and_approver_required(tmp_path: Path) -> None:
    engine = make_engine()
    ctx = make_ctx()
    p = make_proposal()
    intent = engine.authorize(p, engine.evaluate(p, ctx), ctx)
    auth = OperatorAuth(tmp_path / "keys")
    auth.ensure_token()
    with pytest.raises(ValueError, match="ttl"):
        auth.create_approval(_token(auth), intent, approver="v", now=NOW, ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl"):
        auth.create_approval(_token(auth), intent, approver="v", now=NOW, ttl_seconds=601)
    with pytest.raises(ValueError, match="approver"):
        auth.create_approval(_token(auth), intent, approver=" ", now=NOW)
