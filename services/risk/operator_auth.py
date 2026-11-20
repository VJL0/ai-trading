"""Human-operator authentication for the local settings flow and per-order human approvals.

The operator token lives at ``var/<namespace>/keys/operator.token`` (file mode 0600, directory 0700),
one per mode namespace, so a paper token never authenticates in a live namespace.

Who holds it:
* the **human operator** (pastes it into the dashboard settings form / sends it as
  ``X-LAT-Operator-Token``), and the local API process, which reads it to verify requests;
* **never** an AI model, a prompt, the Peekaboo/Playwright GUI agent, logs, screenshots or the
  client bundle. This module never logs, prints or ``repr``s the token value.

Human approvals (``HumanApproval``) are HMAC-SHA256 signatures over the exact order fields and the
statement the human confirmed, keyed by a key derived from the operator token. They bind one
approval to one intent (fresh confirmation per order) and expire quickly.

Limitation (documented): the API process can read the token, so a fully compromised local API could
forge an approval. Live execution remains compiled off; see docs/runbooks for the live release gate.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.contracts.models import ApprovedOrderIntent, HumanApproval
from packages.core.canonical import sha256_hex
from packages.core.ids import new_id

OPERATOR_TOKEN_FILE = "operator.token"  # noqa: S105 - a file name, not a secret
TOKEN_BYTES = 32
MIN_TOKEN_CHARS = 32
APPROVAL_KEY_CONTEXT = b"lat-human-approval-v1"
DEFAULT_APPROVAL_TTL_SECONDS = 120
MAX_APPROVAL_TTL_SECONDS = 600
REDACTED = "[REDACTED]"


class OperatorAuthError(PermissionError):
    """Operator authentication failed or the token store is unusable. Never contains the token."""


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def approval_message(
    *,
    approval_id: str,
    approver: str,
    confirmed_at: datetime,
    expires_at: datetime,
    statement: str,
    intent: ApprovedOrderIntent,
) -> str:
    """Exact string a human approval signs: the approval metadata *and* the order being approved."""
    fields = (
        ("approval_id", approval_id),
        ("intent_id", intent.intent_id),
        ("client_order_id", intent.client_order_id),
        ("mode", intent.mode.value),
        ("gateway", intent.gateway.value),
        ("symbol", intent.symbol),
        ("side", intent.side.value),
        ("quantity", format(intent.quantity, "f")),
        ("order_type", intent.order_type.value),
        ("limit_price", format(intent.limit_price, "f")),
        ("policy_version", intent.policy_version),
        ("approver", approver),
        ("confirmed_at", _iso(confirmed_at)),
        ("expires_at", _iso(expires_at)),
        ("statement_sha256", sha256_hex(statement)),
    )
    for name, value in fields:
        if "|" in value or "\n" in value:
            raise ValueError(f"illegal character in approval field {name}")
    return "lat-approval-v1|" + "|".join(f"{k}={v}" for k, v in fields)


def default_approval_statement(intent: ApprovedOrderIntent) -> str:
    return (
        f"I approve {intent.side.value} {format(intent.quantity, 'f')} {intent.symbol} "
        f"LIMIT {format(intent.limit_price, 'f')} DAY in mode {intent.mode.value} via {intent.gateway.value} "
        f"(client order id {intent.client_order_id})."
    )


class OperatorAuth:
    """File-backed operator token with constant-time verification. Holds no token in attributes."""

    __slots__ = ("keys_dir",)

    def __init__(self, keys_dir: Path) -> None:
        self.keys_dir = keys_dir

    def __repr__(self) -> str:  # never include the token
        return f"OperatorAuth(keys_dir={str(self.keys_dir)!r})"

    @property
    def token_path(self) -> Path:
        return self.keys_dir / OPERATOR_TOKEN_FILE

    # ------------------------------------------------------------------ token lifecycle
    def _ensure_dir(self) -> None:
        self.keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.keys_dir, 0o700)

    def _write_new(self) -> Path:
        """Write a fresh token atomically (temp file 0600 + rename). Returns the token path."""
        self._ensure_dir()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        tmp = self.keys_dir / f".{OPERATOR_TOKEN_FILE}.{secrets.token_hex(6)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(token)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.token_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        os.chmod(self.token_path, 0o600)
        return self.token_path

    def ensure_token(self) -> Path:
        """Create the token if absent (0600). Tightens permissions of an existing token file."""
        self._ensure_dir()
        path = self.token_path
        if path.is_symlink():
            raise OperatorAuthError(f"refusing symlinked operator token at {path}")
        if not path.exists():
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                return self.ensure_token()
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_urlsafe(TOKEN_BYTES))
                fh.flush()
                os.fsync(fh.fileno())
            return path
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            os.chmod(path, 0o600)
        if self._read_stored() is None:
            # Empty / truncated token file: replace it rather than accept a weak secret.
            return self._write_new()
        return path

    def rotate(self) -> Path:
        """Replace the token. All previously issued tokens (and approvals keyed by them) stop verifying."""
        return self._write_new()

    def _read_stored(self) -> str | None:
        try:
            fd = os.open(self.token_path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            return None
        with os.fdopen(fd, "r") as fh:
            value = fh.read().strip()
        if len(value) < MIN_TOKEN_CHARS:
            return None
        return value

    # ------------------------------------------------------------------ verification
    def verify(self, token: str | None) -> bool:
        """Constant-time comparison against the stored token. Missing/unreadable store => False."""
        if not isinstance(token, str) or not token:
            return False
        stored = self._read_stored()
        if stored is None:
            return False
        return hmac.compare_digest(token.strip().encode("utf-8"), stored.encode("utf-8"))

    def require(self, token: str | None) -> None:
        if not self.verify(token):
            raise OperatorAuthError("operator authentication failed")

    def redact(self, text: str, *also: str | None) -> str:
        """``text`` with the stored token and any of ``also`` (e.g. the presented token) replaced by
        ``[REDACTED]``. Used before free text (actor, reason) is written to the immutable audit log, so a
        token pasted into the wrong form field is never persisted."""
        secrets_to_hide = [s for s in (self._read_stored(), *also) if isinstance(s, str) and len(s.strip()) >= 8]
        out = text
        for value in sorted({s.strip() for s in secrets_to_hide}, key=len, reverse=True):
            out = out.replace(value, REDACTED)
        return out

    # ------------------------------------------------------------------ human approvals
    def _approval_key(self) -> bytes | None:
        stored = self._read_stored()
        if stored is None:
            return None
        return hmac.new(stored.encode("utf-8"), APPROVAL_KEY_CONTEXT, hashlib.sha256).digest()

    def create_approval(
        self,
        token: str | None,
        intent: ApprovedOrderIntent,
        *,
        approver: str,
        now: datetime,
        statement: str | None = None,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> HumanApproval:
        """Record a human's confirmation of exactly this intent. Requires the operator token."""
        self.require(token)
        if not approver.strip():
            raise ValueError("approver is required")
        if not 0 < ttl_seconds <= MAX_APPROVAL_TTL_SECONDS:
            raise ValueError(f"approval ttl must be in (0, {MAX_APPROVAL_TTL_SECONDS}] seconds")
        key = self._approval_key()
        if key is None:
            raise OperatorAuthError("operator token store unavailable")
        stmt = statement if statement is not None else default_approval_statement(intent)
        approval_id = new_id("hap")
        expires_at = now + timedelta(seconds=ttl_seconds)
        msg = approval_message(
            approval_id=approval_id, approver=approver, confirmed_at=now, expires_at=expires_at,
            statement=stmt, intent=intent,
        )
        sig = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()
        return HumanApproval(
            approval_id=approval_id, intent_id=intent.intent_id, approver=approver, confirmed_at=now,
            expires_at=expires_at, statement=stmt, signature=sig,
        )

    def verify_approval(self, approval: HumanApproval, intent: ApprovedOrderIntent) -> bool:
        """True iff ``approval`` was signed with the current operator token for exactly ``intent``."""
        if approval.intent_id != intent.intent_id:
            return False
        key = self._approval_key()
        if key is None:
            return False
        try:
            msg = approval_message(
                approval_id=approval.approval_id, approver=approval.approver,
                confirmed_at=approval.confirmed_at, expires_at=approval.expires_at,
                statement=approval.statement, intent=intent,
            )
        except ValueError:
            return False
        expected = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, approval.signature)
