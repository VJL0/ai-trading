"""PolicyStore against a real SQLite ledger (WAL, single writer, all migrations): immutable versions,
operator-token-only changes, unsafe-change rejection, audit trail with field-level diff, read-only view."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core.audit import verify_chain
from packages.core.db import Database
from services.risk.operator_auth import OperatorAuth, OperatorAuthError
from services.risk.policy_store import (
    BOOTSTRAP_ACTOR,
    PolicyChangeRejected,
    PolicyIntegrityError,
    PolicyReader,
    PolicyStore,
    PolicyUnavailableError,
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "var" / "paper" / "ledger.sqlite")
    yield database
    database.close()


@pytest.fixture
def auth(db: Database) -> OperatorAuth:
    a = OperatorAuth(db.path.parent / "keys")
    a.ensure_token()
    return a


@pytest.fixture
def token(auth: OperatorAuth) -> str:
    return auth.token_path.read_text().strip()


@pytest.fixture
def store(db: Database, auth: OperatorAuth) -> PolicyStore:
    s = PolicyStore(db, TradingMode.PAPER)  # default token location: <db dir>/keys/operator.token
    s.ensure_default()
    return s


def audits(db: Database, event_type: str) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM audit_events WHERE event_type=? ORDER BY seq", (event_type,))
    for r in rows:
        r["payload"] = json.loads(r["payload_json"])
    return rows


# ------------------------------------------------------------------------------------------ bootstrap / read


def test_ensure_default_installs_default_once(db: Database, store: PolicyStore) -> None:
    active = store.active()
    assert active == RiskPolicy() and active.version == RiskPolicy().version
    store.ensure_default()
    rows = db.query("SELECT * FROM risk_policies")
    assert len(rows) == 1
    assert rows[0]["created_by"] == BOOTSTRAP_ACTOR and rows[0]["approved_by"] == BOOTSTRAP_ACTOR
    assert rows[0]["is_active"] == 1
    assert len(audits(db, "POLICY_BOOTSTRAPPED")) == 1
    hist = store.history()
    assert hist[0]["integrity_ok"] and hist[0]["is_active"]


def test_no_active_policy_fails_closed(db: Database) -> None:
    with pytest.raises(PolicyUnavailableError):
        PolicyStore(db, TradingMode.PAPER).active()


# ------------------------------------------------------------------------------------------ immutability


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE risk_policies SET policy_json = '{}'",
        "UPDATE risk_policies SET policy_version = 'pol_x'",
        "UPDATE risk_policies SET name = 'x'",
        "UPDATE risk_policies SET created_by = 'x'",
        "UPDATE risk_policies SET approved_by = 'x'",
        "UPDATE risk_policies SET supersedes = 'x'",
        "DELETE FROM risk_policies",
    ],
)
def test_policy_content_is_immutable_in_the_database(db: Database, store: PolicyStore, sql: str) -> None:
    with pytest.raises(sqlite3.IntegrityError, match=r"immutable|never deleted"):
        db.write(lambda c: c.execute(sql))
    assert store.active() == RiskPolicy()


def test_tampered_policy_json_is_detected_and_fails_closed(db: Database, store: PolicyStore) -> None:
    db.close()  # simulate an attacker with raw file access who drops the guard trigger
    raw = sqlite3.connect(db.path)
    raw.execute("DROP TRIGGER risk_policies_immutable")
    evil = RiskPolicy(max_order_notional=Decimal("25000")).canonical_json()
    raw.execute("UPDATE risk_policies SET policy_json = ?", (evil,))
    raw.commit()
    raw.close()
    with pytest.raises(PolicyIntegrityError, match="hashes to"):
        PolicyReader(db.path).active()
    assert PolicyReader(db.path).history()[0]["integrity_ok"] is False


def test_non_canonical_policy_json_is_detected(db: Database, store: PolicyStore) -> None:
    db.close()
    raw = sqlite3.connect(db.path)
    raw.execute("DROP TRIGGER risk_policies_immutable")
    pretty = json.dumps(json.loads(RiskPolicy().canonical_json()), indent=2)
    raw.execute("UPDATE risk_policies SET policy_json = ?", (pretty,))
    raw.commit()
    raw.close()
    with pytest.raises(PolicyIntegrityError, match="canonical"):
        PolicyReader(db.path).active()


def test_change_log_is_append_only(db: Database, store: PolicyStore, token: str) -> None:
    store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token, actor="victor",
                       reason="tighten")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.write(lambda c: c.execute("UPDATE risk_policy_changes SET result='REJECTED'"))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.write(lambda c: c.execute("DELETE FROM risk_policy_changes"))


# ------------------------------------------------------------------------------------------ change flow


def test_wrong_token_is_rejected_and_audited(db: Database, store: PolicyStore, token: str) -> None:
    before = store.active_version()
    with pytest.raises(OperatorAuthError):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token + "x",
                           actor="victor", reason="try")
    with pytest.raises(OperatorAuthError):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token="", actor="model",
                           reason="try")
    assert store.active_version() == before
    rej = audits(db, "POLICY_CHANGE_REJECTED")
    assert len(rej) == 2
    assert rej[0]["payload"]["violations"] == ["operator authentication failed"]
    assert rej[0]["payload"]["operator_token_presented"] is True
    assert rej[1]["payload"]["operator_token_presented"] is False
    changes = store.changes()
    assert [c["result"] for c in changes] == ["REJECTED", "REJECTED"]
    assert len(db.query("SELECT * FROM risk_policies")) == 1


@pytest.mark.parametrize(
    "unsafe",
    [
        {"allow_margin": True},
        {"allow_short": True},
        {"allow_extended_hours": True},
        {"long_only": False},
        {"max_order_notional": Decimal("30000")},
        {"max_daily_loss": Decimal("100000")},
        {"symbol_denylist": ("SQQQ",)},
        {"symbol_allowlist": ("SPY", "TQQQ")},
        {"min_buying_power_buffer": Decimal("0.5")},
        {"min_price": Decimal("0.50")},
    ],
)
def test_unsafe_changes_are_rejected_and_audited(db: Database, store: PolicyStore, token: str,
                                                 unsafe: dict[str, Any]) -> None:
    before = store.active_version()
    with pytest.raises(PolicyChangeRejected) as exc:
        store.apply_change(RiskPolicy(**unsafe), operator_token=token, actor="victor", reason="unsafe edit")
    assert exc.value.violations
    assert store.active_version() == before
    rej = audits(db, "POLICY_CHANGE_REJECTED")
    assert rej and rej[-1]["payload"]["violations"] == exc.value.violations
    assert len(db.query("SELECT * FROM risk_policies")) == 1


def test_float_smuggled_via_model_construct_is_rejected(store: PolicyStore, token: str) -> None:
    smuggled = RiskPolicy.model_construct(**{**RiskPolicy().model_dump(), "max_order_notional": 1000.0})
    with pytest.raises(PolicyChangeRejected, match="float"):
        store.apply_change(smuggled, operator_token=token, actor="victor", reason="float")


@pytest.mark.parametrize(("actor", "reason"), [("", "x"), ("ai", "x"), ("model", "x"), ("system-bootstrap", "x"),
                                               ("victor", ""), ("victor", "y" * 501)])
def test_actor_and_reason_are_required(store: PolicyStore, token: str, actor: str, reason: str) -> None:
    with pytest.raises(PolicyChangeRejected):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token, actor=actor,
                           reason=reason)


def test_accepted_change_creates_new_version_with_diff(db: Database, store: PolicyStore, token: str) -> None:
    old = store.active()
    new = RiskPolicy(max_order_notional=Decimal("1000.00"), max_spread_bps=Decimal("15"),
                     symbol_allowlist=(*RiskPolicy().symbol_allowlist, "BRK.B"))
    version = store.apply_change(new, operator_token=token, actor="victor", reason="tighten for paper week 2")
    assert version == new.version != old.version
    assert store.active() == new
    rows = {r["policy_version"]: r for r in db.query("SELECT * FROM risk_policies")}
    assert rows[old.version]["is_active"] == 0 and rows[version]["is_active"] == 1
    assert rows[version]["supersedes"] == old.version
    assert rows[version]["created_by"] == "victor" and rows[version]["approved_by"] == "victor"
    changed = audits(db, "POLICY_CHANGED")
    assert len(changed) == 1
    payload = changed[0]["payload"]
    assert payload["from_version"] == old.version and payload["to_version"] == version
    assert payload["diff"]["max_order_notional"] == {"from": "2000.00", "to": "1000.00"}
    assert payload["diff"]["max_spread_bps"] == {"from": "25", "to": "15"}
    assert payload["diff"]["symbol_allowlist"]["added"] == ["BRK.B"]
    assert set(payload["diff"]) == {"max_order_notional", "max_spread_bps", "symbol_allowlist"}
    assert changed[0]["policy_version"] == version
    ok, n, detail = verify_chain(db)
    assert ok, detail
    assert n >= 2


def test_identical_policy_is_a_rejected_no_op(store: PolicyStore, token: str) -> None:
    with pytest.raises(PolicyChangeRejected, match="identical"):
        store.apply_change(RiskPolicy(), operator_token=token, actor="victor", reason="noop")


def test_reactivating_an_earlier_version(db: Database, store: PolicyStore, token: str) -> None:
    original = store.active_version()
    store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token, actor="victor",
                       reason="tighten")
    back = store.apply_change(RiskPolicy(), operator_token=token, actor="victor", reason="revert")
    assert back == original and store.active_version() == original
    assert len(db.query("SELECT * FROM risk_policies")) == 2
    assert audits(db, "POLICY_CHANGED")[-1]["payload"]["reactivated_existing_version"] is True
    assert sum(1 for h in store.history() if h["is_active"]) == 1


def test_operator_token_never_stored_or_audited(db: Database, store: PolicyStore, token: str) -> None:
    store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token, actor="victor",
                       reason="tighten")
    with pytest.raises(OperatorAuthError):
        store.apply_change(RiskPolicy(max_order_notional=Decimal("900.00")), operator_token=token + "zz",
                           actor="victor", reason="bad token")
    db.close()
    dump = "\n".join(sqlite3.connect(db.path).iterdump())
    assert token not in dump and (token + "zz") not in dump


def test_concurrent_changes_leave_exactly_one_active(db: Database, store: PolicyStore, token: str) -> None:
    errors: list[BaseException] = []

    def change(n: int) -> None:
        try:
            store.apply_change(RiskPolicy(max_order_notional=Decimal(1000 + n)), operator_token=token,
                               actor="victor", reason=f"concurrent {n}")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=change, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(db.query("SELECT * FROM risk_policies WHERE is_active = 1")) == 1
    assert len(db.query("SELECT * FROM risk_policies")) == 9
    assert verify_chain(db)[0]


# ------------------------------------------------------------------------------------------ read-only view


def test_read_only_view_is_a_detached_json_copy(store: PolicyStore) -> None:
    view = store.read_only_view()
    assert view["read_only"] is True
    assert view["policy_version"] == store.active_version()
    assert view["policy"]["max_order_notional"] == "2000.00"
    assert view["hard_ceilings"]["max_order_notional"] == "25000"
    json.dumps(view)  # plain JSON-safe data
    view["policy"]["max_order_notional"] = "999999"
    assert store.active().max_order_notional == Decimal("2000.00")


def test_ai_facing_reader_has_no_mutators_and_cannot_write(db: Database, store: PolicyStore) -> None:
    reader = store.reader()
    public = {n for n in dir(reader) if not n.startswith("_")}
    assert public == {"active", "active_version", "get", "history", "changes", "read_only_view", "path"}
    assert reader.__slots__ == ("_path",)
    assert not any(isinstance(getattr(reader, s, None), Database) for s in reader.__slots__)
    with pytest.raises(sqlite3.OperationalError, match=r"readonly|read-only|query_only"):
        reader._query("UPDATE risk_policies SET is_active = 0")  # pyright: ignore[reportPrivateUsage]
    assert store.active() == RiskPolicy()


def test_read_only_view_contains_no_secrets(store: PolicyStore, token: str) -> None:
    text = json.dumps(store.read_only_view())
    assert token not in text and "operator.token" in text  # names the procedure, never the value


# ------------------------------------------------------------------------------------------ activation audit


def test_unaudited_activation_flip_fails_closed_until_the_operator_reapplies(db: Database, store: PolicyStore,
                                                                             token: str) -> None:
    default_version = store.active_version()
    tight = RiskPolicy(max_order_notional=Decimal("1000.00"))
    store.apply_change(tight, operator_token=token, actor="victor", reason="tighten")

    def flip(conn: sqlite3.Connection) -> None:  # is_active is the one mutable column: no trigger stops this
        conn.execute("UPDATE risk_policies SET is_active = 0 WHERE policy_version = ?", (tight.version,))
        conn.execute("UPDATE risk_policies SET is_active = 1 WHERE policy_version = ?", (default_version,))

    db.write(flip)
    with pytest.raises(PolicyIntegrityError, match="outside the audited flow"):
        store.active()
    with pytest.raises(PolicyIntegrityError, match="outside the audited flow"):
        store.ensure_default()  # a restart cannot silently accept it either
    # recovery is the audited operator flow itself
    store.apply_change(tight, operator_token=token, actor="victor", reason="re-activate after unaudited flip")
    assert store.active() == tight
    assert audits(db, "POLICY_CHANGED")[-1]["payload"]["from_version"] == default_version


def test_active_policy_without_any_activation_event_is_refused(db: Database) -> None:
    policy = RiskPolicy()

    def install_raw(conn: sqlite3.Connection) -> None:  # e.g. a script writing the table directly
        conn.execute(
            "INSERT INTO risk_policies(policy_version, name, policy_json, created_at, created_by, approved_by,"
            " supersedes, is_active) VALUES (?,?,?,?,?,?,NULL,1)",
            (policy.version, policy.name, policy.canonical_json(), "2026-11-05T15:00:00Z", "script", "script"),
        )

    db.write(install_raw)
    with pytest.raises(PolicyIntegrityError, match="no activation audit event"):
        PolicyStore(db, TradingMode.PAPER).active()


def test_readers_see_a_consistent_snapshot_during_concurrent_changes(db: Database, store: PolicyStore,
                                                                     token: str) -> None:
    stop = threading.Event()
    reader_errors: list[BaseException] = []
    seen: set[str] = set()

    def read_loop() -> None:
        while not stop.is_set():
            try:
                seen.add(store.active_version())
            except BaseException as exc:
                reader_errors.append(exc)
                return

    readers = [threading.Thread(target=read_loop) for _ in range(3)]
    for t in readers:
        t.start()
    try:
        for n in range(12):
            store.apply_change(RiskPolicy(max_order_notional=Decimal(1100 + n)), operator_token=token,
                               actor="victor", reason=f"step {n}")
    finally:
        stop.set()
        for t in readers:
            t.join()
    assert reader_errors == []
    assert len(seen) >= 1


def test_token_pasted_into_reason_or_actor_is_redacted_before_audit(db: Database, store: PolicyStore,
                                                                     token: str) -> None:
    store.apply_change(RiskPolicy(max_order_notional=Decimal("1000.00")), operator_token=token, actor="victor",
                       reason=f"oops pasted {token} here")
    with pytest.raises(OperatorAuthError):  # wrong token presented, real token pasted into the actor field
        store.apply_change(RiskPolicy(max_order_notional=Decimal("900.00")), operator_token="wrong-token-123",
                           actor=f"victor {token}", reason="reason with wrong-token-123")
    changes = store.changes()
    assert any(c["reason"] == "oops pasted [REDACTED] here" for c in changes)
    db.close()
    dump = "\n".join(sqlite3.connect(db.path).iterdump())
    assert token not in dump and "wrong-token-123" not in dump
