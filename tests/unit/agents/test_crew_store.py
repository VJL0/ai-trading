from __future__ import annotations

import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from services.agents.contracts import Provider, TaskKind, TaskState
from services.agents.store import CrewLocked, CrewStore, SingleInstanceLock, task_id_for

from .test_crew_support import T0


def test_enqueue_is_idempotent_by_key(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    tid, created = s.enqueue(TaskKind.HYPOTHESIS, "hypothesis:momentum:1", {"mechanism": "momentum"}, now=T0)
    tid2, created2 = s.enqueue(TaskKind.HYPOTHESIS, "hypothesis:momentum:1", {"mechanism": "other"}, now=T0)
    assert created and not created2 and tid == tid2 == task_id_for(TaskKind.HYPOTHESIS, "hypothesis:momentum:1")
    t = s.get_task(tid)
    assert t is not None and t.payload == {"mechanism": "momentum"} and t.state is TaskState.QUEUED
    assert len(s.tasks()) == 1


def test_atomic_claim_two_stores_one_file(tmp_path: Path) -> None:
    path = tmp_path / "crew.sqlite3"
    a, b = CrewStore(path), CrewStore(path)
    tid, _ = a.enqueue(TaskKind.REVIEW, "review:x", {}, now=T0)
    assert a.claim(tid, T0) is not None
    assert b.claim(tid, T0) is None
    # race many claimers over two connections: exactly one wins each task
    ids = [a.enqueue(TaskKind.HYPOTHESIS, f"h:{i}", {}, now=T0)[0] for i in range(10)]
    wins: list[str] = []
    lock = threading.Lock()

    def worker(store: CrewStore) -> None:
        for i in ids:
            if store.claim(i, T0) is not None:
                with lock:
                    wins.append(i)

    threads = [threading.Thread(target=worker, args=(st,)) for st in (a, b, a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(wins) == sorted(ids)


def test_notes_are_append_only(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    seq = s.add_note("hypothesis", {"x": 1}, now=T0, provider=Provider.CLAUDE, mechanism="momentum")
    assert s.notes()[0].evidence_status == "AGENT_NOTE"
    db = s.raw_connection()
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("UPDATE notes SET body='{}' WHERE seq=?", (seq,))
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("DELETE FROM notes WHERE seq=?", (seq,))
    finally:
        db.close()
    assert s.notes()[0].body == {"x": 1}


def test_recover_running_requeues_without_consuming_attempt(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    tid, _ = s.enqueue(TaskKind.HYPOTHESIS, "h", {}, now=T0)
    assert s.claim(tid, T0) is not None
    rid = s.start_run(tid, Provider.CODEX, T0)
    # crash: a fresh store instance on the same file recovers
    s2 = CrewStore(tmp_path / "crew.sqlite3")
    assert s2.recover(T0 + timedelta(minutes=5)) == [tid]
    t = s2.get_task(tid)
    assert t is not None and t.state is TaskState.QUEUED and t.attempts == 0
    run = s2.task_runs(tid)[0]
    assert run["id"] == rid and run["outcome"] == "INTERRUPTED" and run["ended"]
    assert any(n.body.get("event") == "RECOVERED" for n in s2.notes(kind="event"))


def test_requeue_consumes_attempts_until_failed(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    tid, _ = s.enqueue(TaskKind.HYPOTHESIS, "h", {}, now=T0, max_attempts=2)
    s.claim(tid, T0)
    row = s.requeue(tid, now=T0, consume_attempt=True)
    assert row is not None and row.state is TaskState.QUEUED and row.attempts == 1
    s.claim(tid, T0)
    row = s.requeue(tid, now=T0, consume_attempt=False)
    assert row is not None and row.attempts == 1
    s.claim(tid, T0)
    row = s.requeue(tid, now=T0, consume_attempt=True)
    assert row is not None and row.state is TaskState.FAILED and row.attempts == 2


def test_expire_cancels_review_without_independent_reviewer(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    tid, _ = s.enqueue(TaskKind.REVIEW, "review:h", {}, now=T0, expires_at=T0 + timedelta(hours=24))
    assert s.expire(T0 + timedelta(hours=23)) == []
    assert s.runnable(T0 + timedelta(hours=23))
    assert s.expire(T0 + timedelta(hours=24)) == [tid]
    t = s.get_task(tid)
    assert t is not None and t.state is TaskState.CANCELLED and t.result == {"reason": "no independent reviewer"}


def test_single_instance_lock(tmp_path: Path) -> None:
    path = tmp_path / "crew.lock"
    with SingleInstanceLock(path), pytest.raises(CrewLocked):
        SingleInstanceLock(path).acquire()
    lk = SingleInstanceLock(path)
    lk.acquire()
    lk.release()


def test_runs_today_counts_utc_day(tmp_path: Path) -> None:
    s = CrewStore(tmp_path / "crew.sqlite3")
    s.start_run("t", Provider.CLAUDE, T0 - timedelta(days=1))
    rid = s.start_run("t", Provider.CLAUDE, T0)
    s.end_run(rid, "OK", now=T0)
    s.end_run(s.start_run("t", Provider.CLAUDE, T0), "UNAVAILABLE", now=T0)  # never ran: no quota spent
    assert s.runs_today(Provider.CLAUDE, T0) == 1
    assert s.runs_today(Provider.CODEX, T0) == 0


def test_claim_checks_schedule_at_update_time(tmp_path: Path) -> None:
    store = CrewStore(tmp_path / "crew.sqlite3")
    later = T0 + timedelta(minutes=1)
    future, _ = store.enqueue(TaskKind.HYPOTHESIS, "future", {}, now=T0, not_before=later)
    expired, _ = store.enqueue(TaskKind.REVIEW, "expired", {}, now=T0, expires_at=later)
    assert store.claim(future, T0) is None
    assert store.claim(expired, later) is None
    assert store.claim(future, later) is not None
