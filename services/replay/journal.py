"""Append-only replay journal (JSONL) plus a summary JSON.

Every record carries ``execution_label: "REPLAY"`` (the operator-facing label of this harness; the runtime objects
underneath label their fills ``SIMULATED``), ``is_simulated: true`` and the data provenance of the snapshot.
Records are only ever appended; a resumed replay appends ``RESUMED`` and continues the sequence. A line that a
crash truncated is skipped by the reader and reported (never rewritten).

Record kinds: ``RUN_START``, ``RESUMED``, ``STARTUP_RECONCILE``, ``SESSION_START``, ``DECISION`` (one decision
cycle), ``RISK_DECISION`` (approved) / ``REJECTION`` (with the failing check names), ``INTENT`` (signed intent
recorded), ``ORDER`` (every ledger state transition), ``FILL``, ``LEG_COST`` (runtime cost model, once per filled
order leg), ``STOP`` (protective stop placement), ``EXIT`` (exit trigger / exit order), ``CANCEL``,
``POSITION_ACTION`` (other position-manager actions), ``ROUND_TRIP`` (closed position plan), ``MARK`` (bar-end
equity mark and the journal's resume cursors), ``SESSION_END``, ``RUN_END``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from services.order_ledger.timeutil import iso

EXECUTION_LABEL = "REPLAY"
RUNTIME_LABEL = "SIMULATED"


class Journal:
    def __init__(self, path: Path, provenance: dict[str, Any]) -> None:
        self.path = path
        self.provenance = provenance
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        for rec in self.read(path):
            self.seq = max(self.seq, int(rec.get("seq", 0)))
        self.truncated_lines = self.count_unreadable(path)
        self._fh = path.open("a", encoding="utf-8")
        if path.stat().st_size:
            with path.open("rb") as tail:
                tail.seek(-1, os.SEEK_END)
                if tail.read(1) != b"\n":
                    self._fh.write("\n")  # terminate a crash-truncated line without rewriting it

    @staticmethod
    def read(path: Path) -> Iterator[dict[str, Any]]:
        for record in Journal._lines(path):
            if record is not None:
                yield record

    @staticmethod
    def count_unreadable(path: Path) -> int:
        return sum(record is None for record in Journal._lines(path))

    @staticmethod
    def _lines(path: Path) -> Iterator[dict[str, Any] | None]:
        if not path.is_file():
            return
        with path.open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    record = None
                yield record if isinstance(record, dict) else None

    def records(self) -> list[dict[str, Any]]:
        self._fh.flush()
        return list(self.read(self.path))

    def write(self, kind: str, at: datetime, **payload: Any) -> dict[str, Any]:
        self.seq += 1
        rec = {**payload, "seq": self.seq, "kind": kind, "at": iso(at), "execution_label": EXECUTION_LABEL,
               "runtime_label": RUNTIME_LABEL, "is_simulated": True, "provenance": self.provenance}
        self._fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")
        return rec

    def sync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self.sync()
            self._fh.close()


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        tmp = Path(fh.name)
        try:
            json.dump({**summary, "execution_label": EXECUTION_LABEL, "runtime_label": RUNTIME_LABEL,
                       "is_simulated": True}, fh, indent=1, default=str, sort_keys=True)
            fh.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
