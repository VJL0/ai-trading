"""Fixtures for order-ledger unit tests (deterministic, offline)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.order_ledger.ol_support import Harness, make_harness


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.db.close()
