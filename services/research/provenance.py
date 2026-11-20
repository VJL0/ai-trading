"""Experiment provenance: git-free code version hash, run ids, JSON writers and manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from packages.core.ids import new_id

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

NO_EDGE_NOTE = (
    "NO EDGE CLAIMED. These are research benchmarks under a stated cost model. Results on synthetic "
    "or small sample data are not evidence of a tradable edge, and backtests do not predict paper or "
    "live fills (Alpaca paper fills omit impact, latency slippage, queue position, fees and dividends)."
)


@lru_cache(maxsize=1)
def code_version_hash() -> str:
    """sha256 over the research package's own source files (path + bytes), independent of git."""
    h = hashlib.sha256()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(RESEARCH_DIR):
        dirnames[:] = sorted(d for d in dirnames if d not in {".venv", "__pycache__"})  # never walk the qlib venv
        files.extend(Path(dirpath) / f for f in filenames if f.endswith(".py") or f in {"pyproject.toml", "uv.lock"})
    for path in sorted(files):
        h.update(path.relative_to(RESEARCH_DIR).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return "src-sha256:" + h.hexdigest()[:24]


def new_run_id(prefix: str) -> str:
    return new_id(prefix)


def library_versions(*names: str) -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return format(obj, "f")
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            raise ValueError("naive datetime in experiment output")
        return obj.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(obj, pd.Timestamp):
        if obj.tzinfo is None:
            raise ValueError("naive timestamp in experiment output")
        return obj.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _clean(obj: Any) -> Any:
    """Replace non-finite floats with None (strict JSON)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_clean(obj), indent=2, sort_keys=True, default=_default, allow_nan=False)


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def config_hash(obj: Any) -> str:
    text = json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), default=_default)
    return hashlib.sha256(text.encode()).hexdigest()


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
