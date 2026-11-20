"""Bridge to the isolated Qlib v0.9.8 environment (``services/research/qlib_env``).

The root project never imports ``qlib``. This module (root venv) does two things:

1. ``convert_to_qlib`` writes our point-in-time bars into Qlib's local binary provider layout, exactly as
   ``qlib.data.storage.file_storage`` reads it (and as Qlib's ``scripts/dump_bin.py`` writes it):

   * ``calendars/day.txt`` — one ``YYYY-MM-DD`` NYSE session date per line (union over instruments);
   * ``instruments/all.txt`` — ``SYMBOL<TAB>first_date<TAB>last_date``;
   * ``features/<symbol lower>/<field>.day.bin`` — little-endian float32: the first value is the calendar
     index of the instrument's first date, followed by one value per calendar date (NaN where absent).

   Intraday input is aggregated to daily session bars first (``frames.aggregate_daily``: a daily bar is
   only knowable once its last minute is). Prices become float32 *features* (research only, not accounting).
2. ``run_qlib_workflow`` converts, plans chronological train/valid/test segments separated by ``H + 1``
   sessions, runs ``qlib_env/lat_qlib_workflow.py`` with the isolated interpreter in a subprocess (no network
   needed), and writes ``experiments/qlib/<run_id>/{manifest.json, qlib_results.json, pred_test.csv}``.

If the isolated environment is missing or Qlib fails, the run is recorded as ``BLOCKED``/``FAILED`` with the
exact error text; the native LightGBM pipeline (``services.research.ml``) does not depend on Qlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frames import SegmentPolicy, aggregate_daily, dataset_version_of, is_simulated_of, last_clean_segment, prepare_bars
from .provenance import (
    NO_EDGE_NOTE,
    RESEARCH_DIR,
    code_version_hash,
    library_versions,
    new_run_id,
    relpath,
    utc_now_iso,
    write_json,
)

QLIB_ENV_DIR = RESEARCH_DIR / "qlib_env"
QLIB_PYTHON = QLIB_ENV_DIR / ".venv" / "bin" / "python"
QLIB_WORKFLOW = QLIB_ENV_DIR / "lat_qlib_workflow.py"
QLIB_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "factor")
SETUP_HINT = "cd services/research/qlib_env && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB=0.9.8 uv sync --python 3.11"


def convert_to_qlib(
    bars_df: pd.DataFrame,
    provider_dir: Path | str,
    *,
    segment_policy: SegmentPolicy | None = SegmentPolicy(),  # frozen dataclass: safe default
) -> dict[str, Any]:
    """Write a Qlib provider directory from a bar frame. Returns a manifest with per-file sha256.

    ``segment_policy`` (default on) keeps the last clean contiguous daily segment per symbol so Qlib expressions
    never span a multi-session hole or a ticker change.
    """
    bars = prepare_bars(bars_df)
    daily = aggregate_daily(bars)
    data_quality: dict[str, Any] = {"applied": "none (segment_policy=None)"}
    if segment_policy is not None:
        daily, data_quality = last_clean_segment(daily, segment_policy)
    root = Path(provider_dir)
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(parents=True, exist_ok=True)
    dates = sorted(set(daily["session_date"]))
    if not dates:
        raise ValueError("no sessions to convert")
    index = {d: i for i, d in enumerate(dates)}
    files: dict[str, str] = {}

    def _record(path: Path) -> None:
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    cal_path = root / "calendars" / "day.txt"
    cal_path.write_text("".join(f"{d.isoformat()}\n" for d in dates), encoding="utf-8")
    _record(cal_path)
    inst_lines: list[str] = []
    for sym, g in daily.groupby("symbol", sort=True):
        sym_s = str(sym)
        if not sym_s.isalnum():
            raise ValueError(f"symbol {sym_s!r} not representable as a Qlib instrument directory")
        g_dates = list(g["session_date"])
        first, last = index[g_dates[0]], index[g_dates[-1]]
        inst_lines.append(f"{sym_s}\t{g_dates[0].isoformat()}\t{g_dates[-1].isoformat()}\n")
        fdir = root / "features" / sym_s.lower()
        fdir.mkdir(parents=True, exist_ok=True)
        pos = np.array([index[d] for d in g_dates], dtype=np.int64) - first
        for field in QLIB_FIELDS:
            values = np.full(last - first + 1, np.nan, dtype=np.float64)
            values[pos] = 1.0 if field == "factor" else g[f"{field}_f"].to_numpy(dtype=np.float64)
            path = fdir / f"{field}.day.bin"
            np.hstack([[float(first)], values]).astype("<f").tofile(path)
            _record(path)
    inst_path = root / "instruments" / "all.txt"
    inst_path.write_text("".join(inst_lines), encoding="utf-8")
    _record(inst_path)
    return {
        "provider_dir": relpath(root),
        "symbols": sorted(set(daily["symbol"])),
        "sessions": len(dates),
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "fields": list(QLIB_FIELDS),
        "source_timeframe": str(bars["timeframe"].iloc[0]),
        "aggregated_to_daily": str(bars["timeframe"].iloc[0]) != "1Day",
        "dataset_version": dataset_version_of(bars),
        "is_simulated": is_simulated_of(bars),
        "files_sha256": files,
        "data_quality": data_quality,
        "format": "qlib file_storage: <f float32, first value = calendar start index",
    }


def read_qlib_feature(provider_dir: Path | str, symbol: str, field: str) -> tuple[int, np.ndarray]:
    """Read back one ``.day.bin`` file (start index, values) - used by tests to verify the format."""
    raw = np.fromfile(Path(provider_dir) / "features" / symbol.lower() / f"{field}.day.bin", dtype="<f")
    return int(raw[0]), raw[1:].astype(np.float64)


def plan_segments(
    calendar: list[str], horizon: int, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, list[str]]:
    """Chronological train/valid/test date segments with ``horizon + 1`` session gaps (purge + embargo)."""
    n = len(calendar)
    gap = horizon + 1
    usable = n - 2 * gap - (horizon + 1)  # the last H+1 sessions have no complete label
    if usable < 60:
        raise ValueError(f"only {n} sessions: too short for train/valid/test with {gap}-session gaps")
    n_train = int(usable * fractions[0])
    n_valid = int(usable * fractions[1])
    n_test = usable - n_train - n_valid
    tr = (0, n_train - 1)
    va = (tr[1] + 1 + gap, tr[1] + gap + n_valid)
    te = (va[1] + 1 + gap, va[1] + gap + n_test)
    return {
        "train": [calendar[tr[0]], calendar[tr[1]]],
        "valid": [calendar[va[0]], calendar[va[1]]],
        "test": [calendar[te[0]], calendar[te[1]]],
    }


def qlib_environment_status() -> dict[str, Any]:
    if not QLIB_PYTHON.exists():
        return {"available": False, "reason": f"isolated interpreter missing at {relpath(QLIB_PYTHON)}; run: {SETUP_HINT}"}
    try:
        proc = subprocess.run(
            [str(QLIB_PYTHON), "-I", "-c", "import qlib, lightgbm; print(qlib.__version__, lightgbm.__version__)"],
            capture_output=True, text=True, timeout=180, check=False, env=_child_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"available": False, "reason": (proc.stderr or proc.stdout).strip()[-2000:]}
    qv, lv = proc.stdout.split()[-2:]
    return {"available": qv == "0.9.8", "qlib_version": qv, "lightgbm_version": lv,
            "reason": "" if qv == "0.9.8" else f"expected pyqlib 0.9.8, found {qv}"}


def _child_env() -> dict[str, str]:
    """Minimal environment for the Qlib subprocess: no inherited credentials, telemetry off."""
    keep = ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "1", "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1", "GIT_PYTHON_REFRESH": "quiet"})
    return env


def run_qlib_workflow(
    bars_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    horizon: int = 5,
    seed: int = 20261105,
    run_id: str | None = None,
    timeout_s: float = 1200.0,
    data_adjustments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert bars, run the isolated Qlib LightGBM workflow, write ``<out_dir>/<run_id>/``. Never raises on
    Qlib failures: the returned/recorded ``status`` is ``OK``, ``FAILED`` or ``BLOCKED`` with the exact error."""
    run_id = run_id or new_run_id("qlib")
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_ignore(Path(out_dir))
    started = utc_now_iso()
    provider = run_dir / "qlib_data"
    conv = convert_to_qlib(bars_df, provider)
    calendar = (provider / "calendars" / "day.txt").read_text().split()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "kind": "qlib_workflow",
        "created_at": started,
        "dataset_version": conv["dataset_version"],
        "is_simulated": conv["is_simulated"],
        "code_version": code_version_hash(),
        "conversion": conv,
        "horizon_sessions": horizon,
        "seed": seed,
        "data_adjustments": data_adjustments or {"applied": "none (caller supplied no corporate-action handling)"},
        "qlib_env": relpath(QLIB_ENV_DIR),
        "root_libraries": library_versions("numpy", "pandas"),
        "no_edge_note": NO_EDGE_NOTE,
        "run_dir": relpath(run_dir),
    }
    try:
        segments = plan_segments(calendar, horizon)
    except ValueError as exc:
        manifest.update({"status": "SKIPPED", "reason": str(exc)})
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    manifest["segments"] = segments
    env_status = qlib_environment_status()
    manifest["qlib_environment"] = env_status
    if not env_status.get("available"):
        manifest.update({"status": "BLOCKED", "reason": env_status.get("reason")})
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    cmd = [
        str(QLIB_PYTHON), "-I", str(QLIB_WORKFLOW), "--provider-uri", str(provider.resolve()), "--out",
        str(run_dir.resolve()), "--run-id", run_id, "--segments", json.dumps(segments), "--horizon", str(horizon),
        "--seed", str(seed),
    ]
    results_path = run_dir / "qlib_results.json"
    results_path.unlink(missing_ok=True)  # never read a stale result from an earlier attempt
    returncode: int | None = None
    tail = ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False, env=_child_env(),
                              cwd=str(run_dir.resolve()))
        (run_dir / "qlib_stdout.log").write_text(proc.stdout[-200_000:], encoding="utf-8")
        (run_dir / "qlib_stderr.log").write_text(proc.stderr[-200_000:], encoding="utf-8")
        returncode = proc.returncode
        tail = (proc.stderr or proc.stdout)[-4000:]
    except subprocess.TimeoutExpired as exc:
        manifest.update({"status": "FAILED", "reason": f"qlib workflow timed out after {timeout_s}s: {exc}"})
    except OSError as exc:
        manifest.update({"status": "BLOCKED", "reason": f"cannot launch isolated interpreter: {exc}"})
    if returncode is not None:
        manifest["returncode"] = returncode
        if results_path.exists():
            results = json.loads(results_path.read_text())
            manifest["status"] = results.get("status", "FAILED")
            if manifest["status"] == "OK":
                manifest["metrics"] = {
                    "recorder": results.get("recorder_metrics"),
                    "sig_ana_ic": results.get("sig_ana_ic"),
                    "test_pooled_rank_ic": results.get("test_pooled_rank_ic"),
                    "test_rows_scored": results.get("test_rows_scored"),
                    "best_iteration": results.get("model", {}).get("best_iteration"),
                }
                manifest["qlib_version"] = results.get("qlib_version")
            else:
                manifest["reason"] = f"{results.get('error_type')}: {results.get('error')}"
        else:
            manifest.update({"status": "FAILED", "reason": tail or "workflow produced no results file"})
    files = {p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(run_dir.glob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest["files_sha256"] = files
    manifest["run_dir"] = relpath(run_dir)
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def _write_ignore(base: Path) -> None:
    """Keep bulky derived Qlib provider copies and MLflow stores out of version control."""
    gi = base / ".gitignore"
    if not gi.exists():
        base.mkdir(parents=True, exist_ok=True)
        gi.write_text("# derived, re-creatable artefacts\n*/qlib_data/\n*/mlruns/\n", encoding="utf-8")
