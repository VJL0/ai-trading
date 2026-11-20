"""Minimal Microsoft Qlib v0.9.8 LightGBM workflow over a locally converted provider.

Runs ONLY inside the isolated environment ``services/research/qlib_env/.venv`` (never the root venv):

    services/research/qlib_env/.venv/bin/python -I services/research/qlib_env/lat_qlib_workflow.py \
        --provider-uri <dir> --out <run_dir> --run-id <id> --segments '<json>' [--horizon 5] [--seed N]

It is launched by ``services.research.qlib_bridge.run_qlib_workflow``, which first converts our point-in-time
bars into Qlib's binary provider format. The workflow is the standard Qlib one:

* ``qlib.init`` on the local ``provider_uri`` (region US, single kernel, no redis/expression cache);
* ``DataHandlerLP`` + ``QlibDataLoader`` with a small set of Qlib expressions (returns over 1/5/20 days,
  realised vol, volume z-score, range, distance from the 20-day mean) - a reduced Alpha158-style set because
  the local samples are short;
* label ``Ref($open, -(H+1)) / Ref($open, -1) - 1``: enter at the next session's open, exit ``H`` sessions
  later at the open (the same execution convention as the Python baselines);
* ``DatasetH`` with chronological train / valid / test segments separated by gaps of ``H + 1`` sessions
  (purge/embargo); ``valid`` drives early stopping only; ``test`` is evaluated once, after fitting;
* ``qlib.contrib.model.gbdt.LGBModel`` (deterministic, single-thread, seeded) inside ``R.start`` with
  ``SignalRecord`` + ``SigAnaRecord`` (IC / Rank IC) written to a local file-based MLflow store.

Outputs ``<out>/qlib_results.json`` and ``<out>/pred_test.csv``. No network access is needed or attempted.
Nothing here is evidence of edge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _feature_config() -> tuple[list[str], list[str]]:
    lr = "Log($close/Ref($close, 1))"
    lv = "Log($volume + 1)"
    fields = [
        "Log($close/Ref($close, 1))",
        "Log($close/Ref($close, 5))",
        "Log($close/Ref($close, 20))",
        f"Std({lr}, 20)",
        f"({lv} - Mean({lv}, 20)) / (Std({lv}, 20) + 1e-12)",
        "($high - $low) / $close",
        "$close / Mean($close, 20) - 1",
    ]
    names = ["RET1", "RET5", "RET20", "RV20", "VOLZ20", "RANGE1", "DISTMA20"]
    return fields, names


def _finite(x: Any) -> Any:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _remove_stray_filelock(out: Path) -> None:
    """qlib/workflow/expm.py builds its FileLock path as ``os.path.join(netloc, path.lstrip("/"))`` - a path
    *relative to the CWD*. With CWD = out that leaves ``out/<abs path>/mlruns/filelock``; remove that copy only
    (it never contains anything but the lock file)."""
    stray_rel = Path(str(out / "mlruns").lstrip("/"))
    top = out / stray_rel.parts[0]
    if not top.is_dir():
        return
    files = [p for p in top.rglob("*") if p.is_file()]
    if all(p.name == "filelock" for p in files):
        import shutil

        shutil.rmtree(top)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider-uri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--segments", required=True, help='JSON {"train":[s,e],"valid":[s,e],"test":[s,e]}')
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    started = time.time()
    result: dict[str, Any] = {"run_id": args.run_id, "status": "FAILED"}
    try:
        import lightgbm
        import pandas as pd
        import qlib
        from qlib.constant import REG_US
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.data import D
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R
        from qlib.workflow.record_temp import SigAnaRecord, SignalRecord

        qlib.init(
            provider_uri=str(Path(args.provider_uri).resolve()),
            region=REG_US,
            kernels=1,
            joblib_backend=None,
            expression_cache=None,
            dataset_cache=None,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": (out / "mlruns").as_uri(), "default_exp_name": "lat_qlib"},
            },
        )
        segments = {k: (str(v[0]), str(v[1])) for k, v in json.loads(args.segments).items()}
        h = int(args.horizon)
        fields, names = _feature_config()
        label_expr = f"Ref($open, -{h + 1}) / Ref($open, -1) - 1"
        cal = D.calendar(freq="day")
        instruments = D.instruments("all")
        inst_list = D.list_instruments(instruments, as_list=True)
        handler = DataHandlerLP(
            instruments="all",
            start_time=str(cal[0].date()),
            end_time=str(cal[-1].date()),
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {"config": {"feature": (fields, names), "label": ([label_expr], ["LABEL0"])}, "freq": "day"},
            },
            learn_processors=[{"class": "DropnaLabel"}],
            infer_processors=[],
        )
        dataset = DatasetH(handler, segments=segments)
        model = LGBModel(
            loss="mse",
            early_stopping_rounds=30,
            num_boost_round=300,
            learning_rate=0.03,
            num_leaves=7,
            min_data_in_leaf=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            seed=int(args.seed),
            deterministic=True,
            force_row_wise=True,
            num_threads=1,
        )
        seg_counts: dict[str, int] = {}
        for seg in segments:
            df = dataset.prepare(seg, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            seg_counts[seg] = len(df)
        with R.start(experiment_name="lat_qlib", recorder_name=args.run_id):
            model.fit(dataset, verbose_eval=0)
            rec = R.get_recorder()
            SignalRecord(model=model, dataset=dataset, recorder=rec).generate()
            SigAnaRecord(recorder=rec, ana_long_short=False, ann_scaler=252).generate()
            recorder_id = rec.id
        # metrics are logged asynchronously; read them back after the recorder has been closed and flushed
        done = R.get_recorder(recorder_id=recorder_id, experiment_name="lat_qlib")
        metrics = {k: _finite(v) for k, v in done.list_metrics().items()}
        pred = done.load_object("pred.pkl")
        label = done.load_object("label.pkl")
        ic = done.load_object("sig_analysis/ic.pkl")
        ric = done.load_object("sig_analysis/ric.pkl")
        ic_stats = {
            "dates": int(ic.dropna().shape[0]),
            "ic_mean": _finite(ic.mean()),
            "ic_std": _finite(ic.std()),
            "rank_ic_mean": _finite(ric.mean()),
            "rank_ic_std": _finite(ric.std()),
        }
        joined = pred.join(label, how="inner")
        joined.columns = ["score", "label"]
        joined = joined.dropna()
        pooled_ic = _finite(joined["score"].corr(joined["label"], method="spearman")) if len(joined) > 2 else None
        joined.reset_index().to_csv(out / "pred_test.csv", index=False)
        best_iter = int(getattr(model.model, "best_iteration", 0) or 0)
        result = {
            "run_id": args.run_id,
            "status": "OK",
            "qlib_version": qlib.__version__,
            "lightgbm_version": lightgbm.__version__,
            "pandas_version": pd.__version__,
            "python": sys.version.split()[0],
            "provider_uri": str(Path(args.provider_uri).resolve()),
            "instruments": inst_list,
            "calendar": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "sessions": len(cal)},
            "segments": {k: list(v) for k, v in segments.items()},
            "segment_rows_after_dropna_label": seg_counts,
            "features": dict(zip(names, fields, strict=True)),
            "label": label_expr,
            "horizon_sessions": h,
            "model": {"class": "qlib.contrib.model.gbdt.LGBModel", "params": model.params,
                      "best_iteration": best_iter},
            "seed": int(args.seed),
            "recorder_metrics": metrics,
            "sig_ana_ic": ic_stats,
            "test_pooled_rank_ic": pooled_ic,
            "test_rows_scored": len(joined),
            "mlflow_recorder_id": recorder_id,
            "elapsed_s": round(time.time() - started, 3),
            "notes": [
                "SigAnaRecord IC/Rank IC are per-date cross-sectional statistics; with very few instruments they are "
                "extremely noisy.",
                "test_pooled_rank_ic pools all (date, instrument) test rows; overlapping H-day labels are dependent.",
                "Valid segment drives early stopping only; the test segment is scored once after fitting.",
            ],
        }
    except Exception as exc:  # report the exact obstacle for the bridge (BLOCKED/FAILED), never hide it
        result = {
            "run_id": args.run_id,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.time() - started, 3),
        }
    _remove_stray_filelock(out)
    (out / "qlib_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
