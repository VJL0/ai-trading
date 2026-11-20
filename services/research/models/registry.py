"""Model registry: ``experiments/models/<registry_version>/`` = ``registry.json`` + ``artifacts/``.

* Artifacts use native / plain formats only (LightGBM text, CatBoost ``.cbm``, JSON for Ridge / HAR / EWMA / Markov /
  calibrators). Nothing is ever unpickled on load.
* ``registry.json`` is written LAST (atomic rename): a directory without it is an incomplete (crashed) training run
  and is ignored. It records the dataset version, code hashes, config, selected model per target, metrics,
  calibration, walk-forward parameters, frozen-test usage, ``is_simulated`` and the sha256 of every artifact; its own
  canonical content is hashed into ``registry_sha256``.
* ``load_registry`` returns the newest complete registry after verifying the self-hash, every artifact hash, that
  every deployed artifact is covered by a hash, that the deployed models agree with the recorded selection and that
  the inference-defining source code (``config.INFERENCE_SOURCES``: bar preparation, features, vol inputs, Hamilton
  filter, estimator loading) is byte-identical to the code that trained it. Any mismatch, parse error or load
  failure returns ``None`` (fail closed: tampered, corrupt or stale-code models are never served, and an older
  registry is NOT silently substituted for a corrupt newest one).
* The self-hash detects corruption and accidental edits; it is not a signature (anyone who can write
  ``experiments/models`` can retrain). The registry only feeds predictions/signals - every order still needs the
  independent risk engine and a signed intent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..features import FeatureSpec
from ..provenance import dumps, write_json
from .config import (
    CATBOOST_CLF,
    COST_MODEL_FULL,
    LGBM_RANK,
    MARKOV_2STATE,
    MOMENTUM_RANK,
    SPREAD_BASELINE,
    SPREAD_PLUS_RESIDUAL,
    ZERO_BASELINE,
    feature_spec_from_json,
    feature_spec_to_json,
    inference_code_hash,
    lgbm_reg_id,
    ret_target,
    ridge_id,
)
from .estimators import BoosterModel, LinearModel, MomentumModel, SuccessClassifier, ZeroModel
from .regime import RegimeModel
from .vol import VolModel
from .walkforward import COST_RESIDUAL_ID, FittedSuite, TrainResult

log = logging.getLogger(__name__)

REGISTRY_SCHEMA = "lat-model-registry/1"
REGISTRY_FILE = "registry.json"
ARTIFACTS = "artifacts"
MODELS_SUBDIR = "models"
_VERSION_RE = re.compile(r"^mreg_\d{8}T\d{6}Z_[0-9a-f]{12}$")
TARGET_ORDER = ("ret_5m", "ret_15m", "ret_60m", "success", "rank", "vol", "regime", "cost")


class RegistryError(RuntimeError):
    """The registry cannot be trusted (missing, corrupt or tampered)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(obj: dict[str, Any]) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _write_json_file(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj) + "\n", encoding="utf-8")
    return path


@dataclass
class ModelRegistry:
    registry_version: str
    path: Path
    meta: dict[str, Any]
    fitted: FittedSuite
    feature_spec: FeatureSpec

    @property
    def dataset_version(self) -> str:
        return str(self.meta["dataset_version"])

    @property
    def is_simulated(self) -> bool:
        return bool(self.meta["is_simulated"])

    @property
    def timeframe(self) -> str:
        return str(self.meta["timeframe"])

    @property
    def trained_at(self) -> str:
        return str(self.meta["trained_at"])

    @property
    def probability_calibrated(self) -> bool:
        return bool(self.meta.get("probability_calibrated"))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.meta["feature_columns"])

    def selected(self, target: str) -> str:
        return str(self.meta["selected"][target])

    def target(self, name: str) -> dict[str, Any]:
        return dict(self.meta["targets"][name])

    def model_version_of(self, target: str) -> str:
        sel = self.selected(target)
        for c in self.meta["targets"][target]["candidates"]:
            if c["model_id"] == sel:
                return str(c["model_version"])
        return "n/a"

    def report(self) -> dict[str, Any]:
        """Shape of ``GET /api/v1/models`` (docs/architecture/api-contract.md, 'Quant model suite')."""
        keys = ("model_id", "net_sharpe", "net_return", "turnover", "hit_rate", "brier", "ic", "n_test", "selected",
                "rejected_reason")
        targets: list[dict[str, Any]] = []
        for name in TARGET_ORDER:
            t = self.meta["targets"].get(name)
            if t is None:
                continue
            cands = [{k: c.get(k) for k in keys} | {"model_version": c.get("model_version")} for c in t["candidates"]]
            cal = t.get("calibration")
            targets.append({
                "target": name,
                "selected_model": t["selected_model"],
                "horizon_minutes": t.get("horizon_minutes"),
                "candidates": cands,
                "calibration": None if cal is None else {
                    "method": cal.get("method"),
                    "brier_raw": cal.get("brier_raw"),
                    "brier_calibrated": cal.get("brier_calibrated"),
                    "reliability": [{k: r.get(k) for k in ("bin", "pred", "obs", "n")} for r in cal.get("reliability", [])],
                    "evaluated_on": cal.get("evaluated_on"),
                    "probability_calibrated": cal.get("probability_calibrated"),
                },
            })
        wf = self.meta["walk_forward"]
        ft = wf["frozen_test"]
        mt = self.meta["multiple_testing"]
        return {
            "registry_version": self.registry_version,
            "trained_at": self.trained_at,
            "dataset_version": self.dataset_version,
            "is_simulated": self.is_simulated,
            "timeframe": self.timeframe,
            "selection_metric": "net_sharpe",
            "targets": targets,
            "walk_forward": {
                "folds": wf["folds"],
                "purge_minutes": wf["purge_minutes"],
                "embargo_minutes": wf["embargo_minutes"],
                "frozen_test": {"start": ft.get("start"), "end": ft.get("end"), "used": bool(ft.get("used")),
                                "status": ft.get("status")},
            },
            "multiple_testing": {"method": mt["method"], "alpha": mt["alpha"], "survivors": list(mt["survivors"])},
            "notes": list(self.meta.get("notes", [])),
        }


# --------------------------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------------------------


def _new_version(meta: dict[str, Any]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(f"{meta['run_id']}|{meta['config_hash']}|{meta['dataset_version']}".encode()).hexdigest()[:12]
    return f"mreg_{stamp}_{h}"


def save_registry(result: TrainResult, base_dir: Path | str) -> ModelRegistry:
    """Persist a training result under ``base_dir/<registry_version>/`` and return it re-loaded (verified)."""
    base = Path(base_dir)
    version = _new_version(result.meta)
    rdir = base / version
    adir = rdir / ARTIFACTS
    adir.mkdir(parents=True, exist_ok=False)
    fitted = result.fitted
    files: list[Path] = []
    deployed: dict[str, Any] = {"returns": {}, "success": None, "rank": None, "vol": None, "regime": None, "cost": None}
    for h, (mid, model) in sorted(fitted.returns.items()):
        if isinstance(model, BoosterModel | LinearModel):
            new = model.save(adir)
            files += new
        else:
            new = []
        deployed["returns"][str(h)] = {"model_id": mid, "files": [f.name for f in new]}
    if fitted.success is not None:
        new = fitted.success.save(adir)
        files += new
        deployed["success"] = {"model_id": CATBOOST_CLF, "method": fitted.success.method, "files": [f.name for f in new]}
    if fitted.rank is not None:
        mid, model = fitted.rank
        new = model.save(adir) if isinstance(model, BoosterModel) else []
        files += new
        deployed["rank"] = {"model_id": mid, "files": [f.name for f in new]}
    if fitted.vol is not None:
        p = _write_json_file(adir / "vol.json", fitted.vol.to_json())
        files.append(p)
        deployed["vol"] = {"model_id": fitted.vol.selected, "files": [p.name]}
    if fitted.regime is not None:
        p = _write_json_file(adir / "regime.json", fitted.regime.to_json())
        files.append(p)
        deployed["regime"] = {"model_id": MARKOV_2STATE, "files": [p.name]}
    new = fitted.cost_residual.save(adir) if fitted.cost_residual is not None else []
    files += new
    deployed["cost"] = {"model_id": fitted.cost_selected, "files": [f.name for f in new]}
    fs_path = _write_json_file(adir / "feature_spec.json", {
        "feature_spec": feature_spec_to_json(_spec_from_meta(result.meta)),
        "feature_columns": result.meta["feature_columns"],
    })
    files.append(fs_path)
    payload = json.loads(dumps(result.meta | {
        "schema": REGISTRY_SCHEMA,
        "registry_version": version,
        "deployed": deployed,
        "files_sha256": {f"{ARTIFACTS}/{p.name}": _sha256(p) for p in sorted(files)},
    }))
    payload["registry_sha256"] = _canonical_hash(payload)
    write_json(rdir / REGISTRY_FILE, payload)  # last: marks the registry complete
    reg = load_registry_dir(rdir)
    if reg is None:  # pragma: no cover - would indicate a serialisation bug
        raise RegistryError(f"freshly written registry {rdir} failed verification")
    return reg


def _spec_from_meta(meta: dict[str, Any]) -> FeatureSpec:
    return feature_spec_from_json(meta["feature_spec"])


# --------------------------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------------------------


def _verify(rdir: Path) -> dict[str, Any]:
    path = rdir / REGISTRY_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry.json unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryError("registry.json is not an object")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"unknown registry schema {payload.get('schema')!r}")
    if payload.get("registry_version") != rdir.name or not _VERSION_RE.match(rdir.name):
        raise RegistryError("registry_version does not match its directory")
    claimed = payload.pop("registry_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_hash(payload):
        raise RegistryError("registry.json content hash mismatch (tampered or corrupt)")
    files = payload.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise RegistryError("registry lists no artifact hashes")
    for rel, digest in files.items():
        rel_s = str(rel)
        if rel_s.startswith("/") or ".." in Path(rel_s).parts or not rel_s.startswith(f"{ARTIFACTS}/"):
            raise RegistryError(f"artifact path {rel_s!r} escapes the registry directory")
        fp = rdir / rel_s
        if not fp.is_file():
            raise RegistryError(f"artifact {rel_s} missing")
        if _sha256(fp) != digest:
            raise RegistryError(f"artifact {rel_s} hash mismatch (tampered or corrupt)")
    if payload.get("inference_code_hash") != inference_code_hash():
        raise RegistryError(
            f"registry was trained with inference code {payload.get('inference_code_hash')!r}, current code is "
            f"{inference_code_hash()!r} (features/inputs may differ): retrain (make models)")
    covered = {Path(str(r)).name for r in files}
    deployed = payload.get("deployed") or {}
    listed: list[str] = []
    for v in (deployed.get("returns") or {}).values():
        listed += list(v.get("files", []))
    for k in ("success", "rank", "vol", "regime", "cost"):
        if deployed.get(k):
            listed += list(deployed[k].get("files", []))
    missing = sorted(set(listed) - covered)
    if missing:
        raise RegistryError(f"deployed artifacts without a recorded hash: {missing}")
    return payload


def _load_fitted(payload: dict[str, Any], adir: Path, spec: FeatureSpec, columns: tuple[str, ...]) -> FittedSuite:
    dep = payload["deployed"]
    fitted = FittedSuite()
    for h_s, d in dep["returns"].items():
        h = int(h_s)
        mid = str(d["model_id"])
        if mid != payload["selected"][ret_target(h)]:
            raise RegistryError(f"deployed {mid} for {ret_target(h)} disagrees with the recorded selection")
        if mid == ZERO_BASELINE:
            fitted.returns[h] = (mid, ZeroModel())
        elif mid == lgbm_reg_id(h):
            fitted.returns[h] = (mid, BoosterModel.load(mid, adir))
        elif mid == ridge_id(h):
            fitted.returns[h] = (mid, LinearModel.load(mid, adir))
        else:
            raise RegistryError(f"unknown return model {mid}")
    sel = payload["selected"]
    if bool(dep.get("success")) != (sel["success"] == CATBOOST_CLF):
        raise RegistryError("deployed success classifier disagrees with the recorded selection")
    if dep.get("success"):
        fitted.success = SuccessClassifier.load(CATBOOST_CLF, adir)
        if fitted.success.method != dep["success"].get("method"):
            raise RegistryError("deployed calibrator disagrees with the recorded calibration method")
    if bool(dep.get("rank")) != (sel["rank"] != ZERO_BASELINE):
        raise RegistryError("deployed ranker disagrees with the recorded selection")
    if dep.get("rank"):
        mid = str(dep["rank"]["model_id"])
        if mid != sel["rank"]:
            raise RegistryError(f"deployed ranker {mid} disagrees with the recorded selection {sel['rank']}")
        if mid == LGBM_RANK:
            fitted.rank = (mid, BoosterModel.load(mid, adir))
        elif mid == MOMENTUM_RANK:
            fitted.rank = (mid, MomentumModel(columns, spec))
        else:
            raise RegistryError(f"unknown rank model {mid}")
    if dep.get("vol"):
        fitted.vol = VolModel.from_json(json.loads((adir / "vol.json").read_text(encoding="utf-8")))
        if fitted.vol.selected != sel["vol"] or dep["vol"].get("model_id") != sel["vol"]:
            raise RegistryError("deployed vol model disagrees with the recorded selection")
    elif sel["vol"] != "none":
        raise RegistryError("selected vol model is not deployed")
    if bool(dep.get("regime")) != (sel["regime"] == MARKOV_2STATE):
        raise RegistryError("deployed regime model disagrees with the recorded selection")
    if dep.get("regime"):
        fitted.regime = RegimeModel.from_json(json.loads((adir / "regime.json").read_text(encoding="utf-8")))
    cost = dep.get("cost") or {}
    fitted.cost_selected = str(cost.get("model_id", SPREAD_BASELINE))
    if fitted.cost_selected not in (SPREAD_BASELINE, COST_MODEL_FULL, SPREAD_PLUS_RESIDUAL):
        raise RegistryError(f"unknown cost model {fitted.cost_selected}")
    if fitted.cost_selected != payload["selected"]["cost"]:
        raise RegistryError("deployed cost model disagrees with the recorded selection")
    if fitted.cost_selected == SPREAD_PLUS_RESIDUAL:
        fitted.cost_residual = BoosterModel.load(COST_RESIDUAL_ID, adir)
    return fitted


def load_registry_dir(rdir: Path | str) -> ModelRegistry | None:
    """Verify and load one registry directory; ``None`` on any problem (logged, never raised)."""
    d = Path(rdir)
    try:
        payload = _verify(d)
        adir = d / ARTIFACTS
        fs = json.loads((adir / "feature_spec.json").read_text(encoding="utf-8"))
        spec = feature_spec_from_json(fs["feature_spec"])
        columns = tuple(str(c) for c in fs["feature_columns"])
        if list(columns) != list(payload["feature_columns"]):
            raise RegistryError("feature columns disagree between registry.json and feature_spec.json")
        fitted = _load_fitted(payload, adir, spec, columns)
    except Exception as exc:  # any failure -> fail closed (no model served)
        log.warning("model registry %s rejected: %s: %s", d, type(exc).__name__, exc)
        return None
    return ModelRegistry(registry_version=d.name, path=d, meta=payload, fitted=fitted, feature_spec=spec)


def registry_root(experiments_dir: Path | str) -> Path:
    return Path(experiments_dir) / MODELS_SUBDIR


def list_registry_versions(experiments_dir: Path | str) -> list[str]:
    """Complete registries (``registry.json`` present), newest first."""
    root = registry_root(experiments_dir)
    if not root.is_dir():
        return []
    names = [p.name for p in root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name) and (p / REGISTRY_FILE).is_file()]
    return sorted(names, reverse=True)


def load_registry(experiments_dir: Path | str, registry_version: str | None = None) -> ModelRegistry | None:
    """Newest complete registry under ``<experiments_dir>/models`` (or the named one), verified; else ``None``."""
    if registry_version is not None:
        if not _VERSION_RE.match(registry_version):
            return None
        return load_registry_dir(registry_root(experiments_dir) / registry_version)
    versions = list_registry_versions(experiments_dir)
    if not versions:
        return None
    return load_registry_dir(registry_root(experiments_dir) / versions[0])


def register_research_challenger(experiments_dir: Path, job: str, params: dict[str, Any],
                                 evidence: dict[str, Any], artifacts: Path) -> Path:
    """Hash-bound research catalog beside deployed registries. Never selected by load_registry.

    Sequence models require independent confirmation before a runtime adapter may be enabled.
    Keeping this namespace separate prevents an incomplete challenger replacing the active suite.
    """
    if not re.fullmatch(r'[0-9a-f]{64}', job):
        raise RegistryError('invalid challenger job id')
    files = {str(p.resolve()): _sha256(p) for p in sorted(artifacts.iterdir()) if p.is_file()}
    payload = {'schema': 'lat-research-challenger/1', 'model_id': params['family'], 'job': job,
               'params': params, 'status': 'RESEARCH_ONLY', 'selected': False,
               'dataset_key': evidence['dataset_key'], 'code_hash': evidence['code_hash'],
               'promotion_blockers': evidence['promotion_blockers'], 'files_sha256': files}
    payload['registry_sha256'] = _canonical_hash(payload)
    path = registry_root(experiments_dir) / 'challengers' / f'{job}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path
