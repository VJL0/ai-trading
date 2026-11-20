"""Dataset provenance manifests (``datasets/processed/manifests/<dataset_version>.json``).

Every dataset written through :class:`services.market_data.store.MarketDataStore` gets one manifest recording
source id, feed, symbols, time range, row counts, the sha256 of every Parquet file touched, the generator or
importer name/version, the LEAN commit when relevant, ``created_at``, ``is_synthetic`` and human-readable
limitations.

``dataset_version = "ds_" + sha256(canonical_json(manifest minus volatile keys))[:16]`` where the volatile keys
are ``dataset_version``, ``created_at`` and ``manifest_path``. Identical content therefore always yields the same
version (idempotent fixture generation), and any change to data, parameters or provenance yields a new one.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.contracts.enums import DataFeed
from packages.core.canonical import canonical_json, sha256_hex

MANIFEST_SCHEMA = "lat-dataset-manifest/1"
VOLATILE_KEYS: tuple[str, ...] = ("dataset_version", "created_at", "manifest_path")
VERSION_PREFIX = "ds_"
VERSION_HEX_LEN = 16
SERIES_SNAPSHOT = "snapshot"
SERIES_INCREMENTAL = "incremental"


@dataclass(frozen=True)
class DatasetSpec:
    """Caller-supplied provenance for one write batch (the store adds counts, hashes and the version)."""

    source_id: str
    feed: DataFeed
    generator: str
    generator_version: str
    is_synthetic: bool
    kind: str = "market_data"
    description: str = ""
    limitations: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    lean_commit: str | None = None
    vintage: str | None = None
    adjustment: str = "raw"
    defects_injected: tuple[Mapping[str, Any], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)
    series_mode: str = SERIES_INCREMENTAL
    """``snapshot``: a complete, self-contained series (generated fixture, imported bundle) that becomes the
    *active* series for its (feed, symbol)s; ``incremental``: one batch of an ingested series (IEX/SIP backfill)."""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required for provenance")
        if self.series_mode not in (SERIES_SNAPSHOT, SERIES_INCREMENTAL):
            raise ValueError(f"series_mode must be {SERIES_SNAPSHOT!r} or {SERIES_INCREMENTAL!r}")
        if self.is_synthetic != (self.feed == DataFeed.SYNTHETIC):
            raise ValueError("is_synthetic must be true exactly when feed is SYNTHETIC")

    @classmethod
    def coerce(cls, value: DatasetSpec | Mapping[str, Any]) -> DatasetSpec:
        if isinstance(value, DatasetSpec):
            return value
        data = dict(value)
        data["feed"] = DataFeed(data["feed"])
        for key in ("limitations", "defects_injected"):
            if key in data:
                data[key] = tuple(data[key])
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown dataset spec keys: {sorted(unknown)}")
        return cls(**data)

    @property
    def defects_flag(self) -> bool:
        return bool(self.defects_injected)


FILE_CONTENT_KEYS: tuple[str, ...] = ("path", "content_sha256", "content_rows")
"""Per-file keys that identify *this dataset's own rows*. ``sha256``/``rows`` describe the whole partition file at
write time, which also contains rows merged from other datasets of the same feed, so they are recorded (integrity
check) but excluded from the version hash: the same content always yields the same dataset_version regardless of what
else was stored before it."""


def _version_content(manifest: Mapping[str, Any]) -> dict[str, Any]:
    content = {k: v for k, v in manifest.items() if k not in VOLATILE_KEYS}
    files = content.get("files")
    if isinstance(files, list):
        normalized: list[Any] = []
        for item in files:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(item, Mapping) and "content_sha256" in item:
                normalized.append({k: item[k] for k in FILE_CONTENT_KEYS if k in item})  # pyright: ignore[reportUnknownArgumentType]
            else:
                normalized.append(item)
        content["files"] = normalized
    return content


def compute_dataset_version(manifest: Mapping[str, Any]) -> str:
    return VERSION_PREFIX + sha256_hex(canonical_json(_version_content(manifest)))[:VERSION_HEX_LEN]


def build_manifest(
    spec: DatasetSpec,
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    start: str | None,
    end: str | None,
    row_counts: Mapping[str, int],
    row_counts_by_symbol: Mapping[str, Mapping[str, int]],
    files: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "kind": spec.kind,
        "source_id": spec.source_id,
        "feed": spec.feed.value,
        "is_synthetic": spec.is_synthetic,
        "series_mode": spec.series_mode,
        "adjustment": spec.adjustment,
        "symbols": sorted(set(symbols)),
        "timeframes": sorted(set(timeframes)),
        "start": start,
        "end": end,
        "row_counts": dict(sorted(row_counts.items())),
        "row_counts_by_symbol": {s: dict(sorted(c.items())) for s, c in sorted(row_counts_by_symbol.items())},
        "files": sorted((dict(f) for f in files), key=lambda f: str(f["path"])),
        "generator": {"name": spec.generator, "version": spec.generator_version},
        "lean_commit": spec.lean_commit,
        "vintage": spec.vintage,
        "description": spec.description,
        "limitations": list(spec.limitations),
        "params": json.loads(canonical_json(dict(spec.params))),
        "defects_injected": json.loads(canonical_json([dict(d) for d in spec.defects_injected])),
        "extra": json.loads(canonical_json(dict(spec.extra))),
    }
    manifest["dataset_version"] = compute_dataset_version(manifest)
    manifest["created_at"] = created_at
    return manifest


def write_manifest_file(manifests_dir: Path, manifest: Mapping[str, Any]) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    version = str(manifest["dataset_version"])
    if not version.startswith(VERSION_PREFIX):
        raise ValueError(f"bad dataset_version {version!r}")
    path = manifests_dir / f"{version}.json"
    text = json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=manifests_dir, prefix=f".{version}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_manifest_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manifest_schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a {MANIFEST_SCHEMA} manifest")
    return dict(data)  # pyright: ignore[reportUnknownArgumentType]


def manifest_content_ok(manifest: Mapping[str, Any]) -> bool:
    """True when the stored dataset_version matches the manifest content (tamper check)."""
    return manifest.get("dataset_version") == compute_dataset_version(manifest)


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
