"""Select complete research histories without confusing ingestion batches with histories."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..cli import MODEL_SUITE_EXCLUDED_KINDS

RUNTIME_FEED_ORDER = ("IEX", "SIP")
"""Feeds research prefers, in order: the feed the paper runtime trades on first (research/runtime parity)."""


def _span(manifest: dict[str, Any]) -> tuple[float, datetime] | None:
    try:
        start = datetime.fromisoformat(str(manifest['start']).replace('Z', '+00:00'))
        end = datetime.fromisoformat(str(manifest['end']).replace('Z', '+00:00'))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        return None
    return (end - start).total_seconds(), end


def pick_history(store: Any) -> dict[str, Any] | None:
    """Prefer real, sufficiently long histories; never splice feeds or dataset revisions.

    1. The research snapshot the data pipeline marked *ready* (immutable, cumulative, only switched when enough
       genuinely new complete sessions arrived) for the runtime's feed.
    2. Otherwise the longest real candidate among research snapshots and stored datasets (snapshots win ties).

    540 calendar days is only a selection heuristic. Actual session/trade counts and the independent validation gates
    remain authoritative. When history is shorter, use the longest available dataset for diagnostic research.
    """
    datasets_dir = getattr(store, 'datasets_dir', None)
    if datasets_dir is not None:
        from services.market_data.pipeline import ready_snapshot

        for feed in RUNTIME_FEED_ORDER:
            sid = ready_snapshot(Path(datasets_dir), feed)
            snap = store.get_manifest(sid) if sid else None
            if snap is not None:
                return dict(snap)
    snapshots: list[dict[str, Any]] = []
    lister = getattr(store, 'list_research_snapshots', None)
    if lister is not None:
        snapshots = [dict(s) for s in lister()]
    candidates = []
    for order, manifest in enumerate([*store.list_manifests(include_defective=False), *snapshots]):
        if ('1Min' not in (manifest.get('timeframes') or []) or manifest.get('defects_injected')
                or manifest.get('kind') in MODEL_SUITE_EXCLUDED_KINDS):
            continue
        span = _span(manifest)
        if span is None:
            continue
        is_snapshot = str(manifest.get('dataset_version', '')).startswith('rs_')
        candidates.append((manifest, span[0], span[1], is_snapshot, order))
    real = [row for row in candidates if row[0].get('is_synthetic') is False]
    candidates = real or candidates
    if not candidates:
        return None
    threshold = min(540 * 86400, max(row[1] for row in candidates))
    eligible = [row for row in candidates if row[1] >= threshold]
    # latest end first, then research snapshots over raw ingestion batches, then registration order
    return max(eligible, key=lambda row: (row[2], row[3], row[4]))[0]
