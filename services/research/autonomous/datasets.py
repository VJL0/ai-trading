"""Select complete research histories without confusing ingestion batches with histories."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..cli import MODEL_SUITE_EXCLUDED_KINDS


def pick_history(store: Any) -> dict[str, Any] | None:
    """Prefer real, sufficiently long histories; never splice feeds or dataset revisions.

    540 calendar days is only a selection heuristic. Actual session/trade counts
    and the independent validation gates remain authoritative. When history is
    shorter, use the longest available dataset for diagnostic research.
    """
    candidates = []
    for manifest in store.list_manifests(include_defective=False):
        if ('1Min' not in (manifest.get('timeframes') or []) or manifest.get('defects_injected')
                or manifest.get('kind') in MODEL_SUITE_EXCLUDED_KINDS):
            continue
        try:
            start = datetime.fromisoformat(manifest['start'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(manifest['end'].replace('Z', '+00:00'))
            if start.tzinfo is None or end.tzinfo is None or end <= start:
                continue
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        candidates.append((manifest, (end - start).total_seconds(), end))
    real = [row for row in candidates if row[0].get('is_synthetic') is False]
    candidates = real or candidates
    if not candidates:
        return None
    threshold = min(540 * 86400, max(row[1] for row in candidates))
    eligible = [row for row in candidates if row[1] >= threshold]
    # list_manifests is ingestion ordered; latest ingestion breaks equal-date ties.
    return max(enumerate(eligible), key=lambda item: (item[1][2], item[0]))[1][0]
