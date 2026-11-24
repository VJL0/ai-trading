"""Filesystem layout for quantlab data (``datasets/quantlab``) and results (``experiments/quantlab``)."""

from __future__ import annotations

from pathlib import Path

from packages.core.config import PROJECT_ROOT


def data_root(project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "datasets" / "quantlab"


def results_root(project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "experiments" / "quantlab"


def reports_dir(project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "docs" / "verification"
