"""``python -m services.capabilities probe`` — read-only Alpaca capability snapshot (no orders, no paid calls)."""

from __future__ import annotations

import argparse
import json
import sys

from packages.core.config import Settings

from .http import MissingAlpacaCredentials, ReadOnlyAlpacaHttp
from .probe import CapabilityProbe, load_latest, save_snapshot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m services.capabilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="query account/assets/contracts/data entitlements and save a dated snapshot")
    sub.add_parser("show", help="print the latest saved snapshot")
    args = ap.parse_args(argv)
    settings = Settings()
    if args.cmd == "show":
        snap = load_latest(settings.datasets_dir)
        print(json.dumps(snap, indent=2) if snap else "no snapshot")
        return 0 if snap else 2
    try:
        http = ReadOnlyAlpacaHttp.from_secrets(settings.project_root)
    except MissingAlpacaCredentials as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from .policy import load_active_policy

    active = load_active_policy(settings.project_root)
    with http:
        probe = CapabilityProbe(http, policy=active, policy_source="active paper policy (var/paper/ledger.sqlite)")
        snap = probe.run()
        path = save_snapshot(snap, settings.datasets_dir, probe.asset_frame())
    summary = {k: {"research": v["research_eligible"], "execution": v["execution_eligible"],
                   "policy": v["policy_execution"]["status"]} for k, v in snap["asset_classes"].items()}
    print(json.dumps({"saved": str(path), "requests": http.requests, "asset_classes": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
