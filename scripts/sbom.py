"""Software bill of materials: Python (uv.lock), Node (pnpm lockfiles), .NET (packages.lock.json),
upstream source pins. Writes docs/sources/sbom.json and prints a licence summary.

Usage: uv run --frozen python scripts/sbom.py
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def python_packages() -> list[dict[str, str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    out = []
    for pkg in lock.get("package", []):
        name, version = pkg["name"], pkg.get("version", "")
        lic = ""
        try:
            md = metadata.metadata(name)
            lic = (md.get_all("License-Expression") or [""])[0] or (md.get_all("License") or [""])[0] or ""
            if not lic or len(lic) > 80:
                classifiers = [c for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
                lic = classifiers[0].split("::")[-1].strip() if classifiers else (lic[:80] if lic else "")
        except metadata.PackageNotFoundError:
            lic = "not installed in .venv"
        hashes = [w.get("hash", "") for w in pkg.get("wheels", [])][:1] or [pkg.get("sdist", {}).get("hash", "")]
        out.append({"ecosystem": "pypi", "name": name, "version": version, "license": lic, "hash": hashes[0] if hashes else ""})
    return out


def node_packages(lockfile: Path) -> list[dict[str, str]]:
    if not lockfile.exists():
        return []
    out = []
    text = lockfile.read_text()
    for m in re.finditer(r"^  '?(@?[^@\s']+)@([^:'\s(]+)[^:]*'?:\s*$", text, re.M):
        out.append({"ecosystem": "npm", "name": m.group(1), "version": m.group(2), "lockfile": str(lockfile.relative_to(ROOT))})
    return out


def dotnet_packages() -> list[dict[str, str]]:
    out = []
    for lock in ROOT.glob("trading/**/packages.lock.json"):
        if "/bin/" in str(lock) or "/obj/" in str(lock):
            continue
        data = json.loads(lock.read_text())
        for _tfm, deps in data.get("dependencies", {}).items():
            for name, info in deps.items():
                out.append({"ecosystem": "nuget", "name": name, "version": info.get("resolved", ""),
                            "hash": info.get("contentHash", ""), "lockfile": str(lock.relative_to(ROOT))})
    return out


def upstream_pins() -> list[dict[str, str]]:
    pins = []
    for path, name, lic in [
        ("trading/lean/Lean", "QuantConnect/Lean", "Apache-2.0"),
        ("trading/alpaca_plugin/Lean.Brokerages.Alpaca", "QuantConnect/Lean.Brokerages.Alpaca", "Apache-2.0"),
    ]:
        p = ROOT / path
        if p.exists():
            sha = subprocess.run(
                ["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            pins.append({"ecosystem": "git", "name": name, "version": sha, "license": lic})
    return pins


def main() -> int:
    sbom = {
        "python": python_packages(),
        "node_mac_agent": node_packages(ROOT / "apps/mac-agent/browser/pnpm-lock.yaml"),
        "dotnet": dotnet_packages(),
        "upstream": upstream_pins(),
    }
    out = ROOT / "docs/sources/sbom.json"
    out.write_text(json.dumps(sbom, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    for k, v in sbom.items():
        print(f"  {k:16s} {len(v)} components")
    lic_counts: dict[str, int] = {}
    for p in sbom["python"]:
        lic_counts[p["license"] or "UNKNOWN"] = lic_counts.get(p["license"] or "UNKNOWN", 0) + 1
    print("  python licences:", ", ".join(f"{k}={n}" for k, n in sorted(lic_counts.items(), key=lambda x: -x[1])[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
