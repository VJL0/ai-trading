"""System preflight (spec §1): introspect this Mac and report actionable pass/warn/fail lines.

Usage: uv run --frozen python scripts/preflight.py [--json]
Exit code 1 only when a HARD requirement fails. Never prints secret values (presence only).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    hard: bool = False
    fix: str = ""


def run(argv: list[str], timeout: float = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def tool_version(name: str, argv: list[str]) -> tuple[str | None, str]:
    path = shutil.which(name) if "/" not in argv[0] else (argv[0] if Path(argv[0]).exists() else None)
    if not path:
        return None, "not found"
    code, out = run(argv)
    return (out.splitlines()[0] if out else "") if code == 0 else None, out[:200]


def checks() -> list[Check]:
    out: list[Check] = []
    mac_ver = platform.mac_ver()[0]
    out.append(Check("platform", "PASS", f"{platform.system()} {mac_ver}".strip()))
    arch = platform.machine()
    _, chip = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    out.append(Check("cpu", "PASS", f"{arch} ({chip})"))
    _, mem = run(["sysctl", "-n", "hw.memsize"])
    gb = int(mem) / 2**30 if mem.isdigit() else 0
    out.append(Check("ram", "PASS" if gb >= 16 else ("WARN" if gb >= 8 else "FAIL"), f"{gb:.0f} GiB", hard=gb < 8,
                     fix="LEAN build + ML need >= 8 GiB (16+ recommended)"))
    du = shutil.disk_usage(ROOT)
    free_gb = du.free / 2**30
    out.append(Check("disk", "PASS" if free_gb >= 20 else "FAIL", f"{free_gb:.0f} GiB free at {ROOT}", hard=free_gb < 5,
                     fix="free disk space (LEAN source+build ~3 GiB, datasets grow)"))

    for name, argv, hard, fix in [
        ("brew", ["brew", "--version"], False, "install Homebrew from https://brew.sh"),
        ("uv", ["uv", "--version"], True, "brew install uv"),
        ("git", ["git", "--version"], True, "xcode-select --install"),
        ("docker", ["docker", "--version"], False, "optional: OrbStack or Docker Desktop"),
        ("claude", ["claude", "--version"], False, "optional: Claude Code CLI for the trading MCP server"),
        ("sqlite3", ["sqlite3", "--version"], False, "ships with macOS"),
    ]:
        ver, detail = tool_version(name, argv)
        out.append(
            Check(
                name,
                "PASS" if ver is not None else ("FAIL" if hard else "WARN"),
                ver or detail,
                hard=hard and ver is None,
                fix=fix,
            )
        )

    dotnet = ROOT / ".tools" / "dotnet" / "dotnet"
    ver, detail = tool_version("dotnet", [str(dotnet), "--version"])
    out.append(Check("dotnet (project-local)", "PASS" if ver and ver.startswith("10.") else "FAIL",
                     ver or detail, hard=False,
                     fix="make setup  (installs .NET 10 SDK into .tools/dotnet via the official dotnet-install.sh)"))

    venv_py = ROOT / ".venv" / "bin" / "python"
    ver, detail = tool_version("python", [str(venv_py), "--version"])
    out.append(Check("python 3.11 venv", "PASS" if ver and "3.11" in ver else "FAIL", ver or detail, hard=True,
                     fix="make setup  (uv sync --python 3.11)"))

    lean_dll = ROOT / "trading" / "lean" / "Lean" / "Launcher" / "bin" / "Release" / "QuantConnect.Lean.Launcher.dll"
    out.append(Check("LEAN engine build", "PASS" if lean_dll.exists() else "WARN",
                     str(lean_dll.relative_to(ROOT)) if lean_dll.exists() else "not built",
                     fix="make lean-build"))

    try:
        from packages.core.config import Settings
        from packages.core.secrets import SecretName, SecretStore

        settings = Settings()
        out.append(Check("runtime mode", "PASS", settings.mode.value))
        store = SecretStore(settings.project_root, use_keychain=settings.use_keychain)
        pres = {
            "alpaca_paper": store.alpaca_paper_available(),
            "anthropic": store.has(SecretName.ANTHROPIC_API_KEY),
            "openai": store.has(SecretName.OPENAI_API_KEY),
        }
        missing = [k for k, v in pres.items() if not v]
        out.append(Check("credentials (presence only)", "PASS" if not missing else "WARN",
                         ", ".join(f"{k}={'present' if v else 'absent'}" for k, v in pres.items()),
                         fix="optional; add PAPER keys to Keychain: security add-generic-password -s local-ai-trader "
                             "-a ALPACA_PAPER_KEY_ID -w   (see docs/runbooks/broker-paper-onboarding.md). "
                             "Without them the system runs with explicit SIMULATED adapters."))
        amb = store.ignored_ambiguous_present()
        if amb:
            out.append(Check("ambiguous broker env vars", "WARN", f"ignored: {amb}",
                             fix="use ALPACA_PAPER_KEY_ID/ALPACA_PAPER_SECRET_KEY names; ambiguous names are never read"))
    except Exception as exc:  # preflight must not crash before setup
        out.append(Check("credentials", "WARN", f"could not inspect: {exc}"))

    out.append(Check("project dir", "PASS", str(ROOT)))
    return out


def main() -> int:
    results = checks()
    (ROOT / "var").mkdir(exist_ok=True)
    (ROOT / "var" / "preflight.json").write_text(json.dumps([asdict(c) for c in results], indent=2))
    if "--json" in sys.argv:
        print(json.dumps([asdict(c) for c in results], indent=2))
    else:
        for c in results:
            mark = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL"}[c.status]
            print(f"[{mark}] {c.name:28s} {c.detail}")
            if c.status != "PASS" and c.fix:
                print(f"       -> {c.fix}")
    hard_fail = [c for c in results if c.status == "FAIL" and c.hard]
    if hard_fail:
        print(f"\nPREFLIGHT FAILED: {', '.join(c.name for c in hard_fail)}")
        return 1
    print("\npreflight: hard requirements satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
