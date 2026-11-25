"""Run the local API: uv run --frozen python -m services.api   (binds 127.0.0.1 only)."""

from __future__ import annotations

import uvicorn

from packages.core.config import Settings
from packages.core.modes import LIVE_EXECUTION_COMPILED, OperatorTradingMode

from .app import create_app
from .launcher import ApiStartupError, api_process_lock


def main() -> None:
    settings = Settings()
    if settings.trading_mode == OperatorTradingMode.LIVE and not LIVE_EXECUTION_COMPILED:
        raise SystemExit("Live trading is unavailable in this build")
    try:
        with api_process_lock(settings.run_dir):
            app = create_app(settings)
            uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info", access_log=False)
    except ApiStartupError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
