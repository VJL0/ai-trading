"""Run the local API: uv run --frozen python -m services.api   (binds 127.0.0.1 only)."""

from __future__ import annotations

import uvicorn

from packages.core.config import Settings

from .app import create_app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
