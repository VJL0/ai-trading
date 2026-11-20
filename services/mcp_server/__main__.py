"""``uv run --frozen python -m services.mcp_server`` — the local-ai-trader MCP server over stdio."""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
