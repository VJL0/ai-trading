"""Serve Local AI Trader MCP over stdio, with an optional ``--profile research``."""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
