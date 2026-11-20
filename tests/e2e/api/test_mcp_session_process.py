"""Full API process and MCP subprocess; operator disconnect cannot stop the session."""
from pathlib import Path

from scripts.smoke_paper import smoke


async def test_real_process_session_survives_operator_disconnect(tmp_path: Path) -> None:
    result = await smoke(tmp_path)
    assert result["operator_disconnected_during_session"] is True
    assert result["audit_verified_rows"] > 0
