"""
conftest.py — shared fixtures for llmrp tests.

Sets llmrp_CONFIG to point at test_config.toml before server is imported,
so the real config.toml and real env vars are never required.
"""

import os
import sys
from pathlib import Path

# Must happen before 'import server' anywhere in the test suite.
os.environ["llmrp_CONFIG"] = str(Path(__file__).parent / "test_config.toml")

# Remove leftover session file from previous runs so _load_sessions() is a no-op.
_session_file = Path("/tmp/llmrp_test_sessions.json")
if _session_file.exists():
    _session_file.unlink()

# Now import server — it will read test_config.toml and find no env vars set,
# so all providers get _api_key = None at import time (warnings logged, not raised).
import server  # noqa: E402

import pytest


@pytest.fixture(autouse=True)
def reset_state():
    """Before each test: clear session map and suppress disk writes."""
    server._session_map.clear()
    yield
    server._session_map.clear()
    if _session_file.exists():
        _session_file.unlink()
