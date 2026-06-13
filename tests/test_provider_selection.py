"""
Tests for _pick_provider() filtering logic when providers have missing API keys.

test_config.toml defines one model "test-model" with three providers (p-a, p-b, p-c),
each pointing at env vars TEST_KEY_A / TEST_KEY_B / TEST_KEY_C.

Tests inject resolved keys directly into MODELS[...]["_api_key"] to simulate
the various combinations of present/absent keys.
"""

import hashlib
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL = "test-model"


def _set_keys(**provider_keys: str | None):
    """
    Set _api_key on each provider in MODELS["test-model"].
    Pass provider_keys as name=value (None = missing key).

    Example:
        _set_keys(**{"p-a": "sk-a", "p-b": None, "p-c": None})
    """
    for p in server.MODELS[MODEL]:
        if p["name"] in provider_keys:
            p["_api_key"] = provider_keys[p["name"]]


def _expected_pick(session_id: str, valid_providers: list[dict]) -> str:
    """Replicate the hash logic to predict which provider name gets chosen."""
    slots: list[str] = []
    for p in valid_providers:
        slots.extend([p["name"]] * int(p.get("weight", 1)))
    idx = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % len(slots)
    return slots[idx]


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_literal_key_returned_unchanged(self):
        assert server._resolve_api_key("sk-literal") == "sk-literal"

    def test_env_var_present(self, monkeypatch):
        monkeypatch.setenv("TEST_RESOLVE_KEY", "resolved-value")
        assert server._resolve_api_key("env:TEST_RESOLVE_KEY") == "resolved-value"

    def test_env_var_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("TEST_RESOLVE_KEY", raising=False)
        result = server._resolve_api_key("env:TEST_RESOLVE_KEY")
        assert result is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("TEST_RESOLVE_KEY", "")
        result = server._resolve_api_key("env:TEST_RESOLVE_KEY")
        assert result is None


# ---------------------------------------------------------------------------
# _pick_provider — fresh pick (no sticky session in map)
# ---------------------------------------------------------------------------

class TestPickProviderFreshPick:
    def test_only_one_provider_has_key(self):
        """Only p-c has a key; must always pick p-c regardless of session."""
        _set_keys(**{"p-a": None, "p-b": None, "p-c": "sk-c"})
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("session-1", MODEL)
        assert result["name"] == "p-c"
        assert result["_api_key"] == "sk-c"

    def test_two_providers_missing_one_available(self):
        _set_keys(**{"p-a": None, "p-b": "sk-b", "p-c": None})
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("any-session", MODEL)
        assert result["name"] == "p-b"

    def test_first_provider_missing_picks_from_rest(self):
        """p-a is missing; pick must come from p-b or p-c."""
        _set_keys(**{"p-a": None, "p-b": "sk-b", "p-c": "sk-c"})
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("session-xyz", MODEL)
        assert result["name"] in ("p-b", "p-c")
        assert result["_api_key"] is not None

    def test_all_providers_missing_raises_503(self):
        _set_keys(**{"p-a": None, "p-b": None, "p-c": None})
        with patch.object(server, "_save_sessions"):
            with pytest.raises(HTTPException) as exc_info:
                server._pick_provider("session-1", MODEL)
        assert exc_info.value.status_code == 503
        assert "test-model" in exc_info.value.detail

    def test_all_providers_present_picks_deterministically(self):
        """With all keys set, the same session_id always picks the same provider."""
        _set_keys(**{"p-a": "sk-a", "p-b": "sk-b", "p-c": "sk-c"})
        valid = [p for p in server.MODELS[MODEL] if p["_api_key"]]
        session_id = "deterministic-session-42"
        expected = _expected_pick(session_id, valid)

        with patch.object(server, "_save_sessions"):
            result = server._pick_provider(session_id, MODEL)
        assert result["name"] == expected

    def test_missing_provider_not_in_slots(self):
        """p-b is missing; it must never be chosen across many distinct sessions."""
        _set_keys(**{"p-a": "sk-a", "p-b": None, "p-c": "sk-c"})
        with patch.object(server, "_save_sessions"):
            for i in range(50):
                server._session_map.clear()
                result = server._pick_provider(f"session-{i}", MODEL)
                assert result["name"] != "p-b", f"p-b (no key) was picked for session-{i}"

    def test_pick_stored_in_session_map(self):
        """After a fresh pick, store_key must appear in _session_map."""
        _set_keys(**{"p-a": "sk-a", "p-b": "sk-b", "p-c": "sk-c"})
        with patch.object(server, "_save_sessions"):
            server._pick_provider("my-session", MODEL)
        assert f"my-session:{MODEL}" in server._session_map


# ---------------------------------------------------------------------------
# _pick_provider — sticky session (already in session map)
# ---------------------------------------------------------------------------

class TestPickProviderStickySession:
    def test_sticky_hit_returns_same_provider(self):
        """If session map already has p-a and p-a has a key, return p-a."""
        _set_keys(**{"p-a": "sk-a", "p-b": "sk-b", "p-c": "sk-c"})
        server._session_map[f"my-session:{MODEL}"] = "p-a"
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("my-session", MODEL)
        assert result["name"] == "p-a"

    def test_sticky_provider_lost_key_triggers_repick(self):
        """
        Session was mapped to p-a, but p-a's key disappeared.
        _pick_provider must fall through to a re-pick from valid providers.
        """
        _set_keys(**{"p-a": None, "p-b": "sk-b", "p-c": "sk-c"})
        server._session_map[f"my-session:{MODEL}"] = "p-a"
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("my-session", MODEL)
        # p-a must NOT be returned
        assert result["name"] != "p-a"
        assert result["_api_key"] is not None

    def test_sticky_provider_lost_key_all_missing_raises_503(self):
        """
        Session mapped to p-a, p-a has no key, and no other provider has one either.
        Must raise 503.
        """
        _set_keys(**{"p-a": None, "p-b": None, "p-c": None})
        server._session_map[f"my-session:{MODEL}"] = "p-a"
        with patch.object(server, "_save_sessions"):
            with pytest.raises(HTTPException) as exc_info:
                server._pick_provider("my-session", MODEL)
        assert exc_info.value.status_code == 503

    def test_sticky_provider_removed_from_config_triggers_repick(self):
        """
        Session map references a provider name that no longer exists in MODELS.
        Must re-pick from available providers.
        """
        _set_keys(**{"p-a": "sk-a", "p-b": "sk-b", "p-c": "sk-c"})
        server._session_map[f"ghost-session:{MODEL}"] = "p-ghost"
        with patch.object(server, "_save_sessions"):
            result = server._pick_provider("ghost-session", MODEL)
        assert result["name"] in ("p-a", "p-b", "p-c")
        assert result["_api_key"] is not None
