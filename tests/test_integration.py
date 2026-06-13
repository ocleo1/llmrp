"""
Integration tests for HTTP error handling in llmrp.

These tests use FastAPI's TestClient to drive the full request pipeline:
    client request → header validation → body parsing → model lookup
    → provider pick → upstream proxy (mocked) → response

conftest.py already imports server with test_config.toml, so MODELS contains
one model "test-model" with providers p-a / p-b / p-c.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import server

# ---------------------------------------------------------------------------
# Shared test client & helpers
# ---------------------------------------------------------------------------

client = TestClient(server.app, raise_server_exceptions=False)

# Standard valid headers — accepted by _extract_session_id
VALID_HEADERS = {
    "user-agent": "opencode/1.0",
    "x-session-affinity": "test-session-abc",
}

VALID_ANTHROPIC_BODY = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
VALID_OPENAI_BODY    = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


def _give_p_a_key():
    """Give p-a a valid key so _pick_provider won't 503."""
    for p in server.MODELS["test-model"]:
        p["_api_key"] = "sk-test" if p["name"] == "p-a" else None


def _give_all_keys():
    for p in server.MODELS["test-model"]:
        p["_api_key"] = f"sk-{p['name']}"


def _clear_all_keys():
    for p in server.MODELS["test-model"]:
        p["_api_key"] = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_and_model_list(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "test-model" in data["models"]


# ---------------------------------------------------------------------------
# /sessions
# ---------------------------------------------------------------------------

class TestSessions:
    def test_empty_sessions(self):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_delete_session_removes_entries(self):
        server._session_map["opencode:abc:test-model"] = "p-a"
        server._session_map["opencode:abc:other-model"] = "p-b"
        with patch.object(server, "_save_sessions"):
            resp = client.delete("/sessions/opencode:abc")
        assert resp.status_code == 200
        removed = resp.json()["removed"]
        assert "opencode:abc:test-model" in removed
        assert "opencode:abc:other-model" in removed
        assert "opencode:abc:test-model" not in server._session_map

    def test_delete_session_unknown_id_returns_empty(self):
        with patch.object(server, "_save_sessions"):
            resp = client.delete("/sessions/no-such-session")
        assert resp.status_code == 200
        assert resp.json()["removed"] == {}


# ---------------------------------------------------------------------------
# Header validation errors — _extract_session_id
# ---------------------------------------------------------------------------

class TestHeaderValidation:
    def test_missing_x_session_affinity_returns_400(self):
        resp = client.post(
            "/v1/messages",
            json=VALID_ANTHROPIC_BODY,
            headers={"user-agent": "opencode/1.0"},  # no x-session-affinity
        )
        assert resp.status_code == 400
        assert "x-session-affinity" in resp.json()["detail"]

    def test_missing_user_agent_client_returns_400(self):
        resp = client.post(
            "/v1/messages",
            json=VALID_ANTHROPIC_BODY,
            headers={
                "user-agent": "curl/7.0",           # no opencode/kilo-code
                "x-session-affinity": "s1",
            },
        )
        assert resp.status_code == 400
        assert "user-agent" in resp.json()["detail"]

    def test_kilo_code_user_agent_accepted(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream(200, {"id": "ok"}):
            resp = client.post(
                "/v1/messages",
                json=VALID_ANTHROPIC_BODY,
                headers={
                    "user-agent": "kilo-code/2.3",
                    "x-session-affinity": "kc-session",
                },
            )
        assert resp.status_code == 200

    def test_opencode_in_longer_user_agent_accepted(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream(200, {"id": "ok"}):
            resp = client.post(
                "/v1/messages",
                json=VALID_ANTHROPIC_BODY,
                headers={
                    "user-agent": "Mozilla/5.0 opencode/3.1 compatible",
                    "x-session-affinity": "oc-session",
                },
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Body / model validation errors — _proxy
# ---------------------------------------------------------------------------

class TestBodyValidation:
    def test_invalid_json_returns_400(self):
        resp = client.post(
            "/v1/messages",
            content=b"not-json",
            headers={**VALID_HEADERS, "content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "JSON" in resp.json()["detail"]

    def test_missing_model_field_returns_400(self):
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"]

    def test_unknown_model_returns_400(self):
        resp = client.post(
            "/v1/messages",
            json={"model": "gpt-9000", "messages": []},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 400
        assert "gpt-9000" in resp.json()["detail"]

    def test_unknown_model_lists_known_models(self):
        resp = client.post(
            "/v1/messages",
            json={"model": "unknown-model", "messages": []},
            headers=VALID_HEADERS,
        )
        assert "test-model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Provider unavailability — all keys missing → 503
# ---------------------------------------------------------------------------

class TestProviderUnavailable:
    def test_all_keys_missing_returns_503_anthropic(self):
        _clear_all_keys()
        resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 503
        assert "test-model" in resp.json()["detail"]

    def test_all_keys_missing_returns_503_openai(self):
        _clear_all_keys()
        resp = client.post("/v1/chat/completions", json=VALID_OPENAI_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Upstream error propagation
# ---------------------------------------------------------------------------

class TestUpstreamErrors:
    def test_upstream_401_propagated(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream(401, {"error": "invalid key"}):
            resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 401

    def test_upstream_429_propagated(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream(429, {"error": "rate limited"}):
            resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 429

    def test_upstream_500_propagated(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream(500, {"error": "server error"}):
            resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 500

    def test_upstream_200_success(self):
        _give_p_a_key()
        upstream_body = {"id": "msg_123", "content": [{"text": "hello"}]}
        with patch.object(server, "_save_sessions"), _mock_upstream(200, upstream_body):
            resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["id"] == "msg_123"

    def test_upstream_network_error_raises_502(self):
        """httpx.RequestError (e.g. connection refused) should surface as 502 Bad Gateway."""
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_error(server.httpx.ConnectError("refused")):
            resp = client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Stream timeout scenarios
# ---------------------------------------------------------------------------

class TestStreamTimeoutScenarios:
    def test_stream_setup_timeout_returns_504_messages(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(enter_exc=server.httpx.TimeoutException("setup timeout")):
            resp = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream request timed out"

    def test_stream_setup_timeout_returns_504_chat_completions(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(enter_exc=server.httpx.TimeoutException("setup timeout")):
            resp = client.post(
                "/v1/chat/completions",
                json={**VALID_OPENAI_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream request timed out"

    def test_stream_read_timeout_returns_error_chunk_without_500(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            status_code=200,
            chunks=[b"data: first\\n\\n"],
            read_exc=server.httpx.ReadTimeout("read timeout"),
        ):
            resp = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert resp.status_code == 200
        assert "data: first" in resp.text
        assert "Upstream stream timed out" in resp.text


# ---------------------------------------------------------------------------
# Stream error recovery scenarios
# ---------------------------------------------------------------------------

class TestStreamErrorRecovery:
    def test_recover_after_stream_setup_timeout(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            enter_exc=server.httpx.TimeoutException("setup timeout"),
        ):
            first = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert first.status_code == 504

        with patch.object(server, "_save_sessions"), _mock_upstream(200, {"id": "recovered"}):
            second = client.post(
                "/v1/messages",
                json=VALID_ANTHROPIC_BODY,
                headers=VALID_HEADERS,
            )
        assert second.status_code == 200
        assert second.json()["id"] == "recovered"
        assert server._session_map.get("opencode:test-session-abc:test-model") == "p-a"

    def test_recover_after_stream_setup_connection_error(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            enter_exc=server.httpx.ConnectError("ECONNREFUSED 10.1.2.3:443"),
        ):
            first = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert first.status_code == 502
        assert first.json()["detail"] == "Upstream connection error"

        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            status_code=200,
            chunks=[b"data: recovered\\n\\n"],
        ):
            second = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert second.status_code == 200
        assert "data: recovered" in second.text

    def test_recover_after_stream_read_timeout(self):
        _give_p_a_key()
        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            status_code=200,
            chunks=[b"data: first\\n\\n"],
            read_exc=server.httpx.ReadTimeout("read timeout"),
        ):
            first = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert first.status_code == 200
        assert "Upstream stream timed out" in first.text

        with patch.object(server, "_save_sessions"), _mock_upstream_stream(
            status_code=206,
            chunks=[b"data: healthy\\n\\n"],
        ):
            second = client.post(
                "/v1/messages",
                json={**VALID_ANTHROPIC_BODY, "stream": True},
                headers=VALID_HEADERS,
            )
        assert second.status_code == 206
        assert "data: healthy" in second.text
        assert "Upstream stream timed out" not in second.text


# ---------------------------------------------------------------------------
# API key injection into forwarded headers
# ---------------------------------------------------------------------------

class TestApiKeyInjection:
    def test_resolved_key_injected_into_forwarded_request(self):
        """The provider's _api_key must appear in x-api-key and Authorization headers."""
        _give_all_keys()
        captured_headers: dict = {}

        async def _fake_post(self_client, url, *, headers, json, **kw):
            captured_headers.update(headers)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"id": "captured"}
            return mock_resp

        with patch.object(server, "_save_sessions"), \
             patch("httpx.AsyncClient.post", _fake_post):
            client.post("/v1/messages", json=VALID_ANTHROPIC_BODY, headers=VALID_HEADERS)

        assert captured_headers.get("x-api-key", "").startswith("sk-")
        assert captured_headers.get("authorization", "").startswith("Bearer sk-")

    def test_original_client_auth_header_overwritten(self):
        """Client-supplied Authorization must be replaced with the provider key."""
        _give_all_keys()
        captured_headers: dict = {}

        async def _fake_post(self_client, url, *, headers, json, **kw):
            captured_headers.update(headers)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            return mock_resp

        with patch.object(server, "_save_sessions"), \
             patch("httpx.AsyncClient.post", _fake_post):
            client.post(
                "/v1/messages",
                json=VALID_ANTHROPIC_BODY,
                headers={**VALID_HEADERS, "authorization": "Bearer client-own-key"},
            )

        assert "client-own-key" not in captured_headers.get("authorization", "")


# ---------------------------------------------------------------------------
# Context manager helpers for mocking the upstream httpx call
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _mock_upstream(status_code: int, body: dict):
    """Patch httpx.AsyncClient.post to return a fake response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body

    async def _fake_post(self_client, url, *, headers, json, **kw):
        return mock_resp

    with patch("httpx.AsyncClient.post", _fake_post):
        yield


@contextmanager
def _mock_upstream_error(exc: Exception):
    """Patch httpx.AsyncClient.post to raise an exception."""
    async def _fake_post(self_client, url, *, headers, json, **kw):
        raise exc

    with patch("httpx.AsyncClient.post", _fake_post):
        yield


class _FakeStreamResp:
    def __init__(self, status_code=200, chunks=None, read_exc: Exception | None = None):
        self.status_code = status_code
        self._chunks = chunks or []
        self._read_exc = read_exc

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk
        if self._read_exc:
            raise self._read_exc


class _FakeStreamCtx:
    def __init__(self, enter_exc: Exception | None = None, response: _FakeStreamResp | None = None):
        self._enter_exc = enter_exc
        self._response = response or _FakeStreamResp()

    async def __aenter__(self):
        if self._enter_exc:
            raise self._enter_exc
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


@contextmanager
def _mock_upstream_stream(*, enter_exc: Exception | None = None, status_code=200, chunks=None, read_exc: Exception | None = None):
    def _fake_stream(self_client, method, url, **kw):
        return _FakeStreamCtx(
            enter_exc=enter_exc,
            response=_FakeStreamResp(status_code=status_code, chunks=chunks, read_exc=read_exc),
        )

    with patch("httpx.AsyncClient.stream", _fake_stream):
        yield
