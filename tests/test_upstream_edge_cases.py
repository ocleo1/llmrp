"""
Edge-case tests for malformed / unexpected upstream responses.

Covers the non-streaming path in _proxy:
- Non-JSON body from upstream  → 502
- Empty body                   → 502
- Timeout                      → 504
- Connection-level errors       → 502
- Partial / truncated JSON     → 502
- Upstream returns HTML error   → 502
- Upstream returns plain text   → 502
- Large valid JSON response     → 200 forwarded intact
- Upstream returns null JSON    → 200 forwarded intact
- Upstream returns JSON array   → 200 forwarded intact
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import server

client = TestClient(server.app, raise_server_exceptions=False)

HEADERS = {"user-agent": "opencode/1.0", "x-session-affinity": "edge-session"}
BODY    = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


def _give_key():
    for p in server.MODELS["test-model"]:
        p["_api_key"] = "sk-edge" if p["name"] == "p-a" else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resp(status: int, json_body=None, *, json_raises: Exception | None = None):
    """Build a mock httpx response."""
    mock = MagicMock()
    mock.status_code = status
    if json_raises:
        mock.json.side_effect = json_raises
    else:
        mock.json.return_value = json_body
    return mock


def make_post(mock_resp):
    async def _fake(self_client, url, **kw):
        return mock_resp
    return _fake


@contextmanager
def upstream(mock_resp):
    with patch.object(server, "_save_sessions"), \
         patch("httpx.AsyncClient.post", make_post(mock_resp)):
        yield


def make_raiser(exc):
    async def _raise(self_client, url, **kw):
        raise exc
    return _raise


@contextmanager
def upstream_raises(exc):
    with patch.object(server, "_save_sessions"), \
         patch("httpx.AsyncClient.post", make_raiser(exc)):
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
    def __init__(
        self,
        enter_exc: Exception | None = None,
        response: _FakeStreamResp | None = None,
        tracker: dict | None = None,
    ):
        self._enter_exc = enter_exc
        self._response = response or _FakeStreamResp()
        self._tracker = tracker or {}

    async def __aenter__(self):
        self._tracker["entered"] = self._tracker.get("entered", 0) + 1
        if self._enter_exc:
            raise self._enter_exc
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        self._tracker["exited"] = self._tracker.get("exited", 0) + 1
        return False


@contextmanager
def upstream_stream(*, enter_exc: Exception | None = None, status_code=200, chunks=None, read_exc: Exception | None = None):
    tracker: dict[str, int] = {"entered": 0, "exited": 0}

    def _fake_stream(self_client, method, url, **kw):
        return _FakeStreamCtx(
            enter_exc=enter_exc,
            response=_FakeStreamResp(status_code=status_code, chunks=chunks, read_exc=read_exc),
            tracker=tracker,
        )

    with patch.object(server, "_save_sessions"), \
         patch("httpx.AsyncClient.stream", _fake_stream):
        yield tracker


# ---------------------------------------------------------------------------
# Non-JSON upstream body
# ---------------------------------------------------------------------------

class TestNonJsonUpstreamBody:
    def setup_method(self):
        _give_key()
        server._session_map.clear()

    def test_html_error_page_returns_502(self):
        r = _resp(200, json_raises=Exception("<html>Bad Gateway</html>"))
        with upstream(r):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502
        assert "non-JSON" in resp.json()["detail"]

    def test_plain_text_body_returns_502(self):
        r = _resp(503, json_raises=Exception("Service Unavailable"))
        with upstream(r):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502

    def test_502_detail_includes_upstream_status(self):
        r = _resp(418, json_raises=Exception("I'm a teapot"))
        with upstream(r):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502
        assert "418" in resp.json()["detail"]

    def test_empty_body_returns_502(self):
        """Empty response body — json() raises ValueError."""
        r = _resp(204, json_raises=ValueError("No content"))
        with upstream(r):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502

    def test_truncated_json_returns_502(self):
        """Partial JSON e.g. '{"id": "abc"' — json() raises JSONDecodeError."""
        import json as _json
        r = _resp(200, json_raises=_json.JSONDecodeError("Expecting value", "", 0))
        with upstream(r):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502

    def test_non_json_on_openai_endpoint(self):
        """Same behaviour on /v1/chat/completions."""
        r = _resp(200, json_raises=Exception("not json"))
        with upstream(r):
            resp = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Timeout errors
# ---------------------------------------------------------------------------

class TestTimeoutErrors:
    def setup_method(self):
        _give_key()
        server._session_map.clear()

    def test_timeout_exception_returns_504(self):
        with upstream_raises(httpx.TimeoutException("upstream timed out")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]

    def test_read_timeout_returns_504(self):
        with upstream_raises(httpx.ReadTimeout("read timed out")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 504

    def test_connect_timeout_returns_504(self):
        with upstream_raises(httpx.ConnectTimeout("connect timed out")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 504

    def test_pool_timeout_returns_504(self):
        with upstream_raises(httpx.PoolTimeout("pool timed out")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 504

    def test_timeout_exception_on_chat_completions_returns_504(self):
        with upstream_raises(httpx.TimeoutException("upstream timed out")):
            resp = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Connection-level errors
# ---------------------------------------------------------------------------

class TestConnectionErrors:
    def setup_method(self):
        _give_key()
        server._session_map.clear()

    def test_connect_error_returns_502(self):
        with upstream_raises(httpx.ConnectError("connection refused")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502
        assert "connection error" in resp.json()["detail"].lower()

    def test_remote_protocol_error_returns_502(self):
        with upstream_raises(httpx.RemoteProtocolError("peer closed connection")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502

    def test_network_error_returns_502(self):
        with upstream_raises(httpx.NetworkError("network unreachable")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 502

    def test_502_detail_is_sanitized(self):
        with upstream_raises(httpx.ConnectError("ECONNREFUSED 127.0.0.1:443")):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.json()["detail"] == "Upstream connection error"
        assert "ECONNREFUSED" not in resp.json()["detail"]

    def test_connect_error_on_chat_completions_returns_502_sanitized(self):
        with upstream_raises(httpx.ConnectError("ECONNREFUSED 10.10.1.25:443")):
            resp = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Upstream connection error"
        assert "10.10.1.25" not in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Streaming edge cases
# ---------------------------------------------------------------------------

class TestStreamingEdgeCases:
    def setup_method(self):
        _give_key()
        server._session_map.clear()

    def test_stream_success_preserves_status_content_type_and_chunks(self):
        with upstream_stream(status_code=206, chunks=[b"data: one\\n\\n", b"data: two\\n\\n"]):
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 206
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data: one" in resp.text
        assert "data: two" in resp.text

    def test_stream_success_closes_upstream_context_and_client(self):
        with patch("httpx.AsyncClient.aclose", new=AsyncMock()) as aclose_mock:
            with upstream_stream(chunks=[b"data: ok\\n\\n"]) as tracker:
                resp = client.post(
                    "/v1/messages",
                    json={**BODY, "stream": True},
                    headers=HEADERS,
                )
            assert resp.status_code == 200
            _ = resp.text
        assert tracker["entered"] == 1
        assert tracker["exited"] == 1
        assert aclose_mock.await_count == 1

    def test_stream_setup_timeout_returns_504(self):
        with upstream_stream(enter_exc=httpx.TimeoutException("setup timed out")):
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream request timed out"

    def test_stream_setup_timeout_on_chat_completions_returns_504(self):
        with upstream_stream(enter_exc=httpx.TimeoutException("setup timed out")):
            resp = client.post(
                "/v1/chat/completions",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream request timed out"

    def test_stream_setup_timeout_closes_client(self):
        with patch("httpx.AsyncClient.aclose", new=AsyncMock()) as aclose_mock:
            with upstream_stream(enter_exc=httpx.TimeoutException("setup timed out")) as tracker:
                resp = client.post(
                    "/v1/messages",
                    json={**BODY, "stream": True},
                    headers=HEADERS,
                )
        assert resp.status_code == 504
        assert tracker["entered"] == 1
        assert tracker["exited"] == 0
        assert aclose_mock.await_count == 1

    def test_stream_setup_connection_error_returns_502_sanitized(self):
        with upstream_stream(enter_exc=httpx.ConnectError("ECONNREFUSED 10.0.0.8:443")):
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Upstream connection error"
        assert "10.0.0.8" not in resp.json()["detail"]

    def test_stream_setup_connection_error_on_chat_completions_returns_502_sanitized(self):
        with upstream_stream(enter_exc=httpx.ConnectError("ECONNREFUSED 10.9.0.5:443")):
            resp = client.post(
                "/v1/chat/completions",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Upstream connection error"
        assert "10.9.0.5" not in resp.json()["detail"]

    def test_stream_read_timeout_yields_error_chunk_after_partial_data(self):
        with upstream_stream(chunks=[b"data: first\\n\\n"], read_exc=httpx.ReadTimeout("read timed out")) as tracker:
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 200
        text = resp.text
        assert "data: first" in text
        assert "Upstream stream timed out" in text
        assert text.index("data: first") < text.index("Upstream stream timed out")
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert tracker["entered"] == 1
        assert tracker["exited"] == 1

    def test_stream_read_timeout_on_chat_completions_yields_error_chunk(self):
        with upstream_stream(status_code=206, chunks=[b"data: partial\\n\\n"], read_exc=httpx.ReadTimeout("read timed out")):
            resp = client.post(
                "/v1/chat/completions",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 206
        assert "data: partial" in resp.text
        assert "Upstream stream timed out" in resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_stream_read_connection_error_yields_sanitized_error_chunk(self):
        with upstream_stream(read_exc=httpx.ConnectError("ECONNREFUSED 127.0.0.1:443")) as tracker:
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 200
        assert "Upstream stream connection error" in resp.text
        assert "ECONNREFUSED" not in resp.text
        assert tracker["entered"] == 1
        assert tracker["exited"] == 1

    def test_stream_read_connection_error_preserves_partial_data_order(self):
        with upstream_stream(
            chunks=[b"data: first\\n\\n"],
            read_exc=httpx.ConnectError("ECONNREFUSED 127.0.0.1:443"),
        ):
            resp = client.post(
                "/v1/messages",
                json={**BODY, "stream": True},
                headers=HEADERS,
            )
        assert resp.status_code == 200
        text = resp.text
        assert "data: first" in text
        assert "Upstream stream connection error" in text
        assert text.index("data: first") < text.index("Upstream stream connection error")
        assert "ECONNREFUSED" not in text


# ---------------------------------------------------------------------------
# Valid but unusual JSON responses
# ---------------------------------------------------------------------------

class TestValidUnusualResponses:
    def setup_method(self):
        _give_key()
        server._session_map.clear()

    def test_null_json_response_forwarded(self):
        with upstream(_resp(200, json_body=None)):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json() is None

    def test_json_array_response_forwarded(self):
        payload = [{"id": "1"}, {"id": "2"}]
        with upstream(_resp(200, json_body=payload)):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_large_json_response_forwarded_intact(self):
        large = {"items": [{"index": i, "data": "x" * 100} for i in range(500)]}
        with upstream(_resp(200, json_body=large)):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 500

    def test_upstream_status_preserved_on_valid_json(self):
        """A 206 Partial Content with valid JSON body — status forwarded as-is."""
        with upstream(_resp(206, json_body={"partial": True})):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 206

    def test_error_status_with_valid_json_forwarded(self):
        """Upstream 422 with JSON error body — status and body both forwarded."""
        body = {"error": {"type": "invalid_request", "message": "bad param"}}
        with upstream(_resp(422, json_body=body)):
            resp = client.post("/v1/messages", json=BODY, headers=HEADERS)
        assert resp.status_code == 422
        assert resp.json()["error"]["type"] == "invalid_request"
