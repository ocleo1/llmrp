"""
Standalone sticky-session LLM reverse proxy.
- Config via config.toml  (Python 3.11+ tomllib, zero extra deps)
- Hash table: (session_id, model) → provider, picked once, sticky forever
- session_id comes from request headers (user-agent + x-session-affinity)
- Persists session map to sessions.json across restarts
- No LiteLLM, no database, no Redis
"""

import hashlib
import json
import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()  # load .env before any os.getenv() calls

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("llmrp")

# ── Load config ───────────────────────────────────────────────────────
CONFIG_PATH = Path(os.getenv("llmrp_CONFIG", "config.toml"))

with open(CONFIG_PATH, "rb") as _f:
    _cfg = tomllib.load(_f)

PROXY_CFG: dict[str, Union[str, int]] = _cfg["proxy"]
REQUEST_TIMEOUT = int(PROXY_CFG.get("request_timeout", 120))
SESSION_STORE  = Path(PROXY_CFG.get("session_store", "sessions.json"))

# One dict comprehension — the tradeoff resolution
# [[models]] list → {model_name: [providers]} dict, O(1) lookup
MODELS: dict[str, list[dict]] = {
    m["name"]: m["providers"]
    for m in _cfg["models"]
}

def _resolve_api_key(raw: str) -> str | None:
    """Resolve 'env:VAR_NAME' → actual value from environment, or None if unset."""
    if raw.startswith("env:"):
        var = raw[4:]
        val = os.getenv(var)
        if not val:
            return None
        return val
    return raw

# Pre-resolve all API keys at startup — providers with missing env vars are skipped at pick time
for _model_name, _providers in MODELS.items():
    for _p in _providers:
        _p["_api_key"] = _resolve_api_key(_p["api_key"])
        if _p["_api_key"] is None:
            log.warning("Provider '%s' for model '%s': env var '%s' is not set — will be skipped",
                        _p["name"], _model_name, _p["api_key"][4:])

log.info("Loaded %d model(s): %s", len(MODELS), list(MODELS.keys()))

# ── Session store ─────────────────────────────────────────────────────
# Key:   "{session_id}:{model_name}"
# Value: provider name (string)
_session_map: dict[str, str] = {}

def _load_sessions() -> None:
    if SESSION_STORE.exists():
        try:
            _session_map.update(json.loads(SESSION_STORE.read_text()))
            log.info("Loaded %d session(s) from %s", len(_session_map), SESSION_STORE)
        except Exception as e:
            log.warning("Could not load session store: %s", e)

def _save_sessions() -> None:
    SESSION_STORE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_STORE.write_text(json.dumps(_session_map, indent=2))

_load_sessions()

# ── Session helpers ───────────────────────────────────────────────────
def _extract_session_id(headers: dict) -> str:
    """Build session_id as '<client>:<x-session-affinity>' from headers."""
    user_agent = headers.get("user-agent", "")
    session_affinity = headers.get("x-session-affinity", "")

    if not session_affinity:
        raise HTTPException(status_code=400, detail="Missing 'x-session-affinity' header")

    match = re.search(r"\b(opencode|kilo-code)(?:/[^\s]+)?\b", user_agent, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail="Missing opencode or kilo-code identifier in 'user-agent' header")

    client = match.group(1).lower()
    return f"{client}:{session_affinity}"

def _pick_provider(session_id: str, model: str) -> dict:
    """
    Hash table lookup:
      - If (session_id, model) already mapped → return same provider (sticky)
      - Else → weighted hash pick → store → return
    """
    store_key = f"{session_id}:{model}"

    if store_key in _session_map:
        provider_name = _session_map[store_key]
        providers = MODELS[model]
        provider = next((p for p in providers if p["name"] == provider_name and p["_api_key"]), None)
        if provider:
            log.debug("Session hit  %s → %s", store_key, provider_name)
            return provider
        # Provider was removed from config or has no API key — re-pick
        log.warning("Provider '%s' no longer available, re-picking", provider_name)

    # Weighted pick: build slots list — only include providers with a resolved API key
    providers = [p for p in MODELS[model] if p["_api_key"]]
    if not providers:
        raise HTTPException(status_code=503, detail=f"No available providers for model '{model}'")
    slots: list[str] = []
    for p in providers:
        slots.extend([p["name"]] * int(p.get("weight", 1)))

    # Hash session_id into slots
    idx = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % len(slots)
    chosen_name = slots[idx]
    provider = next(p for p in providers if p["name"] == chosen_name)

    _session_map[store_key] = chosen_name
    _save_sessions()
    log.info("Session new  %s → %s", store_key, chosen_name)
    return provider

# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(title="llmrp", version="0.1")

@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODELS.keys())}

@app.get("/sessions")
async def list_sessions():
    """Inspect all active session → provider mappings."""
    return {"count": len(_session_map), "sessions": _session_map}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Force a session to re-pick provider on next request."""
    removed = {k: v for k, v in _session_map.items() if k.startswith(session_id)}
    for k in removed:
        del _session_map[k]
    _save_sessions()
    return {"removed": removed}

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """Anthropic /v1/messages endpoint."""
    return await _proxy(request, "/v1/messages")

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    log.debug("Request URL: %s", str(request.url))
    log.debug("Request method: %s", request.method)
    log.debug("Request headers: %s", dict(request.headers))
    log.debug("Request body: %s", await request.body())
    return await _proxy(request, "/chat/completions")

async def _proxy(request: Request, path: str):
    # ── Parse body ────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Known: {list(MODELS.keys())}")

    # ── Sticky provider pick ──────────────────────────────────────────
    session_id = _extract_session_id(request.headers)
    provider   = _pick_provider(session_id, model)

    # ── Forward request ───────────────────────────────────────────────
    target_url = provider["base_url"].rstrip("/") + path
    headers = dict(request.headers)

    # Inject resolved API key, strip hop-by-hop headers
    headers["x-api-key"]     = provider["_api_key"]
    headers["authorization"]  = f"Bearer {provider['_api_key']}"
    headers.pop("host", None)
    headers.pop("content-length", None)

    log.info("→ %s  session=%s  provider=%s  model=%s", path, session_id, provider["name"], model)

    stream = body.get("stream", False)

    try:
        if stream:
            client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
            stream_ctx = client.stream("POST", target_url, headers=headers, json=body)

            try:
                upstream_resp = await stream_ctx.__aenter__()
            except httpx.TimeoutException:
                await client.aclose()
                log.warning("Upstream stream setup timed out: path=%s model=%s provider=%s", path, model, provider["name"])
                raise HTTPException(status_code=504, detail="Upstream request timed out")
            except httpx.RequestError:
                await client.aclose()
                log.exception("Upstream stream setup connection error: path=%s model=%s provider=%s", path, model, provider["name"])
                raise HTTPException(status_code=502, detail="Upstream connection error")

            async def _stream_gen():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                except httpx.TimeoutException:
                    log.warning("Upstream stream timed out during read: path=%s model=%s provider=%s", path, model, provider["name"])
                    yield b'data: {"error":{"message":"Upstream stream timed out"}}\n\n'
                except httpx.RequestError:
                    log.exception("Upstream stream connection error during read: path=%s model=%s provider=%s", path, model, provider["name"])
                    yield b'data: {"error":{"message":"Upstream stream connection error"}}\n\n'
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()

            return StreamingResponse(
                _stream_gen(),
                media_type="text/event-stream",
                status_code=upstream_resp.status_code,
            )

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(target_url, headers=headers, json=body)
        try:
            content = resp.json()
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned non-JSON response (status {resp.status_code})",
            )
        return JSONResponse(status_code=resp.status_code, content=content)
    except HTTPException:
        raise
    except httpx.TimeoutException:
        log.warning("Upstream request timed out: path=%s model=%s provider=%s", path, model, provider["name"])
        raise HTTPException(status_code=504, detail="Upstream request timed out")
    except httpx.RequestError:
        log.exception("Upstream connection error: path=%s model=%s provider=%s", path, model, provider["name"])
        raise HTTPException(status_code=502, detail="Upstream connection error")

# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=PROXY_CFG.get("listen_host", "127.0.0.1"),
        port=int(PROXY_CFG.get("listen_port", 4001)),
    )
