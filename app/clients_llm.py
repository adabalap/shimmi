from __future__ import annotations

import httpx, time, asyncio, random, logging
from typing import Optional
from . import config
from .db import inc_usage

logger = logging.getLogger("app.llm")

HTTPX_GROQ: Optional[httpx.AsyncClient] = None

MODEL_CIRCUIT: dict[str, float] = {}
STICKY_MODEL: dict[str, tuple[str, float]] = {}
STICKY_TTL_SEC = 600

_circuit_lock = asyncio.Lock()
GROQ_INFLIGHT = asyncio.Semaphore(int(getattr(config, "GROQ_MAX_INFLIGHT", 5) or 5))


async def init_clients():
    global HTTPX_GROQ
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=30)
    if config.GROQ_API_KEY:
        HTTPX_GROQ = httpx.AsyncClient(
            timeout=config.GROQ_TIMEOUT,
            limits=limits,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        )
    else:
        HTTPX_GROQ = None


async def close_clients():
    try:
        if HTTPX_GROQ:
            await HTTPX_GROQ.aclose()
    except Exception:
        pass


def _model_open(model: str) -> bool:
    return time.monotonic() >= MODEL_CIRCUIT.get(model, 0.0)


async def _open_circuit(model: str, cooldown: float):
    async with _circuit_lock:
        MODEL_CIRCUIT[model] = time.monotonic() + cooldown


def _sticky_or_healthy(chat_id: str, pool: list[str]) -> str:
    key = f"groq:{chat_id}"
    sticky = STICKY_MODEL.get(key)
    if sticky and (time.monotonic() - sticky[1]) < STICKY_TTL_SEC and _model_open(sticky[0]):
        return sticky[0]
    for m in pool:
        if _model_open(m):
            STICKY_MODEL[key] = (m, time.monotonic())
            return m
    STICKY_MODEL[key] = (pool[0], time.monotonic())
    return pool[0]


async def groq_chat(chat_id: str, system_text: str, user_text: str, *, temperature: float = 0.6, max_tokens: int = 800) -> tuple[str, bool, dict]:
    if not HTTPX_GROQ or not config.GROQ_API_KEY:
        return "Sorry, AI unavailable.", False, {}

    model = _sticky_or_healthy(chat_id, config.GROQ_MODEL_POOL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": float(max(0.0, min(temperature, 1.5))),
        "max_tokens": int(max_tokens),
    }

    async with GROQ_INFLIGHT:
        try:
            resp = await HTTPX_GROQ.post("https://api.groq.com/openai/v1/chat/completions", json=payload)

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                cooldown = 30.0
                if retry_after:
                    try:
                        cooldown = max(cooldown, float(retry_after))
                    except Exception:
                        pass
                cooldown += random.random() * 2.0
                await _open_circuit(model, cooldown)
                return "", False, {}

            if resp.status_code in (500, 502, 503):
                await _open_circuit(model, 60.0 + random.random() * 5.0)
                return "", False, {}

            resp.raise_for_status()
            data = resp.json()
            msg = (data.get('choices') or [{}])[0].get('message') or {}
            txt = (msg.get('content') or '').strip()
            usage = data.get('usage') or {}
            await inc_usage(model, usage.get('total_tokens') or 0)
            return txt, True, data

        except httpx.RequestError as e:
            logger.warning("groq.request_error model=%s err=%s", model, str(e)[:200])
            await _open_circuit(model, 15.0)
            return "", False, {}
        except Exception as e:
            logger.warning("groq.error model=%s err=%s", model, str(e)[:200])
            await _open_circuit(model, 15.0)
            return "", False, {}
