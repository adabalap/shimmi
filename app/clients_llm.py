from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional, Dict, Tuple, Any

from groq import Groq

from . import config

logger = logging.getLogger("app.llm")

GROQ_CLIENT: Optional[Groq] = None
MODEL_CIRCUIT: Dict[str, float] = {}
STICKY_MODEL: Dict[str, Tuple[str, float]] = {}
STICKY_TTL_SEC = 600
_circuit_lock = asyncio.Lock()
GROQ_INFLIGHT = asyncio.Semaphore(int(getattr(config, "GROQ_MAX_INFLIGHT", 5) or 5))


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


async def init_clients():
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not config.GROQ_API_KEY:
        logger.warning("🧠 llm.init groq_enabled=0 (missing GROQ_API_KEY)")
        return
    # Groq SDK supports passing api_key explicitly.
    GROQ_CLIENT = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT)
    logger.info("🧠 llm.init groq_enabled=1 models=%s", ",".join(config.GROQ_MODEL_POOL))


async def close_clients():
    # Groq client doesn't require close
    pass


async def groq_chat(chat_id: str, system_text: str, user_text: str, *, temperature: float = 0.3, max_tokens: int = 700) -> tuple[str, bool, dict]:
    if not GROQ_CLIENT or not config.GROQ_API_KEY:
        return "Sorry, AI unavailable.", False, {}

    model = _sticky_or_healthy(chat_id, list(config.GROQ_MODEL_POOL))

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
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            data = resp.model_dump() if hasattr(resp, 'model_dump') else {}
            msg = resp.choices[0].message
            txt = (msg.content or "").strip()
            return txt, True, data
        except Exception as e:
            # best-effort circuit open
            await _open_circuit(model, 12.0 + random.random() * 3.0)
            logger.warning("llm.error model=%s err=%s", model, str(e)[:200])
            return "", False, {}


async def groq_live_search(chat_id: str, user_text: str, *, max_tokens: int = 900) -> tuple[str, bool, dict]:
    """Answer using Groq Compound built-in web_search when enabled."""
    if not GROQ_CLIENT or not config.GROQ_API_KEY:
        return "Sorry, AI unavailable.", False, {}

    model = getattr(config, "LIVE_SEARCH_MODEL", "groq/compound-mini")

    # Enable only web_search tool on Compound systems.
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You may use web search to answer with up-to-date information. Include sources when relevant."},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
    }

    async with GROQ_INFLIGHT:
        try:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            data = resp.model_dump() if hasattr(resp, 'model_dump') else {}
            msg = resp.choices[0].message
            txt = (msg.content or "").strip()
            # executed_tools may exist for compound systems
            try:
                exec_tools = getattr(msg, "executed_tools", None)
                if exec_tools:
                    logger.info("🌐 live.executed_tools n=%s", len(exec_tools))
            except Exception:
                pass
            return txt, True, data
        except Exception as e:
            logger.warning("live_search.error model=%s err=%s", model, str(e)[:200])
            return "", False, {}
