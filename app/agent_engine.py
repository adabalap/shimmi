from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field
from groq import Groq

from .config import settings
from .retry import async_retry
from .prompts import SYSTEM_PROMPT, REPAIR_PROMPT

logger = logging.getLogger("app.agent")
_NEWLINE = "\n"


class ReplyPayload(BaseModel):
    type: str = Field(..., pattern=r"^(text|buttons|list)$")
    text: str


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_update: dict | None = None


GROQ_CLIENT: Optional[Groq] = None
_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


def _needs_live_search(user_text: str) -> bool:
    t = (user_text or "").lower()
    keys = ["weather", "temperature", "forecast", "rain", "today", "latest", "current", "news", "updates"]
    return any(k in t for k in keys)


async def init_llm() -> None:
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not settings.groq_api_key:
        logger.warning("🧠 llm.init disabled (missing GROQ_API_KEY)")
        return
    GROQ_CLIENT = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
    logger.info("🧠 llm.init enabled")


async def close_llm() -> None:
    return


async def _groq_raw(system: str, user: str, *, model: str, temperature: float, max_tokens: int) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            return (resp.choices[0].message.content or "").strip()

    return await async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0)


async def groq_live_search(user_text: str) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Sorry, AI is unavailable right now."

    payload = {
        "model": getattr(settings, "live_search_model", "groq/compound-mini"),
        "messages": [
            {"role": "system", "content": "Use web search for up-to-date answers. Provide a concise answer."},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 900,
        "temperature": 0.2,
        "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
    }

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            return (resp.choices[0].message.content or "").strip()

    return await async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0)


def _format_context(context: list[dict], limit: int = 10) -> str:
    lines = []
    for c in (context or [])[:limit]:
        meta = c.get("metadata") or {}
        ts = meta.get("ts", "?")
        direction = meta.get("direction", "?")
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"- [{ts}] ({direction}): {txt}")
    return "\n".join(lines) if lines else "(none)"


async def run_agent(*, chat_id: str, user_text: str, facts: Dict[str, str], context: list[dict]) -> AgentResult:
    # ✅ safe getattr so config mismatches never crash worker threads
    if getattr(settings, "live_search_enabled", False) and _needs_live_search(user_text):
        ans = await groq_live_search(user_text)
        return AgentResult(reply=ReplyPayload(type="text", text=ans), memory_update=None)

    model = (settings.groq_model_pool or ["llama-3.3-70b-versatile"])[0]

    bundle = (
        "USER:" + _NEWLINE + user_text + (_NEWLINE * 2) +
        "FACTS:" + _NEWLINE + json.dumps(facts, ensure_ascii=False) + (_NEWLINE * 2) +
        "CONTEXT:" + _NEWLINE + _format_context(context) + _NEWLINE
    )

    raw = await _groq_raw(SYSTEM_PROMPT, bundle, model=model, temperature=0.25, max_tokens=900)

    try:
        data = _extract_json(raw)
    except Exception:
        repaired = await _groq_raw(REPAIR_PROMPT, raw, model=model, temperature=0.0, max_tokens=500)
        data = _extract_json(repaired)

    return AgentResult.model_validate(data)
