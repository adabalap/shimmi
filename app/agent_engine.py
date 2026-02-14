from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, Optional, List

from pydantic import BaseModel, Field
from groq import Groq

from .config import settings
from .retry import async_retry
from .prompts import SYSTEM_PROMPT, REPAIR_PROMPT

logger = logging.getLogger("app.agent")
_NEWLINE = "\n"


class MemoryKV(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ReplyPayload(BaseModel):
    type: str = Field(..., pattern=r"^(text|buttons|list)$")
    text: str
    buttons: list[dict] | None = None
    list: dict | None = None


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: list[MemoryKV] | None = None


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
            return json.loads(s[start:end + 1])
        raise


def _needs_live_search(user_text: str) -> bool:
    t = (user_text or "").lower()
    keys = [
        "weather", "temperature", "forecast", "rain",
        "today", "latest", "current", "news", "updates",
        "stocks", "stock", "share price", "market",
        "movie", "movies", "trailer", "showtimes", "ipl", "score",
    ]
    return any(k in t for k in keys)


def _facts_block(facts: Dict[str, str]) -> str:
    if not facts:
        return "(none)"
    return json.dumps(facts, ensure_ascii=False, indent=2)


def _context_block(context: list[dict]) -> str:
    lines = []
    for c in (context or [])[:10]:
        meta = c.get("metadata") or {}
        ts = meta.get("ts", "?")
        direction = meta.get("direction", "?")
        txt = (c.get("text") or "").strip()
        if txt:
            lines.append(f"- [{ts}] ({direction}): {txt}")
    return "\n".join(lines) if lines else "(none)"


def render_whatsapp(text: str) -> str:
    """
    Post-process to avoid ugly tables and make output WhatsApp-friendly.
    - Converts obvious table rows (with many '|') into bullets.
    """
    if not text:
        return ""
    s = text.strip()

    # kill markdown tables if they appear
    if "|" in s and re.search(r"\\|\\s*-{2,}\\s*\\|", s):
        # crude conversion: keep non-separator lines and bullet them
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not re.match(r"^\\|?\\s*-{2,}", ln)]
        bullets = []
        for ln in lines:
            ln = ln.strip("|").strip()
            if ln:
                bullets.append(f"- {ln.replace('|', ' • ')}")
        s = "\n".join(bullets)

    # normalize repeated whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


async def init_llm() -> None:
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not settings.groq_api_key:
        logger.warning("🧠 llm.init disabled (missing GROQ_API_KEY)")
        return
    GROQ_CLIENT = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
    logger.info("🧠 llm.init enabled")


async def _groq_raw(system: str, user: str, *, model: str, temperature: float, max_tokens: int, extra: dict | None = None) -> str:
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
    if extra:
        payload.update(extra)

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            return (resp.choices[0].message.content or "").strip()

    return await async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0)


async def groq_live_search(user_text: str, facts: Dict[str, str]) -> str:
    """
    Live search with FACTS context (location/preferences/etc) without hard-coding patterns.
    """
    model = getattr(settings, "live_search_model", "groq/compound-mini")
    facts_txt = _facts_block(facts)

    prompt = (
        "USER QUESTION:\n" + user_text + "\n\n"
        "KNOWN FACTS (use only if relevant):\n" + facts_txt + "\n\n"
        "INSTRUCTIONS:\n"
        "- Use web search for up-to-date results.\n"
        "- If location/identity is required and present in facts, use it.\n"
        "- If missing, ask ONE short follow-up question.\n"
        "- Respond in WhatsApp-friendly bullets. No tables.\n"
    )

    extra = {"compound_custom": {"tools": {"enabled_tools": ["web_search"]}}}
    out = await _groq_raw(
        system="You may use web search. Keep answers concise and WhatsApp-friendly. No tables.",
        user=prompt,
        model=model,
        temperature=0.2,
        max_tokens=900,
        extra=extra,
    )
    return render_whatsapp(out or "Sorry, I couldn't fetch that right now.")


async def run_agent(*, chat_id: str, user_text: str, facts: Dict[str, str], context: list[dict]) -> AgentResult:
    # Live-search path (recommended for weather/news/stocks/movies)
    if getattr(settings, "live_search_enabled", False) and _needs_live_search(user_text):
        ans = await groq_live_search(user_text, facts)
        return AgentResult(reply=ReplyPayload(type="text", text=ans), memory_updates=[])

    # Normal librarian path
    model = (settings.groq_model_pool or ["llama-3.3-70b-versatile"])[0]
    bundle = (
        "USER:\n" + user_text + "\n\n"
        "FACTS:\n" + _facts_block(facts) + "\n\n"
        "CONTEXT:\n" + _context_block(context) + "\n"
    )

    raw = await _groq_raw(SYSTEM_PROMPT, bundle, model=model, temperature=0.25, max_tokens=900)

    try:
        data = _extract_json(raw)
    except Exception:
        repaired = await _groq_raw(REPAIR_PROMPT, raw, model=model, temperature=0.0, max_tokens=500)
        data = _extract_json(repaired)

    # normalize output
    if isinstance(data.get("reply"), dict) and "text" in data["reply"]:
        data["reply"]["text"] = render_whatsapp(data["reply"]["text"])

    # Back-compat: if model returns memory_update single object, wrap it
    if "memory_updates" not in data and "memory_update" in data and isinstance(data["memory_update"], dict):
        data["memory_updates"] = [data["memory_update"]]
        data.pop("memory_update", None)

    # Ensure memory_updates exists
    if "memory_updates" not in data:
        data["memory_updates"] = []

    return AgentResult.model_validate(data)
