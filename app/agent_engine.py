from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from groq import Groq

from .config import settings
from .retry import async_retry
from .prompts import SYSTEM_PROMPT, REPAIR_PROMPT, VERIFIER_PROMPT
from .utils import sanitize_for_whatsapp

logger = logging.getLogger("app.agent")
_NEWLINE = chr(10)


class MemoryUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


class ApprovedUpdate(BaseModel):
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class VerifyResult(BaseModel):
    approved: List[ApprovedUpdate] = Field(default_factory=list)


GROQ_CLIENT: Optional[Groq] = None
_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))
MODEL_CIRCUIT: Dict[str, float] = {}
STICKY_MODEL: Dict[str, str] = {}
STICKY_TTL_SEC = 600


def _model_open(model: str) -> bool:
    return time.monotonic() >= MODEL_CIRCUIT.get(model, 0.0)


def _pick_model(chat_id: str) -> str:
    pool = list(settings.groq_model_pool or [])
    if not pool:
        return "llama-3.3-70b-versatile"
    sticky = STICKY_MODEL.get(chat_id)
    if sticky and _model_open(sticky):
        return sticky
    for m in pool:
        if _model_open(m):
            STICKY_MODEL[chat_id] = m
            return m
    STICKY_MODEL[chat_id] = pool[0]
    return pool[0]


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


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    return json.loads(s) if s.startswith("{") else json.loads(s[s.find("{"):s.rfind("}")+1])


def _needs_live_search(user_text: str) -> bool:
    t = (user_text or "").lower()
    keys = ["weather", "temperature", "forecast", "rain", "today", "latest", "current", "news", "update", "stock", "stocks", "price", "movie", "movies"]
    return any(k in t for k in keys)


def _enrich_query(user_text: str, facts: Dict[str, str]) -> str:
    t = (user_text or "").strip()
    low = t.lower()

    # Enrich with location/preferences if missing
    city = facts.get("city") or facts.get("location_city") or facts.get("home_city")
    country = facts.get("country")
    topics = facts.get("news_topics") or facts.get("interests")
    stocks = facts.get("watchlist") or facts.get("stock_watchlist")

    if any(k in low for k in ["weather", "forecast", "temperature", "rain"]) and city and (city.lower() not in low):
        t = t + f" in {city}"
        if country and country.lower() not in low:
            t = t + f", {country}"

    if "news" in low and topics:
        t = t + f" about {topics}"

    if any(k in low for k in ["stock", "stocks", "price"]) and stocks:
        t = t + f" for {stocks}"

    return t


async def groq_live_search(chat_id: str, query: str) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Sorry, AI is unavailable right now."

    payload = {
        "model": settings.live_search_model,
        "messages": [
            {"role": "system", "content": "Answer for WhatsApp: short, bullets, no tables, no code blocks."},
            {"role": "user", "content": query},
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


async def _groq_raw(chat_id: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    model = _pick_model(chat_id)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            return (resp.choices[0].message.content or "").strip()

    try:
        return await async_retry(_call, max_attempts=4, base_delay=0.6, max_delay=8.0)
    except Exception:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        raise


async def _verify_updates(chat_id: str, user_text: str, proposed: List[MemoryUpdate]) -> List[MemoryUpdate]:
    if not proposed:
        return []
    if not settings.facts_verification:
        return proposed

    payload = {
        "user_message": user_text,
        "proposed": [u.model_dump() for u in proposed],
    }
    verifier_user = json.dumps(payload, ensure_ascii=False)

    raw = await _groq_raw(chat_id, VERIFIER_PROMPT, verifier_user, temperature=0.0, max_tokens=450)
    try:
        data = _extract_json(raw)
        vr = VerifyResult.model_validate(data)
    except Exception:
        return []

    keep: List[MemoryUpdate] = []
    for a in vr.approved:
        if a.confidence >= float(settings.facts_min_conf or 0.85):
            keep.append(MemoryUpdate(key=str(a.key).strip(), value=str(a.value).strip()))
    return keep


async def run_agent(*, chat_id: str, user_text: str, facts: Dict[str, str], context: List[Dict[str, Any]]) -> AgentResult:
    # Live search path
    if settings.live_search_enabled and _needs_live_search(user_text):
        q = _enrich_query(user_text, facts)
        ans = await groq_live_search(chat_id, q)
        return AgentResult(reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(ans)), memory_updates=[])

    # Grounded path
    bundle = {
        "user": user_text,
        "facts": facts,
        "context": [
            {
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}),
            }
            for c in (context or [])
        ],
    }

    raw = await _groq_raw(chat_id, SYSTEM_PROMPT, json.dumps(bundle, ensure_ascii=False), temperature=0.25, max_tokens=900)
    if settings.debug_agent:
        logger.info("llm.raw_preview %s", raw[:320])

    try:
        data = _extract_json(raw)
    except Exception:
        repaired = await _groq_raw(chat_id, REPAIR_PROMPT, raw, temperature=0.0, max_tokens=550)
        data = _extract_json(repaired)

    result = AgentResult.model_validate(data)
    # whatsapp formatting safety
    result.reply.text = sanitize_for_whatsapp(result.reply.text)

    # verify + keep only deterministic supported updates
    result.memory_updates = await _verify_updates(chat_id, user_text, result.memory_updates)

    return result
