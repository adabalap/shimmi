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
from .prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT,
    REPAIR_PROMPT,
    FORMATTER_PROMPT,
    LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp

logger = logging.getLogger("app.agent")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


class PlannerResult(BaseModel):
    mode: str
    requires_locale: bool = False
    missing_facts: List[str] = Field(default_factory=list)
    question: str = ""
    search_query: str = ""


class ApprovedUpdate(BaseModel):
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class VerifyResult(BaseModel):
    approved: List[ApprovedUpdate] = Field(default_factory=list)


class ExtractResult(BaseModel):
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


class FormatterResult(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Client & circuit-breaker state
# ---------------------------------------------------------------------------

GROQ_CLIENT: Optional[Groq] = None
_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))
MODEL_CIRCUIT: Dict[str, float] = {}   # model -> open-until timestamp
STICKY_MODEL: Dict[str, str] = {}      # chat_id -> preferred model


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
        logger.warning("llm.init disabled (missing GROQ_API_KEY)")
        return
    GROQ_CLIENT = Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
    logger.info("llm.init enabled")


async def close_llm() -> None:
    return


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty")
    if s.startswith("{"):
        return json.loads(s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError("no_json")


# ---------------------------------------------------------------------------
# Core LLM helpers
# ---------------------------------------------------------------------------

async def _groq_raw(
    chat_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Convenience wrapper: single system + single user turn."""
    return await _groq_messages(
        chat_id,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def _groq_messages(
    chat_id: str,
    *,
    system: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    """
    FIX #9 enabler: accepts a full messages array so callers can pass
    ordered conversation history for proper multi-turn context.
    """
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    model = _pick_model(chat_id)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(
                lambda: GROQ_CLIENT.chat.completions.create(**payload)
            )
            return (resp.choices[0].message.content or "").strip()

    try:
        return await async_retry(_call, max_attempts=4, base_delay=0.6, max_delay=8.0)
    except Exception:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        raise


# ---------------------------------------------------------------------------
# WhatsApp formatter  (FIX #5: returns raw text; send_text does final sanitize)
# ---------------------------------------------------------------------------

async def _format_whatsapp(chat_id: str, text: str) -> str:
    raw = await _groq_raw(
        chat_id, FORMATTER_PROMPT, text, temperature=0.0, max_tokens=350
    )
    try:
        data = _extract_json(raw)
        fr = FormatterResult.model_validate(data)
        return fr.text            # send_text will sanitize once at send time
    except Exception:
        return text               # fall back; send_text still sanitizes


# ---------------------------------------------------------------------------
# Live search
# ---------------------------------------------------------------------------

async def groq_live_search(
    chat_id: str, query: str, facts: Dict[str, str]
) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Sorry, AI is unavailable right now."

    user_msg = json.dumps({"query": query, "facts": facts}, ensure_ascii=False)
    payload = {
        # FIX #7: "groq/compound-mini" is not a valid Groq SDK model ID.
        # Correct IDs are "compound-beta" or "compound-beta-mini".
        "model": settings.live_search_model,
        "messages": [
            {"role": "system", "content": LIVE_SEARCH_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 900,
        "temperature": 0.2,
        "compound_custom": {"tools": {"enabled_tools": ["web_search"]}},
    }

    async def _call() -> str:
        async with _inflight:
            resp = await asyncio.to_thread(
                lambda: GROQ_CLIENT.chat.completions.create(**payload)
            )
            return (resp.choices[0].message.content or "").strip()

    return await async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

async def _plan(
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
) -> PlannerResult:
    payload = {"user_message": user_text, "facts": facts, "context": context}
    raw = await _groq_raw(
        chat_id,
        PLANNER_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        temperature=0.0,
        max_tokens=520,
    )
    try:
        data = _extract_json(raw)
        return PlannerResult.model_validate(data)
    except Exception:
        return PlannerResult(mode="answer")


# ---------------------------------------------------------------------------
# Memory extractor
# ---------------------------------------------------------------------------

async def _extract_memory(chat_id: str, user_text: str) -> List[MemoryUpdate]:
    raw = await _groq_raw(
        chat_id, MEMORY_EXTRACTOR_PROMPT, user_text, temperature=0.0, max_tokens=450
    )
    try:
        data = _extract_json(raw)
        er = ExtractResult.model_validate(data)
        return er.memory_updates
    except Exception as e:
        if settings.debug_agent:
            logger.info(
                "memory.extract_failed err=%s raw=%s",
                str(e)[:120],
                (raw or "")[:260],
            )
        return []


# ---------------------------------------------------------------------------
# Memory verifier  — FIX #2: called ONCE at the end, over ALL proposed updates
# ---------------------------------------------------------------------------

async def _verify_updates(
    chat_id: str, user_text: str, proposed: List[MemoryUpdate]
) -> List[MemoryUpdate]:
    if not proposed:
        return []
    if not settings.facts_verification:
        return proposed

    payload = {
        "user_message": user_text,
        "proposed_memory_updates": [u.model_dump() for u in proposed],
    }
    raw = await _groq_raw(
        chat_id,
        VERIFIER_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        temperature=0.0,
        max_tokens=450,
    )
    try:
        data = _extract_json(raw)
        vr = VerifyResult.model_validate(data)
    except Exception as e:
        if settings.debug_agent:
            logger.info(
                "memory.verify_failed err=%s raw=%s",
                str(e)[:120],
                (raw or "")[:260],
            )
        return []

    min_conf = float(settings.facts_min_conf or 0.85)
    return [
        MemoryUpdate(key=str(a.key).strip(), value=str(a.value).strip())
        for a in vr.approved
        if a.confidence >= min_conf
    ]


def _locale_present(facts: Dict[str, str]) -> bool:
    for k in ("city", "country", "postal_code", "locale"):
        if (facts.get(k) or "").strip():
            return True
    return False


def _deduplicate_updates(updates: List[MemoryUpdate]) -> List[MemoryUpdate]:
    """Last-write-wins by key (preserves insertion order of first occurrence)."""
    seen: Dict[str, MemoryUpdate] = {}
    for u in updates:
        seen[u.key] = u   # later entry overrides for the same key
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

async def run_agent(
    *,
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> AgentResult:
    """Run the full agent pipeline for one incoming message.

    Changes vs v2.2:
    - FIX #3: _extract_memory and _plan run in parallel (asyncio.gather).
    - FIX #2: _verify_updates called exactly once, over the union of
              extractor-proposals + LLM-proposals.
    - FIX #9: conversation history passed as ordered turn messages.
    """
    history = history or []

    # FIX #3 — parallel independent LLM calls
    proposed_updates, pr = await asyncio.gather(
        _extract_memory(chat_id, user_text),
        _plan(chat_id, user_text, facts, context),
    )

    logger.info("memory.extracted count=%s", len(proposed_updates))
    logger.info("planner mode=%s requires_locale=%s", pr.mode, pr.requires_locale)

    # --- Locale gate --------------------------------------------------------
    if pr.requires_locale and not _locale_present(facts):
        q = pr.question or "What city and country should I use?"
        verified = await _verify_updates(chat_id, user_text, proposed_updates)
        return AgentResult(
            reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(q)),
            memory_updates=verified,
        )

    # --- Ask-for-facts mode -------------------------------------------------
    if pr.mode == "ask_facts" and pr.question:
        verified = await _verify_updates(chat_id, user_text, proposed_updates)
        return AgentResult(
            reply=ReplyPayload(
                type="text", text=sanitize_for_whatsapp(pr.question)
            ),
            memory_updates=verified,
        )

    # --- Live-search mode ---------------------------------------------------
    if pr.mode == "live_search" and settings.live_search_enabled:
        if pr.missing_facts:
            q = pr.question or "What city and country should I use?"
            verified = await _verify_updates(chat_id, user_text, proposed_updates)
            return AgentResult(
                reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(q)),
                memory_updates=verified,
            )

        query = pr.search_query.strip() if pr.search_query else user_text
        ans = await groq_live_search(chat_id, query, facts)
        ans = await _format_whatsapp(chat_id, ans)

        # FIX #2 — single verify
        verified = await _verify_updates(chat_id, user_text, proposed_updates)
        return AgentResult(
            reply=ReplyPayload(type="text", text=ans),
            memory_updates=verified,
        )

    # --- Standard answer mode — FIX #9: inject conversation history ---------
    bundle = {"user": user_text, "facts": facts, "context": context}
    turn_messages: List[Dict[str, str]] = list(history) + [
        {"role": "user", "content": json.dumps(bundle, ensure_ascii=False)}
    ]

    raw = await _groq_messages(
        chat_id,
        system=SYSTEM_PROMPT,
        messages=turn_messages,
        temperature=0.25,
        max_tokens=900,
    )

    try:
        data = _extract_json(raw)
    except Exception:
        repaired = await _groq_raw(
            chat_id, REPAIR_PROMPT, raw, temperature=0.0, max_tokens=550
        )
        data = _extract_json(repaired)

    result = AgentResult.model_validate(data)
    result.reply.text = await _format_whatsapp(chat_id, result.reply.text)

    # FIX #2 — single verify over union of extractor + LLM proposals
    all_proposed = _deduplicate_updates(proposed_updates + result.memory_updates)
    result.memory_updates = await _verify_updates(chat_id, user_text, all_proposed)

    return result
