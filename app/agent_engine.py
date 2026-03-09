"""
agent_engine.py — Shimmi v2.8.0

Fixes vs v2.7.0:
  ① OrchestratorResult Pydantic crash — `query`/`question` returned as null
    by LLM → Pydantic v2 rejects null for str fields even with defaults.
    Fixed: Optional[str] + field_validator coercing None→"" before validation.
  ② Time context injected into every orchestrator call (current_time,
    time_of_day, greeting) so greetings match actual time of day.
  ③ Reminders field in OrchestratorResult — structured trigger data for the
    background scheduler, separate from the human-readable memory note.
  ④ orchestrate_2+ uses the largest available model (if pool has one) to
    handle large synthesis contexts faster and more accurately.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from groq import AsyncGroq

from .config import settings
from .retry import async_retry
from .prompts import (
    ORCHESTRATOR_PROMPT, MEMORY_EXTRACTOR_PROMPT, VERIFIER_PROMPT,
    REPAIR_PROMPT, FORMATTER_PROMPT, LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp
from .database import normalize_key

logger = logging.getLogger("app.agent")
UTC    = timezone.utc

_SLOW_CALL_WARN_SEC = 5.0
_MAX_ITERATIONS     = 3

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

_CITY_TZ: Dict[str, str] = {
    "hyderabad": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata",
    "chennai": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "pune": "Asia/Kolkata", "ahmedabad": "Asia/Kolkata",
    "jaipur": "Asia/Kolkata", "surat": "Asia/Kolkata",
    "lucknow": "Asia/Kolkata", "kanpur": "Asia/Kolkata",
    "nagpur": "Asia/Kolkata", "indore": "Asia/Kolkata",
    "bhopal": "Asia/Kolkata", "visakhapatnam": "Asia/Kolkata",
    "vizag": "Asia/Kolkata", "patna": "Asia/Kolkata",
    "dubai": "Asia/Dubai", "abu dhabi": "Asia/Dubai",
    "london": "Europe/London", "new york": "America/New_York",
    "singapore": "Asia/Singapore", "sydney": "Australia/Sydney",
}


def _city_to_tz(city: str) -> str:
    return _CITY_TZ.get((city or "").lower().strip(), "UTC")


def _time_context(city: str = "") -> Dict[str, str]:
    """Return a dict of time context fields to inject into the orchestrator."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_city_to_tz(city))
    except Exception:
        tz = UTC
    now = datetime.now(tz)
    h   = now.hour
    if   5 <= h < 12: tod, emoji = "morning",   "☀️"
    elif 12 <= h < 17: tod, emoji = "afternoon", "🌤️"
    elif 17 <= h < 21: tod, emoji = "evening",   "🌆"
    else:              tod, emoji = "night",      "🌙"
    return {
        "current_time": now.strftime("%H:%M ") + now.strftime("%Z").replace("UTC+0530", "IST"),
        "time_of_day":  tod,
        "time_emoji":   emoji,
        "greeting":     f"Good {tod}",
        "day_name":     now.strftime("%A"),
        "date":         now.strftime("%Y-%m-%d"),
        "tz_offset":    now.strftime("%z"),
    }


def _today_str(city: str = "") -> str:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_city_to_tz(city))
    except Exception:
        tz = UTC
    return datetime.now(tz).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    key:   str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ReminderEntry(BaseModel):
    text:        str = ""
    trigger_iso: Optional[str] = None

    @field_validator('text', mode='before')
    @classmethod
    def _coerce_text(cls, v) -> str:
        return str(v) if v else ""


class ReplyPayload(BaseModel):
    type: str = "text"
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply:          ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)
    iterations:     int = 1


class OrchestratorResult(BaseModel):
    action:         str
    reasoning:      str = ""
    text:           str = ""
    query:          Optional[str] = None
    question:       Optional[str] = None
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)

    @field_validator('reasoning', 'text', mode='before')
    @classmethod
    def _str_none(cls, v) -> str:
        """Coerce null/None to empty string — LLMs sometimes return null."""
        return str(v) if v is not None else ""

    @field_validator('query', 'question', mode='before')
    @classmethod
    def _opt_str_none(cls, v) -> Optional[str]:
        """Keep None as None; coerce other falsy to None."""
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator('memory_updates', 'reminders', mode='before')
    @classmethod
    def _list_none(cls, v):
        return v if v is not None else []


class ApprovedUpdate(BaseModel):
    key:        str
    value:      str
    confidence: float = Field(ge=0.0, le=1.0)


class VerifyResult(BaseModel):
    approved: List[ApprovedUpdate] = Field(default_factory=list)


class ExtractResult(BaseModel):
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


class FormatterResult(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# LLM client & circuit breaker
# ---------------------------------------------------------------------------

GROQ_CLIENT: Optional[AsyncGroq] = None
_inflight    = asyncio.Semaphore(int(settings.groq_max_inflight or 5))

MODEL_CIRCUIT: Dict[str, float] = {}
_STICKY_MAX   = 2_000
STICKY_MODEL:  Dict[str, str]   = {}

VALID_GROQ_PREFIXES = (
    "llama-", "mixtral-", "gemma-",
    "compound-beta", "compound-beta-mini",
    "whisper-", "distil-",
)


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
            if len(STICKY_MODEL) >= _STICKY_MAX:
                evict = random.sample(list(STICKY_MODEL.keys()), min(200, len(STICKY_MODEL)))
                for k in evict:
                    STICKY_MODEL.pop(k, None)
            STICKY_MODEL[chat_id] = m
            return m
    STICKY_MODEL[chat_id] = pool[0]
    return pool[0]


def _pick_synthesis_model() -> Optional[str]:
    """
    For orchestrate_2+ (synthesis with large context), prefer the biggest
    model in the pool — it's more accurate and often faster with big inputs.
    """
    pool = [
        m for m in (settings.groq_model_pool or [])
        if _model_open(m) and any(m.startswith(p) for p in VALID_GROQ_PREFIXES)
    ]
    if not pool:
        return None
    # Prefer 70b or larger; fall back to first available
    big = [m for m in pool if "70b" in m or "72b" in m or "65b" in m]
    return big[0] if big else pool[0]


async def init_llm() -> None:
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not settings.groq_api_key:
        logger.warning("🧠 llm.init — GROQ_API_KEY missing, LLM disabled")
        return
    GROQ_CLIENT = AsyncGroq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
    for m in (settings.groq_model_pool or []):
        if not any(m.startswith(p) for p in VALID_GROQ_PREFIXES):
            logger.warning(
                "⚠️  model_pool — %r looks invalid. Valid prefixes: llama-, compound-beta, "
                "mixtral-, gemma-. This WILL fail at inference time. Fix GROQ_MODEL_POOL.", m,
            )
    logger.info("🧠 llm.init — AsyncGroq ready  model_pool=%s", settings.groq_model_pool)


async def close_llm() -> None:
    global GROQ_CLIENT
    if GROQ_CLIENT:
        try:
            await GROQ_CLIENT.close()
        except Exception:
            pass
        GROQ_CLIENT = None


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty_response")
    if s.startswith("{"):
        return json.loads(s)
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError("no_json_found")


async def _groq_raw(
    chat_id:        str,
    system:         str,
    user:           str,
    temperature:    float,
    max_tokens:     int,
    *,
    model_override: Optional[str] = None,
    label:          str = "call",
    timeout_sec:    Optional[float] = None,
) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    model   = model_override or _pick_model(chat_id)
    t_sec   = timeout_sec or settings.groq_timeout
    payload = {
        "model":       model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens":  int(max_tokens),
    }

    t0 = time.perf_counter()

    async def _call() -> str:
        async with _inflight:
            resp = await GROQ_CLIENT.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "").strip()

    try:
        result = await asyncio.wait_for(
            async_retry(_call, max_attempts=3, base_delay=0.6, max_delay=8.0),
            timeout=t_sec,
        )
        elapsed = time.perf_counter() - t0
        if elapsed > _SLOW_CALL_WARN_SEC:
            logger.warning(
                "⚠️  slow_llm  label=%s  model=%s  %.1fs  in≈%d  out≈%d",
                label, model, elapsed, len(user) // 4, len(result) // 4,
            )
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "⚠️  llm_timeout  label=%s  model=%s  timeout=%.0fs",
            label, model, t_sec,
        )
        MODEL_CIRCUIT[model] = time.monotonic() + 15.0
        raise
    except Exception as exc:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        logger.warning("🔴 circuit.tripped  model=%s  label=%s  err=%s", model, label, exc)
        raise


# ---------------------------------------------------------------------------
# WhatsApp formatter
# ---------------------------------------------------------------------------

async def _format_whatsapp(chat_id: str, text: str) -> str:
    needs_fmt = (
        "**" in text or "```" in text
        or "|---" in text or text.count("|") > 3
        or text.count("# ") > 1
    )
    if not needs_fmt:
        return sanitize_for_whatsapp(text)

    raw = await _groq_raw(
        chat_id, FORMATTER_PROMPT,
        json.dumps({"text": text}, ensure_ascii=False),
        temperature=0.0, max_tokens=400, label="format",
    )
    try:
        fr = FormatterResult.model_validate(_extract_json(raw))
        return sanitize_for_whatsapp(fr.text)
    except Exception:
        return sanitize_for_whatsapp(text)


# ---------------------------------------------------------------------------
# Live search
# ---------------------------------------------------------------------------

async def _live_search(chat_id: str, query: str, facts: Dict[str, str], city: str = "") -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Search unavailable."

    tc  = _time_context(city)
    user_payload = json.dumps(
        {
            "query":        query,
            "facts":        facts,
            "today":        tc["date"],
            "current_time": tc["current_time"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model":       settings.live_search_model,
        "messages": [
            {"role": "system", "content": LIVE_SEARCH_PROMPT},
            {"role": "user",   "content": user_payload},
        ],
        "max_tokens":  900,
        "temperature": 0.2,
    }

    t0 = time.perf_counter()

    async def _call() -> str:
        async with _inflight:
            resp = await GROQ_CLIENT.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "").strip()

    result = await asyncio.wait_for(
        async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0),
        timeout=settings.groq_timeout,
    )
    elapsed = time.perf_counter() - t0
    if elapsed > _SLOW_CALL_WARN_SEC:
        logger.warning(
            "⚠️  slow_search  query=%r  %.1fs  result_len=%d",
            query[:80], elapsed, len(result),
        )
    return result


# ---------------------------------------------------------------------------
# Memory pipeline
# ---------------------------------------------------------------------------

def _normalize_updates(updates: List[MemoryUpdate]) -> List[MemoryUpdate]:
    seen: Dict[str, str] = {}
    for u in updates:
        k = normalize_key(u.key)
        if k:
            seen[k] = (u.value or "").strip()
    return [MemoryUpdate(key=k, value=v) for k, v in seen.items() if k and v]


async def _extract_memory(chat_id: str, user_text: str) -> List[MemoryUpdate]:
    raw = await _groq_raw(
        chat_id, MEMORY_EXTRACTOR_PROMPT,
        json.dumps({"user_message": user_text}, ensure_ascii=False),
        temperature=0.0, max_tokens=450, label="extract",
    )
    try:
        er = ExtractResult.model_validate(_extract_json(raw))
        return _normalize_updates(er.memory_updates)
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.extract_failed  err=%s  raw=%.200s", e, raw or "")
        return []


async def _verify_updates(
    chat_id: str, user_text: str, proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    if not proposed:
        return []
    if not settings.facts_verification:
        return _normalize_updates(proposed)

    user_payload = json.dumps(
        {
            "user_message":            user_text,
            "proposed_memory_updates": [u.model_dump() for u in proposed],
        },
        ensure_ascii=False,
    )
    raw = await _groq_raw(
        chat_id, VERIFIER_PROMPT, user_payload,
        temperature=0.0, max_tokens=450, label="verify",
    )
    try:
        vr = VerifyResult.model_validate(_extract_json(raw))
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.verify_failed  err=%s  raw=%.200s", e, raw or "")
        return []

    min_conf = float(settings.facts_min_conf or 0.85)
    return _normalize_updates([
        MemoryUpdate(key=a.key, value=a.value)
        for a in vr.approved
        if a.confidence >= min_conf
    ])


def _merge_memory(
    pre_verified: List[MemoryUpdate], agent_proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    merged = {u.key: u.value for u in pre_verified}
    for u in agent_proposed:
        merged[u.key] = u.value
    return [MemoryUpdate(key=k, value=v) for k, v in merged.items()]


# ---------------------------------------------------------------------------
# Agentic orchestrator
# ---------------------------------------------------------------------------

async def _orchestrate(
    chat_id:        str,
    user_text:      str,
    facts:          Dict[str, str],
    context:        List[Dict[str, Any]],
    search_results: List[Dict[str, str]],
    iteration:      int,
    time_ctx:       Dict[str, str],
) -> OrchestratorResult:
    system = ORCHESTRATOR_PROMPT
    if iteration >= _MAX_ITERATIONS:
        system = system + "\n\nFINAL ITERATION: You MUST use action=answer now."

    # Use the big model for synthesis iterations (iter ≥ 2)
    model_override = None
    if iteration >= 2:
        model_override = _pick_synthesis_model()

    user_payload = json.dumps(
        {
            "user_message":   user_text,
            "facts":          facts,
            "context":        context,
            "search_results": search_results,
            "current_time":   time_ctx.get("current_time", ""),
            "time_of_day":    time_ctx.get("time_of_day", ""),
            "greeting":       time_ctx.get("greeting", ""),
            "today":          time_ctx.get("date", ""),
            "day_name":       time_ctx.get("day_name", ""),
            "tz_offset":      time_ctx.get("tz_offset", ""),
            "iteration":      iteration,
            "max_iterations": _MAX_ITERATIONS,
        },
        ensure_ascii=False,
    )

    raw = await _groq_raw(
        chat_id, system, user_payload,
        temperature=0.25, max_tokens=1000,
        model_override=model_override,
        label=f"orchestrate_{iteration}",
    )

    try:
        result = OrchestratorResult.model_validate(_extract_json(raw))
        logger.info(
            "🤖 orchestrate  iter=%d  action=%-10s  reasoning=%.140s",
            iteration, result.action, result.reasoning or "(none)",
        )
        return result
    except Exception as e:
        logger.warning(
            "orchestrate.parse_failed  iter=%d  err=%s  raw=%.200s",
            iteration, e, raw,
        )
        # One repair attempt
        repaired_raw = await _groq_raw(
            chat_id, REPAIR_PROMPT, raw,
            temperature=0.0, max_tokens=700, label="repair",
        )
        result = OrchestratorResult.model_validate(_extract_json(repaired_raw))
        logger.info("orchestrate.repaired  iter=%d  action=%s", iteration, result.action)
        return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    *,
    chat_id:   str,
    user_text: str,
    facts:     Dict[str, str],
    context:   List[Dict[str, Any]],
    trace:     Any = None,
) -> AgentResult:
    def _step(name: str):
        return trace.step(name) if trace is not None else nullcontext()

    city     = facts.get("city", "")
    time_ctx = _time_context(city)

    # ── kick off memory extraction in background ───────────────────────────
    with _step("memory_extract"):
        extract_task = asyncio.create_task(_extract_memory(chat_id, user_text))

    search_results: List[Dict[str, str]] = []
    orch_result: Optional[OrchestratorResult] = None
    iteration = 0

    # ── agentic loop ───────────────────────────────────────────────────────
    while iteration < _MAX_ITERATIONS:
        iteration += 1

        with _step(f"orchestrate_{iteration}"):
            orch_result = await _orchestrate(
                chat_id, user_text, facts, context,
                search_results, iteration, time_ctx,
            )
            if trace:
                trace.tag(action=orch_result.action, reasoning_len=len(orch_result.reasoning))

        # ask_user → return immediately
        if orch_result.action == "ask_user" and orch_result.question:
            proposed = await extract_task
            verified = await _verify_updates(chat_id, user_text, proposed)
            agent_mu = _normalize_updates(orch_result.memory_updates)
            merged   = _merge_memory(verified, agent_mu)
            if trace:
                trace.tag(total_iterations=iteration, memory_total=len(merged))
            return AgentResult(
                reply=ReplyPayload(text=sanitize_for_whatsapp(orch_result.question)),
                memory_updates=merged,
                reminders=orch_result.reminders,
                iterations=iteration,
            )

        # search → get live data and continue
        if orch_result.action == "search" and settings.live_search_enabled:
            query = (orch_result.query or user_text).strip()
            with _step(f"live_search_{iteration}"):
                try:
                    result_text = await _live_search(chat_id, query, facts, city)
                except asyncio.TimeoutError:
                    result_text = f"[Search timed out for: {query}]"
                    logger.warning("live_search.timeout  query=%r", query[:80])
                if trace:
                    trace.tag(search_query=query[:80], result_len=len(result_text))
            search_results.append({"query": query, "result": result_text})
            logger.info(
                "🔍 search.done  iter=%d  query=%r  result_len=%d",
                iteration, query[:80], len(result_text),
            )
            continue

        if orch_result.action == "answer" or orch_result.text:
            break

        logger.warning("orchestrate.unknown_action  iter=%d  action=%r", iteration, orch_result.action)
        if iteration >= _MAX_ITERATIONS:
            break

    # ── verify memory ──────────────────────────────────────────────────────
    with _step("memory_verify"):
        proposed = await extract_task
        verified = await _verify_updates(chat_id, user_text, proposed) if proposed else []
        pre_keys  = {v.key for v in verified}
        agent_raw = _normalize_updates(orch_result.memory_updates if orch_result else [])
        agent_new = [u for u in agent_raw if u.key not in pre_keys]
        agent_vfd = await _verify_updates(chat_id, user_text, agent_new) if agent_new else []
        final_mem = _merge_memory(verified, agent_vfd)
        if trace:
            trace.tag(
                total_iterations=iteration,
                memory_extracted=len(proposed),
                memory_verified=len(verified),
                memory_total=len(final_mem),
            )

    # ── format ────────────────────────────────────────────────────────────
    raw_text = (orch_result.text if orch_result else "") or "I'm not sure how to respond to that."
    with _step("format"):
        formatted = await _format_whatsapp(chat_id, raw_text)

    return AgentResult(
        reply=ReplyPayload(text=formatted),
        memory_updates=final_mem,
        reminders=orch_result.reminders if orch_result else [],
        iterations=iteration,
    )
