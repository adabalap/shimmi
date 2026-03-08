"""
agent_engine.py — Shimmi v2.7.0

Key improvements vs v2.5.0 (production):
  - normalize_key() applied to all memory updates at source
  - Per-call timeout (GROQ_TIMEOUT env) — no more 10s+ hangs without cancellation
  - Date injected into every orchestrator call so search queries are date-aware
  - Memory verification skipped when extracted count is 0 (saves 100–300ms)
  - Slow call warning threshold logged clearly for monitoring
  - NEVER calls str.format() on any prompt string
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

from pydantic import BaseModel, Field
from groq import AsyncGroq

from .config import settings
from .retry import async_retry
from .prompts import (
    ORCHESTRATOR_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT,
    REPAIR_PROMPT,
    FORMATTER_PROMPT,
    LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp
from .database import normalize_key

logger     = logging.getLogger("app.agent")
UTC        = timezone.utc

_SLOW_CALL_WARN_SEC = 5.0
_MAX_ITERATIONS     = 3


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    key:   str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply:          ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    iterations:     int = 1


class OrchestratorResult(BaseModel):
    action:         str
    reasoning:      str = ""
    text:           str = ""
    query:          str = ""
    question:       str = ""
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


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
_inflight   = asyncio.Semaphore(int(settings.groq_max_inflight or 5))

MODEL_CIRCUIT: Dict[str, float] = {}
_STICKY_MAX  = 2_000
STICKY_MODEL: Dict[str, str]   = {}

# Valid Groq model name prefixes — used by integrity check
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


async def init_llm() -> None:
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not settings.groq_api_key:
        logger.warning("🧠 llm.init — GROQ_API_KEY missing, LLM disabled")
        return
    GROQ_CLIENT = AsyncGroq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)

    # Warn about any invalid model names in the pool
    for m in (settings.groq_model_pool or []):
        if not any(m.startswith(p) for p in VALID_GROQ_PREFIXES):
            logger.warning(
                "⚠️  model_pool — %r looks invalid. Valid prefixes: llama-, compound-beta, etc. "
                "This will fail at inference time. Fix GROQ_MODEL_POOL in .env", m,
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
    system:         str,    # plain string constant — NEVER .format() this
    user:           str,    # built with json.dumps()
    temperature:    float,
    max_tokens:     int,
    *,
    model_override: Optional[str] = None,
    label:          str = "call",
    timeout_sec:    Optional[float] = None,
) -> str:
    """
    Single Groq API call with per-call timeout and circuit breaker.

    `system` is passed to the API as-is — never interpolated.
    `user`   is a json.dumps() result — structured data, not a template.
    """
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    model  = model_override or _pick_model(chat_id)
    t_sec  = timeout_sec or settings.groq_timeout
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
                "⚠️  slow_llm  label=%s  model=%s  %.1fs  "
                "(tokens_in≈%d  tokens_out≈%d)",
                label, model, elapsed, len(user) // 4, len(result) // 4,
            )
        else:
            logger.debug(
                "llm.ok  label=%s  model=%s  %.2fs  in≈%d  out≈%d",
                label, model, elapsed, len(user) // 4, len(result) // 4,
            )
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "⚠️  llm_timeout  label=%s  model=%s  timeout=%.0fs — tripping circuit",
            label, model, t_sec,
        )
        MODEL_CIRCUIT[model] = time.monotonic() + 15.0
        raise
    except Exception as exc:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        logger.warning("🔴 circuit.tripped  model=%s  label=%s  err=%s", model, label, exc)
        raise


# ---------------------------------------------------------------------------
# WhatsApp formatter — only invoked for heavy markdown
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
        chat_id,
        FORMATTER_PROMPT,
        json.dumps({"text": text}, ensure_ascii=False),
        temperature=0.0,
        max_tokens=400,
        label="format",
    )
    try:
        fr = FormatterResult.model_validate(_extract_json(raw))
        return sanitize_for_whatsapp(fr.text)
    except Exception:
        return sanitize_for_whatsapp(text)


# ---------------------------------------------------------------------------
# Live search
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def _live_search(chat_id: str, query: str, facts: Dict[str, str]) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Search unavailable — LLM client not initialised."

    user_payload = json.dumps(
        {"query": query, "facts": facts, "today": _today_str()},
        ensure_ascii=False,
    )
    payload = {
        "model":    settings.live_search_model,
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
    """Normalise keys and deduplicate (last value wins per canonical key)."""
    seen: Dict[str, str] = {}
    for u in updates:
        k = normalize_key(u.key)
        if k:
            seen[k] = (u.value or "").strip()
    return [MemoryUpdate(key=k, value=v) for k, v in seen.items() if k and v]


async def _extract_memory(chat_id: str, user_text: str) -> List[MemoryUpdate]:
    raw = await _groq_raw(
        chat_id,
        MEMORY_EXTRACTOR_PROMPT,
        json.dumps({"user_message": user_text}, ensure_ascii=False),
        temperature=0.0,
        max_tokens=450,
        label="extract",
    )
    try:
        er = ExtractResult.model_validate(_extract_json(raw))
        updates = _normalize_updates(er.memory_updates)
        if updates:
            logger.debug(
                "memory.extracted  count=%d  pairs=%s",
                len(updates),
                "  ".join(f"{u.key}={u.value!r}" for u in updates),
            )
        return updates
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.extract_failed  err=%s  raw=%.200s", e, raw or "")
        return []


async def _verify_updates(
    chat_id:   str,
    user_text: str,
    proposed:  List[MemoryUpdate],
) -> List[MemoryUpdate]:
    if not proposed:
        return []   # skip the LLM call entirely when nothing was extracted
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
    kept = _normalize_updates([
        MemoryUpdate(key=a.key, value=a.value)
        for a in vr.approved
        if a.confidence >= min_conf
    ])
    if kept:
        logger.debug(
            "memory.verified  kept=%d/%d  %s",
            len(kept), len(proposed),
            "  ".join(f"{u.key}={u.value!r}" for u in kept),
        )
    return kept


def _merge_memory(
    pre_verified:   List[MemoryUpdate],
    agent_proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    """Merge two lists — agent proposals win on key conflict."""
    merged: Dict[str, str] = {u.key: u.value for u in pre_verified}
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
) -> OrchestratorResult:
    system = ORCHESTRATOR_PROMPT
    if iteration >= _MAX_ITERATIONS:
        system = system + "\n\nFINAL ITERATION: You MUST use action=answer now."

    # Inject today's date so search queries can be date-stamped
    user_payload = json.dumps(
        {
            "user_message":   user_text,
            "facts":          facts,
            "context":        context,
            "search_results": search_results,
            "today":          _today_str(),
            "iteration":      iteration,
            "max_iterations": _MAX_ITERATIONS,
        },
        ensure_ascii=False,
    )

    raw = await _groq_raw(
        chat_id,
        system,          # constant — not interpolated
        user_payload,    # json.dumps() — not a format string
        temperature=0.25,
        max_tokens=950,
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
            temperature=0.0, max_tokens=600, label="repair",
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

    # ── Stage 1: kick off memory extraction in background ─────────────────
    with _step("memory_extract"):
        extract_task = asyncio.create_task(_extract_memory(chat_id, user_text))

    search_results: List[Dict[str, str]] = []
    orch_result: Optional[OrchestratorResult] = None
    iteration = 0

    # ── Stage 2: agentic loop ──────────────────────────────────────────────
    while iteration < _MAX_ITERATIONS:
        iteration += 1

        with _step(f"orchestrate_{iteration}"):
            orch_result = await _orchestrate(
                chat_id, user_text, facts, context, search_results, iteration,
            )
            if trace:
                trace.tag(action=orch_result.action, reasoning_len=len(orch_result.reasoning))

        # ask_user → return clarifying question immediately
        if orch_result.action == "ask_user" and orch_result.question:
            proposed = await extract_task
            verified = await _verify_updates(chat_id, user_text, proposed)
            merged   = _merge_memory(verified, _normalize_updates(orch_result.memory_updates))
            if trace:
                trace.tag(total_iterations=iteration, memory_total=len(merged))
            return AgentResult(
                reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(orch_result.question)),
                memory_updates=merged,
                iterations=iteration,
            )

        # search → get live data and continue loop
        if orch_result.action == "search" and settings.live_search_enabled:
            query = (orch_result.query or user_text).strip()
            with _step(f"live_search_{iteration}"):
                try:
                    result_text = await _live_search(chat_id, query, facts)
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

        # answer or fallback
        if orch_result.action == "answer" or orch_result.text:
            break

        logger.warning(
            "orchestrate.unknown_action  iter=%d  action=%r",
            iteration, orch_result.action,
        )
        if iteration >= _MAX_ITERATIONS:
            break

    # ── Stage 3: verify memory ─────────────────────────────────────────────
    with _step("memory_verify"):
        proposed = await extract_task

        # Skip the verify LLM call if nothing was extracted — saves 100-300ms
        if proposed:
            verified = await _verify_updates(chat_id, user_text, proposed)
        else:
            verified = []

        pre_keys     = {v.key for v in verified}
        agent_raw    = _normalize_updates(orch_result.memory_updates if orch_result else [])
        agent_new    = [u for u in agent_raw if u.key not in pre_keys]
        agent_vfd    = await _verify_updates(chat_id, user_text, agent_new) if agent_new else []
        final_memory = _merge_memory(verified, agent_vfd)

        if trace:
            trace.tag(
                total_iterations=iteration,
                memory_extracted=len(proposed),
                memory_verified=len(verified),
                memory_total=len(final_memory),
            )

    # ── Stage 4: format ────────────────────────────────────────────────────
    raw_text = (orch_result.text if orch_result else "") or "I'm not sure how to answer that."
    with _step("format"):
        formatted = await _format_whatsapp(chat_id, raw_text)

    return AgentResult(
        reply=ReplyPayload(type="text", text=formatted),
        memory_updates=final_memory,
        iterations=iteration,
    )
