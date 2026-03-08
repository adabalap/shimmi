"""
agent_engine.py — Dynamic agentic loop for Shimmi v2.5.0.

Key fix vs v2.4.0:
  - NEVER call str.format() on prompt strings that contain JSON examples.
    Python interprets every {…} in the string as a format placeholder.
    We use json.dumps() to build the user-turn payload and pass the system
    prompt as a plain constant.  No string interpolation on prompts at all.

Architecture (unchanged from v2.4.0):
  The LLM drives its own reasoning loop. Each iteration it declares:
    action=answer   → produce final reply
    action=search   → run live web search, loop back with results
    action=ask_user → ask one clarifying question, end turn

  Memory extraction runs in parallel with the first orchestrator call.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
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
    iterations: int = 1


class OrchestratorResult(BaseModel):
    action: str                                      # answer | search | ask_user
    reasoning: str = ""
    text: str = ""
    query: str = ""
    question: str = ""
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


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
# LLM client & circuit-breaker
# ---------------------------------------------------------------------------

GROQ_CLIENT: Optional[AsyncGroq] = None
_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))

MODEL_CIRCUIT: Dict[str, float] = {}
_STICKY_MAX = 2_000
STICKY_MODEL: Dict[str, str] = {}

_MAX_ITERATIONS = 3


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
# Low-level LLM helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from an LLM response string."""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty_response")
    # Fast path: response is pure JSON
    if s.startswith("{"):
        return json.loads(s)
    # Slow path: JSON embedded in prose or fences
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError("no_json_found")


async def _groq_raw(
    chat_id: str,
    system: str,     # plain constant — NEVER call str.format() on this
    user: str,       # JSON-dumped payload — built with json.dumps()
    temperature: float,
    max_tokens: int,
    *,
    model_override: Optional[str] = None,
) -> str:
    """
    Single Groq API call.

    IMPORTANT: `system` is passed directly to the API as-is.
    We do NOT call system.format(…) — that would explode on any prompt
    containing JSON schema examples with curly braces.
    """
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    model = model_override or _pick_model(chat_id)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    async def _call() -> str:
        async with _inflight:
            resp = await GROQ_CLIENT.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "").strip()

    try:
        result = await async_retry(_call, max_attempts=4, base_delay=0.6, max_delay=8.0)
        logger.debug(
            "groq.ok model=%s tokens_in≈%d tokens_out≈%d",
            model,
            len(user) // 4,          # rough estimate
            len(result) // 4,
        )
        return result
    except Exception as exc:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        logger.warning("🔴 circuit.tripped model=%s  err=%s", model, exc)
        raise


async def _format_whatsapp(chat_id: str, text: str) -> str:
    """
    Invoke the LLM formatter only when structural cleanup is needed.
    Plain conversational replies are sanitised locally — saves a round-trip.
    """
    needs_fmt = (
        "**" in text
        or "```" in text
        or "|---" in text
        or text.count("|") > 3
    )
    if not needs_fmt:
        return sanitize_for_whatsapp(text)

    # user payload is JSON — system prompt is a plain constant
    raw = await _groq_raw(
        chat_id,
        FORMATTER_PROMPT,          # ← constant, no .format()
        json.dumps({"text": text}, ensure_ascii=False),
        temperature=0.0,
        max_tokens=400,
    )
    try:
        fr = FormatterResult.model_validate(_extract_json(raw))
        return sanitize_for_whatsapp(fr.text)
    except Exception:
        return sanitize_for_whatsapp(text)


# ---------------------------------------------------------------------------
# Live search
# ---------------------------------------------------------------------------

async def _live_search(chat_id: str, query: str, facts: Dict[str, str]) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Search unavailable — no LLM client."

    user_payload = json.dumps({"query": query, "facts": facts}, ensure_ascii=False)

    payload = {
        "model": settings.live_search_model,
        "messages": [
            {"role": "system", "content": LIVE_SEARCH_PROMPT},  # ← constant
            {"role": "user",   "content": user_payload},
        ],
        "max_tokens": 900,
        "temperature": 0.2,
    }

    async def _call() -> str:
        async with _inflight:
            resp = await GROQ_CLIENT.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "").strip()

    result = await async_retry(_call, max_attempts=3, base_delay=0.8, max_delay=10.0)
    logger.debug("live_search.ok query=%r result_len=%d", query[:80], len(result))
    return result


# ---------------------------------------------------------------------------
# Memory pipeline
# ---------------------------------------------------------------------------

async def _extract_memory(chat_id: str, user_text: str) -> List[MemoryUpdate]:
    raw = await _groq_raw(
        chat_id,
        MEMORY_EXTRACTOR_PROMPT,   # ← constant
        json.dumps({"user_message": user_text}, ensure_ascii=False),
        temperature=0.0,
        max_tokens=450,
    )
    try:
        er = ExtractResult.model_validate(_extract_json(raw))
        if er.memory_updates:
            logger.debug(
                "memory.extracted count=%d  keys=%s",
                len(er.memory_updates),
                ", ".join(f"{u.key}={u.value!r}" for u in er.memory_updates),
            )
        return er.memory_updates
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.extract_failed err=%s raw=%.200s", e, raw or "")
        return []


async def _verify_updates(
    chat_id: str,
    user_text: str,
    proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    if not proposed:
        return []
    if not settings.facts_verification:
        return proposed

    user_payload = json.dumps(
        {"user_message": user_text, "proposed_memory_updates": [u.model_dump() for u in proposed]},
        ensure_ascii=False,
    )
    raw = await _groq_raw(
        chat_id,
        VERIFIER_PROMPT,           # ← constant
        user_payload,
        temperature=0.0,
        max_tokens=450,
    )
    try:
        vr = VerifyResult.model_validate(_extract_json(raw))
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.verify_failed err=%s raw=%.200s", e, raw or "")
        return []

    min_conf = float(settings.facts_min_conf or 0.85)
    kept = [
        MemoryUpdate(key=str(a.key).strip(), value=str(a.value).strip())
        for a in vr.approved
        if a.confidence >= min_conf
    ]
    if kept:
        logger.debug(
            "memory.verified kept=%d/%d  pairs=%s",
            len(kept), len(proposed),
            ", ".join(f"{u.key}={u.value!r}" for u in kept),
        )
    return kept


def _merge_memory(
    pre_verified: List[MemoryUpdate],
    agent_proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    merged: Dict[str, str] = {u.key: u.value for u in pre_verified}
    for u in agent_proposed:
        merged[u.key] = u.value
    return [MemoryUpdate(key=k, value=v) for k, v in merged.items()]


# ---------------------------------------------------------------------------
# Agentic orchestrator
# ---------------------------------------------------------------------------

async def _orchestrate(
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
    search_results: List[Dict[str, str]],
    iteration: int,
) -> OrchestratorResult:
    """
    Single orchestrator call.

    The user turn is built with json.dumps() — a plain data structure.
    The system prompt is always passed as a static string constant.
    Dynamic context lives entirely in the user message JSON payload.
    """
    system = ORCHESTRATOR_PROMPT
    if iteration >= _MAX_ITERATIONS:
        # Append to the constant string — still no .format()
        system = system + "\n\nFINAL ITERATION: You MUST use action=answer now."

    # Build the user payload as structured data, then serialise
    user_payload = json.dumps(
        {
            "user_message":   user_text,
            "facts":          facts,
            "context":        context,
            "search_results": search_results,
            "iteration":      iteration,
            "max_iterations": _MAX_ITERATIONS,
        },
        ensure_ascii=False,
    )

    raw = await _groq_raw(
        chat_id,
        system,          # ← constant (possibly with suffix appended)
        user_payload,    # ← json.dumps() result
        temperature=0.25,
        max_tokens=950,
    )

    try:
        result = OrchestratorResult.model_validate(_extract_json(raw))
        logger.info(
            "🤖 orchestrate iter=%d  action=%-10s  reasoning=%.120s",
            iteration,
            result.action,
            result.reasoning or "(none)",
        )
        return result
    except Exception as e:
        logger.warning(
            "orchestrate.parse_failed iter=%d err=%s raw=%.200s",
            iteration, e, raw,
        )
        # Self-repair pass
        repaired_raw = await _groq_raw(
            chat_id,
            REPAIR_PROMPT,    # ← constant
            raw,
            temperature=0.0,
            max_tokens=600,
        )
        result = OrchestratorResult.model_validate(_extract_json(repaired_raw))
        logger.info("orchestrate.repaired iter=%d action=%s", iteration, result.action)
        return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    *,
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
    trace: Any = None,
) -> AgentResult:
    """
    Execute the full agentic pipeline:
      1. Kick off memory extraction in background
      2. Run orchestrator loop (≤ _MAX_ITERATIONS)
      3. Verify and merge all memory updates
      4. Format the reply
    """
    from contextlib import nullcontext

    def _step(name: str):
        return trace.step(name) if trace is not None else nullcontext()

    # ── Stage 1: start memory extraction in background ────────────────────
    with _step("memory_extract"):
        extract_task = asyncio.create_task(_extract_memory(chat_id, user_text))

    search_results: List[Dict[str, str]] = []
    orch_result: Optional[OrchestratorResult] = None
    iteration = 0

    # ── Stage 2: agentic loop ─────────────────────────────────────────────
    while iteration < _MAX_ITERATIONS:
        iteration += 1

        with _step(f"orchestrate_{iteration}"):
            orch_result = await _orchestrate(
                chat_id, user_text, facts, context, search_results, iteration,
            )
            if trace:
                trace.tag(action=orch_result.action, reasoning_len=len(orch_result.reasoning))

        # action: clarifying question → end turn immediately
        if orch_result.action == "ask_user" and orch_result.question:
            proposed = await extract_task
            verified = await _verify_updates(chat_id, user_text, proposed)
            merged = _merge_memory(verified, orch_result.memory_updates)
            if trace:
                trace.tag(total_iterations=iteration, memory_total=len(merged))
            return AgentResult(
                reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(orch_result.question)),
                memory_updates=merged,
                iterations=iteration,
            )

        # action: web search → inject results and loop
        if orch_result.action == "search" and settings.live_search_enabled:
            query = (orch_result.query or user_text).strip()
            with _step(f"live_search_{iteration}"):
                result_text = await _live_search(chat_id, query, facts)
                if trace:
                    trace.tag(search_query=query[:80], result_len=len(result_text))
            search_results.append({"query": query, "result": result_text})
            logger.info(
                "🔍 search.done iter=%d  query=%r  result_len=%d",
                iteration, query[:80], len(result_text),
            )
            continue

        # action: answer (or loop limit reached)
        if orch_result.action == "answer" or orch_result.text:
            break

        # Unknown action — warn and break on next limit check
        logger.warning("orchestrate.unknown_action iter=%d action=%r", iteration, orch_result.action)
        if iteration >= _MAX_ITERATIONS:
            break

    # ── Stage 3: verify all memory ────────────────────────────────────────
    with _step("memory_verify"):
        proposed = await extract_task
        verified = await _verify_updates(chat_id, user_text, proposed)

        pre_keys = {v.key for v in verified}
        agent_new = [u for u in (orch_result.memory_updates if orch_result else []) if u.key not in pre_keys]
        agent_verified = await _verify_updates(chat_id, user_text, agent_new)
        final_memory = _merge_memory(verified, agent_verified)

        if trace:
            trace.tag(
                total_iterations=iteration,
                memory_extracted=len(proposed),
                memory_verified=len(verified),
                memory_total=len(final_memory),
            )

    # ── Stage 4: format ───────────────────────────────────────────────────
    raw_text = (orch_result.text if orch_result else "") or "I'm not sure how to answer that."
    with _step("format"):
        formatted = await _format_whatsapp(chat_id, raw_text)

    return AgentResult(
        reply=ReplyPayload(type="text", text=formatted),
        memory_updates=final_memory,
        iterations=iteration,
    )
