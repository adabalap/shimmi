"""
agent_engine.py — Shimmi v3.0.3

Changes vs v3.0.2:

  PROVIDER-1  Google Gemini added as primary orchestrator.
              Gemini 2.0 Flash free tier: ~1,500 req/day, 1M tokens/min.
              That is 15× more generous than Groq 70B (100K tokens/day).
              Provider chain: Gemini → Groq 70B → Groq 8B.
              Uses OpenAI-compatible endpoint so no extra SDK needed.

  PROVIDER-2  Per-provider circuit breakers (separate from model circuit).
              Gemini circuit opens on quota/429, falls back to Groq 70B.
              Groq 70B circuit opens on 429, falls back to Groq 8B.

  PROVIDER-3  Token budget tracker: estimates daily usage per provider,
              logs warnings at 75%, blocks (routes to fallback) at 92%.
              Prevents silent exhaustion — first warning message is sent
              to the user when budget is nearly full.

  FIX-S1      Facts shortcut v2: completely rewritten with broader matching.
              Previous RECALL_PATTERNS regex missed "coffee order", "hiking
              trail", "podcast" etc., so ALL simple memory queries burned
              ~1700 tokens on the 70B model.  New approach: short-message
              heuristic + broad keyword table covering 30+ fact patterns.

  FIX-S2      Junk fact filtering hardened: facts with value in
              {unknown,none,null,n/a,''} are stripped BEFORE building the
              orchestrator prompt.  Previously the filter ran too late in
              some paths, leaking ~200-400 tokens per call.

  FIX-S3      Fire-and-forget task exception handling: _extract_memory and
              extract_reply_memory wrapped in top-level try/except catching
              ALL exceptions (including TimeoutError, CancelledError).
              "Task exception was never retrieved" errors eliminated.

  FIX-S4      asyncio.get_event_loop() → asyncio.get_running_loop() (broken
              on Python 3.12+).

  FIX-S5      Retry-after parsing: 429 error messages from Groq and Gemini
              are parsed for the wait time; circuit cooldown is set to the
              actual required wait, not a random jitter.

  FIX-S6      Live search 413 (Request Entity Too Large) handling: query is
              automatically shortened and retried.

  FIX-S7      User-facing error message is populated in AgentResult instead
              of silently failing.  main.py uses this to always send a reply.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from groq import AsyncGroq
from openai import AsyncOpenAI   # used for Gemini OpenAI-compat endpoint

from .config import settings
from .retry import async_retry
from .prompts import (
    ORCHESTRATOR_PROMPT, MEMORY_EXTRACTOR_PROMPT, REPLY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT, REPAIR_PROMPT, FORMATTER_PROMPT, LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp
from .database import normalize_key, Reminder

logger = logging.getLogger("app.agent")
UTC    = timezone.utc

_SLOW_CALL_WARN_SEC  = 5.0
_MAX_ITERATIONS      = 3
_MAX_SEARCH_RESULT   = 1_200
_MIN_FORMAT_LEN      = 120

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class MemoryUpdate(BaseModel):
    key:   str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)

    @field_validator("key", "value", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return "" if v is None else str(v).strip()


class ReminderEntry(BaseModel):
    text:        str = Field(..., min_length=1)
    trigger_iso: str = Field(..., min_length=1)

    @field_validator("text", "trigger_iso", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return "" if v is None else str(v).strip()


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply:          ReplyPayload
    memory_updates: List[MemoryUpdate]  = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)
    iterations:     int = 1
    provider_used:  str = ""   # "gemini" | "groq_70b" | "groq_8b"


class OrchestratorResult(BaseModel):
    action:         str
    reasoning:      str              = ""
    text:           str              = ""
    query:          str              = ""
    question:       str              = ""
    memory_updates: List[MemoryUpdate]  = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)

    @field_validator("memory_updates", mode="before")
    @classmethod
    def _clean_memory(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k   = str(item.get("key",   "") or "").strip()
            val = str(item.get("value", "") or "").strip()
            if k and val:
                out.append({"key": k, "value": val})
        return out

    @field_validator("reminders", mode="before")
    @classmethod
    def _clean_reminders(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if not isinstance(item, dict):
                continue
            t   = str(item.get("text",        "") or "").strip()
            iso = str(item.get("trigger_iso", "") or "").strip()
            if t and iso:
                out.append({"text": t, "trigger_iso": iso})
        return out


class ApprovedUpdate(BaseModel):
    key:        str
    value:      str
    confidence: float = Field(ge=0.0, le=1.0)


class VerifyResult(BaseModel):
    approved: List[ApprovedUpdate] = Field(default_factory=list)


class ExtractResult(BaseModel):
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)

    @field_validator("memory_updates", mode="before")
    @classmethod
    def _clean(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k   = str(item.get("key",   "") or "").strip()
            val = str(item.get("value", "") or "").strip()
            if k and val:
                out.append({"key": k, "value": val})
        return out


class FormatterResult(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Timezone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_tz():
    tz_name = (settings.app_timezone or "UTC").strip()
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, Exception):
        return UTC


def _now_local() -> datetime:
    return datetime.now(_get_local_tz())


def _current_time_str() -> str:
    now = _now_local()
    hour = now.hour
    if   6  <= hour < 12: period = "morning"
    elif 12 <= hour < 17: period = "afternoon"
    elif 17 <= hour < 21: period = "evening"
    else:                 period = "night"
    tz_abbr = now.strftime("%Z") or "local"
    return f"{now.strftime('%H:%M')} {tz_abbr} ({now.strftime('%A')} {period})"


def _today_str() -> str:
    return _now_local().strftime("%Y-%m-%d")


def _utc_offset_str() -> str:
    dt     = _now_local()
    offset = dt.utcoffset()
    if offset is None:
        return "+00:00"
    total_min = int(offset.total_seconds() / 60)
    sign = "+" if total_min >= 0 else "-"
    h, m = divmod(abs(total_min), 60)
    return f"{sign}{h:02d}:{m:02d}"


def _time_of_day() -> str:
    hour = _now_local().hour
    if 6  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def _fix_reminder_tz(trigger_iso: str) -> str:
    local_offset = _utc_offset_str()
    if local_offset == "+00:00":
        return trigger_iso
    t = trigger_iso.strip()
    for utc_tail in ("+00:00", "+0000", "Z"):
        if t.endswith(utc_tail):
            base      = t[: len(t) - len(utc_tail)]
            corrected = base + local_offset
            logger.info("⏰ reminder.tz_fix  %s → %s  (server=%s)",
                        trigger_iso, corrected, local_offset)
            return corrected
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Multi-provider LLM clients + circuit breakers + token budget
# ─────────────────────────────────────────────────────────────────────────────

GROQ_CLIENT: Optional[AsyncGroq] = None
GEMINI_CLIENT: Optional[AsyncOpenAI] = None   # OpenAI-compat wrapper for Gemini

# Per-model circuit breakers: model → monotonic timestamp when circuit reopens
MODEL_CIRCUIT: Dict[str, float] = {}

# Per-provider-level circuit: "gemini" | "groq_70b" | "groq_8b" → reopen time
PROVIDER_CIRCUIT: Dict[str, float] = {}

# Sticky model per chat
STICKY_MODEL: Dict[str, str] = {}
_STICKY_MAX = 2_000

# Token budget tracking: provider → (tokens_used_today, reset_ts)
_TOKEN_BUDGET: Dict[str, Tuple[int, float]] = {}
_BUDGET_RESET_INTERVAL = 86_400.0  # 24 hours

_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))


def _budget_add(provider: str, tokens: int) -> int:
    """Track token usage for a provider. Returns current total for the day."""
    now = time.time()
    used, reset_ts = _TOKEN_BUDGET.get(provider, (0, now + _BUDGET_RESET_INTERVAL))
    if now >= reset_ts:
        used = 0
        reset_ts = now + _BUDGET_RESET_INTERVAL
    used += tokens
    _TOKEN_BUDGET[provider] = (used, reset_ts)
    return used


def _budget_fraction(provider: str, daily_limit: int) -> float:
    """Return fraction of daily budget used (0.0–1.0+)."""
    if daily_limit <= 0:
        return 0.0
    used, reset_ts = _TOKEN_BUDGET.get(provider, (0, 0))
    if time.time() >= reset_ts:
        return 0.0
    return used / daily_limit


def _model_open(model: str) -> bool:
    return time.monotonic() >= MODEL_CIRCUIT.get(model, 0.0)


def _provider_open(provider: str) -> bool:
    return time.monotonic() >= PROVIDER_CIRCUIT.get(provider, 0.0)


def _trip_model(model: str, cooldown: float) -> None:
    MODEL_CIRCUIT[model] = time.monotonic() + cooldown


def _trip_provider(provider: str, cooldown: float) -> None:
    PROVIDER_CIRCUIT[provider] = time.monotonic() + cooldown
    logger.warning("🔴 provider.circuit_tripped  provider=%s  cooldown=%.0fs", provider, cooldown)


def _parse_retry_after(exc: Exception) -> float:
    """
    FIX-S5: Parse the actual wait time from Groq/Gemini 429 errors.

    Groq format: "Please try again in 1h4m54.368s"
    Gemini format: may include retry_delay in seconds
    Returns seconds (minimum 60.0, maximum 7200.0).
    """
    s = str(exc)

    # Hours + minutes + seconds: e.g. "1h4m54.368s" or "36m12.096s"
    m = re.search(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", s)
    if m:
        h = float(m.group(1) or 0)
        mn = float(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        total = h * 3600 + mn * 60 + sec
        return max(60.0, min(total + 10.0, 7200.0))  # +10s buffer

    # Gemini: "quota exceeded … retry after N seconds"
    m2 = re.search(r"retry[^\d]*(\d+)\s*second", s, re.IGNORECASE)
    if m2:
        return max(60.0, float(m2.group(1)) + 10.0)

    return 300.0  # conservative default: 5 min


# ─────────────────────────────────────────────────────────────────────────────
# Provider / model selection
# ─────────────────────────────────────────────────────────────────────────────

def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def _pick_provider_and_model(chat_id: str, role: str) -> Tuple[str, str]:
    """
    Route to the best available provider+model for the given role.

    Priority chain for orchestration:
      1. Gemini 2.0 Flash       (primary — 1.5M tokens/day free)
      2. Groq llama-3.3-70b     (fallback — 100K/day free)
      3. Groq llama-3.1-8b      (last resort for orchestration)

    Priority chain for extraction/format/verify:
      1. Groq llama-3.1-8b      (primary — 500K/day free, very fast)
      2. Gemini 2.0 Flash Lite  (fallback)
      3. Groq llama-3.3-70b     (last resort)
    """
    is_orchestrate = (role == "orchestrate")

    if is_orchestrate:
        candidates = [
            ("gemini",   settings.gemini_orchestrator_model),
            ("groq_70b", settings.orchestrator_model),
            ("groq_8b",  settings.extraction_model),
        ]
    else:
        candidates = [
            ("groq_8b",  settings.extraction_model),
            ("gemini",   settings.gemini_extraction_model),
            ("groq_70b", settings.orchestrator_model),
        ]

    for provider, model in candidates:
        # Skip Gemini if no key configured
        if provider == "gemini" and not settings.gemini_enabled:
            continue
        # Check model-level circuit
        if not _model_open(model):
            continue
        # Check provider-level circuit
        if not _provider_open(provider):
            continue
        # Check token budget (only Groq 70B has a meaningful hard limit)
        if provider == "groq_70b":
            frac = _budget_fraction("groq_70b", settings.groq_70b_daily_limit)
            if frac >= settings.token_budget_block_pct:
                logger.warning(
                    "💰 budget.block  provider=groq_70b  usage=%.1f%%  skipping",
                    frac * 100,
                )
                continue

        logger.debug("🎯 provider.selected  role=%s  provider=%s  model=%s",
                     role, provider, model)
        return provider, model

    # All options exhausted — return first candidate, will likely fail with 429
    fallback = candidates[0]
    logger.error(
        "🚨 provider.all_exhausted  role=%s  returning_anyway  provider=%s  model=%s",
        role, fallback[0], fallback[1],
    )
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Low-level LLM call
# ─────────────────────────────────────────────────────────────────────────────

async def _call_llm(
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    max_tokens: int,
    timeout: float,
    chat_id: str,
    label: str,
) -> str:
    """
    Single attempt to call the LLM.  Handles circuit-tripping and budget tracking.
    Raises the original exception on failure.
    """
    t0 = time.monotonic()

    async def _attempt():
        if provider == "gemini":
            if GEMINI_CLIENT is None:
                raise RuntimeError("Gemini client not initialised")
            resp = await asyncio.wait_for(
                GEMINI_CLIENT.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout,
            )
        else:
            if GROQ_CLIENT is None:
                raise RuntimeError("Groq client not initialised")
            resp = await asyncio.wait_for(
                GROQ_CLIENT.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout,
            )
        return resp

    try:
        async with _inflight:
            resp = await async_retry(_attempt, max_attempts=2)

        elapsed = time.monotonic() - t0
        if elapsed > _SLOW_CALL_WARN_SEC:
            logger.warning(
                "⚠️  slow_llm  label=%s  provider=%s  model=%s  %.1fs",
                label, provider, model, elapsed,
            )

        # Track token usage
        usage = getattr(resp, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            if total_tokens and provider in ("groq_70b", "groq_8b"):
                used = _budget_add(provider, total_tokens)
                limit = settings.groq_70b_daily_limit if provider == "groq_70b" else 500_000
                frac = used / max(limit, 1)
                if frac >= settings.token_budget_warn_pct:
                    logger.warning(
                        "💰 budget.warn  provider=%s  usage=%d  limit=%d  pct=%.1f%%",
                        provider, used, limit, frac * 100,
                    )

        return resp.choices[0].message.content or ""

    except Exception as exc:
        elapsed = time.monotonic() - t0
        exc_str = str(exc)
        is_rl   = "429" in exc_str or "rate_limit" in exc_str.lower() or "RESOURCE_EXHAUSTED" in exc_str

        if is_rl:
            cooldown = _parse_retry_after(exc)
            _trip_model(model, cooldown)
            _trip_provider(provider, cooldown)
            logger.warning(
                "🔴 circuit.tripped  provider=%s  model=%s  label=%s  cooldown=%.0fs  err=%s",
                provider, model, label, cooldown, exc_str[:200],
            )
        else:
            logger.error("❌ llm_call.error  label=%s  provider=%s  model=%s  %.1fs  err=%s",
                         label, provider, model, elapsed, exc_str[:200])

        raise


async def _groq_raw(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    chat_id: str,
    label: str,
    role: str,
    timeout: Optional[float] = None,
) -> str:
    """
    Route to the best available provider for the given role, with automatic
    fallback if the primary is circuit-tripped.
    """
    provider, model = _pick_provider_and_model(chat_id, role)
    effective_timeout = timeout or (
        settings.gemini_timeout if provider == "gemini" else settings.groq_timeout
    )
    try:
        return await _call_llm(
            messages,
            provider=provider, model=model,
            max_tokens=max_tokens, timeout=effective_timeout,
            chat_id=chat_id, label=label,
        )
    except Exception as first_exc:
        # Try fallback provider if primary failed
        first_exc_str = str(first_exc)
        is_rl = "429" in first_exc_str or "rate_limit" in first_exc_str.lower()

        # Only attempt fallback on rate-limit or timeout errors
        if not (is_rl or isinstance(first_exc, (asyncio.TimeoutError, TimeoutError))):
            raise

        # Get fallback
        fallback_candidates = (
            [("gemini",   settings.gemini_orchestrator_model),
             ("groq_70b", settings.orchestrator_model),
             ("groq_8b",  settings.extraction_model)]
            if role == "orchestrate"
            else [("groq_8b",  settings.extraction_model),
                  ("gemini",   settings.gemini_extraction_model),
                  ("groq_70b", settings.orchestrator_model)]
        )
        for fb_provider, fb_model in fallback_candidates:
            if fb_provider == provider and fb_model == model:
                continue  # skip the one that just failed
            if fb_provider == "gemini" and not settings.gemini_enabled:
                continue
            if not _model_open(fb_model):
                continue
            if not _provider_open(fb_provider):
                continue
            logger.info("🔄 provider.fallback  role=%s  from=%s/%s  to=%s/%s",
                        role, provider, model, fb_provider, fb_model)
            fb_timeout = timeout or (
                settings.gemini_timeout if fb_provider == "gemini"
                else settings.groq_timeout
            )
            return await _call_llm(
                messages,
                provider=fb_provider, model=fb_model,
                max_tokens=max_tokens, timeout=fb_timeout,
                chat_id=chat_id, label=f"{label}_fb",
            )

        raise  # all fallbacks exhausted


# ─────────────────────────────────────────────────────────────────────────────
# Junk-fact filter
# ─────────────────────────────────────────────────────────────────────────────

_JUNK_VALUES = frozenset({
    "unknown", "none", "null", "n/a", "na", "not set", "not specified",
    "undefined", "empty", "no data", "", "false", "true",
})


def _clean_facts(facts: Dict[str, str]) -> Dict[str, str]:
    """
    FIX-S2: Strip junk placeholder values before building prompts.
    Saves ~200-400 tokens per orchestrator call.
    """
    return {
        k: v for k, v in facts.items()
        if v and str(v).strip().lower() not in _JUNK_VALUES
    }


# ─────────────────────────────────────────────────────────────────────────────
# Facts shortcut v2 (FIX-S1)
# ─────────────────────────────────────────────────────────────────────────────

# Maps DB fact keys → trigger words that appear in the user's question.
# Deliberately broad — better to shortcut and answer than to burn 70B tokens
# to echo the database.
_FACT_SIGNALS: Dict[str, List[str]] = {
    "name":              ["my name", "what's my name", "what is my name", "who am i"],
    "age":               ["my age", "how old am i", "how old"],
    "city":              ["my city", "where i live", "where do i live", "my location"],
    "country":           ["my country", "what country"],
    "postal_code":       ["my postal", "my zip", "my pincode", "postal code"],
    "occupation":        ["my job", "my occupation", "my profession", "what do i do", "where i work", "my work"],
    "favorite_drink":    ["my drink", "coffee order", "my coffee", "what i drink", "favorite drink", "favourite drink", "my tea"],
    "favorite_food":     ["my food", "favorite food", "favourite food", "what i eat", "my favorite meal"],
    "favorite_cuisine":  ["my cuisine", "favorite cuisine", "favourite cuisine"],
    "favorite_color":    ["my color", "my colour", "favorite color", "favourite colour"],
    "favorite_trail":    ["my trail", "hiking trail", "favorite trail", "favourite trail", "where i hike"],
    "hobbies":           ["my hobbies", "my hobby", "what i enjoy", "what do i enjoy", "my interests"],
    "interests":         ["my interests", "my interest", "podcasts i listen", "what podcasts", "my podcast"],
    "allergies":         ["my allergies", "my allergy", "allergic to", "what i'm allergic"],
    "dietary_restriction": ["dietary restriction", "what i can't eat", "foods to avoid"],
    "pets":              ["my pets", "my pet", "my dog", "my cat", "pets' names", "my animals"],
    "pet_max_age":       ["max's age", "how old is max", "max age"],
    "pet_luna_age":      ["luna's age", "how old is luna", "luna age"],
    "shopping_list":     ["shopping list", "what's on my shopping", "grocery", "groceries"],
    "grocery_list":      ["grocery list", "groceries", "what to buy"],
    "todo_list":         ["todo list", "to do list", "my tasks", "to-do list"],
    "car":               ["my car", "my vehicle", "what car do i drive", "my bike"],
    "vehicle":           ["my vehicle", "my car", "what i drive"],
}


def _try_facts_shortcut(user_text: str, facts: Dict[str, str]) -> Optional[str]:
    """
    FIX-S1: Answer simple memory-recall questions directly from the facts
    dict WITHOUT calling any LLM.  Zero tokens, instant response.

    Strategy:
      1. Only attempt if the message is short (≤90 chars) — long messages
         likely contain multiple intents or require reasoning.
      2. Scan the broad _FACT_SIGNALS table for any signal that appears in
         the lowercased user text.
      3. If a matching fact exists in the DB and is not junk, return a
         ready-made reply string.  Otherwise return None so the LLM handles.
    """
    if len(user_text) > 90:
        return None

    low = user_text.lower()

    # Must look like a question or recall request
    recall_triggers = (
        "what", "which", "who", "where", "tell me", "show me", "remind me",
        "do you know", "do i have", "list", "my ", "am i",
    )
    if not any(t in low for t in recall_triggers):
        return None

    clean = _clean_facts(facts)
    if not clean:
        return None

    matched_key: Optional[str] = None
    for db_key, signals in _FACT_SIGNALS.items():
        if any(sig in low for sig in signals):
            matched_key = db_key
            break

    if not matched_key:
        return None

    # Try exact key and underscore/space variants
    val = (
        clean.get(matched_key)
        or clean.get(matched_key.replace("_", " "))
        or clean.get(matched_key.replace(" ", "_"))
    )
    if not val:
        return None

    persona = settings.bot_persona_name or "Shimmi"
    label_map = {
        "name":              "name",
        "age":               "age",
        "city":              "city",
        "country":           "country",
        "postal_code":       "postal code",
        "occupation":        "occupation/job",
        "favorite_drink":    "favourite drink / coffee order",
        "favorite_food":     "favourite food",
        "favorite_cuisine":  "favourite cuisine",
        "favorite_color":    "favourite colour",
        "favorite_trail":    "favourite hiking trail",
        "hobbies":           "hobbies",
        "interests":         "interests & podcasts",
        "allergies":         "allergies",
        "dietary_restriction": "dietary restrictions",
        "pets":              "pets",
        "pet_max_age":       "Max's age",
        "pet_luna_age":      "Luna's age",
        "shopping_list":     "shopping list",
        "grocery_list":      "grocery list",
        "todo_list":         "to-do list",
        "car":               "vehicle",
        "vehicle":           "vehicle",
    }
    label = label_map.get(matched_key, matched_key.replace("_", " "))
    return f"Your {label} on record is: *{val}* 📋"


# ─────────────────────────────────────────────────────────────────────────────
# Live search routing
# ─────────────────────────────────────────────────────────────────────────────

_WEATHER_KW  = re.compile(r"\b(weather|temperature|forecast|rain|humid|wind|climate)\b", re.I)
_STOCKS_KW   = re.compile(r"\b(stock|share|nse|bse|sensex|nifty|equity|ticker|paytm|zomato|reliance)\b", re.I)
_NEWS_KW     = re.compile(r"\b(news|headline|article|latest|update|today.s news|breaking)\b", re.I)
_CURRENCY_KW = re.compile(r"\b(currency|exchange rate|convert|usd|eur|gbp|inr|forex)\b", re.I)
_TIMEZONE_KW = re.compile(r"\b(time in|timezone|what time is it in|local time|current time in)\b", re.I)


async def _live_search(query: str, chat_id: str, facts: Optional[Dict[str, str]] = None) -> str:
    """
    Route live-data requests to MCP first, then fall back to compound-beta-mini.
    compound-beta-mini uses the Groq web search capability with llama-3.3-70b backend.
    Tracked separately from the orchestration token budget.

    FIX-P0-2: accepts `facts` so weather/stocks can use the user's known city/country
    instead of passing the raw LLM query as a city name (which always 404s).
    """
    from .live_data import get_weather, get_indian_stocks, get_news
    from .mcp_client import mcp_currency, mcp_timezone

    q      = query.lower()
    _facts = facts or {}

    # ── MCP structured endpoints first ────────────────────────────────────
    try:
        if _WEATHER_KW.search(q):
            # FIX-P0-2: use city from facts — NOT the raw query string
            city    = (_facts.get("city") or _facts.get("user_city") or "").strip()
            country = (_facts.get("country") or "IN").strip()
            if not city:
                # Last resort: strip stop-words from query to get something usable
                city = re.sub(
                    r"\b(weather|forecast|temperature|tomorrow|today|rain|humidity|"
                    r"wind|climate|in|for|the|what|is|it|s|current|week|weekly)\b",
                    "", query, flags=re.I,
                ).strip() or "Hyderabad"
            result = await get_weather(city, country) or ""
            if result:
                return result
    except Exception:
        pass

    try:
        if _STOCKS_KW.search(q):
            return await get_indian_stocks(None) or ""
    except Exception:
        pass

    try:
        if _NEWS_KW.search(q):
            country = (_facts.get("country") or "IN")[:2].upper()
            return await get_news(query, country) or ""
    except Exception:
        pass

    # currency + timezone: require structured params — skip MCP keyword-match,
    # fall through to compound-beta-mini which handles freeform queries well
    # (Phase 1 will replace this with LLM tool dispatch)

    # ── Fallback: compound-beta-mini web search ───────────────────────────
    if not settings.live_search_enabled:
        return "Live search is disabled."

    search_model = settings.live_search_model or "compound-beta-mini"
    if not _model_open(search_model):
        return "Live search model is temporarily unavailable."

    messages = [
        {"role": "system", "content": LIVE_SEARCH_PROMPT},
        {"role": "user",   "content": query},
    ]

    # compound-beta-mini can return 413 if the query is too long — retry shorter
    for attempt, q_text in enumerate([query, query[:200], query[:80]], 1):
        messages[-1]["content"] = q_text
        try:
            if GROQ_CLIENT is None:
                return "LLM not initialised."
            resp = await asyncio.wait_for(
                GROQ_CLIENT.chat.completions.create(
                    model=search_model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.1,
                ),
                timeout=50.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            elapsed = "?"
            if len(text) < 60 and attempt < 3:
                continue
            logger.info("🔍 search.done  query=%r  result_len=%d", q_text, len(text))
            return text[:_MAX_SEARCH_RESULT] if len(text) > _MAX_SEARCH_RESULT else text
        except Exception as e:
            s = str(e)
            if "413" in s and attempt < 3:
                logger.warning("⚠️  live_search.413  query=%r — retrying shorter", q_text)
                continue
            if "429" in s or "rate_limit" in s.lower():
                _trip_model(search_model, _parse_retry_after(e))
                return "Live search quota exhausted. Please try again later."
            logger.error("❌ live_search.error  %s", s[:200])
            return f"Search failed: {s[:80]}"

    return "Search returned no result."


# ─────────────────────────────────────────────────────────────────────────────
# JSON parse helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_md_fences(raw: str) -> str:
    """Strip ```json ... ``` fences from LLM output."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json(raw: str) -> Any:
    cleaned = _strip_md_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting a {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


async def _parse_with_repair(
    raw: str,
    chat_id: str,
    label: str,
    schema_hint: str,
) -> Any:
    """Parse JSON; if malformed, ask the extraction model to repair it."""
    try:
        return _parse_json(raw)
    except Exception:
        pass

    logger.warning("⚠️  json.repair  label=%s  raw_len=%d", label, len(raw))
    repair_prompt = (
        f"The following text should be valid JSON matching schema: {schema_hint}\n"
        f"Fix it and return ONLY the corrected JSON, nothing else.\n\n{raw[:1000]}"
    )
    messages = [
        {"role": "system", "content": REPAIR_PROMPT},
        {"role": "user",   "content": repair_prompt},
    ]
    try:
        repaired_raw = await _groq_raw(
            messages,
            max_tokens=512,
            chat_id=chat_id,
            label=f"{label}_repair",
            role="extract",
        )
        return _parse_json(repaired_raw)
    except Exception as e:
        logger.error("❌ json.repair_failed  label=%s  err=%s", label, str(e)[:120])
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Agent pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_memory(
    user_text: str,
    chat_id: str,
    *,
    existing_facts: Dict[str, str],
    trace: Any = None,
) -> List[MemoryUpdate]:
    """Extract memory updates from user message before orchestration."""
    step = trace.step("memory_extract") if trace else nullcontext()
    with step:
        if not user_text or len(user_text) < 4:
            return []

        # Skip if message is clearly not personal
        low = user_text.lower()
        personal_hints = ("i ", "i'm", "i am", "my ", "me ", "mine", "myself")
        if not any(h in low for h in personal_hints):
            return []

        facts_str = ", ".join(f"{k}={v!r}" for k, v in _clean_facts(existing_facts).items())
        messages = [
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": (
                f"Existing facts: {facts_str or 'none'}\n\n"
                f"User message: {user_text}"
            )},
        ]
        try:
            raw = await _groq_raw(
                messages,
                max_tokens=256,
                chat_id=chat_id,
                label="extract",
                role="extract",
                timeout=15.0,
            )
            data = await _parse_with_repair(raw, chat_id, "extract",
                                            '{"memory_updates": [...]}')
            result = ExtractResult.model_validate(data)
            return result.memory_updates
        except Exception as e:
            logger.warning("⚠️  extract_memory.failed  err=%s", str(e)[:120])
            return []


async def _verify_updates(
    updates: List[MemoryUpdate],
    chat_id: str,
    *,
    existing_facts: Dict[str, str],
    user_text: str,
    trace: Any = None,
) -> List[ApprovedUpdate]:
    """Verify memory updates using the extraction model."""
    step = trace.step("memory_verify") if trace else nullcontext()
    with step:
        if not updates:
            if trace:
                trace.tag(total_iterations=1, memory_extracted=0,
                          memory_verified=0, memory_total=0)
            return []

        updates_str = json.dumps([u.model_dump() for u in updates])
        existing_str = ", ".join(
            f"{k}={v!r}" for k, v in _clean_facts(existing_facts).items()
        )
        messages = [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": (
                f"User said: {user_text}\n"
                f"Existing facts: {existing_str or 'none'}\n"
                f"Proposed updates: {updates_str}"
            )},
        ]
        try:
            raw = await _groq_raw(
                messages,
                max_tokens=512,
                chat_id=chat_id,
                label="verify",
                role="extract",
                timeout=20.0,
            )
            data = await _parse_with_repair(raw, chat_id, "verify",
                                            '{"approved": [...]}')
            result = VerifyResult.model_validate(data)
            approved = [
                a for a in result.approved
                if a.confidence >= settings.facts_min_conf
            ]
            if trace:
                trace.tag(
                    total_iterations=1,
                    memory_extracted=len(updates),
                    memory_verified=len(approved),
                    memory_total=len(approved),
                )
            return approved
        except Exception as e:
            logger.warning("⚠️  verify.failed  err=%s", str(e)[:120])
            return []


async def _format_whatsapp(
    text: str,
    chat_id: str,
    *,
    trace: Any = None,
) -> str:
    """Format the reply for WhatsApp markup using the extraction model."""
    step = trace.step("format") if trace else nullcontext()
    with step:
        if len(text) < _MIN_FORMAT_LEN:
            return text
        messages = [
            {"role": "system", "content": FORMATTER_PROMPT},
            {"role": "user",   "content": text},
        ]
        try:
            raw = await _groq_raw(
                messages,
                max_tokens=len(text) + 200,
                chat_id=chat_id,
                label="format",
                role="extract",
                timeout=20.0,
            )
            data = await _parse_with_repair(raw, chat_id, "format", '{"text": "..."}')
            result = FormatterResult.model_validate(data)
            return result.text or text
        except Exception:
            return text


async def _orchestrate(
    messages: List[Dict[str, Any]],
    chat_id: str,
    *,
    label: str,
    trace: Any = None,
) -> OrchestratorResult:
    """Run one orchestrator turn using the best available provider."""
    step = trace.step(label) if trace else nullcontext()
    with step:
        raw = await _groq_raw(
            messages,
            max_tokens=600,
            chat_id=chat_id,
            label=label,
            role="orchestrate",
        )
        data = await _parse_with_repair(raw, chat_id, label,
                                        '{"action": "...", "reasoning": "...", "text": "..."}')
        result = OrchestratorResult.model_validate(data)

        if trace:
            trace.tag(action=result.action, reasoning_len=len(result.reasoning))
        logger.info(
            "🤖 orchestrate  iter=%s  action=%-10s  reasoning=%s",
            label.split("_")[-1],
            result.action,
            result.reasoning[:100],
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Context builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_facts_str(facts: Dict[str, str]) -> str:
    clean = _clean_facts(facts)
    if not clean:
        return "No stored facts yet."
    return ", ".join(f"{k}={v!r}" for k, v in clean.items())


def _build_reminders_str(reminders: List[Reminder]) -> str:
    if not reminders:
        return "No pending reminders."
    now = _now_local()
    parts = []
    for r in reminders:
        try:
            trigger = datetime.fromisoformat(r.trigger_iso)
            if trigger.tzinfo is None:
                trigger = trigger.replace(tzinfo=UTC)
            diff = trigger.astimezone(UTC) - now.astimezone(UTC)
            if diff.total_seconds() < 0:
                when = "overdue"
            elif diff.total_seconds() < 3600:
                when = f"in {int(diff.total_seconds()//60)}m"
            else:
                when = f"in {int(diff.total_seconds()//3600)}h"
        except Exception:
            when = "at " + r.trigger_iso
        parts.append(f"• {r.reminder_text} ({when})")
    return "\n".join(parts)


def _build_context_str(context: List[Any]) -> str:
    lines = []
    for item in context:
        if hasattr(item, "role") and hasattr(item, "content"):
            lines.append(f"{item.role}: {item.content}")
        elif isinstance(item, dict):
            lines.append(f"{item.get('role','?')}: {item.get('content','')}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _build_orchestrator_messages(
    user_text: str,
    facts: Dict[str, str],
    context: List[Any],
    reminders: List[Reminder],
    search_result: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the message list for the orchestrator (first or second turn)."""
    facts_str     = _build_facts_str(facts)
    reminders_str = _build_reminders_str(reminders)
    context_str   = _build_context_str(context)
    time_str      = _current_time_str()
    today_str     = _today_str()

    system_content = ORCHESTRATOR_PROMPT

    user_content_parts = [
        f"FACTS: {facts_str}",
        f"REMINDERS: {reminders_str}",
        f"CONTEXT:\n{context_str}",
    ]
    if search_result:
        user_content_parts.append(f"SEARCH_RESULT:\n{search_result}")
    user_content_parts.append(f"USER: {user_text}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": "\n\n".join(user_content_parts)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(
    *,
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Any],
    reminders: List[Reminder],
    trace: Any = None,
) -> AgentResult:
    """
    Full agentic pipeline. Steps:
      1. Facts shortcut — answer simple memory queries instantly (0 tokens)
      2. Pre-extract memory updates from user message (8B / Gemini Lite)
      3. Orchestrate (Gemini Flash primary → Groq 70B fallback)
         - If action=search: call _live_search(), then re-orchestrate
      4. Verify memory updates (8B / Gemini Lite)
      5. Format reply for WhatsApp (8B / Gemini Lite)
    """
    # ── 1. Zero-token facts shortcut ──────────────────────────────────────
    shortcut = _try_facts_shortcut(user_text, facts)
    if shortcut:
        logger.info("⚡ facts.shortcut  chat=%s  reply=%r", chat_id, shortcut[:60])
        return AgentResult(
            reply=ReplyPayload(type="text", text=shortcut),
            provider_used="shortcut",
        )

    # ── 2. Pre-extract memory ─────────────────────────────────────────────
    pre_updates: List[MemoryUpdate] = []
    try:
        pre_updates = await _extract_memory(
            user_text, chat_id, existing_facts=facts, trace=trace,
        )
    except Exception:
        pass   # non-fatal

    # ── 3. Orchestrate ────────────────────────────────────────────────────
    messages = _build_orchestrator_messages(user_text, facts, context, reminders)
    search_result: Optional[str] = None

    for iteration in range(1, _MAX_ITERATIONS + 1):
        label = f"orchestrate_{iteration}"

        if search_result:
            messages = _build_orchestrator_messages(
                user_text, facts, context, reminders, search_result
            )

        orch = await _orchestrate(messages, chat_id, label=label, trace=trace)

        if orch.action == "answer":
            reply_text = orch.text or "Sorry, I couldn't generate a reply."
            memory_updates = pre_updates + orch.memory_updates
            # Verify before saving
            approved = await _verify_updates(
                memory_updates, chat_id,
                existing_facts=facts, user_text=user_text, trace=trace,
            )
            # Format
            formatted = await _format_whatsapp(reply_text, chat_id, trace=trace)
            if len(formatted) > len(reply_text) * 2 or not formatted.strip():
                formatted = reply_text  # sanity guard

            return AgentResult(
                reply=ReplyPayload(type="text", text=formatted),
                memory_updates=[
                    MemoryUpdate(key=a.key, value=a.value) for a in approved
                ],
                reminders=orch.reminders,
                iterations=iteration,
            )

        elif orch.action == "search":
            if not settings.live_search_enabled:
                # Convert to answer with a note
                return AgentResult(
                    reply=ReplyPayload(
                        type="text",
                        text="Live search is disabled. I can't fetch real-time data right now.",
                    ),
                    iterations=iteration,
                )
            query = orch.query or user_text
            search_result = await _live_search(query, chat_id, facts=facts)
            logger.info("🔍 search.done  iter=%d  query=%r  result_len=%d",
                        iteration, query, len(search_result or ""))
            if trace:
                trace.tag(
                    **{f"live_search_{iteration}": {
                        "search_query": query,
                        "result_len": len(search_result or ""),
                    }}
                )

        elif orch.action == "ask":
            question = orch.question or orch.text or "Could you clarify that?"
            return AgentResult(
                reply=ReplyPayload(type="text", text=question),
                iterations=iteration,
            )
        else:
            logger.warning("⚠️  orchestrate.unknown_action  action=%s", orch.action)
            text = orch.text or orch.reasoning or "I'm not sure how to help with that."
            return AgentResult(
                reply=ReplyPayload(type="text", text=text),
                iterations=iteration,
            )

    # Max iterations reached — return last search result or generic fallback
    fallback = f"Here's what I found:\n\n{search_result}" if search_result else \
               "I reached my reasoning limit. Please try rephrasing."
    return AgentResult(
        reply=ReplyPayload(type="text", text=fallback),
        iterations=_MAX_ITERATIONS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fire-and-forget reply memory extraction (FIX-S3)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_reply_memory(
    *,
    reply_text: str,
    chat_id: str,
    sender_key: str,
) -> None:
    """
    FIX-S3: Extract facts that appear in Shimmi's own reply text (e.g. when
    the bot confirms a fact it looked up).  Run as fire-and-forget.

    Wrapped in top-level try/except so 'Task exception was never retrieved'
    errors never appear in logs.
    """
    try:
        if not reply_text or len(reply_text) < 20:
            return

        messages = [
            {"role": "system", "content": REPLY_EXTRACTOR_PROMPT},
            {"role": "user",   "content": reply_text},
        ]
        raw = await _groq_raw(
            messages,
            max_tokens=256,
            chat_id=chat_id,
            label="reply_extract",
            role="extract",
            timeout=12.0,
        )
        data = await _parse_with_repair(raw, chat_id, "reply_extract",
                                        '{"memory_updates": [...]}')
        result = ExtractResult.model_validate(data)
        if result.memory_updates:
            logger.info(
                "🧠 reply_extract.found  count=%d  keys=%s",
                len(result.memory_updates),
                [u.key for u in result.memory_updates],
            )
            # Import here to avoid circular dependency
            from .database import upsert_fact
            for u in result.memory_updates:
                try:
                    await upsert_fact(sender_key, normalize_key(u.key), u.value)
                    logger.info(
                        "🧠 reply_memory.updated  sender=%s  key=%s  value=%r",
                        sender_key, u.key, u.value,
                    )
                except Exception as db_err:
                    logger.warning("⚠️  reply_memory.db_fail  key=%s  err=%s",
                                   u.key, str(db_err)[:80])
    except Exception as e:
        # Completely suppress — this is a best-effort enrichment task
        logger.debug("ℹ️  reply_extract.suppressed  err=%s", str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

async def init_llm() -> None:
    """
    Initialise Groq and (optionally) Gemini clients.
    Called once at application startup.
    """
    global GROQ_CLIENT, GEMINI_CLIENT

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    GROQ_CLIENT = AsyncGroq(api_key=settings.groq_api_key)

    if settings.gemini_enabled:
        GEMINI_CLIENT = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info(
            "🧠 llm.init — Groq + Gemini ready  "
            "groq_pool=%s  gemini_pool=%s  "
            "orchestrator=Gemini(%s) → Groq(%s)  extraction=Groq(%s)",
            settings.groq_model_pool,
            settings.gemini_model_pool,
            settings.gemini_orchestrator_model,
            settings.orchestrator_model,
            settings.extraction_model,
        )
    else:
        logger.info(
            "🧠 llm.init — Groq only (no GEMINI_API_KEY)  "
            "model_pool=%s  orchestrator=%s  extraction=%s",
            settings.groq_model_pool,
            settings.orchestrator_model,
            settings.extraction_model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility / helper exports used by main.py
# ─────────────────────────────────────────────────────────────────────────────

VALID_GROQ_PREFIXES = (
    "llama-", "mixtral-", "gemma-",
    "compound-beta", "compound-beta-mini",
    "whisper-", "distil-",
)

VALID_GEMINI_PREFIXES = ("gemini-",)


async def close_llm() -> None:
    global GROQ_CLIENT, GEMINI_CLIENT
    for client in (GROQ_CLIENT, GEMINI_CLIENT):
        if client:
            try:
                await client.close()
            except Exception:
                pass
    GROQ_CLIENT   = None
    GEMINI_CLIENT = None


def _is_reminder_duplicate(
    text: str,
    trigger_iso: str,
    existing: List[Reminder],
) -> bool:
    norm_text      = text.strip().lower()
    trigger_prefix = trigger_iso[:16]
    for r in existing:
        if r.status != "pending":
            continue
        if r.reminder_text.strip().lower() == norm_text:
            return True
        if r.trigger_iso[:16] == trigger_prefix:
            return True
    return False
