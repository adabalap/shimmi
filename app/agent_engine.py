"""
agent_engine.py — Shimmi Phase 1

Changes vs Phase 0 (v3.0.3):

  P1-FEAT-1: Structured tool dispatch
    • _live_search() removed.  Replaced by tools.ToolDispatcher.dispatch().
    • Orchestrator prompt now contains tool schemas (injected via
      tool_schemas_json()) and is instructed to output a `tool_call` JSON
      object alongside action=search.
    • OrchestratorResult gains a `tool_call` field (raw dict).
    • run_agent() calls parse_tool_call() then dispatcher.dispatch() —
      no more keyword-regex routing.

  P1-FEAT-2: Memory deletion
    • MemoryUpdate.value min_length relaxed to 0 (empty = delete intent).
    • delete=True flag added to MemoryUpdate; orchestrator/extractor can emit
      {"key":"car","value":"","delete":true}.
    • OrchestratorResult._clean_memory accepts delete entries.
    • Verifier prompt updated to accept deletion intent.
    • run_agent() routes delete updates to database.delete_fact().
    • Prompts updated: ORCHESTRATOR_PROMPT, MEMORY_EXTRACTOR_PROMPT,
      VERIFIER_PROMPT gain deletion examples.

  P1-FEAT-3: Gemini RPM rate limiter (ISSUE-6)
    • _GeminiRPMLimiter class — rolling 60-second window counter.
    • Capped at settings.gemini_rpm_limit (default 12).
    • _pick_provider_and_model() consults limiter before selecting Gemini.
    • When limiter is full, Gemini is skipped and Groq 70B is used instead.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import re
import time
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from groq import AsyncGroq
from openai import AsyncOpenAI

from .config import settings
from .retry import async_retry
from .prompts import (
    ORCHESTRATOR_PROMPT_P1,
    MEMORY_EXTRACTOR_PROMPT_P1,
    REPLY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT_P1,
    REPAIR_PROMPT,
    FORMATTER_PROMPT,
    LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp
from .database import normalize_key, Reminder
from .tools import tool_schemas_json, parse_tool_call, dispatcher as tool_dispatcher

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
    key:    str  = Field(..., min_length=1)
    value:  str  = Field(default="")          # P1-FEAT-2: empty = delete
    delete: bool = Field(default=False)       # P1-FEAT-2: explicit delete flag

    @field_validator("key", "value", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        return "" if v is None else str(v).strip()

    @field_validator("delete", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    @property
    def is_delete(self) -> bool:
        """True if this update should delete the key rather than upsert it."""
        return self.delete or not self.value


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
    provider_used:  str = ""


class OrchestratorResult(BaseModel):
    action:         str
    reasoning:      str              = ""
    text:           str              = ""
    query:          str              = ""
    question:       str              = ""
    tool_call:      Optional[Any]    = None   # P1-FEAT-1: raw dict from LLM
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
            k      = str(item.get("key",   "") or "").strip()
            val    = str(item.get("value", "") or "").strip()
            delete = item.get("delete", False)
            if k:
                # P1-FEAT-2: accept delete entries (value may be "")
                if val or delete or not val:
                    out.append({"key": k, "value": val, "delete": delete})
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
    delete:     bool  = False            # P1-FEAT-2
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
            delete = item.get("delete", False)
            if k:
                out.append({"key": k, "value": val, "delete": delete})
        return out


class FormatterResult(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Timezone helpers (unchanged)
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
# P1-FEAT-3: Gemini RPM rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class _GeminiRPMLimiter:
    """
    Rolling-window RPM limiter for Gemini.

    Tracks request timestamps in a deque.  Before each Gemini call,
    _pick_provider_and_model() calls .allow() — returns False if the
    count of requests in the past 60 seconds >= limit.
    """

    def __init__(self, limit: int = 12, window_sec: float = 60.0):
        self._limit      = limit
        self._window_sec = window_sec
        self._timestamps: Deque[float] = collections.deque()

    @property
    def limit(self) -> int:
        return self._limit

    @limit.setter
    def limit(self, v: int) -> None:
        self._limit = max(1, v)

    def allow(self) -> bool:
        """Return True and record the request if under the RPM cap, else False."""
        now = time.monotonic()
        # Prune stale entries
        while self._timestamps and now - self._timestamps[0] > self._window_sec:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._limit:
            return False
        self._timestamps.append(now)
        return True

    def current_rpm(self) -> int:
        """Return the number of Gemini requests in the last 60 s."""
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > self._window_sec:
            self._timestamps.popleft()
        return len(self._timestamps)


# Module-level limiter — limit read from settings at first call (settings may
# not be fully populated at import time).
_gemini_rpm_limiter = _GeminiRPMLimiter(limit=12)


def _update_gemini_rpm_limit() -> None:
    """Sync limiter limit from settings (called lazily)."""
    try:
        lim = int(settings.gemini_rpm_limit or 12)
        _gemini_rpm_limiter.limit = lim
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Multi-provider LLM clients + circuit breakers + token budget (mostly unchanged)
# ─────────────────────────────────────────────────────────────────────────────

GROQ_CLIENT:   Optional[AsyncGroq]    = None
GEMINI_CLIENT: Optional[AsyncOpenAI]  = None

MODEL_CIRCUIT:    Dict[str, float] = {}
PROVIDER_CIRCUIT: Dict[str, float] = {}

STICKY_MODEL: Dict[str, str] = {}
_STICKY_MAX = 2_000

_TOKEN_BUDGET: Dict[str, Tuple[int, float]] = {}
_BUDGET_RESET_INTERVAL = 86_400.0

_inflight = asyncio.Semaphore(int(settings.groq_max_inflight or 5))


def _budget_add(provider: str, tokens: int) -> int:
    now = time.time()
    used, reset_ts = _TOKEN_BUDGET.get(provider, (0, now + _BUDGET_RESET_INTERVAL))
    if now >= reset_ts:
        used = 0
        reset_ts = now + _BUDGET_RESET_INTERVAL
    used += tokens
    _TOKEN_BUDGET[provider] = (used, reset_ts)
    return used


def _budget_fraction(provider: str, daily_limit: int) -> float:
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
    s = str(exc)
    m = re.search(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", s)
    if m:
        h = float(m.group(1) or 0)
        mn = float(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        total = h * 3600 + mn * 60 + sec
        return max(60.0, min(total + 10.0, 7200.0))
    m2 = re.search(r"retry[^\d]*(\d+)\s*second", s, re.IGNORECASE)
    if m2:
        return max(60.0, float(m2.group(1)) + 10.0)
    return 300.0


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def _pick_provider_and_model(chat_id: str, role: str) -> Tuple[str, str]:
    """
    Route to the best available provider+model.

    P1-FEAT-3: Gemini is skipped when the rolling RPM limiter is full
    (i.e. ≥ gemini_rpm_limit requests in the past 60 seconds).
    """
    _update_gemini_rpm_limit()

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
        if provider == "gemini" and not settings.gemini_enabled:
            continue
        if not _model_open(model):
            continue
        if not _provider_open(provider):
            continue
        # P1-FEAT-3: Gemini RPM guard
        if provider == "gemini":
            if not _gemini_rpm_limiter.allow():
                logger.warning(
                    "🚦 gemini.rpm_limit  current=%d/min  skipping_gemini",
                    _gemini_rpm_limiter.current_rpm(),
                )
                continue
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

    fallback = candidates[0]
    logger.error(
        "🚨 provider.all_exhausted  role=%s  returning_anyway  provider=%s  model=%s",
        role, fallback[0], fallback[1],
    )
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Low-level LLM call (unchanged from Phase 0)
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
            elapsed = time.monotonic() - t0
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
        first_exc_str = str(first_exc)
        is_rl = "429" in first_exc_str or "rate_limit" in first_exc_str.lower()

        if not (is_rl or isinstance(first_exc, (asyncio.TimeoutError, TimeoutError))):
            raise

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
                continue
            if fb_provider == "gemini" and not settings.gemini_enabled:
                continue
            if not _model_open(fb_model):
                continue
            if not _provider_open(fb_provider):
                continue
            # P1-FEAT-3: also check RPM on fallback path
            if fb_provider == "gemini" and not _gemini_rpm_limiter.allow():
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

        raise


# ─────────────────────────────────────────────────────────────────────────────
# Junk-fact filter (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_JUNK_VALUES = frozenset({
    "unknown", "none", "null", "n/a", "na", "not set", "not specified",
    "undefined", "empty", "no data", "", "false", "true",
})


def _clean_facts(facts: Dict[str, str]) -> Dict[str, str]:
    return {
        k: v for k, v in facts.items()
        if v and str(v).strip().lower() not in _JUNK_VALUES
    }


# ─────────────────────────────────────────────────────────────────────────────
# Facts shortcut v2 (unchanged from Phase 0)
# ─────────────────────────────────────────────────────────────────────────────

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
    "shopping_list":     ["shopping list", "what's on my shopping", "grocery", "groceries"],
    "grocery_list":      ["grocery list", "groceries", "what to buy"],
    "todo_list":         ["todo list", "to do list", "my tasks", "to-do list"],
    "car":               ["my car", "my vehicle", "what car do i drive", "my bike"],
    "vehicle":           ["my vehicle", "my car", "what i drive"],
}


def _try_facts_shortcut(user_text: str, facts: Dict[str, str]) -> Optional[str]:
    if len(user_text) > 90:
        return None
    low = user_text.lower()
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
        "shopping_list":     "shopping list",
        "grocery_list":      "grocery list",
        "todo_list":         "to-do list",
        "car":               "vehicle",
        "vehicle":           "vehicle",
    }
    label = label_map.get(matched_key, matched_key.replace("_", " "))
    return f"Your {label} on record is: *{val}* 📋"


# ─────────────────────────────────────────────────────────────────────────────
# JSON parse helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_md_fences(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json(raw: str) -> Any:
    cleaned = _strip_md_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
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
    step = trace.step("memory_extract") if trace else nullcontext()
    with step:
        if not user_text or len(user_text) < 4:
            return []
        low = user_text.lower()
        personal_hints = ("i ", "i'm", "i am", "my ", "me ", "mine", "myself", "forget", "remove", "delete")
        if not any(h in low for h in personal_hints):
            return []
        facts_str = ", ".join(f"{k}={v!r}" for k, v in _clean_facts(existing_facts).items())
        messages = [
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT_P1},
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
    step = trace.step("memory_verify") if trace else nullcontext()
    with step:
        if not updates:
            if trace:
                trace.tag(total_iterations=1, memory_extracted=0,
                          memory_verified=0, memory_total=0)
            return []

        updates_str  = json.dumps([u.model_dump() for u in updates])
        existing_str = ", ".join(
            f"{k}={v!r}" for k, v in _clean_facts(existing_facts).items()
        )
        messages = [
            {"role": "system", "content": VERIFIER_PROMPT_P1},
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
    step = trace.step(label) if trace else nullcontext()
    with step:
        raw = await _groq_raw(
            messages,
            max_tokens=700,                # +100 for tool_call field
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
            when = "overdue" if diff.total_seconds() < 0 else (
                f"in {int(diff.total_seconds()//60)}m"
                if diff.total_seconds() < 3600
                else f"in {int(diff.total_seconds()//3600)}h"
            )
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
    """Build orchestrator message list.

    P1-FEAT-1: Uses ORCHESTRATOR_PROMPT_P1 which includes tool schemas.
    """
    facts_str     = _build_facts_str(facts)
    reminders_str = _build_reminders_str(reminders)
    context_str   = _build_context_str(context)

    user_content_parts = [
        f"FACTS: {facts_str}",
        f"REMINDERS: {reminders_str}",
        f"CONTEXT:\n{context_str}",
    ]
    if search_result:
        user_content_parts.append(f"SEARCH_RESULT:\n{search_result}")
    user_content_parts.append(f"USER: {user_text}")

    return [
        {"role": "system", "content": ORCHESTRATOR_PROMPT_P1},
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
    Full agentic pipeline (Phase 1).

    Steps:
      1. Facts shortcut — zero-token fast path for simple recall
      2. Pre-extract memory from user message (8B / Gemini Lite)
      3. Orchestrate (Gemini Flash → Groq 70B):
           - action=search: parse tool_call → ToolDispatcher.dispatch() → re-orchestrate
           - action=answer: proceed to verify + format
      4. Verify memory updates (8B)
      5. Format reply for WhatsApp (8B)
      6. Route deletes vs upserts in memory_updates
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
        pass

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
                formatted = reply_text

            return AgentResult(
                reply=ReplyPayload(type="text", text=formatted),
                memory_updates=[
                    MemoryUpdate(key=a.key, value=a.value, delete=a.delete)
                    for a in approved
                ],
                reminders=orch.reminders,
                iterations=iteration,
            )

        elif orch.action == "search":
            if not settings.live_search_enabled:
                return AgentResult(
                    reply=ReplyPayload(
                        type="text",
                        text="Live search is disabled. I can't fetch real-time data right now.",
                    ),
                    iterations=iteration,
                )

            # P1-FEAT-1: parse LLM tool_call → typed ToolCall → dispatch
            query = orch.query or user_text
            tc = parse_tool_call(
                orch.tool_call,
                fallback_query=query,
                facts=facts,
            )
            logger.info(
                "🔧 tool.selected  tool=%s  chat=%s",
                getattr(tc, "tool", "?"), chat_id,
            )
            search_result = await tool_dispatcher.dispatch(
                tc,
                groq_client=GROQ_CLIENT,
                live_search_model=settings.live_search_model or "compound-beta-mini",
                live_search_enabled=settings.live_search_enabled,
            )
            logger.info("🔍 search.done  iter=%d  tool=%s  result_len=%d",
                        iteration, getattr(tc, "tool", "?"), len(search_result or ""))
            if trace:
                trace.tag(**{f"live_search_{iteration}": {
                    "tool":       getattr(tc, "tool", "?"),
                    "result_len": len(search_result or ""),
                }})

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

    fallback = f"Here's what I found:\n\n{search_result}" if search_result else \
               "I reached my reasoning limit. Please try rephrasing."
    return AgentResult(
        reply=ReplyPayload(type="text", text=fallback),
        iterations=_MAX_ITERATIONS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fire-and-forget reply memory extraction (unchanged from Phase 0)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_reply_memory(
    *,
    reply_text: str,
    chat_id: str,
    sender_key: str,
) -> None:
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
            from .database import upsert_fact
            for u in result.memory_updates:
                try:
                    await upsert_fact(sender_key, normalize_key(u.key), u.value)
                except Exception as db_err:
                    logger.warning("⚠️  reply_memory.db_fail  key=%s  err=%s",
                                   u.key, str(db_err)[:80])
    except Exception as e:
        logger.debug("ℹ️  reply_extract.suppressed  err=%s", str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

async def init_llm() -> None:
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
            "🧠 llm.init — Groq + Gemini ready  gemini_rpm_limit=%d  "
            "orchestrator=Gemini(%s) → Groq(%s)  extraction=Groq(%s)",
            _gemini_rpm_limiter.limit,
            settings.gemini_orchestrator_model,
            settings.orchestrator_model,
            settings.extraction_model,
        )
    else:
        logger.info(
            "🧠 llm.init — Groq only  orchestrator=%s  extraction=%s",
            settings.orchestrator_model,
            settings.extraction_model,
        )


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


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility exports used by main.py
# ─────────────────────────────────────────────────────────────────────────────

VALID_GROQ_PREFIXES = (
    "llama-", "mixtral-", "gemma-",
    "compound-beta", "compound-beta-mini",
    "whisper-", "distil-",
)

VALID_GEMINI_PREFIXES = ("gemini-",)


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
