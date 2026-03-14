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
    KEY_CONSOLIDATION_PROMPT,
)
from .utils import sanitize_for_whatsapp
from .database import normalize_key, Reminder

logger = logging.getLogger("app.agent")
UTC    = timezone.utc

_SLOW_CALL_WARN_SEC  = 5.0
_MAX_ITERATIONS      = 3
_MAX_SEARCH_RESULT   = 1_200
_MIN_FORMAT_LEN      = 120

# Consolidation runs at most once per hour per user — prevents burning 8B quota
# on every single message when the user's facts are already clean.
_CONSOLIDATION_COOLDOWN_SEC: float = 3_600.0   # 1 hour
_CONSOLIDATION_LAST_RUN: Dict[str, float] = {}  # whatsapp_id → monotonic timestamp

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class MemoryUpdate(BaseModel):
    key:     str  = Field(..., min_length=1)
    value:   str  = Field(default="")   # empty string + delete=True → delete the key
    delete:  bool = Field(default=False, description="P1-FEAT-2: if True, delete this key from DB")
    confirm: bool = Field(
        default=False,
        description=(
            "P1-GUARD: must be True for high-stakes list deletions "
            "(shopping_list, grocery_list, todo_list). "
            "Only set True after the user has explicitly confirmed (e.g. 'yes, clear it')."
        ),
    )

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
    # FIX-B1: Optional — LLM sometimes returns null/missing on action=answer,
    # which previously caused a hard ValidationError crash (no reply sent to user).
    question:       Optional[str]    = ""
    memory_updates: List[MemoryUpdate]  = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)
    # P1-FEAT-1: LLM-decided tool dispatch — replaces keyword regex in _live_search
    tool_call:      Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("question", mode="before")
    @classmethod
    def _coerce_question(cls, v):
        """Coerce None → empty string so the field is never None at runtime."""
        return "" if v is None else str(v)

    @field_validator("memory_updates", mode="before")
    @classmethod
    def _clean_memory(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k       = str(item.get("key",    "") or "").strip()
            val     = str(item.get("value",  "") or "").strip()
            delete  = bool(item.get("delete",  False))
            confirm = bool(item.get("confirm", False))
            if k and (val or delete):
                out.append({"key": k, "value": val, "delete": delete, "confirm": confirm})
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
    delete:     bool  = Field(default=False)
    confirm:    bool  = Field(default=False)


class VerifyResult(BaseModel):
    approved: List[ApprovedUpdate] = Field(default_factory=list)

    @field_validator("approved", mode="before")
    @classmethod
    def _clean_approved(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k          = str(item.get("key",        "") or "").strip()
            val        = str(item.get("value",       "") or "").strip()
            conf_raw   = item.get("confidence", 0.0)
            delete     = bool(item.get("delete",  False))
            confirm    = bool(item.get("confirm", False))
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = 0.0
            if k and (val or delete):
                out.append({
                    "key": k, "value": val,
                    "confidence": confidence,
                    "delete": delete, "confirm": confirm,
                })
        return out


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
            k       = str(item.get("key",    "") or "").strip()
            val     = str(item.get("value",  "") or "").strip()
            delete  = bool(item.get("delete",  False))
            confirm = bool(item.get("confirm", False))
            if k and (val or delete):
                out.append({"key": k, "value": val, "delete": delete, "confirm": confirm})
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
    already_tripped = not _provider_open(provider)
    PROVIDER_CIRCUIT[provider] = time.monotonic() + cooldown
    if not already_tripped:
        # First time tripping this provider — log at WARNING so it's visible
        logger.warning(
            "🔴 provider.circuit_tripped  provider=%s  cooldown=%.0fs", provider, cooldown
        )
    # (subsequent re-trips are silently absorbed — _call_llm logs circuit.tripped already)


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
        # FIX-B5: Gemini free tier quota exhausts immediately in practice
        # (trips on the very first message each session). Groq 70b is now primary
        # for orchestration; Gemini is kept as a fallback so it absorbs overflow
        # once Groq 70b hits its own daily budget ceiling.
        candidates = [
            ("groq_70b", settings.orchestrator_model),
            ("gemini",   settings.gemini_orchestrator_model),
            ("groq_8b",  settings.extraction_model),
        ]
    else:
        candidates = [
            ("groq_8b",  settings.extraction_model),
            ("gemini",   settings.gemini_extraction_model),
            ("groq_70b", settings.orchestrator_model),
        ]

    gemini_skipped = False
    for provider, model in candidates:
        # Skip Gemini if no key configured
        if provider == "gemini" and not settings.gemini_enabled:
            continue
        # Check model-level circuit
        if not _model_open(model):
            if provider == "gemini":
                gemini_skipped = True
            continue
        # Check provider-level circuit
        if not _provider_open(provider):
            if provider == "gemini" and not gemini_skipped:
                gemini_skipped = True
                reopen_in = max(0, PROVIDER_CIRCUIT.get("gemini", 0) - time.monotonic())
                logger.info(
                    "⏭️  gemini.rate_limited  routing=groq  reopen_in=%.0fs",
                    reopen_in,
                )
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
            # Strip verbose URLs from the error string (e.g. Gemini quota messages)
            err_short = re.sub(r"https?://\S+", "", exc_str).strip()[:160]
            _trip_model(model, cooldown)
            _trip_provider(provider, cooldown)
            # Demote to INFO — the fallback handles this cleanly, it's not an error
            logger.info(
                "⚡ rate_limit  provider=%s  model=%s  label=%s  cooldown=%.0fs  fallback=auto  err=%s",
                provider, model, label, cooldown, err_short,
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

        # Get fallback — same priority order as _pick_provider_and_model
        fallback_candidates = (
            [("groq_70b", settings.orchestrator_model),
             ("gemini",   settings.gemini_orchestrator_model),
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
# P1-GUARD: Pending-delete confirmation cache
# ─────────────────────────────────────────────────────────────────────────────

# Maps  f"{sender_key}:{key}"  →  (original_value: str, queued_at: float)
# An entry here means the agent has asked the user to confirm deletion of
# a high-stakes key (shopping_list / grocery_list / todo_list) and is waiting
# for a yes/no reply.  Entries expire after _PENDING_DELETE_TTL seconds.
_PENDING_DELETES: Dict[str, tuple[str, float]] = {}
_PENDING_DELETE_TTL = 120.0   # 2 minutes to confirm

# Keywords that constitute user confirmation of a pending delete
_CONFIRM_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "go ahead", "confirm", "do it", "yes please", "definitely",
    "absolutely", "clear it", "delete it", "remove it",
})
# Keywords that cancel a pending delete
_CANCEL_WORDS  = frozenset({
    "no", "nope", "nah", "cancel", "stop", "keep it", "never mind",
    "nevermind", "don't", "dont",
})


def _register_pending_delete(sender_key: str, key: str, current_value: str) -> None:
    """Record that we've asked the user to confirm deletion of key."""
    cache_key = f"{sender_key}:{key}"
    _PENDING_DELETES[cache_key] = (current_value, time.monotonic())
    logger.info("⏳ pending_delete.registered  sender=%s  key=%s", sender_key, key)


def _check_pending_delete(sender_key: str, user_text: str) -> Optional[tuple[str, str]]:
    """
    Check if the user's message is confirming or cancelling a pending delete.

    Returns:
        ("confirm", key) — user said yes to deleting key
        ("cancel",  key) — user said no
        None             — not a pending-delete response
    """
    now = time.monotonic()
    # Prune expired entries
    expired = [
        k for k, (_, ts) in _PENDING_DELETES.items()
        if now - ts > _PENDING_DELETE_TTL
    ]
    for k in expired:
        _PENDING_DELETES.pop(k, None)

    # Find entries for this sender
    sender_entries = {
        k: v for k, v in _PENDING_DELETES.items()
        if k.startswith(f"{sender_key}:")
    }
    if not sender_entries:
        return None

    low = user_text.strip().lower()

    # If the message clearly confirms or cancels, act on the most recent entry
    is_confirm = any(w in low for w in _CONFIRM_WORDS)
    is_cancel  = any(w in low for w in _CANCEL_WORDS)

    if not is_confirm and not is_cancel:
        return None

    # Pick the most recently registered pending delete for this sender
    cache_key = max(sender_entries, key=lambda k: sender_entries[k][1])
    _, fact_key = cache_key.split(":", 1)

    _PENDING_DELETES.pop(cache_key, None)   # consume it

    if is_confirm:
        logger.info("✅ pending_delete.confirmed  sender=%s  key=%s", sender_key, fact_key)
        return ("confirm", fact_key)
    else:
        logger.info("❌ pending_delete.cancelled  sender=%s  key=%s", sender_key, fact_key)
        return ("cancel", fact_key)


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
# P1-FEAT-1: Tool dispatch — LLM-decided, replaces keyword regex
# ─────────────────────────────────────────────────────────────────────────────

async def _dispatch_tool(
    tool_call_raw: Optional[Dict[str, Any]],
    query: str,
    chat_id: str,
    facts: Optional[Dict[str, str]] = None,
) -> str:
    """
    P1-FEAT-1: Dispatch an LLM-chosen tool to its live-data backend.

    Replaces the brittle keyword-regex _live_search() from Phase 0.
    The orchestrator now embeds a ``tool_call`` JSON block in its output when
    action=search.  We validate it with Pydantic and route it correctly.

    Flow::
        OrchestratorResult.tool_call (dict)
            ↓ parse_tool_call()
        ToolCall (typed Pydantic model)
            ↓ tool_dispatcher.dispatch()
        str result (injected as SEARCH_RESULT into next orchestrator turn)

    If tool_call is absent/invalid we fall back to compound-beta-mini
    (WebSearchTool path) using the LLM's ``query`` string.  This preserves
    100% backward compat with any orchestrator output that omits tool_call.

    Args:
        tool_call_raw:  Raw dict from OrchestratorResult.tool_call (may be None).
        query:          OrchestratorResult.query — used as fallback search text.
        chat_id:        Chat identifier for logging.
        facts:          User fact dict for city/country defaults.

    Returns:
        String result for the orchestrator SEARCH_RESULT block.
    """
    from .tools import parse_tool_call, tool_dispatcher, _WEB_SEARCH_SENTINEL

    _facts = facts or {}

    # ── Parse the tool_call from orchestrator output ──────────────────────
    tool_call = parse_tool_call(tool_call_raw)

    if tool_call is None:
        # Orchestrator did not provide a valid tool_call.
        # Fall back to compound-beta-mini web search with the query string.
        logger.info(
            "🔍 tool_dispatch.no_tool_call  chat=%s  query=%r → web_search fallback",
            chat_id, query[:80],
        )
        from .tools import WebSearchTool
        tool_call = WebSearchTool(tool="web_search", query=query or "")

    logger.info(
        "🔧 tool_dispatch  chat=%s  tool=%s",
        chat_id, tool_call.tool,
    )

    # ── Dispatch to the correct backend ──────────────────────────────────
    raw_result = await tool_dispatcher.dispatch(tool_call, facts=_facts)

    # ── Handle web_search sentinel (routes to compound-beta-mini) ─────────
    if raw_result.startswith(_WEB_SEARCH_SENTINEL):
        search_query = raw_result[len(_WEB_SEARCH_SENTINEL):]
        return await _compound_beta_search(search_query, chat_id)

    return raw_result


async def _compound_beta_search(query: str, chat_id: str) -> str:
    """
    Fallback web search using Groq compound-beta-mini.
    Handles 413 (query too long) with automatic truncation + retry.
    """
    if not settings.live_search_enabled:
        return "Live search is disabled."

    search_model = settings.live_search_model or "compound-beta-mini"
    if not _model_open(search_model):
        return "Live search model is temporarily unavailable."

    messages = [
        {"role": "system", "content": LIVE_SEARCH_PROMPT},
        {"role": "user",   "content": query},
    ]

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
            if len(text) < 60 and attempt < 3:
                continue
            logger.info(
                "🔍 compound_beta.done  chat=%s  query=%r  result_len=%d",
                chat_id, q_text, len(text),
            )
            return text[:_MAX_SEARCH_RESULT] if len(text) > _MAX_SEARCH_RESULT else text
        except Exception as e:
            s = str(e)
            if "413" in s and attempt < 3:
                logger.warning(
                    "⚠️  compound_beta.413  query=%r — retrying shorter", q_text
                )
                continue
            if "429" in s or "rate_limit" in s.lower():
                _trip_model(search_model, _parse_retry_after(e))
                return "Live search quota exhausted. Please try again later."
            logger.error("❌ compound_beta.error  %s", s[:200])
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

        updates_str = json.dumps([u.model_dump() for u in updates])  # includes delete+confirm fields
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

        # FIX-B1: If action=answer but text is empty and question is also empty,
        # the LLM returned a degenerate answer (e.g. null question field).
        # Treat it as a recoverable failure: force action=search so the loop
        # re-tries rather than returning an empty/confusing reply to the user.
        if result.action == "answer" and not (result.text or "").strip():
            logger.warning(
                "⚠️  orchestrate.empty_answer  label=%s  — forcing re-search", label
            )
            result = OrchestratorResult(
                action="search",
                reasoning="Empty answer received — retrying with web search",
                query=data.get("query") or "",
                tool_call=data.get("tool_call"),
            )

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


def _relative_ts_label(ts_iso: Optional[str], now: Optional[datetime] = None) -> str:
    """
    Convert an ISO-8601 timestamp string to a human-readable relative label
    so the orchestrator can reason about temporal distance.

    Examples:
      "today 14:32"        — same calendar day in user's timezone
      "yesterday 09:15"    — previous calendar day
      "2 days ago 18:04"   — older messages
      "Mon 08:50"          — if > 6 days ago, show weekday name

    Falls back to the raw ts string if parsing fails.
    """
    if not ts_iso:
        return ""
    try:
        # Parse ISO with optional timezone offset (e.g. "+05:30" or "Z")
        ts_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if now is None:
            try:
                tz = ZoneInfo(settings.app_timezone)
            except (ZoneInfoNotFoundError, AttributeError):
                tz = UTC
            now = datetime.now(tz)
        # Normalise both to user's local timezone for calendar-day comparison
        try:
            tz = ZoneInfo(settings.app_timezone)
        except (ZoneInfoNotFoundError, AttributeError):
            tz = UTC
        local_ts  = ts_dt.astimezone(tz)
        local_now = now.astimezone(tz)

        delta_days = (local_now.date() - local_ts.date()).days
        time_str   = local_ts.strftime("%H:%M")

        if delta_days == 0:
            return f"today {time_str}"
        elif delta_days == 1:
            return f"yesterday {time_str}"
        elif delta_days < 7:
            return f"{delta_days} days ago {time_str}"
        else:
            return local_ts.strftime("%a %d %b %H:%M")   # e.g. "Mon 03 Mar 08:50"
    except (ValueError, TypeError):
        return ts_iso[:16] if ts_iso else ""


def _build_context_str(context: List[Any]) -> str:
    """
    Render context items as a timestamped conversation feed.

    Each item is expected to be a dict with keys:
      text      — the message text
      metadata  — dict containing:
          direction  "in" | "out"
          ts         ISO-8601 timestamp (e.g. "2026-03-11T09:15:30+05:30")

    Output is sorted chronologically (oldest first) so the LLM reads a
    natural conversation arc, with a relative timestamp prefix on each line:

      [yesterday 09:15] user: I went for a run this morning
      [yesterday 09:15] shimmi: Great! How far did you go?
      [today 14:32] user: Can you remind me about the marathon?

    The orchestrator can then correctly answer "did we talk about X yesterday?"
    """
    now = datetime.now(UTC)

    # ── Normalise each item into (ts_iso, direction, text) ───────────────────
    entries: List[Tuple[str, str, str]] = []
    for item in context:
        if isinstance(item, dict):
            meta      = item.get("metadata") or {}
            text      = (item.get("text") or "").strip()
            direction = meta.get("direction", "in")
            ts_iso    = meta.get("ts") or ""
        elif hasattr(item, "metadata"):
            meta      = item.metadata or {}
            text      = (getattr(item, "text", None) or "").strip()
            direction = meta.get("direction", "in")
            ts_iso    = meta.get("ts") or ""
        elif hasattr(item, "role") and hasattr(item, "content"):
            text      = getattr(item, "content", "").strip()
            direction = "in" if getattr(item, "role", "user") == "user" else "out"
            ts_iso    = ""
        else:
            entries.append(("", "in", str(item)))
            continue

        if text:
            entries.append((ts_iso, direction, text))

    # ── Sort chronologically by ts (empty ts floats to top as oldest) ─────────
    entries.sort(key=lambda e: e[0] or "")

    # ── Render with relative timestamp prefix ─────────────────────────────────
    lines = []
    for ts_iso, direction, text in entries:
        label  = _relative_ts_label(ts_iso, now=now)
        role   = "user" if direction == "in" else "shimmi"
        prefix = f"[{label}] " if label else ""
        lines.append(f"{prefix}{role}: {text}")

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
    tz_offset     = _utc_offset_str()
    time_of_day   = _time_of_day()

    system_content = ORCHESTRATOR_PROMPT

    # Inject temporal context as a JSON block so the model can use it
    # for reminders (trigger_iso), date-relative queries, and time queries.
    temporal = (
        f"current_time={time_str!r}  today={today_str!r}  "
        f"tz_offset={tz_offset!r}  time_of_day={time_of_day!r}"
    )

    user_content_parts = [
        f"TIME: {temporal}",
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
    sender_key: str = "",           # P1-GUARD: authenticated sender (from WAHA webhook, not LLM)
    user_text: str,
    facts: Dict[str, str],
    context: List[Any],
    reminders: List[Reminder],
    trace: Any = None,
) -> AgentResult:
    """
    Full agentic pipeline. Steps:
      0. Pending-delete intercept — if user is confirming/cancelling a queued delete
      1. Facts shortcut — answer simple memory queries instantly (0 tokens)
      2. Pre-extract memory updates from user message (8B / Gemini Lite)
      3. Orchestrate (Gemini Flash primary → Groq 70B fallback)
         - If action=search: _dispatch_tool() with LLM-chosen tool, then re-orchestrate
      4. Verify memory updates (8B / Gemini Lite)
      5. Format reply for WhatsApp (8B / Gemini Lite)

    Args:
        sender_key: Authenticated sender identifier derived from the WAHA webhook
                    payload — NEVER from LLM output. Used only for pending-delete
                    confirmation matching; all DB writes in main.py use their own
                    sender_key derived from the same source.
    """
    # ── 0. Pending-delete intercept (P1-GUARD) ────────────────────────────
    # Check if this message is a yes/no response to a queued list-delete confirmation.
    # This runs BEFORE the LLM to avoid burning tokens on a one-word confirm/cancel.
    if sender_key:
        pending = _check_pending_delete(sender_key, user_text)
        if pending:
            action, key = pending
            if action == "confirm":
                logger.info(
                    "✅ pending_delete.executing  sender=%s  key=%s", sender_key, key
                )
                # Return a MemoryUpdate with delete=True AND confirm=True so main.py
                # can pass confirmed=True to delete_fact() and bypass the confirmation guard.
                return AgentResult(
                    reply=ReplyPayload(
                        type="text",
                        text=f"Done — your *{key.replace('_', ' ')}* has been cleared. 🗑️",
                    ),
                    memory_updates=[
                        MemoryUpdate(key=key, value="", delete=True, confirm=True)
                    ],
                    provider_used="pending_delete",
                )
            else:  # cancel
                return AgentResult(
                    reply=ReplyPayload(
                        type="text",
                        text=f"No problem — your *{key.replace('_', ' ')}* is kept as-is. ✅",
                    ),
                    provider_used="pending_delete_cancel",
                )

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
    # FIX-B2: Track (tool, args) pairs already dispatched this loop to prevent
    # the agent from re-issuing the exact same tool call on every iteration when
    # a tool returns empty (e.g. GNews 400 → result_len=0 every time).
    _dispatched_calls: set = set()

    for iteration in range(1, _MAX_ITERATIONS + 1):
        label = f"orchestrate_{iteration}"

        if search_result:
            messages = _build_orchestrator_messages(
                user_text, facts, context, reminders, search_result
            )

        orch = await _orchestrate(messages, chat_id, label=label, trace=trace)

        if orch.action == "answer":
            reply_text = orch.text or "Sorry, I couldn't generate a reply."
            # Orchestrator-decided updates are already verified by the orchestrator
            # itself (it knows the full context and explicitly chose to save them).
            # Only run the verifier on pre_updates (extracted from user message alone)
            # where LLM inference may be less reliable.
            # Orchestrator updates are accepted directly (they pass through normalize_key).
            orch_approved = [
                ApprovedUpdate(
                    key=mu.key,
                    value=mu.value,
                    confidence=1.0,
                    delete=getattr(mu, "delete", False),
                    confirm=getattr(mu, "confirm", False),
                )
                for mu in orch.memory_updates
            ]
            # Pre-extracted updates still go through verifier
            pre_verified = await _verify_updates(
                pre_updates, chat_id,
                existing_facts=facts, user_text=user_text, trace=trace,
            ) if pre_updates else []
            approved = pre_verified + orch_approved
            if trace:
                trace.tag(
                    memory_extracted=len(pre_updates) + len(orch.memory_updates),
                    memory_verified=len(approved),
                    memory_total=len(approved),
                )
            # Format
            formatted = await _format_whatsapp(reply_text, chat_id, trace=trace)
            if len(formatted) > len(reply_text) * 2 or not formatted.strip():
                formatted = reply_text  # sanity guard

            return AgentResult(
                reply=ReplyPayload(type="text", text=formatted),
                memory_updates=[
                    MemoryUpdate(
                        key=a.key,
                        value=a.value,
                        delete=getattr(a, "delete", False),
                        confirm=getattr(a, "confirm", False),
                    )
                    for a in approved
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

            # FIX-B2: Deduplicate tool dispatches — if the LLM issues the exact
            # same (tool, query) pair it already tried, fall through to web_search
            # instead of looping with guaranteed-empty results.
            dispatch_key = f"{(orch.tool_call or {}).get('tool','web_search')}::{query}"
            if dispatch_key in _dispatched_calls:
                logger.warning(
                    "⚠️  search.duplicate_dispatch  iter=%d  key=%r — forcing web_search fallback",
                    iteration, dispatch_key,
                )
                # Override with a web_search on the same query instead
                search_result = await _compound_beta_search(query, chat_id)
            else:
                _dispatched_calls.add(dispatch_key)
                search_result = await _dispatch_tool(orch.tool_call, query, chat_id, facts=facts)

            logger.info("🔍 search.done  iter=%d  query=%r  result_len=%d",
                        iteration, query, len(search_result or ""))
            if trace:
                trace.tag(
                    **{f"live_search_{iteration}": {
                        "search_query": query,
                        "result_len": len(search_result or ""),
                    }}
                )

            # FIX-B3: If we have a non-trivial result, force action=answer next
            # iteration instead of letting the LLM re-search with the same query.
            # This prevents the "I don't have real-time data" hallucination when
            # search_result actually contains the answer.
            if search_result and len(search_result.strip()) > 60:
                # Rebuild messages with the result — the next _orchestrate call
                # will see SEARCH_RESULT and should choose action=answer.
                pass  # normal flow: loop continues, messages rebuilt at top with search_result

            # ── Early exit: tool returned an "unavailable" notice ────────────
            # When tool returns a non-empty message that signals unavailability
            # (stock data unavailable, market closed, etc.), relay it to the user
            # immediately rather than looping. Prevents the 3-iteration spin seen
            # when yfinance has no price data for a ticker.
            _sr_lower = (search_result or "").lower()
            if search_result and (
                "unavailable" in _sr_lower
                or "could not fetch" in _sr_lower
                or "not recognised" in _sr_lower
                or "market may be closed" in _sr_lower
            ):
                logger.info("🔍 search.unavailable  iter=%d  — replying directly", iteration)
                return AgentResult(
                    reply=ReplyPayload(type="text", text=search_result),
                    iterations=iteration,
                )

        elif orch.action in ("ask", "ask_user", "clarify"):
            question = orch.question or orch.text or "Could you clarify that?"
            return AgentResult(
                reply=ReplyPayload(type="text", text=question),
                iterations=iteration,
            )
        else:
            logger.info("ℹ️  orchestrate.unknown_action  action=%s  (treating as answer)", orch.action)
            text = orch.text or orch.reasoning or "I'm not sure how to help with that."
            return AgentResult(
                reply=ReplyPayload(type="text", text=text),
                iterations=iteration,
            )

    # Max iterations reached — return last search result or generic fallback
    if search_result and len(search_result.strip()) > 60:
        # We have results but the LLM kept re-searching — present what we have.
        fallback = search_result.strip()
    else:
        fallback = (
            "I wasn't able to find live data for that right now. "
            "The news/search service may be temporarily unavailable — please try again in a moment."
        )
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

        # FIX-B4: Keys managed by dedicated subsystems must never be overwritten
        # from reply text. reminder_notes is written by the scheduler/orchestrator
        # directly; extract_reply_memory scanning the reply would overwrite it with
        # the reminder's description string, causing value drift on every message.
        _REPLY_EXTRACT_SKIP = frozenset({"reminder_notes", "recent_activity"})
        if result.memory_updates:
            before = len(result.memory_updates)
            result.memory_updates = [
                u for u in result.memory_updates
                if u.key not in _REPLY_EXTRACT_SKIP
            ]
            dropped = before - len(result.memory_updates)
            if dropped:
                logger.debug(
                    "reply_extract.skip_protected  dropped=%d  (reminder_notes / recent_activity)",
                    dropped,
                )

        if result.memory_updates:
            logger.info(
                "🧠 reply_extract.found  count=%d  keys=%s",
                len(result.memory_updates),
                [u.key for u in result.memory_updates],
            )
            # Import here to avoid circular dependency
            from .database import upsert_fact, delete_fact
            for u in result.memory_updates:
                try:
                    if getattr(u, "delete", False):
                        await delete_fact(sender_key, u.key)
                    else:
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
            "orchestrator=Groq(%s) → Gemini(%s)  extraction=Groq(%s)",
            settings.groq_model_pool,
            settings.gemini_model_pool,
            settings.orchestrator_model,
            settings.gemini_orchestrator_model,
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
# LLM-driven fact key consolidation  (replaces brittle _KEY_ALIASES expansion)
# ─────────────────────────────────────────────────────────────────────────────

async def consolidate_user_facts(whatsapp_id: str) -> None:
    """
    LLM-driven deduplication of a user's fact keys.

    Instead of an ever-growing hand-written alias map, we give the LLM the
    user's full key→value dict and ask it to identify semantic duplicates
    (favourite_colour / favorite_color, fitness_goal / fitness_goals, etc.)
    and return a merge plan.  We then apply the plan directly to SQLite.

    Design decisions:
    • Uses the 8B extraction model (cheapest / fastest — no orchestrator quota)
    • Fire-and-forget: caller wraps in asyncio.create_task(); all exceptions
      are caught internally so nothing can break the main request path
    • Called from main.py after facts_load when the user has ≥2 facts
    • Idempotent: merging already-canonical keys produces an empty merge list
    • Rate-limited: runs at most once per hour per user (cooldown gate below)
    • The _KEY_ALIASES dict in database.py stays as a fast synchronous path
      for known aliases; this function handles unknown / novel variants

    Merge plan applied:
      1. For each merge group: upsert canonical_key = best_value
      2. Delete all absorbed (alias) keys that differ from canonical
    """
    # ── Cooldown gate: skip if run recently ──────────────────────────────────
    now_mono = time.monotonic()
    last_run = _CONSOLIDATION_LAST_RUN.get(whatsapp_id, 0.0)
    if now_mono - last_run < _CONSOLIDATION_COOLDOWN_SEC:
        logger.debug(
            "consolidate.skipped  sender=%s  next_in=%.0fs",
            whatsapp_id, _CONSOLIDATION_COOLDOWN_SEC - (now_mono - last_run),
        )
        return
    _CONSOLIDATION_LAST_RUN[whatsapp_id] = now_mono

    from .database import sqlite_store, normalize_key

    if sqlite_store is None:
        return

    try:
        facts = await sqlite_store.get_all_facts(whatsapp_id)
    except Exception as exc:
        logger.debug("consolidate.get_facts_fail  sender=%s  err=%s", whatsapp_id, exc)
        return

    if len(facts) < 2:
        return  # nothing to deduplicate

    payload = json.dumps({"facts": facts}, ensure_ascii=False)

    try:
        raw = await _groq_raw(
            [
                {"role": "system", "content": KEY_CONSOLIDATION_PROMPT},
                {"role": "user",   "content": payload},
            ],
            max_tokens=512,
            chat_id=whatsapp_id,
            label="key_consolidation",
            role="extract",         # uses cheap 8B model
            timeout=20.0,
        )
    except Exception as exc:
        logger.debug("consolidate.llm_fail  sender=%s  err=%s", whatsapp_id, str(exc)[:120])
        return

    # ── parse response ────────────────────────────────────────────────────────
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"```[^\n]*\n?", "", clean).strip()
        data  = json.loads(clean)
        merges: list = data.get("merges") or []
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.debug("consolidate.parse_fail  sender=%s  err=%s  raw=%r",
                     whatsapp_id, exc, raw[:200])
        return

    if not merges:
        logger.debug("consolidate.no_duplicates  sender=%s  facts=%d", whatsapp_id, len(facts))
        return

    # ── apply merge plan ──────────────────────────────────────────────────────
    applied = 0
    for merge in merges:
        try:
            canonical = normalize_key(str(merge.get("canonical", "")))
            absorb    = [normalize_key(str(k)) for k in (merge.get("absorb") or [])]
            value     = str(merge.get("value") or "").strip()

            if not canonical or not value:
                continue

            # Write canonical key with the best value
            await sqlite_store.upsert_fact(whatsapp_id, canonical, value)

            # Delete alias keys (skip if they already equal canonical — normalize_key
            # may have already mapped them, in which case the row doesn't exist)
            for alias in absorb:
                if alias and alias != canonical:
                    # Direct delete bypassing the allowlist guard — this is an
                    # internal maintenance operation, not a user-initiated deletion.
                    await _consolidation_delete(whatsapp_id, alias)

            logger.info(
                "🔑 consolidate.merged  sender=%s  canonical=%s  absorbed=%s  value=%r",
                whatsapp_id, canonical, absorb, value[:60],
            )
            applied += 1
        except Exception as exc:
            logger.debug("consolidate.merge_fail  sender=%s  err=%s", whatsapp_id, str(exc)[:80])

    if applied:
        logger.info(
            "✅ consolidate.done  sender=%s  merges_applied=%d  original_keys=%d",
            whatsapp_id, applied, len(facts),
        )


async def _consolidation_delete(whatsapp_id: str, key: str) -> None:
    """
    Internal-only delete that bypasses the _DELETABLE_KEYS allowlist guard.
    Used exclusively by consolidate_user_facts() to remove alias rows after
    their value has been merged into the canonical key.
    """
    from .database import sqlite_store
    import asyncio as _asyncio
    import sqlite3 as _sqlite3

    if sqlite_store is None:
        return

    async with sqlite_store._lock:
        def _do():
            with _sqlite3.connect(sqlite_store.path) as conn:
                conn.execute(
                    "DELETE FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                    (whatsapp_id, key),
                )
                conn.commit()
        await _asyncio.to_thread(_do)



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
