"""
agent_engine.py — Shimmi v3.15.6

Changes vs v3.3.0:

  FIX-FACTS   _clean_facts() now strips 9 ephemeral/bloat keys and truncates
              fact values to 180 chars before building orchestrator prompts.
              Reduces prompt size by ~40% for power users with 60+ facts.
              Dropped keys: recent_query, recent_search, conversation_since_morning,
              last_summary, recent_article_details, favorite_news_source_details,
              arrival_time, destination, next_meeting_*, social_security_number.

  FIX-SUMMARY New _try_summary_shortcut() intercepts "summarise today/yesterday/
              last week" queries. Reads actual timestamped rows from SQLite
              message_log (time-windowed), calls 8B to generate summary, saves
              back to DB. Zero orchestrator tokens. Replaces the broken approach
              where a stale conversation_summary fact (frozen at March 17) was
              fed to the orchestrator which then guessed or hallucinated.

  FIX-STOCKS  Broader stock keyword trigger regex. Better ticker filter set.
              Symbol always gets .NS suffix if unqualified (PAYTM → PAYTM.NS).

  FIX-2       str() coerce on tool_dispatcher return value (AttributeError fix).

Changes vs v3.1.0 (this release):

  FIX-CHAIN   CRITICAL: _groq_raw fallback loop was not exhausting all
              providers. When Gemini + Groq 70B both fail, the loop did
              `return await _call_llm(groq_70b)` — if that raised, the
              exception escaped immediately and Groq 8B was never tried.
              Each candidate is now wrapped in its own try/except; the loop
              continues to the next provider on rate-limit or timeout.
              Impact: fatal 429 errors when Groq 70B is exhausted are now
              handled gracefully using Groq 8B (500K tokens/day headroom).

  FIX-RPD     Gemini daily-quota (RPD) now gets a 2-hour cooldown instead
              of the default 300s. The message "You exceeded your current
              quota" identifies RPD exhaustion — there is no retry-after
              hint. Previously Gemini was retried on every message for the
              rest of the day, generating a WARNING on every request.

  FIX-NOISE   _clean_facts() now strips 15 ephemeral/junk keys before
              building the orchestrator prompt (result_*, recent_activity,
              next_meeting_*, semester, year, etc.). These stay in the DB
              for audit/consolidation but don't burn tokens on every call.

  FIX-TOOL    _dispatch_tool() now has a keyword-based routing layer before
              the web_search fallback. When Groq 70B acts as fallback
              orchestrator it frequently omits the tool_call JSON block —
              the new _keyword_tool_from_query() parses the query string to
              route weather / stocks / news / currency / timezone to the
              correct MCP endpoint without an LLM call. Structured tools
              now work even when Gemini is fully exhausted.

  FIX-2       CRITICAL: _dispatch_tool() called raw_result.startswith() on the
              return value of tool_dispatcher.dispatch() without guarding against
              non-str returns. When the MCP timezone/currency client short-circuited
              (dict response), this crashed with AttributeError and the user got no
              reply. Fixed with str() coercion before .startswith().

  FIX-TIME    Zero-token time/date shortcut. LLMs hallucinate the current
              time by reading stale timestamps from conversation context.
              _try_time_shortcut() intercepts short time/date queries and
              answers from the server clock — zero tokens, always accurate.

Changes vs v3.0.3 (previous session):

  PROVIDER-1  Gemini 2.0 Flash as primary orchestrator.
  PROVIDER-2  Per-provider circuit breakers.
  PROVIDER-3  Token budget tracker.
  FIX-S1      Facts shortcut v2.
  FIX-S2      Junk fact filtering.
  FIX-S3      Fire-and-forget exception handling.
  FIX-S5      Retry-after parsing.
  FIX-S6      Live search 413 handling.
  FIX-S7      User-facing error messages.
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
    question:       str              = ""
    memory_updates: List[MemoryUpdate]  = Field(default_factory=list)
    reminders:      List[ReminderEntry] = Field(default_factory=list)
    # P1-FEAT-1: LLM-decided tool dispatch — replaces keyword regex in _live_search
    tool_call:      Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("reasoning", "text", "query", "question", mode="before")
    @classmethod
    def _coerce_opt_str(cls, v: Any) -> str:
        """
        FIX-NULL: Pydantic v2 does NOT fall back to the field default when the
        key IS present in JSON with value null — it raises ValidationError.
        This coerces None → "" for all optional string fields so the model
        never crashes on incomplete LLM output.

        Root cause of the fatal error for "Red Tape running shoes" query:
          ValidationError: question — Input should be a valid string
            [input_value=None, input_type=NoneType]
        LLM returned {"action":"answer","question":null,...} and the bot
        crashed without sending any reply to the user.
        """
        return "" if v is None else str(v).strip()

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
    FIX-S5 + FIX-RPD: Parse the actual wait time from Groq/Gemini 429 errors.

    Groq:        "Please try again in 1h4m54.368s"  → extract duration
    Gemini RPM:  "retry after N seconds"            → extract N
    Gemini RPD:  "You exceeded your current quota"  → no retry hint → 2h cooldown
                 Daily quota resets at midnight Google-time; 2h is a safe floor
                 that avoids hammering the API for the rest of the day.

    Returns seconds (minimum 60.0, maximum 7200.0).
    """
    s = str(exc)

    # ── Gemini RPD: daily quota exhausted, no retry hint ─────────────────
    # These phrases appear in Gemini's 429 when the daily cap is hit:
    if ("you exceeded your current quota" in s.lower()
            or ("quota" in s.lower() and "billing" in s.lower())):
        logger.info("📅 quota.rpd  daily_quota_exhausted — 2h cooldown")
        return 7_200.0

    # ── Hours + minutes + seconds: e.g. "1h4m54.368s" or "36m12.096s" ───
    m = re.search(r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", s)
    if m:
        h = float(m.group(1) or 0)
        mn = float(m.group(2) or 0)
        sec = float(m.group(3) or 0)
        total = h * 3600 + mn * 60 + sec
        if total > 5:  # ignore tiny matches from URLs like "...5s..."
            return max(60.0, min(total + 10.0, 7200.0))  # +10s buffer

    # ── Gemini RPM: "quota exceeded … retry after N seconds" ─────────────
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
                logger.debug(
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
    json_mode: bool = True,
) -> str:
    """
    Single attempt to call the LLM.  Handles circuit-tripping and budget tracking.
    Raises the original exception on failure.

    json_mode=True  (default) — forces response_format=json_object. Required for all
                                structured extraction/orchestration calls.
    json_mode=False            — plain text response. Required when the prompt does
                                not contain the word "json" (e.g. summary shortcut).
                                Groq returns HTTP 400 if json_mode=True but the prompt
                                has no "json" mention.
    """
    t0 = time.monotonic()
    fmt = {"type": "json_object"} if json_mode else {"type": "text"}

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
                    response_format=fmt,
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
                    response_format=fmt,
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
    json_mode: bool = True,
) -> str:
    """
    Route to the best available provider for the given role, with automatic
    fallback if the primary is circuit-tripped.

    json_mode=False must be passed for plain-text calls (e.g. summary_shortcut)
    where the prompt contains no "json" mention — Groq 400s otherwise.
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
            chat_id=chat_id, label=label, json_mode=json_mode,
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
        last_fb_exc: Optional[Exception] = None
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
            try:
                return await _call_llm(
                    messages,
                    provider=fb_provider, model=fb_model,
                    max_tokens=max_tokens, timeout=fb_timeout,
                    chat_id=chat_id, label=f"{label}_fb", json_mode=json_mode,
                )
            except Exception as fb_exc:
                fb_exc_str = str(fb_exc)
                fb_is_rl = (
                    "429" in fb_exc_str
                    or "rate_limit" in fb_exc_str.lower()
                    or "RESOURCE_EXHAUSTED" in fb_exc_str
                )
                last_fb_exc = fb_exc
                if fb_is_rl or isinstance(fb_exc, (asyncio.TimeoutError, TimeoutError)):
                    # This fallback is also exhausted — continue to the next one
                    logger.info(
                        "⚡ fallback.exhausted  provider=%s  model=%s  trying_next",
                        fb_provider, fb_model,
                    )
                    continue
                raise  # non-rate-limit error — propagate immediately

        # All fallbacks exhausted — raise the last exception we got
        if last_fb_exc is not None:
            logger.warning(
                "🚨 all_providers_exhausted  role=%s  — groq_8b also at limit",
                role,
            )
            raise last_fb_exc
        raise  # original first_exc (no fallback was even attempted)


# ─────────────────────────────────────────────────────────────────────────────
# Junk-fact filter
# ─────────────────────────────────────────────────────────────────────────────

# Memory key definitions — imported from single source of truth.
from .memory_schema import (
    JUNK_VALUES as _JUNK_VALUES,
    PROMPT_SKIP_KEYS as _PROMPT_SKIP_KEYS,
    CONSOLIDATION_PROTECTED as _CONSOLIDATION_PROTECTED,
)

# Max characters for a single fact value in the orchestrator prompt.
# Long values like conversation_summary, favorite_quote get truncated.
_FACT_VALUE_MAX_CHARS = 180


def _clean_facts(facts: Dict[str, str]) -> Dict[str, str]:
    """
    Filter and trim facts before building LLM prompts.

    - Drops junk/empty values
    - Drops ephemeral or security-sensitive keys (_PROMPT_SKIP_KEYS)
    - Truncates long values to _FACT_VALUE_MAX_CHARS so a 63-fact user
      doesnt burn 3K+ tokens on facts alone every single message
    """
    out = {}
    for k, v in facts.items():
        if not v or str(v).strip().lower() in _JUNK_VALUES:
            continue
        if k in _PROMPT_SKIP_KEYS:
            continue
        v_str = str(v)
        if len(v_str) > _FACT_VALUE_MAX_CHARS:
            v_str = v_str[:_FACT_VALUE_MAX_CHARS] + "…"
        out[k] = v_str
    return out



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
# ─────────────────────────────────────────────────────────────────────────────
# Facts shortcut — zero-token instant recall for direct personal questions
# ─────────────────────────────────────────────────────────────────────────────
#
# Design principle: keep this minimal and unambiguous. Only fire on queries
# where the intent is 100% clear AND the fact key is 100% determinable.
# Every entry in _FACT_SYNONYMS has been chosen because:
#   (a) there is only one possible fact it could refer to
#   (b) the user phrase is common enough to be worth optimising
# Anything beyond these cases goes to the orchestrator — it handles it in 1
# iteration at ~800 tokens and gets nuance right.
#
# This is not a keyword table — it is a compact alias dictionary for the
# ~15 fact keys that users ask about most often.
# No regex, no guards, no word-overlap fallback. Two structural patterns only.

# Maps user subject phrase → canonical DB key
_FACT_SYNONYMS: Dict[str, str] = {
    # Identity
    "name": "name",
    "age": "age",       "old": "age",
    "birthday": "birthday",
    "occupation": "occupation", "job": "occupation", "work": "occupation",
    # Location
    "city": "city",     "location": "city",    "live": "city",
    "country": "country",
    # Preferences
    "coffee": "favorite_drink",  "drink": "favorite_drink",  "tea": "favorite_drink",
    "favorite drink": "favorite_drink",    "favourite drink": "favorite_drink",
    "food": "favorite_food",     "eat": "favorite_food",
    "favorite food": "favorite_food",      "favourite food": "favorite_food",
    "color": "favorite_color",   "colour": "favorite_color",
    "favorite color": "favorite_color",    "favourite color": "favorite_color",
    "genre": "favorite_genre",
    "trail": "favorite_trail",
    # Possessions
    "car": "car",        "drive": "car",     "vehicle": "car",    "bike": "car",
    # People / social
    "pets": "pets",      "pet": "pets",      "dog": "pets",       "cat": "pets",
    # Health
    "allergies": "allergies",  "allergic": "allergies",
    # Lists
    "shopping": "shopping_list",   "shopping list": "shopping_list",
    "grocery": "grocery_list",     "grocery list": "grocery_list",
    "todo": "todo_list",           "todo list": "todo_list",
    # Plans
    "travel": "travel_plans",      "trip": "travel_plans",
    "travel plans": "travel_plans",
}

# Special-case interrogative forms that don't fit "what's my X"
_SPECIAL_RECALL_FORMS: List[Tuple[str, str]] = [
    ("how old am i",         "age"),
    ("what do i drive",      "car"),
    ("what car do i",        "car"),
    ("where do i live",      "city"),
    ("where am i from",      "city"),
    ("what am i allergic to", "allergies"),
    ("what do i do for work", "occupation"),
    ("what do i do for a living", "occupation"),
]


def _try_facts_shortcut(user_text: str, facts: Dict[str, str]) -> Optional[str]:
    """
    Answer the most common direct recall questions instantly from the facts dict.
    Zero tokens, zero LLM call.

    Fires ONLY on two structural patterns:
      • "what's/what is/what are my <subject>" where subject maps to a fact key
      • A handful of special-case forms (how old am I, what do I drive, etc.)

    Anything else — writes, corrections, analytical questions, anything ambiguous —
    passes through to the orchestrator which handles it correctly in 1 iteration.
    This is deliberately conservative: it is better to use 800 tokens on the
    orchestrator than to return a stale/wrong value here.
    """
    if len(user_text) > 60:
        return None

    low = user_text.lower().strip().rstrip("?").strip()

    def _reply(key: str) -> Optional[str]:
        val = facts.get(key, "")
        if not val or str(val).strip().lower() in _JUNK_VALUES:
            return None
        label = key.replace("_", " ")
        return f"Your {label} on record is: *{val}* 📋"

    # Pattern 1: "what's my X" / "what is my X" / "what are my X"
    for prefix in ("what's my ", "what is my ", "what are my "):
        if low.startswith(prefix):
            subject = low[len(prefix):].strip()
            key = _FACT_SYNONYMS.get(subject)
            if key:
                return _reply(key)
            return None  # e.g. "what's my favourite book?" → not in map → orchestrator

    # Pattern 2: special-case forms
    for phrase, key in _SPECIAL_RECALL_FORMS:
        if low.startswith(phrase):
            return _reply(key)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Zero-token time / date shortcut
# ─────────────────────────────────────────────────────────────────────────────

_TIME_SIGNALS = frozenset({
    "what time", "what's the time", "whats the time", "current time",
    "time now", "time is it", "time right now", "what is the time",
})
_DATE_SIGNALS = frozenset({
    "what date", "what's the date", "whats the date", "today's date",
    "what day is it", "what day today", "today date", "current date",
    "what is today", "what is the date",
})


# Geographic qualifier: "in Japan", "in Tokyo", "at London", etc.
# Used by both _try_time_shortcut and _try_facts_shortcut to detect world-clock
# queries that need the timezone MCP tool, not the local server clock.
# Matches "in/at/for [Capitalised]" and a curated list of lowercase country names
# so "What time is it in Tokyo?" and "time in japan" are both caught.
_GEO_QUALIFIER_RE = re.compile(
    r"\b(?:in|at|for)\s+(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|"
    r"japan|tokyo|london|paris|berlin|dubai|singapore|sydney|"
    r"new\s+(?:york|zealand|delhi)|usa|uk|australia|china|india|europe)\b",
    re.IGNORECASE,
)


def _try_time_shortcut(user_text: str) -> Optional[str]:
    """
    Answer LOCAL time/date queries directly from the server clock — zero tokens.

    LLMs reliably hallucinate the current time by reading stale timestamps
    from conversation context instead of the injected current_time value.
    Intercepting these queries here is both more accurate and free.

    BUG-1 FIX: "What's the time in Japan right now?" fired this shortcut
    because "what's the time" is a substring — returning IST instead of JST.
    Guard: if the message contains a location qualifier ("time in X", "time
    at X"), step aside so _keyword_tool_from_query routes it to the timezone
    MCP tool, which gives the correct local time for that city.

    Only fires on short, unambiguous LOCAL time/date messages (≤70 chars).
    Returns a WhatsApp-formatted reply or None to fall through.
    """
    if len(user_text) > 70:
        return None
    low = user_text.lower().strip()

    # Block location-qualified time queries — they need the timezone MCP tool
    if _GEO_QUALIFIER_RE.search(user_text):  # check original (preserves case for capital letters)
        return None

    is_time = any(sig in low for sig in _TIME_SIGNALS)
    is_date = any(sig in low for sig in _DATE_SIGNALS)

    if not is_time and not is_date:
        return None

    now     = _now_local()
    tz_abbr = now.strftime("%Z") or "local"
    hour    = now.hour
    if   6  <= hour < 12: period = "morning"
    elif 12 <= hour < 17: period = "afternoon"
    elif 17 <= hour < 21: period = "evening"
    else:                 period = "night"

    if is_time and not is_date:
        return (
            f"🕰️ It's *{now.strftime('%H:%M')} {tz_abbr}* "
            f"({now.strftime('%A')} {period})"
        )
    if is_date and not is_time:
        return f"📅 Today is *{now.strftime('%A, %d %B %Y')}*"

    # both
    return (
        f"🕰️ *{now.strftime('%H:%M')} {tz_abbr}*  "
        f"📅 *{now.strftime('%A, %d %B %Y')}*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# P1-FEAT-1: Tool dispatch — LLM-decided, replaces keyword regex
# ─────────────────────────────────────────────────────────────────────────────

def _is_stock_query(query: str) -> bool:
    """
    Returns True if the query is clearly asking for a stock PRICE / market data.
    Returns False for portfolio edit/correction messages — those need memory writes.
    Used to intercept mis-routed tool choices and prevent hallucination on price queries.
    """
    if not query:
        return False
    low = query.lower()

    # Explicit portfolio edit patterns — these are MEMORY UPDATES, not price queries.
    # The hallucination guard must NOT fire on these or the correction gets lost.
    _EDIT_PATTERNS = (
        r"\b(update|correct|change|fix|edit|modify|revise)\b.*"
        r"\b(portfolio|purchase.?price|avg.?price|average.?price|bought.?at|shares?)\b",
        r"\b(purchase.?price|avg.?price|bought.?at)\b.*\b(was|is|should.?be|correct)\b",
        r"\b(correction|mistake|error|wrong)\b.*\b(price|portfolio|stock|shares?)\b",
        r"\b(add|remove|delete)\b.*\b(portfolio|holdings?|stocks?)\b",
    )
    for pat in _EDIT_PATTERNS:
        if re.search(pat, low):
            return False   # portfolio edit — don't force search

    return bool(re.search(
        r"\b(stock|share|price|equity|nse|bse|nifty|sensex|"
        r"paytm|reliance|tcs|infy|infosys|wipro|hdfc|icici|sbi|"
        r"adani|airtel|zomato|bajaj|ongc|kotak|hcl|"
        r"gold.?price|gold.?rate|silver.?price|silver.?rate|"
        r"how.*(stock|share|market|doing|performing)|"
        r"(stock|share|market).*(update|today|now|current|latest))\b",
        low
    ))


async def _dispatch_tool(
    tool_call_raw: Optional[Dict[str, Any]],
    query: str,
    chat_id: str,
    facts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Dispatch an LLM-chosen tool to its live-data backend.

    Primary path: orchestrator embeds a ``tool_call`` JSON block → parsed and
    dispatched to the correct MCP endpoint.

    Keyword fallback: when Groq 70B acts as fallback orchestrator it often omits
    tool_call. We parse the query string with lightweight keyword heuristics to
    still route weather / stocks / news / currency / timezone to MCP correctly.
    This means structured tools work even when Gemini is fully down.

    Final fallback: compound-beta-mini web search.
    """
    from .tools import parse_tool_call, tool_dispatcher, _WEB_SEARCH_SENTINEL

    _facts = facts or {}
    tool_call = parse_tool_call(tool_call_raw)

    if tool_call is None:
        # Try keyword routing before falling back to web_search
        tool_call = _keyword_tool_from_query(query, _facts)
        if tool_call is not None:
            logger.info(
                "🔑 tool_dispatch.keyword  chat=%s  tool=%s  query=%r",
                chat_id, tool_call.tool, query[:60],
            )
        else:
            logger.info(
                "🔍 tool_dispatch.web_search  chat=%s  query=%r",
                chat_id, query[:60],
            )
            from .tools import WebSearchTool
            tool_call = WebSearchTool(tool="web_search", query=query or "")
    else:
        # FIX-TOOL-OVERRIDE: Groq 8B fallback frequently mis-routes stock/price
        # queries to web_search or news instead of stocks. The keyword router
        # correctly identifies these but is skipped when tool_call is not None.
        #
        # Extended override: run keyword router whenever LLM picks web_search OR news,
        # and replace with stocks if the query pattern matches a stock/price/commodity.
        # This prevents "how is reliance doing" → news, "TCS share price" → web_search.
        from .tools import WebSearchTool, NewsTool, StocksTool
        _should_override = (
            isinstance(tool_call, WebSearchTool)
            or (isinstance(tool_call, NewsTool)
                and _is_stock_query(query))  # news chosen for a stock query
        )
        if _should_override:
            better = _keyword_tool_from_query(query, _facts)
            if better is not None:
                original_tool = tool_call.tool
                logger.info(
                    "🔑 tool_dispatch.override  chat=%s  %s→%s  query=%r",
                    chat_id, original_tool, better.tool, query[:60],
                )
                tool_call = better

    logger.info("🔧 tool_dispatch  chat=%s  tool=%s", chat_id, tool_call.tool)

    # FIX-2: dispatch() occasionally returns a non-str (e.g. a dict from MCP) when
    # a tool's inner HTTP call short-circuits before the str-coercion layer.
    # Guard with explicit str() so .startswith() never raises AttributeError.
    raw_result = str(await tool_dispatcher.dispatch(tool_call, facts=_facts))

    if raw_result.startswith(_WEB_SEARCH_SENTINEL):
        search_query = raw_result[len(_WEB_SEARCH_SENTINEL):]
        return await _compound_beta_search(search_query, chat_id)

    return raw_result


def _keyword_tool_from_query(query: str, facts: Dict[str, str]) -> Optional[Any]:
    """
    Lightweight keyword-based tool routing for when the LLM doesn't emit tool_call.
    Covers the most common structured-data query types.
    """
    from .tools import WeatherTool, NewsTool, StocksTool, CurrencyTool, TimezoneTool, FetchUrlTool
    if not query:
        return None
    low = query.lower()

    # ── URL fetch — highest priority: if message contains a URL, fetch it ─
    _url_match = re.search(r"https?://[^\s]{8,}", query)
    if _url_match:
        return FetchUrlTool(tool="fetch_url", url=_url_match.group(0))

    # ── Weather ───────────────────────────────────────────────────────────
    if re.search(r"\b(weather|forecast|temperature|rain|humidity|wind|monsoon)\b", low):
        city = facts.get("city") or "Hyderabad"
        # Try extracting a capitalized city word from the query
        for w in query.split():
            w = w.strip(".,?!")
            if (len(w) >= 4 and w[0].isupper()
                    and w.lower() not in {"what", "tell", "give", "show",
                                          "today", "weather", "forecast", "india"}):
                city = w
                break
        return WeatherTool(
            tool="weather", city=city,
            country=facts.get("country", "IN")[:2].upper(), days=3,
        )

    # ── Commodities (gold / silver) — check BEFORE generic stocks ───────
    if re.search(
        r"\b(gold|silver|precious metal|bullion|commodity|commodities|"
        r"xau|xag|GC=F|SI=F|gold price|gold rate|silver price|silver rate)\b",
        low
    ):
        # Gold futures (COMEX) → USD price; bot will offer INR conversion
        if re.search(r"\b(silver|xag|SI=F)\b", low):
            return StocksTool(tool="stocks", symbols=["SI=F"])
        return StocksTool(tool="stocks", symbols=["GC=F"])

    # ── Portfolio query — user asks about their own holdings ─────────────
    if re.search(r"\b(portfolio|my stocks|my holdings|my shares|"
                 r"how.*my stock|stocks.*doing|holdings.*doing)\b", low):
        _facts = facts or {}

        # Prefer structured holdings (has qty + avg_price → full P&L review)
        holdings_json = _facts.get("portfolio_holdings", "")
        if holdings_json:
            # Signal to _dispatch_tool that we want the P&L review path
            # Use a special sentinel symbol that live_data recognises
            return StocksTool(tool="stocks", symbols=["__PORTFOLIO_REVIEW__"])

        # Fall back to flat ticker list (no cost basis, just price check)
        portfolio_str = _facts.get("portfolio_stocks", "")
        if portfolio_str:
            portfolio_tickers = [t.strip() for t in portfolio_str.split(",") if t.strip()]
            portfolio_tickers = [t if "." in t or t.startswith("^")
                                  else t + ".NS"
                                  for t in portfolio_tickers[:10]]
            if portfolio_tickers:
                return StocksTool(tool="stocks", symbols=portfolio_tickers)

        # No portfolio stored → return general market overview
        return StocksTool(tool="stocks", symbols=[])

    # ── Stocks / markets ──────────────────────────────────────────────────
    if re.search(
        r"\b(stock|share|price|nifty|sensex|bse|nse|market|equity|"
        r"paytm|reliance|tcs|infy|infosys|wipro|hdfc|icici|sbi|zomato|"
        r"adani|airtel|bajaj|ongc|kotak|ltimindtree|hcl|tech|mahindra)\b",
        low
    ):
        tickers = re.findall(r"\b([A-Z]{2,12}(?:\.NS|\.BO)?)\b", query)
        _SKIP = {"NSE", "BSE", "IPO", "MF", "ETF", "WHAT", "HOW", "THE",
                 "FOR", "AND", "OF", "IN", "ON", "AT", "TO", "BY",
                 # Currency/unit abbreviations — not tickers
                 "RS", "INR", "USD", "EUR", "GBP", "PER", "EACH",
                 "MY", "IS", "IT", "AS", "OR", "IF", "UP", "NO"}
        filtered = [t for t in tickers if t not in _SKIP]
        # Always append .NS for unqualified Indian tickers
        symbols = [t if "." in t or t.startswith("^") else t + ".NS"
                   for t in filtered[:5]]
        # If no explicit ticker found, return empty → MCP returns top indices
        return StocksTool(tool="stocks", symbols=symbols)

    # ── News / Sports scores ─────────────────────────────────────────────
    # Include sports because cricket/football scores are best served by GNews.
    # FIX-CRICKET: Groq 70B returns tool_call=web_search for score queries;
    # the override in _dispatch_tool redirects them here via keyword routing.
    if re.search(r"\b(news|headline|headlines|breaking|latest|current events|"
                 r"cricket|football|tennis|score|scorecard|ipl|match result|"
                 r"sports update|morning.*news|news.*round)\b", low):
        country = (facts.get("country") or "IN")[:2].upper()
        return NewsTool(tool="news", query=query[:200], country=country)

    # ── Currency ──────────────────────────────────────────────────────────
    m = re.search(r"\b([A-Z]{3})\s+(?:to|in)\s+([A-Z]{3})\b", query, re.IGNORECASE)
    if m or re.search(r"\b(exchange rate|forex|convert currency)\b", low):
        if m:
            return CurrencyTool(
                tool="currency",
                from_currency=m.group(1).upper(),
                to_currency=m.group(2).upper(),
                amount=1.0,
            )

    # ── Timezone / world clock ─────────────────────────────────────────────
    if re.search(r"\b(time in|timezone|local time in|what time is it in)\b", low):
        tm = re.search(r"(?:time in|timezone of|clock in)\s+([A-Za-z ]{3,25})", low)
        if tm:
            return TimezoneTool(tool="timezone", city=tm.group(1).strip().title())

    return None


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

        # FIX-COMPLAINT: Complaint / correction messages about the bot's behaviour
        # must NOT trigger memory extraction. They contain personal pronouns ("I asked
        # you...") but are describing the bot's error, not declaring personal facts.
        # Evidence: "I asked you for Japan time and you gave me India time" →
        # LLM extracted travel_plans='Japan' — wrong, harmful overwrite.
        _COMPLAINT_SIGNALS = re.compile(
            r"\b(you gave me|you told me|you said|i asked you|you gave|"
            r"you made a mistake|that.s wrong|that is wrong|incorrect|wrong answer|"
            r"you were wrong|that.s not right|you got it wrong|you misunderstood|"
            r"you ignored|you didn.t|you should have|you need to|"
            r"why did you|how could you|you failed|you messed up)\b",
            re.IGNORECASE,
        )
        if _COMPLAINT_SIGNALS.search(low):
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

        # ── ARCH-5: Value-presence pre-filter ───────────────────────────────
        # Before spending tokens on LLM verification, reject any proposed update
        # whose value does not appear (even approximately) in the user message.
        # This blocks hallucinated facts like travel_plans='Japan' from the
        # message "I asked you for Japan time" — 'Japan' is present but the
        # context is not a travel declaration.
        # Rule: if an existing non-junk value for this key already exists AND
        # the proposed new value is not a word-for-word substring of user_text,
        # require the value to be explicit — not just incidentally mentioned.
        # Note: this only runs for UPDATES to existing facts, not new creations,
        # to avoid blocking legitimate first-time saves.
        _user_lower = user_text.lower()
        pre_filtered: List[MemoryUpdate] = []
        for u in updates:
            if getattr(u, "delete", False):
                pre_filtered.append(u)
                continue
            existing_val = (existing_facts.get(u.key) or "").strip()
            proposed_val = (u.value or "").strip()
            # Only apply the guard when CHANGING an existing fact (not creating)
            if existing_val and existing_val.lower() != proposed_val.lower():
                # Check if the proposed value appears meaningfully in the message
                # (at least one substantive word from the value is present)
                value_words = set(re.findall(r"\b[a-z]{3,}\b", proposed_val.lower()))
                context_words = set(re.findall(r"\b[a-z]{3,}\b", _user_lower))
                overlap = value_words & context_words
                if not overlap:
                    # Value has no word overlap with the message at all — likely
                    # a hallucination or extraction from a prior context window.
                    logger.debug(
                        "verify.pre_filter  key=%s  value=%r  not in message  dropping",
                        u.key, proposed_val[:40],
                    )
                    continue
                # Check if this looks like an incidental mention rather than a
                # declaration. Complaint/correction patterns that contain the
                # value word but aren't asserting it as a personal fact.
                _complaint_about_bot = re.compile(
                    r"\b(you gave|you told|you said|i asked you|wrong|mistake|"
                    r"incorrect|should have|didn.t|didn't)\b",
                    re.IGNORECASE,
                )
                if _complaint_about_bot.search(user_text) and len(value_words) <= 2:
                    # Short value (1-2 words) found in a complaint → almost
                    # certainly incidental, not a declaration.
                    logger.debug(
                        "verify.pre_filter  key=%s  value=%r  incidental in complaint  dropping",
                        u.key, proposed_val[:40],
                    )
                    continue
            pre_filtered.append(u)

        if not pre_filtered:
            if trace:
                trace.tag(total_iterations=1, memory_extracted=len(updates),
                          memory_verified=0, memory_total=0)
            return []
        updates = pre_filtered
        # ────────────────────────────────────────────────────────────────────

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
    """
    Format the reply for WhatsApp markup.

    MCP-first approach: try POST /format on the local MCP sidecar — pure Python,
    deterministic, zero LLM tokens. Falls back to the Groq 8B LLM call only if
    MCP is unreachable. This alone saves ~50-100K Groq tokens/day.
    """
    step = trace.step("format") if trace else nullcontext()
    with step:
        if len(text) < _MIN_FORMAT_LEN:
            return text

        # ── Try MCP /format first (zero tokens) ──────────────────────────
        try:
            from .mcp_client import _client as _mcp_client
            resp = await _mcp_client().post("/format", json={"text": text}, timeout=4.0)
            resp.raise_for_status()
            result = resp.json()
            formatted = result.get("text") or text
            if result.get("changed"):
                logger.debug("🎨 format.mcp  chat=%s  len=%d→%d", chat_id, len(text), len(formatted))
            return formatted
        except Exception as mcp_err:
            logger.debug("format.mcp_unavailable  err=%s — falling back to LLM", str(mcp_err)[:80])

        # ── Fallback: Groq 8B LLM ─────────────────────────────────────────
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
    sender_name: str = "",
    is_group: bool = False,
) -> List[Dict[str, Any]]:
    """Build the message list for the orchestrator (first or second turn).

    sender_name / is_group are injected for group chats so the LLM knows
    WHICH member of the group is speaking. Without this, in a group with
    multiple users the LLM would answer "What's my name?" with the primary
    user's name regardless of who actually asked.
    """
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

    # Group chat: tell the LLM who is speaking so memory queries are scoped
    # to the correct person. Facts loaded are for the actual sender, not the
    # primary account, so "What's my name?" returns their name not the bot owner's.
    if is_group and sender_name:
        user_content_parts.append(f"SPEAKER: {sender_name} (group member asking this message)")
    elif is_group:
        user_content_parts.append("SPEAKER: a group member (name unknown)")

    user_content_parts.append(f"USER: {user_text}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": "\n\n".join(user_content_parts)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Conversation summary shortcut — zero orchestrator tokens
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_RE = re.compile(
    r"\b(summarize|summarise|summary|recap|what happened|what did we|"
    r"what have we|what was|catch me up|brief me|tell me about our|"
    r"our conversation|what did i|what have i)\b",
    re.IGNORECASE,
)

_WINDOW_RE = re.compile(
    r"\b(today|this morning|this afternoon|this evening|tonight|"
    r"last (hour|\d+ hours?)|yesterday|last (night|week|month)|"
    r"past (\d+)\s*(hours?|days?|weeks?)|last (24|48|72) hours?|"
    r"since (yesterday|last week|this morning|today)|"
    r"so far|just now|recently)\b",
    re.IGNORECASE,
)


def _parse_window_since(text: str) -> datetime:
    """
    Parse the start of the requested time window as a UTC datetime.
    Returns a UTC datetime representing "since when" to pull messages.
    """
    low  = text.lower()
    tz   = _get_local_tz()
    now  = datetime.now(tz)

    if any(w in low for w in ("this morning", "today", "so far", "this afternoon",
                               "this evening", "tonight")):
        # Since local midnight today
        return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)

    if any(w in low for w in ("yesterday", "last night")):
        # Since midnight the day before yesterday → covers all of yesterday
        yesterday_midnight = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        return yesterday_midnight.astimezone(UTC)

    if "last week" in low:
        return (now - timedelta(days=7)).astimezone(UTC)

    if "last month" in low:
        return (now - timedelta(days=30)).astimezone(UTC)

    # "last N hours" / "past N hours"
    m = re.search(r"\b(last|past)\s+(\d+)\s*hours?\b", low)
    if m:
        return (now - timedelta(hours=float(m.group(2)))).astimezone(UTC)

    # "last N days"
    m = re.search(r"\b(last|past)\s+(\d+)\s*days?\b", low)
    if m:
        return (now - timedelta(days=float(m.group(2)))).astimezone(UTC)

    # "last 24 / 48 / 72 hours"
    m = re.search(r"\b(24|48|72)\s*hours?\b", low)
    if m:
        return (now - timedelta(hours=float(m.group(1)))).astimezone(UTC)

    # Default: since local midnight today
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def _window_label(since_utc: datetime) -> str:
    """Human label for the window, e.g. 'today', 'yesterday', 'the last 7 days'."""
    tz    = _get_local_tz()
    now   = datetime.now(tz)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    since = since_utc.astimezone(tz)
    delta = now - since
    days  = delta.days

    if since.date() == today.date():
        return "today"
    if since.date() == (today - timedelta(days=1)).date():
        return "yesterday"
    if days <= 7:
        return f"the last {days} days"
    return f"the last {days} days"


async def _try_summary_shortcut(
    user_text: str,
    chat_id:   str,
    facts:     Dict[str, str],
) -> Optional[str]:
    """
    Intercept conversation-summary requests and answer from SQLite message_log.

    Returns a WhatsApp reply string, or None if not a summary request.
    Uses _parse_window_since() for accurate local-timezone windowing.
    Errors are logged at WARNING so they are visible without being fatal.
    """
    if not _SUMMARY_RE.search(user_text):
        return None
    if not _WINDOW_RE.search(user_text) and "conversation" not in user_text.lower():
        return None

    from . import database
    if not database.sqlite_store:
        logger.warning("summary_shortcut.skip  reason=no_sqlite_store")
        return None

    since_utc = _parse_window_since(user_text)
    since_iso = since_utc.isoformat()
    label     = _window_label(since_utc)

    logger.info(
        "📋 summary_shortcut.start  chat=%s  window=%r  since=%s",
        chat_id, label, since_iso[:19],
    )

    try:
        rows = await database.sqlite_store.get_messages_since(
            chat_id=chat_id, since_iso=since_iso, limit=150,
        )
    except Exception as exc:
        logger.warning("summary_shortcut.db_fail  chat=%s  err=%s", chat_id, exc)
        return None

    logger.info("📋 summary_shortcut.rows  chat=%s  count=%d", chat_id, len(rows))

    if not rows:
        user_name = facts.get("name", "")
        greeting  = _time_of_day_greeting()
        name_part = f", *{user_name}*" if user_name else ""
        return f"{greeting}{name_part}. We haven't chatted {label} yet. 🙂"

    # Build a readable transcript
    tz    = _get_local_tz()
    lines = []
    for direction, text, ts in rows:
        who = "You" if direction == "in" else "Shimmi"
        try:
            ts_dt  = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
            ts_str = ts_dt.strftime("%H:%M")
        except Exception:
            ts_str = ""
        prefix = f"[{ts_str}] " if ts_str else ""
        lines.append(f"{prefix}{who}: {text[:300]}")

    transcript = chr(10).join(lines[-80:])

    # Summarise with 8B model
    user_name = facts.get("name", "")
    name_part = f" The user's name is {user_name}." if user_name else ""
    try:
        raw = await _groq_raw(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are a WhatsApp assistant summarising a conversation.{name_part} "
                        "Write a concise bullet-point summary of key topics, decisions and facts. "
                        "Use past tense. Max 8 bullets. Use • for bullets. *bold* for key terms. "
                        "Start directly — no preamble, no filler."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            max_tokens=400,
            chat_id=chat_id,
            label="summary_shortcut",
            role="extract",
            timeout=20.0,
            json_mode=False,   # plain text — no JSON structure needed for summaries
        )
    except Exception as exc:
        logger.warning("summary_shortcut.llm_fail  chat=%s  err=%s", chat_id, str(exc)[:150])
        return None

    summary = (raw or "").strip()
    if not summary or len(summary) < 20:
        logger.warning("summary_shortcut.empty_reply  chat=%s  raw=%r", chat_id, (raw or "")[:80])
        return None

    greeting  = _time_of_day_greeting()
    user_name = facts.get("name", "")
    name_part = f", *{user_name}*" if user_name else ""
    header    = f"{greeting}{name_part}. Here's what we covered {label}:"
    reply     = header + chr(10) + chr(10) + summary

    logger.info(
        "📋 summary_shortcut.done  chat=%s  rows=%d  reply_len=%d",
        chat_id, len(rows), len(reply),
    )

    # Persist updated summary for future context
    try:
        from .database import normalize_key
        await database.sqlite_store.upsert_fact(
            chat_id.split("@")[0],
            normalize_key("conversation_summary"),
            summary,
            source="bot_inferred",
        )
    except Exception:
        pass

    return reply


def _time_of_day_greeting() -> str:
    """Return an emoji greeting based on current local time."""
    h = _now_local().hour
    if   6  <= h < 12: return "☀️ Good morning"
    elif 12 <= h < 17: return "🌤️ Good afternoon"
    elif 17 <= h < 21: return "🌆 Good evening"
    else:              return "🌙 Good night"


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

    # ── 1b. Zero-token time/date shortcut ─────────────────────────────────
    # LLMs frequently hallucinate the current time by reading stale timestamps
    # from conversation context. Handle these queries directly from the server
    # clock — zero tokens, always accurate.
    time_reply = _try_time_shortcut(user_text)
    if time_reply:
        logger.info("⚡ time.shortcut  chat=%s  reply=%r", chat_id, time_reply[:60])
        return AgentResult(
            reply=ReplyPayload(type="text", text=time_reply),
            provider_used="shortcut",
        )

    # ── 1b. Summary shortcut — answer time-windowed summary requests directly
    #        from SQLite message_log, zero orchestrator tokens
    summary_reply = await _try_summary_shortcut(user_text, chat_id, facts)
    if summary_reply is not None:
        logger.info("📋 summary.shortcut  chat=%s  reply_len=%d", chat_id, len(summary_reply))
        return AgentResult(
            reply=ReplyPayload(type="text", text=summary_reply),
            provider_used="summary_shortcut",
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
    # Detect group chats: chat_id ends in @g.us, sender_key is an individual
    _is_group   = chat_id.endswith("@g.us") if chat_id else False
    _sender_name = facts.get("name", "") if not _is_group else ""
    # In group chats, load the sender's own name from their facts if available
    # (facts are already loaded per-sender by main.py, so this is correct)

    messages = _build_orchestrator_messages(
        user_text, facts, context, reminders,
        sender_name=_sender_name, is_group=_is_group,
    )
    search_result: Optional[str] = None

    for iteration in range(1, _MAX_ITERATIONS + 1):
        label = f"orchestrate_{iteration}"

        _NO_MEM_TAG = "[SHIMMI_INTERNAL:NO_MEMORY_EXTRACT]"

        # Strip internal tag from search_result before it reaches the LLM
        if search_result and _NO_MEM_TAG in search_result:
            search_result = search_result.replace(_NO_MEM_TAG, "").rstrip()
            # Suppress pre-extracted memory updates — news content is not user data
            pre_updates = []
            logger.info("🔇 memory.suppressed  reason=news_result  iter=%d", iteration)

        if search_result:
            messages = _build_orchestrator_messages(
                user_text, facts, context, reminders, search_result,
                sender_name=_sender_name, is_group=_is_group,
            )

        orch = await _orchestrate(messages, chat_id, label=label, trace=trace)

        # ── Hallucination guard: force search for live-data queries ──────────
        # Groq 8B (last-resort fallback) sometimes answers action=answer for
        # stock / gold / price queries, hallucinating prices from training data.
        # Evidence: "TCS share price" → action=answer, reply="₹1,200.00" (fake).
        # Fix: if action=answer on iteration 1 AND the query is a stock/price
        # query, override to action=search so the correct MCP tool is called.
        # We only apply this on iteration 1 (iteration 2 has real search data).
        if (orch.action == "answer"
                and iteration == 1
                and settings.live_search_enabled
                and _is_stock_query(user_text)):
            logger.info(
                "🛡️  hallucination_guard  iter=%d  overriding action=answer→search  query=%r",
                iteration, user_text[:80],
            )
            # Build a keyword-routed tool_call and inject it into orch
            kw_tool = _keyword_tool_from_query(user_text, facts or {})
            if kw_tool is not None:
                # Patch orch to be a search action with the keyword-routed tool
                # Use object.__setattr__ since OrchestratorResult is Pydantic
                object.__setattr__(orch, "action", "search")
                object.__setattr__(orch, "query", user_text)
                object.__setattr__(orch, "tool_call", kw_tool.model_dump())

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
            search_result = await _dispatch_tool(orch.tool_call, query, chat_id, facts=facts)
            logger.info("🔍 search.done  iter=%d  query=%r  result_len=%d",
                        iteration, query, len(search_result or ""))

            # TAG news results so the memory extractor skips them.
            # News content (headlines, article snippets) is NOT personal user data
            # and must never be saved as facts. Without this, Groq 8B extracts
            # stock prices from financial headlines as user portfolio facts.
            _tool_name = (orch.tool_call or {}).get("tool", "") if isinstance(orch.tool_call, dict)                 else getattr(orch.tool_call, "tool", "")
            if _tool_name == "news":
                search_result = (search_result or "") + "\n\n[SHIMMI_INTERNAL:NO_MEMORY_EXTRACT]"
            if trace:
                trace.tag(
                    **{f"live_search_{iteration}": {
                        "search_query": query,
                        "result_len": len(search_result or ""),
                    }}
                )
            # ── Early exit: empty result after a structured MCP tool ─────────
            # When the news/weather/stocks MCP tool returned empty, there is no
            # point asking the LLM to "try again" — it will just dispatch the same
            # tool again and get empty again (which is what caused the 3-iteration
            # "morning news round up" and "cricket score" loops in production).
            # On an empty result from a non-web-search tool, answer honestly
            # rather than burning 2 more iterations.
            if not search_result and orch.tool_call is not None:
                from .tools import WebSearchTool
                # Only short-circuit for structured tools, not web search
                # (web search returning empty is handled by compound-beta internally)
                if not isinstance(orch.tool_call, WebSearchTool) if hasattr(orch, "tool_call") else True:
                    logger.info(
                        "🔍 search.empty_exit  iter=%d  tool=%s  — no data, answering directly",
                        iteration, getattr(orch.tool_call, "tool", "?"),
                    )
                    return AgentResult(
                        reply=ReplyPayload(
                            type="text",
                            text="I couldn't find any results for that right now. "
                                 "Please try rephrasing or try again in a moment.",
                        ),
                        iterations=iteration,
                    )

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
    ARCH-3: This function is intentionally a no-op.

    Previously this extracted facts from the bot's own replies — a circular design
    that caused real production bugs:
      • "favorite_biography='Wings of Fire'" saved from bot's own analysis
      • "conversation_summary" filled with hallucinated dates
      • "travel_plans='Japan'" saved from a user complaint, not a travel declaration

    The root problem: the orchestrator already returns memory_updates with everything
    it wants to remember. A second extraction pass on the bot's own output reads LLM
    inferences as if they were user-stated facts, then saves them to the DB, then
    they appear in future prompts, then the LLM hallucinates from them.

    ARCH-1 (source column) and the orchestrator's own memory_updates field together
    replace everything this function was trying to do — without the pollution.
    """
    # Deliberately empty. See docstring.
    return
    try:
        if not reply_text or len(reply_text) < 30:
            return

        stripped = reply_text.strip()

        # Skip questions — bot is seeking clarification, not confirming facts
        if stripped.rstrip().endswith("?"):
            return

        # Skip live-search result replies
        _live_markers = (
            "Open-Meteo", "Yahoo Finance", "GNews", "Google News",
            "📰 *Latest News*", "📈 *Indian Markets",
            "_Source:", "Source: ",
        )
        if any(m in reply_text for m in _live_markers):
            return

        # Skip template/placeholder text that reply_extract would mislabel as facts
        _placeholder_pats = (
            r"'s title\b",      # "book's title"
            r"'s name\b",       # "person's name"
            r"\[.*?\]",         # [city], [placeholder]
            r"\{[a-z_]+\}",    # {key}
        )
        for pat in _placeholder_pats:
            if re.search(pat, reply_text, re.IGNORECASE):
                logger.debug("reply_extract.skip_placeholder  pat=%r", pat)
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
            from .database import upsert_fact, delete_fact
            for u in result.memory_updates:
                try:
                    if getattr(u, "delete", False):
                        await delete_fact(sender_key, u.key)
                    else:
                        # Reply-extracted facts are bot observations from its own
                        # reply — never explicit user declarations. Mark bot_inferred
                        # so they never surface in LLM prompts via source_filter.
                        await upsert_fact(
                            sender_key, normalize_key(u.key), u.value,
                            source="bot_inferred",
                        )
                    logger.info(
                        "🧠 reply_memory.saved  sender=%s  key=%s  value=%r",
                        sender_key, u.key, u.value[:60],
                    )
                except Exception as db_err:
                    logger.warning("⚠️  reply_memory.db_fail  key=%s  err=%s",
                                   u.key, str(db_err)[:80])
    except Exception as e:
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
    original_keys = set(facts.keys())   # keys that actually exist before merging

    for merge in merges:
        try:
            canonical = normalize_key(str(merge.get("canonical", "")))
            absorb_raw = [normalize_key(str(k)) for k in (merge.get("absorb") or [])]
            value     = str(merge.get("value") or "").strip()

            if not canonical or not value:
                continue

            # Protection: never merge protected keys (shopping lists, SSN, etc.)
            # These have distinct semantics and must never be absorbed into
            # another key — a shopping list is not a book title.
            if canonical in _CONSOLIDATION_PROTECTED:
                logger.debug(
                    "consolidate.skip_protected  sender=%s  canonical=%s  (protected key)",
                    whatsapp_id, canonical,
                )
                continue

            # Safety: only absorb keys that actually exist in the DB AND differ from
            # canonical. Prevents the LLM from hallucinating keys or deleting a key
            # it incorrectly labeled as a duplicate of something else.
            absorb = [
                k for k in absorb_raw
                if k and k != canonical and k in original_keys
                and k not in _CONSOLIDATION_PROTECTED  # never absorb protected keys
            ]

            # Safety: canonical must also exist OR be a known alias of an existing key.
            # If neither the canonical nor any absorb key is in the DB, skip entirely.
            if canonical not in original_keys and not absorb:
                logger.debug(
                    "consolidate.skip_phantom  sender=%s  canonical=%s  (not in DB)",
                    whatsapp_id, canonical,
                )
                continue

            # FIX-CONSOL-NOOP: absorb=[] with an unchanged value is pure wasted work.
            # Evidence: 36 consolidate.merged log lines with absorbed=[] in one session
            # — the LLM was "merging" single keys with no duplicates, just re-writing
            # the same value. This burns ~512 tokens per consolidation run for nothing.
            # Skip the upsert when nothing would actually change.
            if not absorb and facts.get(canonical, "").strip() == value:
                logger.debug(
                    "consolidate.skip_noop  sender=%s  canonical=%s  (value unchanged, nothing to absorb)",
                    whatsapp_id, canonical,
                )
                continue

            # Write canonical key. Consolidation preserves provenance:
            # if the key already existed (user_stated), keep it user_stated.
            # If it's a new canonical from merging, mark bot_inferred.
            consol_source = "user_stated" if canonical in original_keys else "bot_inferred"
            status = await sqlite_store.upsert_fact(whatsapp_id, canonical, value, source=consol_source)

            # Delete only verified alias keys
            for alias in absorb:
                await _consolidation_delete(whatsapp_id, alias)

            if status != "unchanged" or absorb:
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
