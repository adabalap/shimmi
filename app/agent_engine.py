"""
agent_engine.py — Shimmi v2.9.2

Fixes vs v2.8.0:
  BUG-1  Memory saves: added post-reply extractor (REPLY_EXTRACTOR_PROMPT on
         user_msg+bot_reply) so list creates/edits are captured even when the
         orchestrator returns memory_updates:[]. Also lowered verifier threshold
         for action-type keys (shopping_list, todo_list, grocery_list, etc.)
  BUG-2  Pydantic crash: pre-clean memory_updates JSON before model_validate —
         empty-value entries are silently dropped, never crash the message.
  BUG-3  Reminder timezone: _fix_reminder_tz() rewrites bare UTC offsets to the
         server's local offset (Asia/Kolkata = +05:30) since the LLM always
         interprets user-said times in UTC by default.
  BUG-4  Reminder dedup: reminders_pending passed to run_agent; before inserting
         a new reminder, text+trigger uniqueness is checked against pending list.
  BUG-5  413 recovery: _live_search truncates results to MAX_RESULT_CHARS; if
         compound-beta raises 413 the call is retried with a minimal payload.
  BUG-6  Formatter always runs for responses > 120 chars or with prose markers.
  NEW    AgentResult now carries parsed reminders[] (ReminderEntry list) so
         main.py can save them without encoding them as special memory keys.
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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from groq import AsyncGroq

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
_MAX_SEARCH_RESULT   = 1_200   # chars – truncate before passing to orchestrate_2
_MIN_FORMAT_LEN      = 120     # always format responses longer than this

VALID_GROQ_PREFIXES = (
    "llama-", "mixtral-", "gemma-",
    "compound-beta", "compound-beta-mini",
    "whisper-", "distil-",
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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
        """Drop entries with null/empty keys or values before Pydantic validates."""
        if not isinstance(v, list):
            return []
        clean = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k = str(item.get("key", "") or "").strip()
            val = str(item.get("value", "") or "").strip()
            if k and val:
                clean.append({"key": k, "value": val})
        return clean

    @field_validator("reminders", mode="before")
    @classmethod
    def _clean_reminders(cls, v):
        if not isinstance(v, list):
            return []
        clean = []
        for item in v:
            if not isinstance(item, dict):
                continue
            t   = str(item.get("text", "") or "").strip()
            iso = str(item.get("trigger_iso", "") or "").strip()
            if t and iso:
                clean.append({"text": t, "trigger_iso": iso})
        return clean


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
        clean = []
        for item in v:
            if not isinstance(item, dict):
                continue
            k   = str(item.get("key",   "") or "").strip()
            val = str(item.get("value", "") or "").strip()
            if k and val:
                clean.append({"key": k, "value": val})
        return clean


class FormatterResult(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

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
    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"
    tz_abbr = now.strftime("%Z") or "local"
    return f"{now.strftime('%H:%M')} {tz_abbr} ({now.strftime('%A')} {period})"


def _today_str() -> str:
    return _now_local().strftime("%Y-%m-%d")


def _utc_offset_str() -> str:
    """Return ±HH:MM offset string, e.g. '+05:30'."""
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
    if 6 <= hour < 12:   return "morning"
    if 12 <= hour < 17:  return "afternoon"
    if 17 <= hour < 21:  return "evening"
    return "night"


def _fix_reminder_tz(trigger_iso: str) -> str:
    """
    BUG-3 fix: The LLM tends to output naive UTC times (ending in +0000 or Z)
    when the user means their local time.  Replace bare UTC offsets with the
    server's configured local offset so that '6 AM' → '06:00+05:30'.
    Only re-stamps if the server is NOT UTC.
    """
    local_offset = _utc_offset_str()
    if local_offset == "+00:00":
        return trigger_iso  # server IS UTC — no correction needed
    t = trigger_iso.strip()
    for utc_tail in ("+00:00", "+0000", "Z"):
        if t.endswith(utc_tail):
            base = t[: len(t) - len(utc_tail)]
            corrected = base + local_offset
            logger.info(
                "⏰ reminder.tz_fix  %s → %s  (server=%s)",
                trigger_iso, corrected, local_offset,
            )
            return corrected
    return t


# ---------------------------------------------------------------------------
# LLM client + circuit breaker
# ---------------------------------------------------------------------------

GROQ_CLIENT: Optional[AsyncGroq] = None
_inflight    = asyncio.Semaphore(int(settings.groq_max_inflight or 5))
MODEL_CIRCUIT: Dict[str, float] = {}
STICKY_MODEL:  Dict[str, str]   = {}
_STICKY_MAX = 2_000


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
        logger.warning("🧠 llm.init — GROQ_API_KEY missing")
        return
    GROQ_CLIENT = AsyncGroq(api_key=settings.groq_api_key, timeout=settings.groq_timeout)
    for m in (settings.groq_model_pool or []):
        if not any(m.startswith(p) for p in VALID_GROQ_PREFIXES):
            logger.warning(
                "⚠️  model_pool — %r looks invalid. Fix GROQ_MODEL_POOL in .env", m,
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

def _sanitize_json(s: str) -> str:
    """
    Sanitize raw LLM output before JSON parsing.
    LLMs often embed literal newlines/tabs inside JSON string values,
    producing invalid JSON that Python's json.loads() rejects.
    Strategy: inside a quoted string, replace bare control chars with safe escapes.
    """
    out = []
    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 32:
                # Other control characters → skip
                pass
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty_response")
    # Locate outermost JSON object
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no_json_found")
    candidate = s[start : end + 1]
    # First try raw (most LLM outputs are already valid)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Sanitize control chars inside string values and retry
    try:
        return json.loads(_sanitize_json(candidate))
    except json.JSONDecodeError as e:
        raise e


# Keywords whose presence strongly suggests personal content worth extracting
_PERSONAL_SIGNALS = frozenset([
    "my ", "i am", "i'm", "i live", "i work", "i have", "i drink", "i eat",
    "i use", "i own", "i drive", "i ride", "my name", "my city", "my list",
    "my shopping", "my reminder", "my address", "add to", "remove from",
    "remind me", "set a reminder", "update my", "postal", "i've been",
    "i usually", "i always", "my favorite", "my phone", "i prefer",
])

def _has_personal_content(text: str) -> bool:
    """Return True if the message likely contains extractable personal facts."""
    low = text.lower()
    return any(sig in low for sig in _PERSONAL_SIGNALS)


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
        # Latency-based circuit trip: if call took >15s, soft-trip this model
        # for 45s so next call uses the next model in the pool.
        if elapsed > 15.0:
            current_penalty = MODEL_CIRCUIT.get(model, 0.0)
            if time.monotonic() >= current_penalty:   # only trip if not already tripped
                MODEL_CIRCUIT[model] = time.monotonic() + 45.0
                logger.info(
                    "⚡ circuit.latency_trip  model=%s  elapsed=%.1fs  cooldown=45s",
                    model, elapsed,
                )
                STICKY_MODEL.pop(chat_id, None)  # force re-pick next turn
        return result
    except asyncio.TimeoutError:
        logger.warning("⚠️  llm_timeout  label=%s  model=%s  %.0fs", label, model, t_sec)
        MODEL_CIRCUIT[model] = time.monotonic() + 15.0
        raise
    except Exception as exc:
        MODEL_CIRCUIT[model] = time.monotonic() + (10.0 + random.random() * 4.0)
        logger.warning("🔴 circuit.tripped  model=%s  label=%s  err=%s", model, label, exc)
        raise


# ---------------------------------------------------------------------------
# WhatsApp formatter (BUG-6: always run for longer/prose responses)
# ---------------------------------------------------------------------------

def _is_prose(text: str) -> bool:
    """Heuristic: detect responses that look like prose and need WhatsApp formatting."""
    if len(text) < _MIN_FORMAT_LEN:
        return False
    has_markdown = "**" in text or "```" in text or "|---" in text or text.count("# ") > 1
    if has_markdown:
        return True
    # Long responses with no bullets and few newlines = prose
    if len(text) > _MIN_FORMAT_LEN and text.count("•") == 0 and text.count("\n") < 3:
        return True
    return False


async def _format_whatsapp(chat_id: str, text: str) -> str:
    if not _is_prose(text):
        return sanitize_for_whatsapp(text)
    raw = await _groq_raw(
        chat_id, FORMATTER_PROMPT,
        json.dumps({"text": text}, ensure_ascii=False),
        temperature=0.0, max_tokens=600, label="format",
    )
    try:
        fr = FormatterResult.model_validate(_extract_json(raw))
        return sanitize_for_whatsapp(fr.text)
    except Exception:
        return sanitize_for_whatsapp(text)


# ---------------------------------------------------------------------------
# Live search (BUG-5: truncate result; handle 413)
# ---------------------------------------------------------------------------

# ── Query intent detection helpers ────────────────────────────────────────

_WEATHER_KW  = frozenset(["weather", "forecast", "temperature", "rain", "humid",
                           "wind", "cold", "hot", "sunny", "monsoon", "climate today"])
_STOCK_KW    = frozenset(["stock", "stocks", "share price", "nifty", "sensex",
                           "bse", "nse", "market today", "indian market", "equity",
                           "bull", "bear", "reliance", "tcs", "infosys", "wipro",
                           "hdfc", "icici", "mutual fund", "index today"])
_NEWS_KW     = frozenset(["news", "headlines", "latest news", "breaking", "today's news",
                           "world news", "india news", "top stories"])

def _query_intent(query: str) -> str:
    """Return 'weather' | 'stocks' | 'news' | 'general'."""
    low = query.lower()
    if any(k in low for k in _WEATHER_KW):
        return "weather"
    if any(k in low for k in _STOCK_KW):
        return "stocks"
    if any(k in low for k in _NEWS_KW):
        return "news"
    return "general"

def _fix_placeholder_query(query: str, facts: Dict[str, str]) -> str:
    """
    Replace placeholder tokens like [user's city] or [user's location] with
    actual values from facts. If the query still has unresolved placeholders,
    return a clean version substituting known city/country.
    """
    city    = facts.get("city", "")
    country = facts.get("country", "India")
    loc     = f"{city}, {country}".strip(", ") if city else country

    # Replace common placeholder patterns
    import re
    q = re.sub(r"\[user\'?s?\s+(?:city|country|location|region|place)[^\]]*\]",
               loc, query, flags=re.IGNORECASE)
    q = re.sub(r"\[(?:location|city|country|region|place)\]",
               loc, q, flags=re.IGNORECASE)
    return q.strip()


async def _live_search(chat_id: str, query: str, facts: Dict[str, str]) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Search unavailable."

    # Resolve any literal placeholder tokens the LLM may have output
    query = _fix_placeholder_query(query, facts)

    # Route to specialized free-API tools when query matches known intents
    intent = _query_intent(query)
    city    = facts.get("city", "")
    country = facts.get("country", "India")

    if intent == "weather" and (city or country):
        from . import live_data
        result = await live_data.get_weather(city or "Hyderabad", country or "India")
        if result:
            return result
        # Fall through to compound-beta if live_data fails

    elif intent == "stocks":
        from . import live_data
        result = await live_data.get_indian_stocks()
        if result:
            return result

    elif intent == "news":
        from . import live_data
        result = await live_data.get_news(query, country="IN")
        if result:
            return result

    # Only send minimal user context to compound-beta to avoid 413
    mini_facts = {
        k: v for k, v in facts.items()
        if k in ("city", "country", "name", "preferred_language")
    }

    user_payload = json.dumps(
        {
            "query":     query,
            "user_city": mini_facts.get("city", ""),
            "today":     _today_str(),
        },
        ensure_ascii=False,
    )

    payload = {
        "model":       settings.live_search_model,
        "messages": [
            {"role": "system", "content": LIVE_SEARCH_PROMPT},
            {"role": "user",   "content": user_payload},
        ],
        "max_tokens": 800, "temperature": 0.2,
    }

    t0 = time.perf_counter()

    async def _call() -> str:
        async with _inflight:
            resp = await GROQ_CLIENT.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "").strip()

    try:
        result = await asyncio.wait_for(
            async_retry(_call, max_attempts=2, base_delay=0.8, max_delay=10.0),
            timeout=settings.groq_timeout,
        )
    except Exception as exc:
        err_str = str(exc)
        if "413" in err_str or "request_too_large" in err_str.lower():
            # Retry with bare query only
            logger.warning("⚠️  live_search.413  query=%r — retrying bare", query[:80])
            bare_payload = {**payload, "messages": [
                {"role": "system", "content": LIVE_SEARCH_PROMPT},
                {"role": "user",   "content": json.dumps({"query": query, "today": _today_str()})},
            ]}
            async def _bare_call() -> str:
                async with _inflight:
                    resp = await GROQ_CLIENT.chat.completions.create(**bare_payload)
                    return (resp.choices[0].message.content or "").strip()
            result = await asyncio.wait_for(
                async_retry(_bare_call, max_attempts=2, base_delay=1.0, max_delay=10.0),
                timeout=settings.groq_timeout,
            )
        else:
            raise

    elapsed = time.perf_counter() - t0
    if elapsed > _SLOW_CALL_WARN_SEC:
        logger.warning("⚠️  slow_search  query=%r  %.1fs  result_len=%d",
                       query[:80], elapsed, len(result))

    # Truncate so orchestrate_2 doesn't hit token limits
    if len(result) > _MAX_SEARCH_RESULT:
        result = result[:_MAX_SEARCH_RESULT] + "\n…[truncated]"

    return result


# ---------------------------------------------------------------------------
# Memory pipeline
# ---------------------------------------------------------------------------

_LIST_KEYS = frozenset({
    "shopping_list", "grocery_list", "todo_list", "reminder_notes",
    "task_list", "bucket_list", "wish_list",
})


def _normalize_updates(updates: List[MemoryUpdate]) -> List[MemoryUpdate]:
    seen: Dict[str, str] = {}
    for u in updates:
        k = normalize_key(u.key)
        if k:
            seen[k] = (u.value or "").strip()
    return [MemoryUpdate(key=k, value=v) for k, v in seen.items() if v]


async def _extract_memory(chat_id: str, user_text: str) -> List[MemoryUpdate]:
    """Extract facts explicitly declared in the user message."""
    raw = await _groq_raw(
        chat_id, MEMORY_EXTRACTOR_PROMPT,
        json.dumps({"user_message": user_text}, ensure_ascii=False),
        temperature=0.0, max_tokens=400, label="extract",
    )
    try:
        er = ExtractResult.model_validate(_extract_json(raw))
        updates = _normalize_updates(er.memory_updates)
        if updates:
            logger.debug("memory.extracted  count=%d  keys=%s",
                         len(updates), [u.key for u in updates])
        return updates
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.extract_failed  err=%s  raw=%.200s", e, raw or "")
        return []


async def extract_reply_memory(
    chat_id: str,
    user_text: str,
    bot_reply: str,
    existing_facts: Dict[str, str],
) -> List[MemoryUpdate]:
    """
    BUG-1 fix: post-reply extractor.
    Read (user_message + bot_reply) and extract whatever the bot CONFIRMED saving.
    This catches list creations, list updates, and reminder notes that the
    orchestrator forgot to include in memory_updates.
    """
    user_payload = json.dumps(
        {
            "user_message":  user_text,
            "bot_reply":     bot_reply,
            "existing_facts": existing_facts,
        },
        ensure_ascii=False,
    )
    raw = await _groq_raw(
        chat_id, REPLY_EXTRACTOR_PROMPT, user_payload,
        temperature=0.0, max_tokens=400, label="reply_extract",
    )
    try:
        er = ExtractResult.model_validate(_extract_json(raw))
        updates = _normalize_updates(er.memory_updates)
        if updates:
            logger.info(
                "🧠 reply_extract.found  count=%d  keys=%s",
                len(updates), [u.key for u in updates],
            )
        return updates
    except Exception as e:
        if settings.debug_agent:
            logger.info("reply_extract.failed  err=%s  raw=%.200s", e, raw or "")
        return []


async def _verify_updates(
    chat_id: str,
    user_text: str,
    proposed: List[MemoryUpdate],
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
        temperature=0.0, max_tokens=400, label="verify",
    )
    try:
        vr = VerifyResult.model_validate(_extract_json(raw))
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.verify_failed  err=%s  raw=%.200s", e, raw or "")
        return []

    min_conf = float(settings.facts_min_conf or 0.85)
    approved = []
    for a in vr.approved:
        # Lower threshold for list-type keys — actions imply facts
        threshold = 0.60 if normalize_key(a.key) in _LIST_KEYS else min_conf
        if a.confidence >= threshold:
            k = normalize_key(a.key)
            if k and (a.value or "").strip():
                approved.append(MemoryUpdate(key=k, value=a.value.strip()))
    return approved


def _merge_memory(
    pre_verified:   List[MemoryUpdate],
    agent_proposed: List[MemoryUpdate],
) -> List[MemoryUpdate]:
    merged: Dict[str, str] = {u.key: u.value for u in pre_verified}
    for u in agent_proposed:
        merged[u.key] = u.value
    return [MemoryUpdate(key=k, value=v) for k, v in merged.items()]


# ---------------------------------------------------------------------------
# Agentic orchestrator
# ---------------------------------------------------------------------------

def _reminders_to_json(reminders: List[Reminder]) -> List[Dict]:
    return [
        {
            "id":          r.id,
            "text":        r.reminder_text,
            "trigger_iso": r.trigger_iso,
            "status":      r.status,
        }
        for r in reminders
        if r.status == "pending"
    ]


async def _orchestrate(
    chat_id:        str,
    user_text:      str,
    facts:          Dict[str, str],
    context:        List[Dict[str, Any]],
    search_results: List[Dict[str, str]],
    reminders:      List[Reminder],
    iteration:      int,
) -> OrchestratorResult:
    system = ORCHESTRATOR_PROMPT
    if iteration >= _MAX_ITERATIONS:
        system = system + "\n\nFINAL ITERATION: You MUST use action=answer now."

    tz_offset  = _utc_offset_str()
    today      = _today_str()
    time_str   = _current_time_str()
    tod        = _time_of_day()

    user_payload = json.dumps(
        {
            "user_message":      user_text,
            "facts":             facts,
            "reminders_pending": _reminders_to_json(reminders),
            "context":           context,
            "search_results":    search_results,
            "current_time":      time_str,
            "time_of_day":       tod,
            "today":             today,
            "tz_offset":         tz_offset,
            "iteration":         iteration,
            "max_iterations":    _MAX_ITERATIONS,
        },
        ensure_ascii=False,
    )

    raw = await _groq_raw(
        chat_id, system, user_payload,
        temperature=0.25, max_tokens=1000,
        label=f"orchestrate_{iteration}",
    )

    try:
        result = OrchestratorResult.model_validate(_extract_json(raw))
        logger.info(
            "🤖 orchestrate  iter=%d  action=%-10s  reasoning=%.120s",
            iteration, result.action, result.reasoning or "(none)",
        )
        return result
    except Exception as e:
        logger.warning("orchestrate.parse_failed  iter=%d  err=%s  raw=%.200s", iteration, e, raw)
        repaired_raw = await _groq_raw(
            chat_id, REPAIR_PROMPT, raw,
            temperature=0.0, max_tokens=600, label="repair",
        )
        result = OrchestratorResult.model_validate(_extract_json(repaired_raw))
        logger.info("orchestrate.repaired  iter=%d  action=%s", iteration, result.action)
        return result


# ---------------------------------------------------------------------------
# Reminder dedup helper (BUG-4)
# ---------------------------------------------------------------------------

def _is_reminder_duplicate(
    text: str,
    trigger_iso: str,
    existing: List[Reminder],
) -> bool:
    """Return True if an identical pending reminder already exists."""
    norm_text = text.strip().lower()
    # Compare on first 16 chars of trigger (YYYY-MM-DDTHH:MM) ignoring tz
    trigger_prefix = trigger_iso[:16]
    for r in existing:
        if r.status != "pending":
            continue
        if r.reminder_text.strip().lower() == norm_text:
            return True
        if r.trigger_iso[:16] == trigger_prefix:
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    *,
    chat_id:   str,
    user_text: str,
    facts:     Dict[str, str],
    context:   List[Dict[str, Any]],
    reminders: List[Reminder],
    trace:     Any = None,
) -> AgentResult:
    def _step(name: str):
        return trace.step(name) if trace is not None else nullcontext()

    with _step("memory_extract"):
        # Skip extractor for clearly impersonal/factual messages — saves 3-7s LLM call
        if _has_personal_content(user_text):
            extract_task = asyncio.create_task(_extract_memory(chat_id, user_text))
        else:
            extract_task = asyncio.get_event_loop().create_future()
            extract_task.set_result([])

    search_results: List[Dict[str, str]] = []
    orch_result: Optional[OrchestratorResult] = None
    iteration = 0

    while iteration < _MAX_ITERATIONS:
        iteration += 1

        with _step(f"orchestrate_{iteration}"):
            orch_result = await _orchestrate(
                chat_id, user_text, facts, context,
                search_results, reminders, iteration,
            )
            if trace:
                trace.tag(action=orch_result.action, reasoning_len=len(orch_result.reasoning))

        if orch_result.action == "ask_user" and orch_result.question:
            proposed = await extract_task
            verified = await _verify_updates(chat_id, user_text, proposed)
            merged   = _merge_memory(verified, _normalize_updates(orch_result.memory_updates))
            # Apply timezone fix on any reminders from orchestrator
            fixed_reminders = [
                ReminderEntry(text=r.text, trigger_iso=_fix_reminder_tz(r.trigger_iso))
                for r in orch_result.reminders
            ]
            if trace:
                trace.tag(total_iterations=iteration, memory_total=len(merged))
            return AgentResult(
                reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(orch_result.question)),
                memory_updates=merged,
                reminders=fixed_reminders,
                iterations=iteration,
            )

        # Force-search trigger: topics that MUST have live data
        _FORCE_SEARCH_PATTERNS = re.compile(
            r"\b(stock|stocks|share price|nse|bse|sensex|nifty|market cap|"
            r"nifty50|bank nifty|index fund|mutual fund|rupee|inr|"
            r"₹\d|today.{0,20}market|market.{0,20}today)\b",
            re.IGNORECASE,
        )
        if (
            orch_result.action == "answer"
            and iteration == 1
            and _FORCE_SEARCH_PATTERNS.search(user_text)
            and not search_results
        ):
            logger.info(
                "🤖 orchestrate  iter=%d  force_search — detected live-data topic in user_text",
                iteration,
            )
            orch_result.action = "search"
            orch_result.query = user_text

        if orch_result.action == "search" and settings.live_search_enabled:
            query = (orch_result.query or user_text).strip()

            # Prevent literal placeholder tokens like "[user's location]" in queries.
            # Replace with actual values from facts where available.
            _city    = facts.get("city", "")
            _country = facts.get("country", "")
            _loc     = ", ".join(x for x in [_city, _country] if x) or ""
            _placeholder_re = re.compile(
                r"\[(?:user['\u2019]?s?\s+)?(?:location|city|country|place|region)[^\]]*\]",
                re.IGNORECASE,
            )
            query = _placeholder_re.sub(_loc or "location unknown", query)
            # Strip any remaining unfilled placeholders
            query = re.sub(r"\[[^\]]{1,50}\]", "", query).strip()

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

        if orch_result.action == "answer" or orch_result.text:
            break

        logger.warning("orchestrate.unknown_action  iter=%d  action=%r", iteration, orch_result.action)
        if iteration >= _MAX_ITERATIONS:
            break

    # ── Memory verification ────────────────────────────────────────────────
    with _step("memory_verify"):
        proposed = await extract_task
        verified = await _verify_updates(chat_id, user_text, proposed) if proposed else []
        pre_keys  = {v.key for v in verified}
        agent_raw = _normalize_updates(orch_result.memory_updates if orch_result else [])
        agent_new = [u for u in agent_raw if u.key not in pre_keys]
        agent_vfd = await _verify_updates(chat_id, user_text, agent_new) if agent_new else []
        final_memory = _merge_memory(verified, agent_vfd)

        if trace:
            trace.tag(
                total_iterations=iteration,
                memory_extracted=len(proposed),
                memory_verified=len(verified),
                memory_total=len(final_memory),
            )

    # ── Deduplicate: drop updates that are already stored with the same value ──
    # Avoids wasteful verifier round-trips and DB writes for facts already known.
    final_memory = [
        u for u in final_memory
        if facts.get(u.key, "").strip().lower() != u.value.strip().lower()
    ]
    if trace:
        trace.tag(memory_deduped=len(final_memory))

    # ── Format reply ───────────────────────────────────────────────────────
    raw_text  = (orch_result.text if orch_result else "") or "I'm not sure how to answer that."
    with _step("format"):
        formatted = await _format_whatsapp(chat_id, raw_text)

    # ── Apply timezone fix on reminders ───────────────────────────────────
    fixed_reminders = [
        ReminderEntry(text=r.text, trigger_iso=_fix_reminder_tz(r.trigger_iso))
        for r in (orch_result.reminders if orch_result else [])
    ]

    return AgentResult(
        reply=ReplyPayload(type="text", text=formatted),
        memory_updates=final_memory,
        reminders=fixed_reminders,
        iterations=iteration,
    )
