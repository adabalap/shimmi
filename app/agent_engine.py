"""
agent_engine.py — Shimmi v3.0.2

Changes vs v3.0.1:
  FIX-1  Role-based model routing: orchestration uses ORCHESTRATOR_MODEL
         (llama-3.3-70b-versatile); extract/verify/format/reply_extract
         use EXTRACTION_MODEL (llama-3.1-8b-instant). Config fields were
         defined in config.py but never read here — now they are.
         Impact: 70B token usage drops by ~65% per message.

  FIX-2  Retry-after parsing: 429 errors include "Please try again in Xm Ys".
         Circuit cooldown is now set to the actual Groq retry-after time
         instead of a fixed 10-14s jitter. Prevents hammering an exhausted
         model and ensures recovery at the right moment.

  FIX-3  compound-beta-mini removed from orchestrator pool routing.
         compound-beta-mini is the LIVE SEARCH model only. It internally
         uses llama-3.3-70b and shares the same daily token bucket — having
         it in the orchestrator pool drains that bucket even faster.

  FIX-4  Unknown-value fact filtering: facts where value is 'unknown',
         'none', '' are stripped before passing to orchestrator prompt,
         saving ~200-400 tokens per message.

  FIX-5  Facts recall short-circuit: queries that are clearly memory
         lookups ("what's my X?", "do you know my Y?") and have all the
         needed facts already loaded skip the LLM orchestrator and answer
         directly. This prevents burning 70B tokens to echo the DB.

  FIX-6  asyncio.get_event_loop() deprecated call → asyncio.get_running_loop()
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
_MIN_FORMAT_LEN      = 120

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

    @field_validator("reminders", mode="before")
    @classmethod
    def _clean_reminders(cls, v):
        if not isinstance(v, list):
            return []
        clean = []
        for item in v:
            if not isinstance(item, dict):
                continue
            t   = str(item.get("text",        "") or "").strip()
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
    """Pick the next available model from the pool (excluding compound-beta-mini)."""
    pool = [
        m for m in (settings.groq_model_pool or [])
        # compound-beta-mini is the live-search model; never use it for orchestration.
        # It shares the llama-3.3-70b daily token bucket and would drain it faster.
        if not m.startswith("compound-beta")
    ]
    if not pool:
        pool = ["llama-3.3-70b-versatile"]

    sticky = STICKY_MODEL.get(chat_id)
    if sticky and not sticky.startswith("compound-beta") and _model_open(sticky):
        return sticky

    for m in pool:
        if _model_open(m):
            if len(STICKY_MODEL) >= _STICKY_MAX:
                evict = random.sample(list(STICKY_MODEL.keys()), min(200, len(STICKY_MODEL)))
                for k in evict:
                    STICKY_MODEL.pop(k, None)
            STICKY_MODEL[chat_id] = m
            return m

    # All exhausted — return first in pool (will likely 429, but caller handles it)
    STICKY_MODEL[chat_id] = pool[0]
    return pool[0]


def _pick_model_for_role(chat_id: str, role: str) -> str:
    """
    FIX-1: Route calls to the correct model tier based on role.

    - "orchestrate"                    → ORCHESTRATOR_MODEL (70b) with pool fallback
    - "extract"|"verify"|"format"|
      "reply_extract"|"repair"         → EXTRACTION_MODEL (8b) with pool fallback

    The 70B model has 100K tokens/day on Groq free tier.
    The 8B model has a much higher (500K+) daily limit.
    By routing lightweight calls to 8B, we preserve 70B budget for reasoning.
    """
    if role == "orchestrate":
        preferred = settings.orchestrator_model or "llama-3.3-70b-versatile"
    else:
        # All lightweight calls (extract, verify, format, repair, reply_extract)
        # use the fast small model.
        preferred = settings.extraction_model or "llama-3.1-8b-instant"

    if preferred and _model_open(preferred):
        return preferred

    # Preferred is circuit-tripped — fall back to pool
    fallback = _pick_model(chat_id)
    logger.info(
        "🔄 model.fallback  role=%s  preferred=%s  using=%s",
        role, preferred, fallback,
    )
    return fallback


def _parse_retry_after(exc: Exception) -> float:
    """
    FIX-2: Parse 'Please try again in Xh Ym Z.Ws' from Groq 429 error text.
    Returns the number of seconds to wait, or 120.0 if not parseable.
    """
    msg = str(exc)
    m = re.search(
        r"try again in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:([\d.]+)s)?",
        msg, re.IGNORECASE,
    )
    if m:
        h    = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = float(m.group(3) or 0)
        total = h * 3600 + mins * 60 + secs
        return max(total, 60.0)   # at least 60s cooldown
    return 120.0   # safe default: 2 min


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
    logger.info(
        "🧠 llm.init — AsyncGroq ready  orchestrator=%s  extraction=%s  live_search=%s  pool=%s",
        settings.orchestrator_model, settings.extraction_model,
        settings.live_search_model, settings.groq_model_pool,
    )


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
    out = []
    in_string = False
    escaped   = False
    for ch in s:
        if escaped:
            out.append(ch); escaped = False; continue
        if ch == "\\":
            out.append(ch); escaped = True; continue
        if ch == '"' and not escaped:
            in_string = not in_string
            out.append(ch); continue
        if in_string:
            if   ch == "\n": out.append("\\n")
            elif ch == "\r": out.append("\\r")
            elif ch == "\t": out.append("\\t")
            elif ord(ch) < 32: pass
            else: out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty_response")
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no_json_found")
    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_sanitize_json(candidate))
    except json.JSONDecodeError as e:
        raise e


_PERSONAL_SIGNALS = frozenset([
    "my ", "i am", "i'm", "i live", "i work", "i have", "i drink", "i eat",
    "i use", "i own", "i drive", "i ride", "my name", "my city", "my list",
    "my shopping", "my reminder", "my address", "add to", "remove from",
    "remind me", "set a reminder", "update my", "postal", "i've been",
    "i usually", "i always", "my favorite", "my phone", "i prefer",
])

def _has_personal_content(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in _PERSONAL_SIGNALS)


# FIX-4: Strip junk/unknown values before passing facts to LLM
_JUNK_VALUES = frozenset({"unknown", "none", "null", "n/a", "", "-", "—"})

def _clean_facts(facts: Dict[str, str]) -> Dict[str, str]:
    """Remove entries whose value is empty/unknown — they waste tokens."""
    return {
        k: v for k, v in facts.items()
        if (v or "").strip().lower() not in _JUNK_VALUES
    }


async def _groq_raw(
    chat_id:        str,
    system:         str,
    user:           str,
    temperature:    float,
    max_tokens:     int,
    *,
    model_override: Optional[str] = None,
    role:           str = "extract",   # FIX-1: role for model selection
    label:          str = "call",
    timeout_sec:    Optional[float] = None,
) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return ""

    # FIX-1: pick model by role (orchestrate vs extract/verify/format)
    model  = model_override or _pick_model_for_role(chat_id, role)
    t_sec  = timeout_sec or settings.groq_timeout

    t0 = time.perf_counter()

    async def _call() -> str:
        payload = {
            "model":       model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens":  int(max_tokens),
        }
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
        # Latency-based circuit trip: >15s → soft-trip 45s
        if elapsed > 15.0:
            current_penalty = MODEL_CIRCUIT.get(model, 0.0)
            if time.monotonic() >= current_penalty:
                MODEL_CIRCUIT[model] = time.monotonic() + 45.0
                logger.info(
                    "⚡ circuit.latency_trip  model=%s  elapsed=%.1fs  cooldown=45s",
                    model, elapsed,
                )
                STICKY_MODEL.pop(chat_id, None)
        return result

    except asyncio.TimeoutError:
        logger.warning("⚠️  llm_timeout  label=%s  model=%s  %.0fs", label, model, t_sec)
        MODEL_CIRCUIT[model] = time.monotonic() + 15.0
        raise

    except Exception as exc:
        # FIX-2: use actual retry-after time from 429 response
        cooldown = _parse_retry_after(exc) if ("429" in str(exc) or "rate_limit" in str(exc).lower()) else (10.0 + random.random() * 4.0)
        MODEL_CIRCUIT[model] = time.monotonic() + cooldown
        logger.warning(
            "🔴 circuit.tripped  model=%s  label=%s  cooldown=%.0fs  err=%s",
            model, label, cooldown, exc,
        )
        raise


# ---------------------------------------------------------------------------
# WhatsApp formatter
# ---------------------------------------------------------------------------

def _is_prose(text: str) -> bool:
    if len(text) < _MIN_FORMAT_LEN:
        return False
    has_markdown = "**" in text or "```" in text or "|---" in text or text.count("# ") > 1
    if has_markdown:
        return True
    if len(text) > _MIN_FORMAT_LEN and text.count("•") == 0 and text.count("\n") < 3:
        return True
    return False


async def _format_whatsapp(chat_id: str, text: str) -> str:
    if not _is_prose(text):
        return sanitize_for_whatsapp(text)
    raw = await _groq_raw(
        chat_id, FORMATTER_PROMPT,
        json.dumps({"text": text}, ensure_ascii=False),
        temperature=0.0, max_tokens=600,
        role="format", label="format",  # FIX-1: use extraction_model
    )
    try:
        fr = FormatterResult.model_validate(_extract_json(raw))
        return sanitize_for_whatsapp(fr.text)
    except Exception:
        return sanitize_for_whatsapp(text)


# ---------------------------------------------------------------------------
# Live search  (compound-beta-mini ONLY — not in the orchestrator pool)
# ---------------------------------------------------------------------------

_WEATHER_KW  = frozenset(["weather", "forecast", "temperature", "rain", "humid",
                           "wind", "cold", "hot", "sunny", "monsoon", "climate today"])
_STOCK_KW    = frozenset(["stock", "stocks", "share price", "nifty", "sensex",
                           "bse", "nse", "market today", "indian market", "equity",
                           "bull", "bear", "reliance", "tcs", "infosys", "wipro",
                           "hdfc", "icici", "mutual fund", "index today"])
_NEWS_KW     = frozenset(["news", "headlines", "latest news", "breaking", "today's news",
                           "world news", "india news", "top stories"])
_CURRENCY_KW = frozenset(["exchange rate", "currency", "convert", "usd to", "inr to",
                           "eur to", "gbp to", "dollar to", "rupee to", "how much is",
                           "forex", "₹ to $", "$ to ₹"])
_TIMEZONE_KW = frozenset(["time in", "what time is it in", "current time in", "timezone",
                           "time zone", "local time in", "what's the time in"])


def _query_intent(query: str) -> str:
    low = query.lower()
    if any(k in low for k in _WEATHER_KW):  return "weather"
    if any(k in low for k in _STOCK_KW):    return "stocks"
    if any(k in low for k in _NEWS_KW):     return "news"
    if any(k in low for k in _CURRENCY_KW): return "currency"
    if any(k in low for k in _TIMEZONE_KW): return "timezone"
    return "general"


def _fix_placeholder_query(query: str, facts: Dict[str, str]) -> str:
    city    = facts.get("city",    "")
    country = facts.get("country", "India")
    loc     = f"{city}, {country}".strip(", ") if city else country
    q = re.sub(
        r"\[user\'?s?\s+(?:city|country|location|region|place)[^\]]*\]",
        loc, query, flags=re.IGNORECASE,
    )
    q = re.sub(r"\[(?:location|city|country|region|place)\]", loc, q, flags=re.IGNORECASE)
    return q.strip()


async def _live_search(chat_id: str, query: str, facts: Dict[str, str]) -> str:
    if not GROQ_CLIENT:
        await init_llm()
    if not GROQ_CLIENT:
        return "Search unavailable."

    query  = _fix_placeholder_query(query, facts)
    intent = _query_intent(query)
    city    = facts.get("city",    "")
    country = facts.get("country", "India")

    # Route structured queries to MCP / live_data first (free, no token cost)
    if intent == "weather" and (city or country):
        from . import live_data
        result = await live_data.get_weather(city or "Hyderabad", country or "India")
        if result:
            return result

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

    elif intent == "currency":
        # Parse "USD to INR 100" or "convert 50 USD to EUR"
        from .mcp_client import mcp_currency
        import re as _re
        _m = _re.search(
            r"(?:(\d+(?:\.\d+)?)\s+)?([A-Z]{3})\s+to\s+([A-Z]{3})",
            query, _re.IGNORECASE,
        )
        if _m:
            amount    = float(_m.group(1) or 1)
            from_cur  = _m.group(2).upper()
            to_cur    = _m.group(3).upper()
            data = await mcp_currency(from_cur=from_cur, to_cur=to_cur, amount=amount)
            if data:
                return (
                    f"💱 *{from_cur} → {to_cur}*\n"
                    f"{amount} {from_cur} = *{data['converted']} {to_cur}*\n"
                    f"Rate: 1 {from_cur} = {data['rate']} {to_cur}\n"
                    f"_Source: Frankfurter (ECB rates, as of {data.get('as_of', 'today')})_"
                )

    elif intent == "timezone":
        # Extract city from "what time is it in Tokyo"
        from .mcp_client import mcp_timezone
        import re as _re
        _m = _re.search(r"(?:time in|time is it in|current time in)\s+(.+?)(?:\?|$)", query, _re.IGNORECASE)
        city_q = (_m.group(1).strip() if _m else city) or "London"
        data = await mcp_timezone(city=city_q)
        if data:
            lt  = data.get("local_time", "")
            tz  = data.get("timezone",   "")
            off = data.get("utc_offset", "")
            dow = data.get("day_of_week", "")
            time_part = ""
            if lt:
                try:
                    from datetime import datetime as _dt
                    parsed    = _dt.fromisoformat(lt.replace("Z",""))
                    time_part = parsed.strftime("%I:%M %p")
                except Exception:
                    time_part = lt[:16]
            return (
                f"🕐 *{data.get('city', city_q)}*\n"
                f"Local time: *{time_part}* ({dow})\n"
                f"Timezone: {tz}  (UTC{off})"
            )

    # Fallback: compound-beta-mini web search
    mini_facts = {k: v for k, v in facts.items() if k in ("city", "country", "name")}
    user_payload = json.dumps(
        {"query": query, "user_city": mini_facts.get("city", ""), "today": _today_str()},
        ensure_ascii=False,
    )

    # compound-beta-mini uses its own dedicated model (NOT the orchestrator pool)
    payload = {
        "model":       settings.live_search_model,   # compound-beta-mini
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
        logger.warning(
            "⚠️  slow_search  query=%r  %.1fs  result_len=%d",
            query[:80], elapsed, len(result),
        )

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
    raw = await _groq_raw(
        chat_id, MEMORY_EXTRACTOR_PROMPT,
        json.dumps({"user_message": user_text}, ensure_ascii=False),
        temperature=0.0, max_tokens=400,
        role="extract", label="extract",  # FIX-1: uses extraction_model (8b)
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
    user_payload = json.dumps(
        {"user_message": user_text, "bot_reply": bot_reply, "existing_facts": existing_facts},
        ensure_ascii=False,
    )
    raw = await _groq_raw(
        chat_id, REPLY_EXTRACTOR_PROMPT, user_payload,
        temperature=0.0, max_tokens=400,
        role="extract", label="reply_extract",  # FIX-1: uses extraction_model (8b)
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
        temperature=0.0, max_tokens=400,
        role="verify", label="verify",  # FIX-1: uses extraction_model (8b)
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
# FIX-5: Facts recall short-circuit
# ---------------------------------------------------------------------------

_RECALL_PATTERNS = re.compile(
    r"\b(?:what(?:'s|\ is)|do you know|tell me|remind me of|recall|"
    r"show me|list|what are|do i have)\b.{0,40}"
    r"\b(?:my|me)\b.{0,40}"
    r"\b(?:name|city|age|birthday|hobbies|interests|favorite|allergies|"
    r"occupation|job|car|vehicle|pets?|shopping.?list|grocery.?list|"
    r"todo.?list|reminder|country|postal)\b",
    re.IGNORECASE,
)

def _try_facts_shortcut(user_text: str, facts: Dict[str, str]) -> Optional[str]:
    """
    FIX-5: For simple memory-recall questions we can answer directly from
    the facts dict without spending a 70B token call on orchestration.

    Returns a pre-formatted reply string, or None if LLM is needed.
    """
    if not _RECALL_PATTERNS.search(user_text):
        return None

    low = user_text.lower()

    # Only short-circuit if question asks about a single well-known key
    _RECALL_KEYS = {
        "name": ["name"],
        "city": ["city", "where i live", "my city"],
        "age":  ["age", "how old"],
        "birthday": ["birthday", "born"],
        "hobbies": ["hobbie", "hobby"],
        "interests": ["interest"],
        "favorite drink": ["drink", "coffee", "tea"],
        "favorite food": ["food", "eat", "cuisine"],
        "allergies": ["allerg"],
        "occupation": ["job", "work", "occupation", "profession"],
        "car": ["car", "vehicle", "drive", "bike"],
        "pets": ["pet", "dog", "cat"],
        "shopping_list": ["shopping list", "shopping_list"],
        "grocery_list":  ["grocery list", "grocery_list", "groceries"],
        "todo_list":     ["todo list", "todo_list", "to do list", "tasks"],
        "country": ["country"],
    }

    matched_key = None
    for db_key, signals in _RECALL_KEYS.items():
        if any(sig in low for sig in signals):
            matched_key = db_key
            break

    if not matched_key:
        return None

    val = facts.get(matched_key) or facts.get(matched_key.replace(" ", "_"))
    if not val or val.strip().lower() in _JUNK_VALUES:
        return None  # not stored — let LLM handle gracefully

    # Simple, factual answer
    persona = settings.bot_persona_name or "Shimmi"
    label_map = {
        "name": "name",
        "city": "city",
        "age":  "age",
        "birthday": "birthday",
        "hobbies": "hobbies",
        "interests": "interests",
        "favorite drink": "favourite drink",
        "favorite food": "favourite food",
        "allergies": "allergies",
        "occupation": "occupation/job",
        "car": "vehicle",
        "pets": "pets",
        "shopping_list": "shopping list",
        "grocery_list": "grocery list",
        "todo_list": "to-do list",
        "country": "country",
    }
    label = label_map.get(matched_key, matched_key)
    return f"Your {label}: *{val}*"


# ---------------------------------------------------------------------------
# Agentic orchestrator
# ---------------------------------------------------------------------------

def _reminders_to_json(reminders: List[Reminder]) -> List[Dict]:
    return [
        {"id": r.id, "text": r.reminder_text, "trigger_iso": r.trigger_iso, "status": r.status}
        for r in reminders if r.status == "pending"
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

    # FIX-4: strip junk facts before building the prompt
    clean_facts = _clean_facts(facts)

    user_payload = json.dumps(
        {
            "user_message":      user_text,
            "facts":             clean_facts,
            "reminders_pending": _reminders_to_json(reminders),
            "context":           context,
            "search_results":    search_results,
            "current_time":      _current_time_str(),
            "time_of_day":       _time_of_day(),
            "today":             _today_str(),
            "tz_offset":         _utc_offset_str(),
            "iteration":         iteration,
            "max_iterations":    _MAX_ITERATIONS,
        },
        ensure_ascii=False,
    )

    raw = await _groq_raw(
        chat_id, system, user_payload,
        temperature=0.25, max_tokens=1000,
        role="orchestrate", label=f"orchestrate_{iteration}",  # FIX-1: uses orchestrator_model (70b)
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
            temperature=0.0, max_tokens=600,
            role="repair", label="repair",  # FIX-1: uses extraction_model (8b)
        )
        result = OrchestratorResult.model_validate(_extract_json(repaired_raw))
        logger.info("orchestrate.repaired  iter=%d  action=%s", iteration, result.action)
        return result


# ---------------------------------------------------------------------------
# Reminder dedup helper
# ---------------------------------------------------------------------------

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

    # FIX-5: answer simple fact-recall questions without LLM
    shortcut = _try_facts_shortcut(user_text, facts)
    if shortcut:
        logger.info("⚡ facts.shortcut  text=%r  reply=%r", user_text[:80], shortcut[:80])
        if trace:
            trace.tag(facts_shortcut=True)
        return AgentResult(
            reply=ReplyPayload(type="text", text=shortcut),
            memory_updates=[],
            reminders=[],
            iterations=0,
        )

    with _step("memory_extract"):
        if _has_personal_content(user_text):
            extract_task = asyncio.create_task(_extract_memory(chat_id, user_text))
        else:
            # FIX-6: deprecated asyncio.get_event_loop() → get_running_loop()
            loop = asyncio.get_running_loop()
            extract_task = loop.create_future()
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

        # Force-search for live-data topics
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
                "🤖 orchestrate  iter=%d  force_search — detected live-data topic", iteration,
            )
            orch_result.action = "search"
            orch_result.query  = user_text

        if orch_result.action == "search" and settings.live_search_enabled:
            query = (orch_result.query or user_text).strip()

            _city    = facts.get("city",    "")
            _country = facts.get("country", "")
            _loc     = ", ".join(x for x in [_city, _country] if x) or ""
            _placeholder_re = re.compile(
                r"\[(?:user['']?s?\s+)?(?:location|city|country|place|region)[^\]]*\]",
                re.IGNORECASE,
            )
            query = _placeholder_re.sub(_loc or "location unknown", query)
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

        logger.warning(
            "orchestrate.unknown_action  iter=%d  action=%r", iteration, orch_result.action,
        )
        if iteration >= _MAX_ITERATIONS:
            break

    # ── Memory verification ────────────────────────────────────────────────
    with _step("memory_verify"):
        proposed  = await extract_task
        verified  = await _verify_updates(chat_id, user_text, proposed) if proposed else []
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

    # Deduplicate: drop updates already in DB with same value
    final_memory = [
        u for u in final_memory
        if facts.get(u.key, "").strip().lower() != u.value.strip().lower()
    ]
    if trace:
        trace.tag(memory_deduped=len(final_memory))

    # ── Format reply ───────────────────────────────────────────────────────
    raw_text = (orch_result.text if orch_result else "") or "I'm not sure how to answer that."
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
