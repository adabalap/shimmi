"""
Token-Optimized Utilities
Includes WhatsApp ID normalization and response caching

Refactor goals:
- STRICT allowlist: respond only to chat IDs listed in ALLOWED_GROUP_JIDS
- Prefix detection anywhere in sentence: BOT_COMMAND_PREFIX can appear anywhere
- Robust prefix stripping
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Optional, Dict, Tuple, List

from .config import settings

# ---------------------------------------------------------------------------
# Hashing / IDs
# ---------------------------------------------------------------------------

def sha1_hex(data: str | bytes) -> str:
    """
    Return SHA1 hex digest of input.
    Accepts str (UTF-8 encoded) or bytes.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def normalize_whatsapp_id(raw_id: str) -> str:
    """
    Normalize WhatsApp ID to ensure consistency for USER IDs (not chat IDs).
    - Keeps groups as-is (@g.us)
    - Strips suffixes for user ids: @lid, @c.us, @s.whatsapp.net
    """
    if not raw_id:
        return ""
    if "@g.us" in raw_id:
        return raw_id
    return raw_id.split("@")[0]


# Compatibility alias used by main.py
canonical_user_key = normalize_whatsapp_id


def _normalize_chat_id(chat_id: Optional[str]) -> Optional[str]:
    """
    Normalize CHAT IDs for allowlist comparison.
    - WA sometimes sends @s.whatsapp.net; normalize to @c.us
    - Keep @lid, @c.us, @g.us as-is otherwise
    """
    if not chat_id:
        return None
    if chat_id.endswith("@s.whatsapp.net"):
        return chat_id.replace("@s.whatsapp.net", "@c.us")
    return chat_id


# ---------------------------------------------------------------------------
# Response Cache (kept mostly as-is)
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    In-memory response cache to reduce token usage
    """
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, user_id: str, query: str, facts: Dict[str, str]) -> str:
        normalized_query = query.lower().strip()
        facts_str = "\n".join(f"{k}:{v}" for k, v in sorted(facts.items()))
        raw = f"{user_id}:{normalized_query}:{facts_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, user_id: str, query: str, facts: Dict[str, str]) -> Optional[str]:
        key = self._make_key(user_id, query, facts)
        if key in self.cache:
            response, timestamp = self.cache[key]
            age = time.time() - timestamp
            if age < self.ttl:
                self.hits += 1
                return response
            del self.cache[key]
        self.misses += 1
        return None

    def set(self, user_id: str, query: str, facts: Dict[str, str], response: str):
        key = self._make_key(user_id, query, facts)
        self.cache[key] = (response, time.time())
        if len(self.cache) > 100:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            for k, _ in sorted_items[:20]:
                del self.cache[k]

    def stats(self) -> Dict[str, str]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {"hits": str(self.hits), "misses": str(self.misses), "hit_rate": f"{hit_rate:.1f}%", "size": str(len(self.cache))}


response_cache = ResponseCache(ttl_seconds=300)


# ---------------------------------------------------------------------------
# Token / context helpers (kept as-is)
# ---------------------------------------------------------------------------

def should_use_cache(query: str) -> bool:
    no_cache_patterns = [
        "what time",
        "current time",
        "right now",
        "today's",
        "latest",
        "news",
        "weather",
        "forecast",
    ]
    q = (query or "").lower()
    return not any(p in q for p in no_cache_patterns)


def estimate_tokens(text: str) -> int:
    return len(text or "") // 4


def needs_context(query: str) -> bool:
    fact_only_patterns = [
        "what do you know about me",
        "my favorite",
        "my favourite",
        "do i have",
        "do i like",
        "where do i",
        "when did i",
        "what's my",
        "what is my",
        "tell me about myself",
        "what kind of",
        "do i own",
    ]
    q = (query or "").lower()
    return not any(p in q for p in fact_only_patterns)


def extract_fact_key(query: str) -> Optional[str]:
    q = (query or "").lower()
    patterns = {
        "favorite drink": "favorite_drink",
        "favourite drink": "favorite_drink",
        "morning drink": "favorite_morning_drink",
        "bike": "bike_model",
        "vehicle": "vehicle_model",
        "car": "vehicle_model",
        "city": "city",
        "where.*live": "city",
        "location": "location",
        "name": "name",
    }
    for pattern, fact_key in patterns.items():
        if re.search(pattern, q):
            return fact_key
    return None


def compress_context(messages: list, max_tokens: int = 500) -> str:
    if not messages:
        return ""
    if len(messages) <= 2:
        return "\n".join(m.get("text", "") for m in messages)

    topics: List[str] = []
    for msg in messages[-5:]:
        txt = msg.get("text", "")
        if len(txt) > 100:
            topics.append(txt.split("\n")[0][:100])
        else:
            topics.append(txt)

    summary = "Recent conversation:\n" + "\n".join(f"- {t}" for t in topics)
    if estimate_tokens(summary) > max_tokens:
        summary = summary[: max_tokens * 4]
    return summary


# ---------------------------------------------------------------------------
# WhatsApp formatting / signature
# ---------------------------------------------------------------------------

def sanitize_for_whatsapp(text: str) -> str:
    return (text or "").strip()


def verify_signature(raw_body: bytes, signature: Optional[str]) -> bool:
    # Placeholder; you can implement HMAC verification later if needed
    return True


def canonical_text(text: str) -> str:
    return (text or "").lower().strip()


# ---------------------------------------------------------------------------
# STRICT allowlist
# ---------------------------------------------------------------------------

def chat_is_allowed(chat_id: Optional[str]) -> bool:
    """
    Strict: respond only if chat_id is listed in ALLOWED_GROUP_JIDS.
    """
    cid = _normalize_chat_id(chat_id)
    if not cid:
        return False
    allow = settings.allowed_chat_jids or []
    # Normalize allowlist entries too
    allow_norm = {_normalize_chat_id(a.strip()) for a in allow if a and a.strip()}
    return cid in allow_norm


# ---------------------------------------------------------------------------
# Prefix detection anywhere (compiled regex)
# ---------------------------------------------------------------------------

_PREFIX_RE: Optional[re.Pattern] = None
_PREFIX_TOKENS: List[str] = []

def compile_prefix_re() -> None:
    """
    Compile prefix regex from settings.bot_command_prefix.
    Supports prefix tokens anywhere in the sentence, not only at start.

    We match tokens when bounded by start/space/punctuation to avoid matching inside words.
    """
    global _PREFIX_RE, _PREFIX_TOKENS

    raw = (settings.bot_command_prefix or "").strip()
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    # Fallback safety
    if not tokens:
        tokens = ["@shimmi", "shimmi", "@spock", "spock"]

    _PREFIX_TOKENS = tokens

    # Sort longer first to prevent partial matches (e.g., @shimmi before shimmi)
    tokens_sorted = sorted(tokens, key=len, reverse=True)
    alts = "|".join(re.escape(t) for t in tokens_sorted)

    # Boundary: start OR whitespace/punct before and after token
    # Works better than \b for non-latin tokens too.
    before = r"(^|[\s,.:;!?()\[\]{}<>\"'“”‘’\-])"
    after = r"($|[\s,.:;!?()\[\]{}<>\"'“”‘’\-])"

    _PREFIX_RE = re.compile(before + r"(" + alts + r")" + after, flags=re.IGNORECASE)


def _ensure_prefix_re():
    global _PREFIX_RE
    if _PREFIX_RE is None:
        compile_prefix_re()


def has_prefix(text: str) -> bool:
    """
    True if any BOT_COMMAND_PREFIX token appears anywhere in text.
    """
    if not text:
        return False
    _ensure_prefix_re()
    assert _PREFIX_RE is not None
    return _PREFIX_RE.search(text) is not None


def strip_invocation(text: str) -> str:
    """
    Remove the first occurrence of a prefix token (anywhere) and tidy punctuation/spacing.
    Example:
      "Hey Spock, what do you know?" -> "Hey what do you know?"
    """
    if not text:
        return ""

    _ensure_prefix_re()
    assert _PREFIX_RE is not None

    m = _PREFIX_RE.search(text)
    if not m:
        return text.strip()

    # Remove matched token including surrounding boundary char if it is punctuation/space
    start, end = m.span(2)  # token group
    # Remove token only, then clean up
    out = (text[:start] + text[end:]).strip()

    # Clean up doubled spaces and leftover punctuation patterns like ", ,"
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.:;!?])", r"\1", out)     # no space before punctuation
    out = re.sub(r"([,.:;!?]){2,}", r"\1", out)    # collapse repeated punctuation
    out = out.strip(" ,:;")
    return out.strip()
