"""
utils.py — Shimmi v2.4.0

New in this version:
  - bot_signature()  — appends emoji + rotating AI tagline to every reply
  - Dynamic tagline pool (deterministic rotation by message count / hash)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
from typing import List, Optional

from .config import settings

# ---------------------------------------------------------------------------
# Bot signature & dynamic taglines
# ---------------------------------------------------------------------------

_TAGLINES = [
    "Intelligence that listens",
    "Thinking in milliseconds, speaking your language",
    "AI that remembers, learns, evolves",
    "Context-aware. Always.",
    "Your AI, personalised",
    "Not just smart — present",
    "Learning with every message",
    "Where language meets intelligence",
    "Powered by curiosity",
    "AI that grows with you",
    "Fluent in human",
    "Every answer, thoughtfully crafted",
]

_tagline_counter: int = 0


def _next_tagline(seed: str = "") -> str:
    """
    Return a deterministic-but-rotating tagline.
    Uses a seed (e.g. chat_id + event_id) so the same event always gets
    the same tagline, but different chats/events get different ones.
    """
    if seed:
        idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(_TAGLINES)
    else:
        global _tagline_counter
        idx = _tagline_counter % len(_TAGLINES)
        _tagline_counter += 1
    return _TAGLINES[idx]


def bot_signature(seed: str = "") -> str:
    """
    Returns the standard bot footer to append to every outgoing message.
    Format:
        ─────────────────────
        🤖 _Thinking in milliseconds, speaking your language_
    """
    tagline = _next_tagline(seed)
    return f"\n─────────────────────\n🤖 _{tagline}_"


def sign_message(text: str, seed: str = "") -> str:
    """
    Append the bot signature to `text`.
    Respects the 3800-char limit: signature is only added if the result fits.
    """
    if not text:
        return text
    sig = bot_signature(seed)
    combined = text + sig
    # Hard cap — truncate body, keep signature
    if len(combined) > 3800:
        max_body = 3800 - len(sig) - 1
        combined = text[:max_body].rstrip() + "…" + sig
    return combined


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------

def canonical_text(text: str, cap: int = 4000) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:cap]


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def verify_signature(raw: bytes, header_value: Optional[str]) -> bool:
    secret = settings.webhook_secret
    if not secret:
        return True
    secret_b = secret.encode("utf-8")

    sig_hdr = (header_value or "").strip()
    if not sig_hdr:
        return False

    normalized = sig_hdr
    if normalized.lower().startswith("sha256="):
        normalized = normalized.split("=", 1)[1].strip()

    mac = hmac.new(secret_b, raw, hashlib.sha256).digest()
    mac_hex = mac.hex()
    mac_b64 = base64.b64encode(mac).decode("ascii").strip()

    return (
        hmac.compare_digest(normalized.lower(), mac_hex.lower())
        or hmac.compare_digest(normalized, mac_hex)
        or hmac.compare_digest(normalized, mac_b64)
        or hmac.compare_digest(normalized, mac_b64.lower())
    )


def canonical_user_key(jid: Optional[str]) -> str:
    if not jid:
        return ""
    head = jid.split("@", 1)[0]
    digits = re.sub(r"\D+", "", head)
    return digits


def prefixes() -> List[str]:
    raw = settings.bot_command_prefix or ""
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _compile_prefix_alternation() -> str:
    alts = [re.escape(p.lstrip("@")) for p in prefixes()]
    return "|".join(alts) if alts else ""


_PREFIX_ANY_RE: Optional[re.Pattern] = None
_PREFIX_TOKEN_RE: Optional[re.Pattern] = None


def compile_prefix_re() -> None:
    global _PREFIX_ANY_RE, _PREFIX_TOKEN_RE
    alt = _compile_prefix_alternation()
    if not alt:
        _PREFIX_ANY_RE = re.compile(r"a^")
        _PREFIX_TOKEN_RE = re.compile(r"a^")
        return
    _PREFIX_ANY_RE = re.compile(r"(?i)@?(?:%s)\b" % alt)
    _PREFIX_TOKEN_RE = re.compile(r"(?i)(?:^|[\s,;:–—-]+)@?(?:%s)\b[\s,;:!?\.]*" % alt)


def has_prefix(text: Optional[str]) -> bool:
    if not text:
        return False
    if _PREFIX_ANY_RE is None:
        compile_prefix_re()
    return bool(_PREFIX_ANY_RE.search(text))


def strip_invocation(text: str) -> str:
    if not text:
        return ""
    if _PREFIX_TOKEN_RE is None:
        compile_prefix_re()
    out = _PREFIX_TOKEN_RE.sub(" ", text)
    out = re.sub(r"\s+([,;:!?\.])", r"\1", out)
    out = re.sub(r"([,;:!?\.])\s+", r"\1 ", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def chat_is_allowed(chat_id: Optional[str]) -> bool:
    if settings.allow_all_chats:
        return bool(chat_id)
    allow = settings.allowed_chat_jids
    if not allow:
        return False
    return bool(chat_id) and chat_id in allow


def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return ""

    out = html.unescape(text).strip()
    out = out.replace("```", "")
    out = out.replace("`", "")
    out = out.replace("\\*", "*")

    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out)
    out = re.sub(r"(?m)^\s*[-*]\s+", "• ", out)

    lines = out.splitlines()
    looks_like_table = (
        any("|" in ln for ln in lines)
        and any(set(ln.strip()) <= set("|:- ") for ln in lines)
    )
    if looks_like_table:
        cleaned: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s or set(s) <= set("|:- "):
                continue
            if "|" in s:
                cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
                if cells:
                    cleaned.append("• " + " — ".join(cells))
            else:
                cleaned.append(s)
        out = "\n".join(cleaned)

    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)

    if len(out) > 3800:
        out = out[:3800].rstrip() + "…"

    return out.strip()
