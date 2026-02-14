from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
from typing import List, Optional

from .config import settings


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
    """Normalize sender identity across @c.us, @lid, @s.whatsapp.net.

    Returns a stable key like '919573717667'. If unknown, returns empty string.
    """
    if not jid:
        return ""
    # common formats: 919573717667@c.us, 4930656034916@lid
    head = jid.split('@', 1)[0]
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
    looks_like_table = any('|' in ln for ln in lines) and any(set(ln.strip()) <= set('|:- ') for ln in lines)
    if looks_like_table:
        cleaned: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if set(s) <= set('|:- '):
                continue
            if '|' in s:
                cells = [c.strip() for c in s.strip('|').split('|') if c.strip()]
                if cells:
                    cleaned.append('• ' + ' — '.join(cells))
            else:
                cleaned.append(s)
        out = "\n".join(cleaned)

    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)

    if has_prefix(out):
        out = strip_invocation(out)

    if len(out) > 3800:
        out = out[:3800].rstrip() + "…"

    return out.strip()


# Backwards-compatible alias
sanitize_for_whatsapp = sanitize_for_whatsapp
