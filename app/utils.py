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


def prefixes() -> List[str]:
    raw = settings.bot_command_prefix or ""
    return [p.strip() for p in str(raw).split(",") if p.strip()]


_PREFIX_RE: Optional[re.Pattern] = None


def compile_prefix_re() -> None:
    global _PREFIX_RE
    alts = [re.escape(p.lstrip("@")) for p in prefixes()]
    if not alts:
        _PREFIX_RE = re.compile(r"a^")
        return
    _PREFIX_RE = re.compile(r"(?i)(?:^|\s)@?(%s)\b" % "|".join(alts))


def has_prefix(text: Optional[str]) -> bool:
    if not text:
        return False
    if _PREFIX_RE is None:
        compile_prefix_re()
    return bool(_PREFIX_RE.search(text))


def strip_prefix(text: str) -> str:
    if _PREFIX_RE is None:
        compile_prefix_re()
    s = text or ""
    m = _PREFIX_RE.search(s)
    if not m:
        return s.strip()
    start, end = m.span()
    before = s[:start].strip()
    after = s[end:].lstrip(" ,:;-\t")
    if not before:
        return after.strip()
    joined = (before + " " + after).strip()
    return re.sub(r"\s+", " ", joined)


def chat_is_allowed(chat_id: Optional[str]) -> bool:
    allow = settings.allowed_chat_jids
    if not allow:
        return False
    return bool(chat_id) and chat_id in allow


def sanitize_for_whatsapp(text: str) -> str:
    """Make output WhatsApp friendly: remove code fences, convert tables to bullets, keep concise."""
    if not text:
        return ""

    out = html.unescape(text).strip()

    # Remove backticks/code fences
    out = out.replace("```", "")
    out = out.replace("`", "")

    lines = out.splitlines()

    looks_like_table = any('|' in ln for ln in lines) and any(set(ln.strip()) <= set('|:- ') for ln in lines)
    if looks_like_table:
        cleaned_lines: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if set(s) <= set('|:- '):
                continue
            if '|' in s:
                cells = [c.strip() for c in s.strip('|').split('|') if c.strip()]
                if cells:
                    cleaned_lines.append('• ' + ' — '.join(cells))
            else:
                cleaned_lines.append(s)
        out = "\n".join(cleaned_lines)

    # normalize whitespace
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)

    # Avoid starting with invocation tokens
    if has_prefix(out):
        out = strip_prefix(out)

    if len(out) > 3800:
        out = out[:3800].rstrip() + "…"

    return out.strip()
