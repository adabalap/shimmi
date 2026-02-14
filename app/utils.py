from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
from typing import Optional

from .config import settings


def canonical_text(text: str, cap: int = 4000) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:cap]


def sha1_hex(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def verify_signature(raw: bytes, header_value: Optional[str]) -> bool:
    secret = settings.webhook_secret
    if not secret:
        return True

    sig_hdr = (header_value or "").strip()
    if not sig_hdr:
        return False

    normalized = sig_hdr
    if normalized.lower().startswith("sha256="):
        normalized = normalized.split("=", 1)[1].strip()

    secret_b = secret.encode("utf-8")
    mac = hmac.new(secret_b, raw, hashlib.sha256).digest()
    mac_hex = mac.hex()
    mac_b64 = base64.b64encode(mac).decode("ascii").strip()

    return (
        hmac.compare_digest(normalized.lower(), mac_hex.lower())
        or hmac.compare_digest(normalized, mac_hex)
        or hmac.compare_digest(normalized, mac_b64)
        or hmac.compare_digest(normalized, mac_b64.lower())
    )


def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return ""
    out = html.unescape(text)
    out = out.replace("```", "").replace("`", "")
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_jid(jid: Optional[str]) -> Optional[str]:
    if not jid:
        return None
    if jid.endswith("@s.whatsapp.net"):
        return jid.replace("@s.whatsapp.net", "@c.us")
    return jid


def looks_group(jid: Optional[str]) -> bool:
    return bool(jid and jid.endswith("@g.us"))


def looks_broadcast(jid: Optional[str]) -> bool:
    return bool(jid and jid.endswith("@broadcast"))


def looks_channel(jid: Optional[str]) -> bool:
    return bool(jid and jid.endswith("@newsletter"))


def is_groupish(jid: Optional[str]) -> bool:
    return looks_group(jid) or looks_broadcast(jid) or looks_channel(jid)


def group_allowed(chat_id: Optional[str]) -> bool:
    if not chat_id:
        return False
    if looks_channel(chat_id):
        return True
    if not looks_group(chat_id):
        return True
    allow = settings.allowed_group_jids
    if not allow:
        return True
    return chat_id in allow


_PREFIX_RE: Optional[re.Pattern] = None


def compile_prefix_re() -> None:
    global _PREFIX_RE
    raw = settings.bot_command_prefix or ""
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    alts = [re.escape(p.lstrip("@")) for p in parts]
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
