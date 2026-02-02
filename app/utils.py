# app/utils.py
from __future__ import annotations
import html, re, hmac, hashlib, base64
from typing import Optional
from . import config

LOG_TRUNCATE_DEFAULT = 200
LOG_TRUNCATE_TEXT = 1200

def canonical_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip())

def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return text
    out = html.unescape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out)
    out = out.replace("```", "").replace("`", "")
    out = re.sub(r"^\s*[-•]\s*[-•]\s*", "• ", out, flags=re.MULTILINE)
    out = re.sub(r"^\s*[-•]\s*", "• ", out, flags=re.MULTILINE)
    out = re.sub(r"•\s*", "• ", out)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()

def format_bot_text(text: str) -> str:
    if not config.BOT_EMOJI_PREFIX_ENABLED:
        return text
    t = (text or '').lstrip()
    if t.startswith(config.BOT_EMOJI):
        return text
    return f"{config.BOT_EMOJI} {text}"

def verify_signature(raw: bytes, header_value: str | None) -> bool:
    if not config.WEBHOOK_SECRET:
        return True
    sig_hdr = (header_value or '').strip()
    if not sig_hdr:
        return False
    normalized = sig_hdr.lower().removeprefix('sha256=').strip()
    mac = hmac.new(config.WEBHOOK_SECRET, raw, hashlib.sha256).digest()
    mac_hex = mac.hex()
    mac_b64 = base64.b64encode(mac).decode('ascii').strip()
    return (
        hmac.compare_digest(normalized, mac_hex)
        or hmac.compare_digest(sig_hdr, mac_hex)
        or hmac.compare_digest(sig_hdr, mac_b64)
        or hmac.compare_digest(normalized, mac_b64.lower())
    )

