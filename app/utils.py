from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
from typing import Optional

from . import config


def canonical_text(text: str, cap: int = 4000) -> str:
    t = re.sub(r"\s+", " ", (text or '').strip())
    return t[:cap]


def verify_signature(raw: bytes, header_value: Optional[str]) -> bool:
    secret = (config.WEBHOOK_SECRET or '').strip()
    if not secret:
        return True
    sig = (header_value or '').strip()
    if not sig:
        return False
    if sig.lower().startswith('sha256='):
        sig = sig.split('=', 1)[1].strip()

    mac = hmac.new(secret.encode('utf-8'), raw, hashlib.sha256).digest()
    return (
        hmac.compare_digest(sig.lower(), mac.hex().lower()) or
        hmac.compare_digest(sig, base64.b64encode(mac).decode('ascii').strip())
    )


def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return ''
    out = html.unescape(text)
    out = out.replace('```', '').replace('`', '')
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()
