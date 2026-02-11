from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
from typing import Optional

from . import config


def canonical_text(text: str, cap: int = 4000) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:cap]


def verify_signature(raw: bytes, header_value: Optional[str]) -> bool:
    secret = getattr(config, "WEBHOOK_SECRET", None)
    if not secret:
        return True

    if isinstance(secret, str):
        secret_b = secret.encode("utf-8")
    else:
        secret_b = secret

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


def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return ""
    out = html.unescape(text)
    # strip common markdown that confuses WA
    out = out.replace("```", "").replace("`", "")
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()
