from __future__ import annotations

import re, hmac, hashlib, base64, html
from . import config

LOG_TRUNCATE_DEFAULT = 200
LOG_TRUNCATE_TEXT = 1200


def canonical_text(text: str, cap: int = 4000) -> str:
    """Normalize whitespace + cap length (used for echo/idempotency hashing)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:cap]


def verify_signature(raw: bytes, header_value: str | None) -> bool:
    """Verify WAHA webhook HMAC SHA256.

    Accepts:
      - sha256=<hex>
      - <hex>
      - <base64>

    If WEBHOOK_SECRET is unset, returns True (dev-friendly).
    """
    secret = getattr(config, "WEBHOOK_SECRET", None)
    if not secret:
        return True

    if isinstance(secret, str):
        secret = secret.encode("utf-8")

    sig_hdr = (header_value or "").strip()
    if not sig_hdr:
        return False

    normalized = sig_hdr.strip()
    if normalized.lower().startswith("sha256="):
        normalized = normalized.split("=", 1)[1].strip()

    mac = hmac.new(secret, raw, hashlib.sha256).digest()
    mac_hex = mac.hex()
    mac_b64 = base64.b64encode(mac).decode("ascii").strip()

    return (
        hmac.compare_digest(normalized.lower(), mac_hex.lower())
        or hmac.compare_digest(normalized, mac_hex)
        or hmac.compare_digest(normalized, mac_b64)
        or hmac.compare_digest(normalized, mac_b64.lower())
    )


# Back-compat helper; outbound formatting is handled in clients_waha.py
def sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return text
    out = html.unescape(text)
    out = out.replace("```", "").replace("`", "")
    out = re.sub(r"[ 	]+", " ", out)
    return out.strip()
