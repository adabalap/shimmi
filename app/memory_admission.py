from __future__ import annotations

import re
from typing import Dict, Any

KEY_OK = re.compile(r"^[a-z0-9_:\-]{1,64}$")
PII_DENY = re.compile(r"(password|passcode|otp|cvv|ssn|aadhaar|pan\s*card|token|secret|api[_-]?key|pin)", re.IGNORECASE)
ALLOWED_NAMESPACES = {"default", "prefs", "location", "work", "family", "contact", "custom"}


def _is_safe_text(s: str, max_len: int = 256) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if len(s) < 2 or len(s) > max_len:
        return False
    return all((ch.isprintable() or ch.isspace()) for ch in s)


def admit_fact(f: Dict[str, Any], *, user_text: str | None = None) -> bool:
    k = (f.get("key") or "").strip().lower()
    v = (f.get("value") or "").strip()
    ns = (f.get("namespace") or "").strip().lower()

    if not KEY_OK.match(k):
        return False
    if ns not in ALLOWED_NAMESPACES:
        return False
    if PII_DENY.search(k) or PII_DENY.search(v):
        return False
    if not _is_safe_text(v, 256):
        return False

    span = (f.get("span") or "").strip()
    if not span or not _is_safe_text(span, 256):
        return False

    if user_text:
        ut = " ".join((user_text or "").split()).lower()
        sp = " ".join(span.split()).lower()
        if sp not in ut:
            return False

    return True


def normalize_fact(f: Dict[str, Any]) -> Dict[str, Any]:
    k = (f.get("key") or "").strip().lower()
    v = (f.get("value") or "").strip()
    ns = (f.get("namespace") or "default").strip().lower()
    t  = (f.get("type") or "text").strip().lower()

    # Canonicalize: location -> city
    if k == "location" and v:
        k = "city"

    if k == "country":
        v_l = v.lower()
        if v_l in {"usa", "us", "u.s.", "u.s.a."}:
            v = "United States"
        elif v_l in {"uk", "u.k."}:
            v = "United Kingdom"
        elif 2 <= len(v) <= 64:
            v = v.title()

    if k == "city" and 2 <= len(v) <= 64:
        v = v.title()

    if k in {"favorite_color", "colour", "color"}:
        k = "favorite_color"
        v = v.lower()

    f["key"] = k
    f["value"] = v
    f["namespace"] = ns
    f["type"] = t
    return f


def admit_holding(r: Dict[str, Any]) -> bool:
    if (r.get("type") or "").lower() != "holding":
        return False
    symbol = (r.get("symbol") or "").strip()
    if not symbol:
        return False
    span = (r.get("span") or "").strip()
    if not _is_safe_text(span, 256):
        return False

    shares = r.get("shares")
    if shares is not None:
        try:
            if float(shares) <= 0:
                return False
        except Exception:
            return False

    price = r.get("purchase_price")
    if price is not None:
        try:
            if float(price) <= 0:
                return False
        except Exception:
            return False

    return True


def normalize_holding(r: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(r, dict):
        return None
    t = (r.get("type") or "").strip().lower()
    if t != "holding":
        return None
    r["symbol"] = (r.get("symbol") or "").strip().upper()
    if r.get("currency"):
        r["currency"] = (str(r["currency"]).strip().upper() or None)

    def _to_num(x):
        try:
            return float(x) if x is not None and str(x).strip() != "" else None
        except Exception:
            return None

    for key in ("shares", "purchase_price", "avg_price"):
        if key in r:
            r[key] = _to_num(r.get(key))
    return r
