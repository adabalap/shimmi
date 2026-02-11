from __future__ import annotations

import os
import re
from typing import Any, Dict

KEY_OK = re.compile(r"^[a-z0-9_:\-]{1,64}$")
PII_DENY = re.compile(r"(password|passcode|otp|cvv|ssn|aadhaar|pan\s*card|token|secret|api[_-]?key|pin)", re.IGNORECASE)

DEFAULT_ALLOWED_KEYS = {
    "name", "age", "birthdate", "birthday",
    "city", "state", "country",
    "occupation", "current_company", "current_job_role", "career_goal",
    "coffee_order", "favorite_color", "hobbies", "favorite_trail",
    "favorite_podcasts", "allergy",
}

ALLOW_FREEFORM_KEYS = os.getenv("ALLOW_FREEFORM_MEMORY_KEYS", "0").lower() in ("1","true","yes","on")


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

    if not KEY_OK.match(k):
        return False
    if PII_DENY.search(k) or PII_DENY.search(v):
        return False
    if not _is_safe_text(v, 256):
        return False

    if (k not in DEFAULT_ALLOWED_KEYS) and (not ALLOW_FREEFORM_KEYS) and (not k.startswith("custom:")):
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

    if k == "location" and v:
        k = "city"

    if k == "city" and 2 <= len(v) <= 64:
        v = v.title()

    if k in {"favorite_color", "colour", "color"}:
        k = "favorite_color"
        v = v.lower()

    f["key"] = k
    f["value"] = v
    return f


def admit_holding(r: Dict[str, Any]) -> bool:
    return False


def normalize_holding(r: Dict[str, Any]) -> Dict[str, Any] | None:
    return None
