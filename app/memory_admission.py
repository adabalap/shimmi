
# app/memory_admission.py
from __future__ import annotations
import re
from typing import Dict, Any

KEY_OK = re.compile(r"^[a-z0-9_:\-]{1,64}$")
PRINTABLE = re.compile(r"^[\x20-\x7E\s]{1,256}$", re.ASCII)
PII_DENY = re.compile(r"(password|passcode|otp|cvv|ssn|aadhaar|pan\s*card)", re.IGNORECASE)
ALLOWED_NAMESPACES = {"prefs", "location", "work", "family", "contact", "custom"}

def admit_fact(f: Dict[str, Any]) -> bool:
    k = (f.get("key") or "").strip().lower()
    v = (f.get("value") or "").strip()
    ns = (f.get("namespace") or "").strip().lower()

    if not KEY_OK.match(k): return False
    if ns not in ALLOWED_NAMESPACES: return False
    if not PRINTABLE.match(v): return False
    if len(v) < 2 or len(v) > 256: return False
    if PII_DENY.search(v): return False
    # ensure we have an evidence span and it's printable
    span = (f.get("span") or "").strip()
    if not span or not PRINTABLE.match(span):
        return False
    return True

def normalize_fact(f: Dict[str, Any]) -> Dict[str, Any]:
    # light-touch canonicalization; LLM does the heavy lifting
    k = (f.get("key") or "").strip().lower()
    v = (f.get("value") or "").strip()
    ns = (f.get("namespace") or "").strip().lower()
    t  = (f.get("type") or "text").strip().lower()

    if k == "country":
        v_l = v.lower()
        if v_l in {"usa", "us", "u.s.", "u.s.a."}: v = "United States"
        elif v_l in {"uk", "u.k."}: v = "United Kingdom"
        elif 2 <= len(v) <= 64: v = v.title()
    if k == "city":
        if 2 <= len(v) <= 64: v = v.title()
    if k == "favorite_color":
        v = v.lower()

    f["key"] = k; f["value"] = v; f["namespace"] = ns; f["type"] = t
    return f

# ------- holdings records admission ------------------------------------------

def admit_holding(r: Dict[str, Any]) -> bool:
    # minimal required: type=holding, symbol, span
    if (r.get("type") or "").lower() != "holding":
        return False
    symbol = (r.get("symbol") or "").strip()
    if not symbol: return False
    span = (r.get("span") or "").strip()
    if not span or not PRINTABLE.match(span): return False

    # Optional numeric sanity: shares/purchase_price if present must be positive
    shares = r.get("shares")
    if shares is not None:
        try:
            if float(shares) <= 0: return False
        except Exception:
            return False

    price = r.get("purchase_price")
    if price is not None:
        try:
            if float(price) <= 0: return False
        except Exception:
            return False

    return True

def normalize_holding(r: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(r, dict): return None
    t = (r.get("type") or "").strip().lower()
    if t != "holding": return None
    # normalize trivial
    r["symbol"] = (r.get("symbol") or "").strip().upper()
    if r.get("currency"):
        r["currency"] = (str(r["currency"]).strip().upper() or None)
    if r.get("name"):
        nm = str(r["name"]).strip()
        r["name"] = nm if nm else None
    if r.get("exchange"):
        ex = str(r["exchange"]).strip().upper()
        r["exchange"] = ex if ex else None
    # coerce numerics
    def _to_num(x):
        try:
            return float(x) if x is not None and str(x).strip() != "" else None
        except Exception:
            return None
    for key in ("shares", "purchase_price", "avg_price"):
        if key in r:
            r[key] = _to_num(r.get(key))
    return r

