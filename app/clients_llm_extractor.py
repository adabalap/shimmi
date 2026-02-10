# app/clients_llm_extractor.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .clients_llm import groq_chat

ALLOWED_NAMESPACES = {"default", "prefs", "location", "work", "family", "contact", "custom"}

_FENCE_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def _extract_fenced_json(s: str) -> Optional[str]:
    if not s:
        return None
    m = _FENCE_BLOCK.search(s)
    return m.group(1).strip() if m else None

def _extract_first_json_object(s: str) -> Optional[str]:
    """Balanced brace extraction for the first JSON object."""
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1].strip()
    return None

def _parse_json_strict(s: str) -> Optional[Any]:
    if not s:
        return None
    fenced = _extract_fenced_json(s)
    candidate = fenced if fenced else s.strip()
    try:
        return json.loads(candidate)
    except Exception:
        obj = _extract_first_json_object(candidate)
        if obj:
            try:
                return json.loads(obj)
            except Exception:
                return None
    return None

def _clip_conf(x: Any, default: float = 0.7) -> float:
    try:
        v = float(x)
        if v > 1.0:
            return 1.0
        if v < 0.0:
            return 0.0
        return v
    except Exception:
        return default

def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)

def _normalize_fact_shape(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _as_str(f.get("key")).strip().lower()
    val = _as_str(f.get("value")).strip()
    ns = (_as_str(f.get("namespace")).strip().lower() or "default")
    typ = (_as_str(f.get("type")).strip().lower() or "text")
    conf = _clip_conf(f.get("confidence", 0.7))
    span = _as_str(f.get("span")).strip()

    if not key or not val:
        return None
    if ns not in ALLOWED_NAMESPACES:
        ns = "default"
    return {"key": key, "value": val, "namespace": ns, "type": typ, "confidence": conf, "span": span}

def _normalize_holding_shape(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    typ = _as_str(r.get("type")).strip().lower()
    if typ != "holding":
        return None
    symbol = _as_str(r.get("symbol")).strip().upper()
    span = _as_str(r.get("span")).strip()
    if not symbol or not span:
        return None

    def _num(x):
        try:
            return float(x) if x is not None and str(x).strip() != "" else None
        except Exception:
            return None

    return {
        "type": "holding",
        "symbol": symbol,
        "shares": _num(r.get("shares")),
        "purchase_price": _num(r.get("purchase_price")),
        "avg_price": _num(r.get("avg_price")),
        "currency": _as_str(r.get("currency")).strip().upper() or None,
        "span": span,
        "confidence": _clip_conf(r.get("confidence", 0.7)),
    }

async def llm_extract_facts_open(chat_id: str, user_text: str, known_facts: Dict[str, str] | None = None) -> Dict[str, Any]:
    sys = """You extract durable personal facts and typed records from ONE WhatsApp message.
Output STRICT JSON only:
{
  "facts": [
    {"key":"...","value":"...","type":"text|number|date|...","namespace":"default|prefs|location|work|family|contact|custom","confidence":0..1,"span":"exact fragment from the message"}
  ],
  "records": [
    {"type":"holding","symbol":"PAYTM","shares":38,"purchase_price":2150,"currency":"INR","confidence":0.9,"span":"..."}
  ]
}
- If unsure, return {"facts":[],"records":[]}.
- NEVER output secrets (passwords, OTPs, pins, tokens).
- Do not include any prose outside JSON.
"""

    hint = ""
    if known_facts:
        kv = ", ".join(f"{k}={v}" for k, v in known_facts.items())
        hint = f"KNOWN (do not restate unless updated): {kv}\n"

    prompt = f"{hint}MESSAGE:\n{user_text}"

    reply, ok, _ = await groq_chat(chat_id, sys, prompt, temperature=0.2, max_tokens=600)
    if not ok or not reply:
        return {"facts": [], "records": []}

    parsed = _parse_json_strict(reply)
    if not isinstance(parsed, dict):
        return {"facts": [], "records": []}

    rf = parsed.get("facts", []) or []
    rr = parsed.get("records", []) or []

    facts: List[Dict[str, Any]] = []
    for f in rf:
        if isinstance(f, dict):
            shaped = _normalize_fact_shape(f)
            if shaped:
                facts.append(shaped)

    records: List[Dict[str, Any]] = []
    for r in rr:
        if isinstance(r, dict):
            shaped = _normalize_holding_shape(r)
            if shaped:
                records.append(shaped)

    return {"facts": facts, "records": records}

async def llm_judge_facts(chat_id: str, user_text: str, normalized: Dict[str, Any]) -> Tuple[bool, float, str]:
    sys = """Verify that provided JSON facts/records are supported by the user's message.
Reply STRICT JSON only: {"approved": true|false, "confidence": 0..1, "why": "..."}
No prose outside JSON.
"""

    try:
        facts_json = json.dumps(normalized, ensure_ascii=False)
    except Exception:
        return False, 0.0, "serialize_error"

    prompt = f"MESSAGE:\n{user_text}\n\nFACTS_AND_RECORDS:\n{facts_json}"

    reply, ok, _ = await groq_chat(chat_id, sys, prompt, temperature=0.0, max_tokens=256)
    parsed = _parse_json_strict(reply) if (ok and reply) else None
    if not isinstance(parsed, dict):
        return False, 0.0, "json_parse_error"

    approved = bool(parsed.get("approved", False))
    conf = _clip_conf(parsed.get("confidence", 0.0), 0.0)
    why = str(parsed.get("why", "") or "")
    return approved, conf, why
