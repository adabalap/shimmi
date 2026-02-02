
# app/clients_llm_extractor.py
from __future__ import annotations
import json, re
from typing import Dict, Any, Tuple, List, Optional

from .clients_llm import groq_chat

ALLOWED_NAMESPACES = {"prefs", "location", "work", "family", "contact", "custom"}

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def _strip_code_fences(s: str) -> str:
    if not s: return s
    return _CODE_FENCE_RE.sub("", s).strip()

def _extract_json_object(s: str) -> Optional[str]:
    if not s: return None
    m = _JSON_BLOCK_RE.search(s)
    return m.group(0).strip() if m else None

def _parse_json_strict(s: str) -> Optional[Any]:
    if not s: return None
    txt = _strip_code_fences(s)
    try:
        return json.loads(txt)
    except Exception:
        block = _extract_json_object(txt)
        if block:
            try: return json.loads(block)
            except Exception: return None
    return None

def _clip_conf(x: Any, default: float = 0.85) -> float:
    try:
        v = float(x)
        return 1.0 if v > 1.0 else 0.0 if v < 0.0 else v
    except Exception:
        return default

def _as_str(x: Any) -> str:
    if x is None: return ""
    return x if isinstance(x, str) else str(x)

def _normalize_fact_shape(f: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _as_str(f.get("key")).strip().lower()
    val = _as_str(f.get("value")).strip()
    ns  = (_as_str(f.get("namespace")).strip().lower() or "custom")
    typ = (_as_str(f.get("type")).strip().lower() or "text")
    conf = _clip_conf(f.get("confidence", 0.85))
    span = _as_str(f.get("span")).strip()
    if not key or not val: return None
    if ns not in ALLOWED_NAMESPACES: ns = "custom"
    return {"key":key,"value":val,"namespace":ns,"type":typ,"confidence":conf,"span":span}

def _normalize_holding_shape(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    typ = _as_str(r.get("type")).strip().lower()
    if typ != "holding": return None
    symbol = _as_str(r.get("symbol")).strip().upper()
    span = _as_str(r.get("span")).strip()
    if not symbol or not span: return None

    def _num(x):
        try: return float(x) if x is not None and str(x).strip() != "" else None
        except Exception: return None

    out = {
        "type":"holding","symbol":symbol,
        "name":_as_str(r.get("name")).strip() or None,
        "exchange":_as_str(r.get("exchange")).strip().upper() or None,
        "shares":_num(r.get("shares")),
        "purchase_price":_num(r.get("purchase_price")),
        "avg_price":_num(r.get("avg_price")),
        "currency":_as_str(r.get("currency")).strip().upper() or None,
        "span":span,"confidence":_clip_conf(r.get("confidence", 0.85))
    }
    return out

async def llm_extract_facts_open(chat_id: str, user_text: str, known_facts: Dict[str, str] | None = None) -> Dict[str, Any]:
    sys = (
        "You extract durable personal facts and typed records from ONE WhatsApp message.\n"
        "Output STRICT JSON only:\n"
        "{ \"facts\": [\n"
        "  {\"key\":\"...\",\"value\":\"...\",\"type\":\"string|number|date|...\",\n"
        "   \"namespace\":\"prefs|location|work|family|contact|custom\",\n"
        "   \"confidence\":0..1,\n"
        "   \"span\":\"exact fragment from the message\"}\n"
        "],\n"
        "\"records\": [\n"
        "  {\"type\":\"holding\",\"symbol\":\"PAYTM\",\"shares\":38,\"purchase_price\":2150,\n"
        "   \"currency\":\"INR\",\"confidence\":0.9,\n"
        "   \"span\":\"38 shares of paytm each share with a purchase value of 2150\"}\n"
        "]}\n"
        "- If unsure, return {\"facts\":[],\"records\":[]}.\n"
        "- Do not include any prose outside JSON."
    )
    hint = ""
    if known_facts:
        kv = ", ".join(f"{k}={v}" for k, v in known_facts.items())
        hint = f"KNOWN: {kv}\n"
    prompt = f"{hint}MESSAGE:\n{user_text}"

    reply, ok, _meta = await groq_chat(chat_id, sys, prompt)
    if not ok or not reply: return {"facts": [], "records": []}
    parsed = _parse_json_strict(reply)
    if not isinstance(parsed, dict): return {"facts": [], "records": []}

    rf = parsed.get("facts", []) or []
    rr = parsed.get("records", []) or []

    facts: List[Dict[str, Any]] = []
    for f in rf:
        if not isinstance(f, dict): continue
        shaped = _normalize_fact_shape(f)
        if shaped: facts.append(shaped)
    facts = [f for f in facts if (f.get("namespace") or "").lower() in ALLOWED_NAMESPACES]

    records: List[Dict[str, Any]] = []
    for r in rr:
        if not isinstance(r, dict): continue
        shaped = _normalize_holding_shape(r)
        if shaped: records.append(shaped)

    return {"facts": facts, "records": records}

async def llm_judge_facts(chat_id: str, user_text: str, normalized: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Judge JSON strictly + safe fallback.
    """
    def _parse(ans: str): 
        from_json = _parse_json_strict(ans)
        return from_json if isinstance(from_json, dict) else None

    sys = (
        "Verify that provided JSON facts/records are supported by the user's message.\n"
        "Reply STRICT JSON only: {\"approved\": true|false, \"confidence\": 0..1, \"why\": \"...\"}\n"
        "No prose outside JSON."
    )
    import json as _json
    try:
        facts_json = _json.dumps(normalized, ensure_ascii=False)
    except Exception:
        return False, 0.0, "serialize_error"

    prompt = f"MESSAGE:\n{user_text}\n\nFACTS_AND_RECORDS:\n{facts_json}"

    reply, ok, _m = await groq_chat(chat_id, sys, prompt)
    parsed = _parse(reply) if (ok and reply) else None

    if not parsed:
        sys2 = sys + "\nReturn JSON only. Temperature must be 0."
        reply2, ok2, _m2 = await groq_chat(chat_id, sys2, prompt)
        parsed = _parse(reply2) if (ok2 and reply2) else None

    if not parsed:
        facts = normalized.get("facts", []) or []
        recs = normalized.get("records", []) or []
        simple = all(f.get("span") and len(str(f.get("value",""))) <= 128 for f in facts)
        if simple and (facts or recs):
            return True, 0.80, "fallback_simple_approve"
        return False, 0.0, "json_parse_error"

    approved = bool(parsed.get("approved", False))
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = 1.0 if conf > 1.0 else 0.0 if conf < 0.0 else conf
    why = str(parsed.get("why", "") or "")

    if not approved and conf >= 0.8 and not why.strip():
        facts = normalized.get("facts", []) or []
        recs = normalized.get("records", []) or []
        if recs:
            return False, conf, "judge_no_for_records"
        if len(facts) == 1:
            f = facts[0]
            val = str(f.get("value","")); span = str(f.get("span",""))
            if span and len(val) <= 64:
                return True, 0.80, "judge_inconsistent_high_conf"

    return approved, conf, why

