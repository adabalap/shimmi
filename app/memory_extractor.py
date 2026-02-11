from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from . import config
from .clients_llm import groq_chat

_PATTERNS = [
    ('name', re.compile(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z .'-]{1,48})\b", re.I)),
    ('city', re.compile(r"\b(?:i\s+live\s+in|i\s+am\s+from|my\s+city\s+is)\s+([A-Za-z][A-Za-z .'-]{1,48})\b", re.I)),
]


def extract_deterministic(text: str) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    if not text:
        return out
    for key, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            out.append((key, m.group(1).strip(), 0.75))
    return out


async def extract_with_llm(chat_id: str, user_text: str) -> List[Tuple[str, str, float]]:
    sys = (
        'Extract durable personal facts the user explicitly states in ONE message. '
        'Return STRICT JSON only: {"facts":[{"key":"...","value":"...","confidence":0.0}]}. '
        'If none, return {"facts":[]}. Never extract secrets (passwords, OTPs, tokens, API keys).'
    )
    reply, ok, meta, model = await groq_chat(chat_id, system=sys, user=user_text, temperature=0.0, max_tokens=260)
    if not ok or not reply:
        return []
    try:
        start = reply.find('{'); end = reply.rfind('}')
        payload = reply[start:end+1] if start != -1 and end != -1 else reply
        obj = json.loads(payload)
        facts = obj.get('facts', []) if isinstance(obj, dict) else []
        out: List[Tuple[str, str, float]] = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            k = str(f.get('key','')).strip().lower()
            v = str(f.get('value','')).strip()
            c = float(f.get('confidence', 0.7) or 0.7)
            if not k or not v:
                continue
            if any(x in k for x in ('password','otp','token','secret','api_key','key')):
                continue
            out.append((k, v, max(0.0, min(1.0, c))))
        return out
    except Exception:
        return []


async def extract_facts(chat_id: str, user_text: str) -> List[Tuple[str, str, float]]:
    mode = (config.FACTS_EXTRACTION_MODE or 'hybrid').lower()
    det = extract_deterministic(user_text) if mode in ('deterministic','hybrid') else []
    if mode == 'deterministic':
        return det
    if mode in ('llm','hybrid'):
        llm = await extract_with_llm(chat_id, user_text)
        merged: Dict[str, Tuple[str, float]] = {}
        for k, v, c in det + llm:
            if k not in merged or c > merged[k][1]:
                merged[k] = (v, c)
        return [(k, v, c) for k, (v, c) in merged.items()]
    return []
