from __future__ import annotations

import re
from typing import Dict, List, Tuple

HI_NAME = re.compile(r"^(hi|hello|hey)\s+([A-Z][a-z]{1,30})([,!\.]|\s)", re.IGNORECASE)
AGE_PAT = re.compile(r"\b(\d{1,3})\s*(years\s*old|yo)\b", re.IGNORECASE)


def _facts_map(facts: List[Tuple[str, str, float]]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for k, v, _c in facts:
        kk = str(k).strip().lower()
        if kk and kk not in d:
            d[kk] = str(v).strip()
    return d


def sanitize_reply(reply: str, facts: List[Tuple[str, str, float]]) -> str:
    if not reply:
        return reply

    f = _facts_map(facts)
    name_fact = f.get('name', '').strip()
    has_location = bool(f.get('city') or f.get('state') or f.get('country'))
    has_age = bool(f.get('age') or f.get('birthdate') or f.get('birthday'))

    out = reply.strip()

    if HI_NAME.match(out) and not name_fact:
        out = HI_NAME.sub(lambda mm: (mm.group(1).capitalize() + ', '), out, count=1).strip()

    if not has_age:
        parts = re.split(r"(?<=[.!?])\s+", out)
        parts = [p for p in parts if not AGE_PAT.search(p)]
        out = ' '.join(parts).strip()

    if not has_location:
        parts = re.split(r"(?<=[.!?])\s+", out)
        parts2 = []
        for p in parts:
            if 'you' in p.lower() and re.search(r"\b(from|in)\s+[A-Z][A-Za-z .'-]{2,50}\b", p):
                continue
            parts2.append(p)
        out = ' '.join(parts2).strip()

    return out if out else reply.strip()
