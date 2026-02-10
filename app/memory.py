from __future__ import annotations

import re, json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone

from .db import fetchall, execute, insert_user_record as db_insert_user_record


async def get_user_facts(sender_id: str, namespace: str = "default") -> List[Tuple[str, str]]:
    rows = await fetchall(
        """SELECT attr_key, attr_value
             FROM user_facts
            WHERE sender_id=? AND namespace=?
            ORDER BY updated_at DESC""",
        (sender_id, namespace),
    )
    return [(r[0], r[1]) for r in rows]


async def upsert_user_fact(sender_id: str, key: str, value: str, *,
                           namespace: str = "default", value_type: str = "text",
                           confidence: float = 0.95, source_msg_id: Optional[int] = None) -> bool:
    rows = await fetchall(
        "SELECT attr_value FROM user_facts WHERE sender_id=? AND namespace=? AND attr_key=?",
        (sender_id, namespace, key),
    )
    if rows and str(rows[0][0]) == str(value):
        return False

    now_iso = datetime.now(timezone.utc).isoformat()
    await execute(
        """INSERT INTO user_facts
           (sender_id, namespace, attr_key, attr_value, value_type, confidence, source_msg_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(sender_id, namespace, attr_key) DO UPDATE SET
             attr_value=excluded.attr_value,
             value_type=excluded.value_type,
             confidence=MIN(excluded.confidence, 1.0),
             source_msg_id=COALESCE(excluded.source_msg_id, user_facts.source_msg_id),
             updated_at=excluded.updated_at""",
        (sender_id, namespace, key, value, value_type, confidence, source_msg_id, now_iso, now_iso),
    )
    return True


async def insert_user_record(sender_id: str, record_type: str, data: Dict[str, Any], *,
                             confidence: float = 0.9,
                             occurred_at_iso: str | None = None,
                             source_msg_id: int | None = None) -> bool:
    obj = dict(data or {})
    obj["_confidence"] = float(confidence)
    rid = await db_insert_user_record(sender_id, record_type, obj, occurred_at_iso=occurred_at_iso, source_msg_id=source_msg_id)
    return bool(rid)


_CITY_PAT = re.compile(r"(?:my\s+)?city\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z .,'-]{1,48})", re.IGNORECASE)
_COUNTRY_PAT = re.compile(r"my\s+country\s*(?:is|=|:)?\s*(India|United\s+States|United\s+Kingdom|UK|USA|Canada|Australia)", re.IGNORECASE)


def _clean_value(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_personal_facts(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    if not text:
        return facts
    t = text.strip()
    mc = _CITY_PAT.search(t)
    if mc:
        facts["city"] = _clean_value(mc.group(1))
    mco = _COUNTRY_PAT.search(t)
    if mco:
        country = mco.group(1)
        if re.fullmatch(r"UK|United\s+Kingdom", country, re.IGNORECASE):
            country = "United Kingdom"
        elif re.fullmatch(r"USA|United\s+States", country, re.IGNORECASE):
            country = "United States"
        facts["country"] = _clean_value(country)
    return facts


async def classify_and_persist(chat_id: str, sender_id: str, text: str) -> Dict[str, int]:
    summary = {"facts": 0, "records": 0}
    if not sender_id or not text:
        return summary
    facts = extract_personal_facts(text)
    persisted = 0
    for k in ("city", "country"):
        v = facts.get(k)
        if v:
            ok = await upsert_user_fact(sender_id, k, v, namespace="default", value_type="text", confidence=0.75)
            if ok:
                persisted += 1
    summary["facts"] = persisted
    return summary


def _fmt_listish(value: str) -> str:
    s = (value or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return "; ".join(str(x) for x in obj)
    except Exception:
        pass
    return s


def _add_if(lines: List[str], key: str, label: str, facts: Dict[str, str], transform=None):
    if key in facts:
        val = facts[key]
        if transform:
            val = transform(val)
        lines.append(f"{label}: {val}")


async def build_profile_snapshot_text(sender_id: str) -> str:
    merged: Dict[str, str] = {}
    for ns in ("default", "prefs", "location", "work", "family", "contact", "custom"):
        rows = await get_user_facts(sender_id, namespace=ns)
        for k, v in rows:
            if k not in merged:
                merged[k] = v

    facts = merged
    lines: List[str] = []

    core = []
    _add_if(core, "name", "NAME", facts)
    city, state, country = facts.get("city"), facts.get("state"), facts.get("country")
    if city or state or country:
        loc = ", ".join([x for x in [city, state, country] if x])
        core.append(f"LOCATION: {loc}")
    _add_if(core, "favorite_color", "FAVORITE COLOR", facts)
    if core:
        lines.append("PROFILE")
        lines.extend(core)

    work = []
    _add_if(work, "occupation", "OCCUPATION", facts)
    _add_if(work, "current_company", "CURRENT COMPANY", facts)
    _add_if(work, "current_job_role", "CURRENT ROLE", facts)
    _add_if(work, "career_goal", "CAREER GOAL", facts)
    if work:
        lines.append("WORK")
        lines.extend(work)

    pref = []
    _add_if(pref, "hobbies", "HOBBIES", facts, _fmt_listish)
    _add_if(pref, "favorite_podcasts", "PODCASTS", facts, _fmt_listish)
    _add_if(pref, "coffee_order", "COFFEE", facts)
    if pref:
        lines.append("PREFERENCES")
        lines.extend(pref)

    personal = []
    _add_if(personal, "birthdate", "BIRTHDATE", facts)
    _add_if(personal, "age", "AGE", facts)
    _add_if(personal, "allergy", "ALLERGY", facts)
    if personal:
        lines.append("PERSONAL")
        lines.extend(personal)

    return " ".join(lines).strip()
