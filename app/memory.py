
# app/memory.py
from __future__ import annotations
import re, json, logging
import aiosqlite
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from . import config

logger = logging.getLogger("app")
DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")

async def _db_exec(query: str, args: Tuple = ()) -> int:
    async with aiosqlite.connect(DB_FILE) as db:
        cur = await db.execute(query, args)
        await db.commit()
        return cur.rowcount if hasattr(cur, "rowcount") else 0

async def _db_fetchall(query: str, args: Tuple = ()) -> List[Tuple]:
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(query, args) as cur:
            return await cur.fetchall()

async def _table_exists(name: str) -> bool:
    rows = await _db_fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return bool(rows)

# ---------- facts ----------
async def _current_fact_value(sender_id: str, key: str, namespace: str = "default") -> Optional[str]:
    rows = await _db_fetchall(
        "SELECT attr_value FROM user_facts WHERE sender_id=? AND namespace=? AND attr_key=?",
        (sender_id, namespace, key),
    )
    return rows[0][0] if rows else None

async def upsert_user_fact(
    sender_id: str, key: str, value: str, *,
    namespace: str = "default", value_type: str = "text",
    confidence: float = 0.95, source_msg_id: Optional[int] = None,
) -> bool:
    existing = await _current_fact_value(sender_id, key, namespace)
    if existing is not None and str(existing) == str(value):
        return False
    now_iso = datetime.now(timezone.utc).isoformat()
    upd = await _db_exec(
        """
        UPDATE user_facts
           SET attr_value=?, value_type=?, confidence=?,
               source_msg_id=COALESCE(?, source_msg_id), updated_at=?
         WHERE sender_id=? AND namespace=? AND attr_key=?
        """,
        (value, value_type, confidence, source_msg_id, now_iso, sender_id, namespace, key),
    )
    if upd and upd > 0:
        return True
    ins = await _db_exec(
        """
        INSERT INTO user_facts
          (sender_id, namespace, attr_key, attr_value, value_type, confidence, source_msg_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sender_id, namespace, key, value, value_type, confidence, source_msg_id, now_iso, now_iso),
    )
    return bool(ins and ins > 0)

async def get_user_facts(sender_id: str, namespace: str = "default") -> List[Tuple[str, str]]:
    rows = await _db_fetchall(
        """
        SELECT attr_key, attr_value
          FROM user_facts
         WHERE sender_id=? AND namespace=?
         ORDER BY updated_at DESC
        """,
        (sender_id, namespace),
    )
    return [(r[0], r[1]) for r in rows]

# ---------- records ----------
async def insert_user_record(
    sender_id: str, record_type: str, data: Dict[str, Any], *,
    namespace: str = "default", confidence: float = 0.9,
) -> bool:
    now_iso = datetime.now(timezone.utc).isoformat()
    has_records = await _table_exists("user_records")
    if has_records:
        try:
            payload = json.dumps(data, ensure_ascii=False)
            ins = await _db_exec(
                """
                INSERT INTO user_records
                    (sender_id, namespace, record_type, data, confidence, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sender_id, namespace, record_type, payload, confidence, now_iso, now_iso),
            )
            return bool(ins and ins > 0)
        except Exception as e:
            logger.warning("records.insert.failed %s", e)
    # fact fallback
    try:
        base = f"record:{record_type}:"
        persisted = 0
        for k, v in data.items():
            if v is None: continue
            key = f"{base}{k}"
            if await upsert_user_fact(sender_id, key, str(v), value_type="json", confidence=confidence):
                persisted += 1
        return persisted > 0
    except Exception as e:
        logger.error("records.fallback.failed %s", e)
        return False

# ---------- narrow deterministic fallback ----------
_CITY_PAT = re.compile(r"\b(?:my\s+)?city\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z .,'-]{1,48})", re.IGNORECASE)
_ZIP_HINT_PAT = re.compile(r"\b(?:zip|zipcode|postal\s*code|pin\s*code|pincode)\b[:=]?\s*([A-Za-z0-9-]{4,10})\b", re.IGNORECASE)
_COUNTRY_PAT = re.compile(r"\bmy\s+country\s*(?:is|=|:)?\s*(India|United\s+States|United\s+Kingdom|UK|USA|Canada|Australia)\b", re.IGNORECASE)
_BULLET_LINE = re.compile(r"^\s*(?:[-•*]\s+|\d+\)\s+|option\d+\s*:)", re.IGNORECASE)

def _clean_value(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def looks_like_bullet_line(text: str) -> bool:
    return bool(_BULLET_LINE.match(text or ""))

def extract_personal_facts(text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    if not text: return facts
    t = text.strip()
    mc = _CITY_PAT.search(t)
    if mc: facts["city"] = _clean_value(mc.group(1))
    mz = _ZIP_HINT_PAT.search(t)
    if mz: facts["postal_code"] = _clean_value(mz.group(1))
    mco = _COUNTRY_PAT.search(t)
    if mco:
        country = mco.group(1)
        if re.fullmatch(r"UK|United\s+Kingdom", country, re.IGNORECASE): country = "United Kingdom"
        elif re.fullmatch(r"USA|United\s+States", country, re.IGNORECASE): country = "United States"
        facts["country"] = _clean_value(country)
    return facts

async def classify_and_persist(chat_id: str, sender_id: str, text: str) -> Dict[str, int]:
    summary = {"facts": 0, "records": 0}
    if not sender_id or not text: return summary
    if looks_like_bullet_line(text): return summary
    facts = extract_personal_facts(text)
    persisted = 0
    for k in ("city", "postal_code", "country"):
        v = facts.get(k)
        if v:
            ok = await upsert_user_fact(sender_id, k, v, value_type="text", confidence=0.95)
            if ok: persisted += 1
    summary["facts"] = persisted
    return summary

# ---------- snapshot ----------
def _fmt_listish(value: str) -> str:
    s = (value or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, tuple)):
            return "; ".join(str(x) for x in obj)
    except Exception:
        pass
    m = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
    if m:
        parts = [a or b for a, b in m if (a or b)]
        if parts:
            return "; ".join(parts)
    return s

def _add_if(lines: List[str], key: str, label: str, facts: Dict[str, str], transform=None):
    if key in facts:
        val = facts[key]
        if transform: val = transform(val)
        lines.append(f"{label}: {val}")

async def build_profile_snapshot_text(sender_id: str) -> str:
    """
    Compact WhatsApp-friendly snapshot. Plain lines, grouped sections.
    """
    facts = dict(await get_user_facts(sender_id, namespace="default"))
    lines: List[str] = []

    # PROFILE
    core = []
    _add_if(core, "name", "NAME", facts)
    city, state, country = facts.get("city"), facts.get("state"), facts.get("country")
    if city or state or country:
        loc = ", ".join([x for x in [city, state, country] if x])
        core.append(f"LOCATION: {loc}")
    _add_if(core, "favorite_color", "FAVORITE COLOR", facts)
    if core:
        lines.append("PROFILE"); lines.extend(core)

    # WORK
    work = []
    _add_if(work, "occupation", "OCCUPATION", facts)
    _add_if(work, "current_company", "CURRENT COMPANY", facts)
    _add_if(work, "current_job_role", "CURRENT ROLE", facts)
    _add_if(work, "career_goal", "CAREER GOAL", facts)
    if work:
        lines.append("WORK"); lines.extend(work)

    # EDUCATION
    edu = []
    _add_if(edu, "educational_background", "EDUCATION", facts)
    _add_if(edu, "university", "UNIVERSITY", facts)
    _add_if(edu, "degree", "DEGREE", facts)
    _add_if(edu, "graduation_year", "GRADUATION YEAR", facts)
    _add_if(edu, "internship_company", "INTERNSHIP COMPANY", facts)
    _add_if(edu, "internship_role", "INTERNSHIP ROLE", facts)
    if edu:
        lines.append("EDUCATION"); lines.extend(edu)

    # TECH
    tech = []
    _add_if(tech, "programming_languages", "PROGRAMMING LANGUAGES", facts, _fmt_listish)
    _add_if(tech, "project_description", "PROJECT", facts)
    if tech:
        lines.append("TECH"); lines.extend(tech)

    # ENTERTAINMENT / PREFERENCES
    pref = []
    _add_if(pref, "favorite_podcasts", "FAVORITE PODCASTS", facts, _fmt_listish)
    _add_if(pref, "favorite_bands", "FAVORITE BANDS", facts, _fmt_listish)
    _add_if(pref, "music_genres", "MUSIC GENRES", facts, _fmt_listish)
    _add_if(pref, "coffee_type", "COFFEE", facts)
    _add_if(pref, "coffee_size", "COFFEE SIZE", facts)
    _add_if(pref, "hobbies", "HOBBIES", facts, _fmt_listish)
    _add_if(pref, "favorite_trail", "FAVORITE TRAIL", facts)
    if pref:
        lines.append("PREFERENCES"); lines.extend(pref)

    # PETS
    pets = []
    _add_if(pets, "pet_names", "PET NAMES", facts, _fmt_listish)
    _add_if(pets, "type_of_pets", "PET TYPES", facts, _fmt_listish)
    _add_if(pets, "name_of_pets", "PET NAMES", facts, _fmt_listish)
    _add_if(pets, "age_of_pets", "PET AGES", facts, _fmt_listish)
    _add_if(pets, "personality_of_pets", "PET PERSONALITIES", facts, _fmt_listish)
    if pets:
        lines.append("PETS"); lines.extend(pets)

    # TRANSPORT
    transport = []
    _add_if(transport, "car_year", "CAR YEAR", facts)
    _add_if(transport, "car_model", "CAR MODEL", facts)
    _add_if(transport, "commute_time_minutes", "COMMUTE (MIN)", facts)
    _add_if(transport, "neighborhood", "NEIGHBORHOOD", facts)
    if transport:
        lines.append("TRANSPORT"); lines.extend(transport)

    # FITNESS
    fit = []
    _add_if(fit, "running_mileage_per_week", "WEEKLY RUNNING MILEAGE", facts)
    _add_if(fit, "weekly_running_distance", "WEEKLY RUNNING DISTANCE", facts)
    if fit:
        lines.append("FITNESS"); lines.extend(fit)

    # GOALS / LIFE
    goals = []
    _add_if(goals, "trip_destination", "TRIP DESTINATION", facts)
    _add_if(goals, "marathon_goal", "MARATHON GOAL", facts)
    _add_if(goals, "marathon_location", "MARATHON LOCATION", facts)
    _add_if(goals, "language_goal", "LANGUAGE GOAL", facts)
    _add_if(goals, "dream_project", "DREAM PROJECT", facts)
    _add_if(goals, "best_friend", "BEST FRIEND", facts)
    if goals:
        lines.append("GOALS"); lines.extend(goals)

    # PERSONAL
    personal = []
    _add_if(personal, "birthdate", "BIRTHDATE", facts)
    _add_if(personal, "age", "AGE", facts)
    _add_if(personal, "allergy", "ALLERGY", facts)
    if personal:
        lines.append("PERSONAL"); lines.extend(personal)

    return "\n".join(lines).strip()

