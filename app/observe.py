from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

from . import config
from .utils import canonical_text

logger = logging.getLogger('app.observe')

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d \-]{7,}\d)\b")
URL_RE = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)
LINK_ONLY_RE = re.compile(r"^\s*(https?://\S+\s*)+$", re.IGNORECASE)

TOPIC_RULES = [
    ('MOVIES', re.compile(r"\b(movie|watch|stream|trailer|imax|netflix|prime|hotstar)\b", re.I)),
    ('TRIP',   re.compile(r"\b(trip|travel|itinerary|hotel|flight|booking|visa)\b", re.I)),
    ('MUSIC',  re.compile(r"\b(music|album|playlist|song|band|spotify)\b", re.I)),
    ('SPORT',  re.compile(r"\b(cricket|football|soccer|nba|ipl|match|score)\b", re.I)),
    ('WORK',   re.compile(r"\b(meeting|jira|release|deployment|incident|prod)\b", re.I)),
    ('LIFE',   re.compile(r"\b(today|bought|purchased|ordered|paid|went|visited|met|ate)\b", re.I)),
]

_BUCKET: dict[str, tuple[int, int]] = {}


def redact(text: str) -> str:
    t = text or ''
    t = EMAIL_RE.sub('[email]', t)
    t = PHONE_RE.sub('[phone]', t)
    t = URL_RE.sub('[link]', t)
    return t


def topic_tag(text: str) -> str | None:
    for name, pat in TOPIC_RULES:
        if pat.search(text or ''):
            return name
    return None


def _allow_embed(chat_id: str) -> bool:
    now_min = int(time.time() // 60)
    m, c = _BUCKET.get(chat_id, (now_min, 0))
    if m != now_min:
        m, c = now_min, 0
    if c >= int(config.OBSERVE_MAX_EMBED_PER_MIN):
        _BUCKET[chat_id] = (m, c)
        return False
    _BUCKET[chat_id] = (m, c + 1)
    return True


def should_observe_group(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if len(s) < int(config.OBSERVE_MIN_TEXT_LEN):
        return False
    if LINK_ONLY_RE.match(s):
        return False
    return True


def should_observe_dm(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if LINK_ONLY_RE.match(s):
        return False
    if len(s) >= int(config.OBSERVE_MIN_TEXT_LEN):
        return True
    return bool(re.search(r"\b(today|bought|purchased|ordered|paid|went|visited|met|ate)\b", s, re.I))


async def enforce_retention(chat_id: str, retention_days: int, *, db_exec):
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    await db_exec('DELETE FROM rag_chunks WHERE chat_id=? AND created_at < ?', (chat_id, cutoff.isoformat()))


async def ingest(*, chat_id: str, sender_id: str, text: str, mode: str, chroma_add_text, db_exec) -> bool:
    if mode == 'off':
        return False
    if not _allow_embed(chat_id):
        logger.info('👁️ observe.rate_limit chat_id=%s', chat_id)
        return False

    msg = canonical_text(text)
    if str(config.OBSERVE_REDACTION_DEFAULT).strip().lower() in ('1','true','yes','on'):
        msg = redact(msg)

    tag = topic_tag(msg) if mode == 'topics' else None
    if tag:
        msg = f'TOPIC: {tag}\n{msg}'

    try:
        await chroma_add_text(chat_id=chat_id, sender_id=sender_id, text=msg)
        await enforce_retention(chat_id, int(config.OBSERVE_RETENTION_DEFAULT_DAYS), db_exec=db_exec)
        logger.info('👁️ observe.add chat_id=%s mode=%s tag=%s', chat_id, mode, tag or '')
        return True
    except Exception as e:
        logger.warning('👁️ observe.fail chat_id=%s err=%s', chat_id, str(e)[:160])
        return False


async def observe_ingest_group(chat_id: str, sender_id: str, text: str, *, chroma_add_text, db_exec, get_prefs) -> bool:
    prefs = await get_prefs(chat_id)
    mode = (prefs.get('observe_mode') or 'off').lower()
    if mode == 'off':
        return False
    if not should_observe_group(text):
        return False
    return await ingest(chat_id=chat_id, sender_id=sender_id, text=text, mode=mode, chroma_add_text=chroma_add_text, db_exec=db_exec)


async def observe_ingest_dm(chat_id: str, sender_id: str, text: str, *, chroma_add_text, db_exec) -> bool:
    mode = (config.OBSERVE_DMS_DEFAULT or 'on').lower()
    if mode == 'off':
        return False
    if not should_observe_dm(text):
        return False
    return await ingest(chat_id=chat_id, sender_id=sender_id, text=text, mode=mode, chroma_add_text=chroma_add_text, db_exec=db_exec)


async def handle_observe_command(chat_id: str, raw_text: str, *, get_prefs, set_prefs) -> str | None:
    tl = (raw_text or '').strip().lower()
    if not tl.startswith('/observe'):
        return None

    parts = tl.split()
    if len(parts) == 1 or (len(parts) == 2 and parts[1] == 'status'):
        prefs = await get_prefs(chat_id)
        return f"Observe: {prefs.get('observe_mode','off')} | Retention: {prefs.get('retention_days',30)}d | Redaction: {'on' if int(prefs.get('redaction_enabled',1))==1 else 'off'}"

    cmd = parts[1]
    if cmd in ('on','topics'):
        await set_prefs(chat_id, observe_mode='topics', redaction_enabled=1)
        prefs = await get_prefs(chat_id)
        return f"✅ Ambient observe ON. Retention: {prefs.get('retention_days',30)}d."
    if cmd == 'off':
        await set_prefs(chat_id, observe_mode='off')
        return '✅ Ambient observe OFF.'
    return 'Usage: /observe on|off|status'
