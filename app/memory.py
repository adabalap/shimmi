from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from zoneinfo import ZoneInfo

from . import config
from .db import fetchall, execute


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_recent_messages(chat_id: str, limit: int = 12) -> List[Tuple[str, str, str]]:
    rows = await fetchall('SELECT timestamp, role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?', (chat_id, int(limit)))
    rows = list(reversed(rows))
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


async def get_messages_in_window(chat_id: str, *, start_iso: str, end_iso: str, role: str = 'user', limit: int = 200) -> List[Tuple[str, str, str]]:
    rows = await fetchall(
        'SELECT timestamp, role, content FROM messages WHERE chat_id=? AND role=? AND timestamp>=? AND timestamp<? ORDER BY timestamp ASC LIMIT ?',
        (chat_id, role, start_iso, end_iso, int(limit)),
    )
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def day_window_iso(*, tz_name: str) -> tuple[str, str]:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    start = datetime(now.year, now.month, now.day, tzinfo=tz)
    end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start.isoformat(), end.isoformat()


async def get_user_facts(sender_id: str, namespace: str = 'default') -> List[Tuple[str, str, float]]:
    rows = await fetchall('SELECT attr_key, attr_value, confidence FROM user_facts WHERE sender_id=? AND namespace=? ORDER BY updated_at DESC', (sender_id, namespace))
    return [(str(r[0]), str(r[1]), float(r[2] or 0.0)) for r in rows]


async def upsert_user_fact(sender_id: str, key: str, value: str, *, namespace: str = 'default', value_type: str = 'text', confidence: float = 0.9, source_msg_id: Optional[int] = None) -> bool:
    existing = await fetchall('SELECT attr_value FROM user_facts WHERE sender_id=? AND namespace=? AND attr_key=?', (sender_id, namespace, key))
    if existing and str(existing[0][0]) == str(value):
        return False

    now = _now_iso_utc()
    sql = (
        'INSERT INTO user_facts (sender_id, namespace, attr_key, attr_value, value_type, confidence, source_msg_id, created_at, updated_at) '
        'VALUES (?,?,?,?,?,?,?,?,?) '
        'ON CONFLICT(sender_id, namespace, attr_key) DO UPDATE SET '
        'attr_value=excluded.attr_value, value_type=excluded.value_type, confidence=excluded.confidence, '
        'source_msg_id=COALESCE(excluded.source_msg_id, user_facts.source_msg_id), updated_at=excluded.updated_at'
    )
    await execute(sql, (sender_id, namespace, key, value, value_type, float(confidence), source_msg_id, now, now))
    return True


async def clear_user_memory(sender_id: str) -> None:
    await execute('DELETE FROM user_facts WHERE sender_id=?', (sender_id,))
