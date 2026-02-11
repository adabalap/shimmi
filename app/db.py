from __future__ import annotations

import aiosqlite
import logging
import os
from datetime import datetime
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo

from . import config

logger = logging.getLogger('app.db')
TZ = ZoneInfo(config.APP_TIMEZONE)
DB: Optional[aiosqlite.Connection] = None
DB_TRACE = os.getenv('DB_TRACE', '0').lower() in ('1','true','yes','on')

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=3000;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  chat_id TEXT,
  sender_id TEXT,
  timestamp TEXT,
  role TEXT,
  content TEXT,
  meta_json TEXT,
  event_id TEXT,
  UNIQUE(chat_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, timestamp);

CREATE TABLE IF NOT EXISTS user_facts (
  id INTEGER PRIMARY KEY,
  sender_id TEXT NOT NULL,
  namespace TEXT NOT NULL DEFAULT 'default',
  attr_key TEXT NOT NULL,
  attr_value TEXT NOT NULL,
  value_type TEXT DEFAULT 'text',
  confidence REAL DEFAULT 0.95,
  source_msg_id INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(sender_id, namespace, attr_key)
);

CREATE TABLE IF NOT EXISTS chat_prefs (
  chat_id TEXT PRIMARY KEY,
  observe_mode TEXT NOT NULL DEFAULT 'off',
  retention_days INTEGER NOT NULL DEFAULT 30,
  redaction_enabled INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _trace(sql: str, params: Tuple):
    if DB_TRACE:
        logger.info('🗄️ db.sql %s params=%s', sql.replace('\n',' ')[:220], params)


async def _table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    cols = set()
    async with db.execute(f'PRAGMA table_info({table})') as cur:
        rows = await cur.fetchall()
    for r in rows:
        if r and len(r) >= 2:
            cols.add(str(r[1]))
    return cols


async def _migrate_messages(db: aiosqlite.Connection) -> None:
    cols = await _table_columns(db, 'messages')
    if 'meta_json' not in cols:
        await db.execute("ALTER TABLE messages ADD COLUMN meta_json TEXT DEFAULT ''")
        logger.info('🧱 db.migrate messages.add_column name=meta_json')
    if 'event_id' not in cols:
        await db.execute("ALTER TABLE messages ADD COLUMN event_id TEXT")
        logger.info('🧱 db.migrate messages.add_column name=event_id')
    await db.commit()


async def init_db():
    global DB
    DB = await aiosqlite.connect(config.DB_FILE)
    await DB.executescript(SCHEMA_SQL)
    try:
        await _migrate_messages(DB)
    except Exception as e:
        logger.warning('🧱 db.migrate.fail err=%s', str(e)[:180])
    await DB.commit()
    logger.info('🗄️ db.ready file=%s', config.DB_FILE)


async def close_db():
    global DB
    if DB:
        await DB.close()
        DB = None
        logger.info('🗄️ db.closed')


async def fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    if not DB:
        return []
    _trace(query, params)
    async with DB.execute(query, params) as cur:
        return await cur.fetchall()


async def execute(query: str, params: Tuple = ()) -> None:
    if not DB:
        return
    _trace(query, params)
    await DB.execute(query, params)
    await DB.commit()


async def save_message(chat_id: str, sender_id: str, role: str, content: str, *, event_id: str = '', meta_json: str = '') -> Optional[int]:
    if not DB:
        return None
    ts = _now_iso()
    sql = 'INSERT INTO messages (chat_id, sender_id, timestamp, role, content, meta_json, event_id) VALUES (?,?,?,?,?,?,?)'
    params = (chat_id, sender_id, ts, role, content, meta_json or '', event_id or None)
    try:
        cur = await DB.execute(sql, params)
        await DB.commit()
        return cur.lastrowid
    except aiosqlite.OperationalError as e:
        if 'meta_json' in str(e):
            sql2 = 'INSERT INTO messages (chat_id, sender_id, timestamp, role, content, event_id) VALUES (?,?,?,?,?,?)'
            cur = await DB.execute(sql2, (chat_id, sender_id, ts, role, content, event_id or None))
            await DB.commit()
            return cur.lastrowid
        raise
    except aiosqlite.IntegrityError:
        return None


async def get_chat_prefs(chat_id: str) -> dict:
    default_mode = (config.OBSERVE_GROUPS_DEFAULT or 'off').strip().lower()
    default_days = int(config.OBSERVE_RETENTION_DEFAULT_DAYS)
    default_red = 1 if str(config.OBSERVE_REDACTION_DEFAULT).strip().lower() in ('1','true','yes','on') else 0
    rows = await fetchall('SELECT observe_mode, retention_days, redaction_enabled FROM chat_prefs WHERE chat_id=?', (chat_id,))
    if not rows:
        return {'observe_mode': default_mode, 'retention_days': default_days, 'redaction_enabled': default_red}
    return {'observe_mode': rows[0][0], 'retention_days': int(rows[0][1]), 'redaction_enabled': int(rows[0][2])}


async def set_chat_prefs(chat_id: str, *, observe_mode=None, retention_days=None, redaction_enabled=None) -> None:
    now = _now_iso()
    cur = await get_chat_prefs(chat_id)
    om = observe_mode if observe_mode is not None else cur['observe_mode']
    rd = int(retention_days) if retention_days is not None else int(cur['retention_days'])
    re_ = int(redaction_enabled) if redaction_enabled is not None else int(cur['redaction_enabled'])
    sql = (
        'INSERT INTO chat_prefs (chat_id, observe_mode, retention_days, redaction_enabled, updated_at) VALUES (?,?,?,?,?) '
        'ON CONFLICT(chat_id) DO UPDATE SET observe_mode=excluded.observe_mode, retention_days=excluded.retention_days, '
        'redaction_enabled=excluded.redaction_enabled, updated_at=excluded.updated_at'
    )
    await execute(sql, (chat_id, om, rd, re_, now))
