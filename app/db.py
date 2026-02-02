# app/db.py
from __future__ import annotations
import aiosqlite, json
from typing import Optional, List, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from . import config

TZ = ZoneInfo(config.APP_TIMEZONE)
DB: Optional[aiosqlite.Connection] = None

SCHEMA_SQL = '''
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  chat_id TEXT,
  sender_id TEXT,
  timestamp DATETIME,
  role TEXT,
  content TEXT,
  summary TEXT,
  event_id TEXT,
  UNIQUE(chat_id, timestamp, role, content)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_chat_summary ON messages(chat_id, summary);

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  sender_id TEXT UNIQUE,
  name TEXT,
  city TEXT,
  created_at DATETIME,
  updated_at DATETIME
);

CREATE TABLE IF NOT EXISTS user_facts (
  id INTEGER PRIMARY KEY,
  sender_id TEXT NOT NULL,
  namespace TEXT NOT NULL DEFAULT 'default',
  attr_key TEXT NOT NULL,
  attr_value TEXT NOT NULL,
  value_type TEXT DEFAULT 'text',
  confidence REAL DEFAULT 0.95,
  source_msg_id INTEGER,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  UNIQUE(sender_id, namespace, attr_key)
);
CREATE INDEX IF NOT EXISTS idx_user_facts_sender_ns_key ON user_facts(sender_id, namespace, attr_key);

CREATE TABLE IF NOT EXISTS user_collections (
  id INTEGER PRIMARY KEY,
  sender_id TEXT NOT NULL,
  name TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  UNIQUE(sender_id, name)
);

CREATE TABLE IF NOT EXISTS collection_items (
  id INTEGER PRIMARY KEY,
  collection_id INTEGER NOT NULL,
  item_text TEXT NOT NULL,
  status TEXT DEFAULT 'open',
  qty REAL,
  unit TEXT,
  meta_json TEXT,
  source_msg_id INTEGER,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  FOREIGN KEY(collection_id) REFERENCES user_collections(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_collection_items_unique_open ON collection_items(collection_id, item_text, status);
CREATE INDEX IF NOT EXISTS idx_collection_items_coll_status ON collection_items(collection_id, status);

CREATE TABLE IF NOT EXISTS record_types (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  schema_json TEXT NOT NULL,
  UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS user_records (
  id INTEGER PRIMARY KEY,
  sender_id TEXT NOT NULL,
  record_type_id INTEGER NOT NULL,
  record_json TEXT NOT NULL,
  occurred_at DATETIME,
  source_msg_id INTEGER,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  FOREIGN KEY(record_type_id) REFERENCES record_types(id)
);
CREATE INDEX IF NOT EXISTS idx_user_records_sender_type ON user_records(sender_id, record_type_id);

CREATE TABLE IF NOT EXISTS usage_ledger (
  date_pt TEXT NOT NULL,
  model TEXT NOT NULL,
  req INTEGER NOT NULL DEFAULT 0,
  tokens INTEGER NOT NULL DEFAULT 0,
  updated_at DATETIME,
  PRIMARY KEY (date_pt, model)
);
'''

async def init_db():
    global DB
    DB = await aiosqlite.connect(config.DB_FILE)
    await DB.executescript(SCHEMA_SQL)
    await DB.commit()

async def fetchall(query: str, params: Tuple = ()) -> List[Tuple]:
    if not DB: return []
    async with DB.execute(query, params) as cur:
        return await cur.fetchall()

async def execute(query: str, params: Tuple = ()) -> None:
    if not DB: return
    await DB.execute(query, params)
    await DB.commit()

async def save_message(chat_id: str, sender_id: str, role: str, content: str,
                       summary: str | None = None, event_id: str | None = None) -> Optional[int]:
    if not DB: return None
    ts = datetime.now(TZ).isoformat()
    try:
        cur = await DB.execute(
            "INSERT INTO messages (chat_id, sender_id, timestamp, role, content, summary, event_id) VALUES (?,?,?,?,?,?,?)",
            (chat_id, sender_id, ts, role, content, summary, event_id)
        )
        await DB.commit()
        return cur.lastrowid
    except aiosqlite.IntegrityError:
        return None

async def upsert_display_name(sender_id: str, name: str) -> None:
    if not DB: return
    now = datetime.now(TZ).isoformat()
    await DB.execute(
        """INSERT INTO users (sender_id, name, created_at, updated_at)
           VALUES (?,?,?,?)
           ON CONFLICT(sender_id) DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at""",
        (sender_id, name, now, now)
    )
    await DB.commit()

async def upsert_fact(sender_id: str, key: str, value: str, value_type: str = 'text',
                      confidence: float = 0.95, namespace: str = 'default') -> None:
    if not DB or not key: return
    now = datetime.now(TZ).isoformat()
    await DB.execute(
        """INSERT INTO user_facts (sender_id, namespace, attr_key, attr_value, value_type, confidence, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(sender_id, namespace, attr_key) DO UPDATE SET
             attr_value=excluded.attr_value,
             value_type=excluded.value_type,
             confidence=min(?, 1.0),
             updated_at=excluded.updated_at""",
        (sender_id, namespace, key, value, value_type, confidence, now, now, confidence)
    )
    await DB.commit()

async def ensure_collection(sender_id: str, name: str) -> Optional[int]:
    if not DB: return None
    now = datetime.now(TZ).isoformat()
    await DB.execute(
        """INSERT INTO user_collections (sender_id, name, created_at, updated_at)
           VALUES (?,?,?,?) ON CONFLICT(sender_id,name) DO UPDATE SET updated_at=excluded.updated_at""",
        (sender_id, name, now, now)
    )
    await DB.commit()
    rows = await fetchall("SELECT id FROM user_collections WHERE sender_id=? AND name=?", (sender_id, name))
    return int(rows[0][0]) if rows else None

async def add_collection_items(sender_id: str, name: str, items: list[dict]) -> int:
    coll_id = await ensure_collection(sender_id, name)
    if not DB or not coll_id or not items:
        return 0
    now = datetime.now(TZ).isoformat()
    changed = 0
    await DB.execute("BEGIN IMMEDIATE")
    try:
        for it in items:
            item_text = (it.get('item_text') or '').strip()
            if not item_text:
                continue
            qty = it.get('qty'); unit = it.get('unit'); meta_json = it.get('meta_json')
            await DB.execute(
                """INSERT INTO collection_items (collection_id, item_text, status, qty, unit, meta_json, created_at, updated_at)
                   VALUES (?,?,'open',?,?,?, ?, ?)
                   ON CONFLICT(collection_id, item_text, status) DO UPDATE SET
                      qty=COALESCE(excluded.qty, collection_items.qty),
                      unit=COALESCE(excluded.unit, collection_items.unit),
                      meta_json=COALESCE(excluded.meta_json, collection_items.meta_json),
                      updated_at=excluded.updated_at""",
                (coll_id, item_text, qty, unit, meta_json, now, now)
            )
            changed += 1
        await DB.commit()
        return changed
    except Exception:
        await DB.execute("ROLLBACK")
        return 0

async def mark_collection_item(sender_id: str, name: str, item_text: str, status: str = 'done') -> bool:
    coll_id = await ensure_collection(sender_id, name)
    if not DB or not coll_id:
        return False
    now = datetime.now(TZ).isoformat()
    await DB.execute("UPDATE collection_items SET status=?, updated_at=? WHERE collection_id=? AND item_text=? AND status='open'",
                     (status, now, coll_id, item_text))
    await DB.commit()
    rows = await fetchall("SELECT changes()")
    return bool(int(rows[0][0]) if rows else 0)

async def clear_done(sender_id: str, name: str) -> int:
    coll_id = await ensure_collection(sender_id, name)
    if not coll_id: return 0
    await DB.execute("DELETE FROM collection_items WHERE collection_id=? AND status='done'", (coll_id,))
    await DB.commit()
    rows = await fetchall("SELECT changes()")
    return int(rows[0][0]) if rows else 0

async def ensure_record_type(name: str, schema_json: str) -> Optional[int]:
    if not DB: return None
    await DB.execute(
        """INSERT INTO record_types (name, schema_json) VALUES (?,?)
            ON CONFLICT(name) DO UPDATE SET schema_json=excluded.schema_json""",
        (name, schema_json)
    )
    await DB.commit()
    rows = await fetchall("SELECT id FROM record_types WHERE name=?", (name,))
    return int(rows[0][0]) if rows else None

async def insert_user_record(sender_id: str, record_type: str, obj: dict,
                             occurred_at_iso: str | None = None,
                             source_msg_id: int | None = None) -> Optional[int]:
    type_id = await ensure_record_type(record_type, json.dumps({"type":"object"}))
    if not DB or not type_id: return None
    now = datetime.now(TZ).isoformat()
    cur = await DB.execute(
        """INSERT INTO user_records (sender_id, record_type_id, record_json, occurred_at, source_msg_id, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (sender_id, type_id, json.dumps(obj, ensure_ascii=False), occurred_at_iso, source_msg_id, now, now)
    )
    await DB.commit()
    return int(cur.lastrowid)

async def list_records(sender_id: str, record_type: str) -> list[dict]:
    rows = await fetchall("""SELECT ur.record_json FROM user_records ur
                             JOIN record_types rt ON rt.id=ur.record_type_id
                             WHERE ur.sender_id=? AND rt.name=?
                             ORDER BY ur.updated_at DESC""", (sender_id, record_type))
    return [json.loads(r[0]) for r in rows]

# simple usage ledger (optional)
USAGE_LEDGER: dict[str, dict[str, int]] = {}
LEDGER_DATE_PT: str = ''
from datetime import datetime as _dt

def _pt_today() -> str:
    return _dt.now(TZ).strftime('%Y-%m-%d')

async def inc_usage(model: str, tokens: int):
    global LEDGER_DATE_PT, USAGE_LEDGER
    today = _pt_today()
    if LEDGER_DATE_PT != today:
        USAGE_LEDGER = {}
        LEDGER_DATE_PT = today
    m = USAGE_LEDGER.setdefault(model, {"req":0,"tokens":0})
    m["req"] += 1
    m["tokens"] += max(0, int(tokens or 0))
    now = _dt.now(TZ).isoformat()
    await execute("""INSERT INTO usage_ledger (date_pt, model, req, tokens, updated_at)
                     VALUES (?,?,?,?,?)
                     ON CONFLICT(date_pt,model) DO UPDATE SET
                       req = usage_ledger.req + excluded.req,
                       tokens = usage_ledger.tokens + excluded.tokens,
                       updated_at = excluded.updated_at""",
                  (today, model, 1, max(0, int(tokens or 0)), now))

