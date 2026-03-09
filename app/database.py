"""
database.py — Shimmi v2.9.2

Changes vs v2.7.0:
  ① reminders table + ReminderStore methods
     - add_reminder(), get_due_reminders(), mark_reminder_sent()
     - get_user_reminders(), cancel_reminder()
  ② Dataclass Reminder for typed reminder records
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger("app.database")
UTC    = timezone.utc

# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

_KEY_ALIASES: Dict[str, str] = {
    "user_name": "name", "username": "name", "first_name": "name", "full_name": "name",
    "user_city": "city", "user_location": "city", "location": "city", "hometown": "city",
    "user_country": "country",
    "user_favorite_drink": "favorite_drink", "preferred_drink": "favorite_drink",
    "user_drink": "favorite_drink", "drink": "favorite_drink",
    "user_interests": "interests", "user_interest": "interests",
    "user_hobby": "hobbies", "user_hobbies": "hobbies",
    "user_occupation": "occupation", "user_job": "occupation", "job": "occupation",
    "user_age": "age",
    "user_language": "preferred_language", "language": "preferred_language",
    "grocery": "grocery_list", "groceries": "grocery_list",
    "shopping": "shopping_list",
    "todo": "todo_list", "todos": "todo_list",
}

_SPECIAL_PREFIXES = ("_reminder", "_cancel_reminder")


def normalize_key(raw: str) -> str:
    k = (raw or "").strip().lower()
    k = re.sub(r"[\s\-]+", "_", k)
    k = re.sub(r"[^\w]", "", k)
    k = re.sub(r"_+", "_", k)
    if k.startswith("user_"):
        k = k[5:]
    k = k.strip("_")
    if not k or k == "user":
        return ""
    return _KEY_ALIASES.get(k, k)


# ---------------------------------------------------------------------------
# Reminder dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reminder:
    id:            int
    whatsapp_id:   str
    chat_id:       str
    reminder_text: str
    trigger_iso:   str          # ISO 8601 with tz offset, e.g. "2026-03-09T06:00+05:30"
    created_at:    str
    sent_at:       Optional[str] = None
    cancelled:     bool          = False
    failed:        bool          = False
    user_name:     str           = ""  # populated from facts at send time

    @property
    def status(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.failed:
            return "failed"
        if self.sent_at:
            return "sent"
        return "pending"


# ---------------------------------------------------------------------------
# SQLite memory + reminder store
# ---------------------------------------------------------------------------

class SQLiteMemory:
    def __init__(self, path):
        self.path  = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")

            # ── user facts KV ──────────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id  TEXT NOT NULL,
                    fact_key     TEXT NOT NULL,
                    fact_value   TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
            """)
            self._migrate(conn)

            # ── message log ───────────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id     TEXT NOT NULL,
                    whatsapp_id TEXT,
                    direction   TEXT NOT NULL,
                    event_id    TEXT,
                    text        TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    UNIQUE(chat_id, event_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_log_chat_ts "
                "ON message_log(chat_id, ts DESC)"
            )

            # ── reminders ─────────────────────────────────────────────────
            # NOTE: _migrate() may have already rebuilt this table from an older
            # schema. CREATE TABLE IF NOT EXISTS is a no-op in that case — the
            # migrated table with the correct schema is already in place.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id    TEXT NOT NULL,
                    chat_id        TEXT NOT NULL,
                    reminder_text  TEXT NOT NULL,
                    trigger_iso    TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    sent_at        TEXT,
                    cancelled      INTEGER NOT NULL DEFAULT 0,
                    failed         INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_trigger "
                "ON reminders(trigger_iso) WHERE sent_at IS NULL AND cancelled=0 AND failed=0"
            )
            conn.commit()

        logger.info("🗄️  sqlite.ready  path=%s", self.path)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        # ── user_memory migrations ─────────────────────────────────────────
        um_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_memory)").fetchall()
        }
        if "created_at" not in um_cols:
            logger.info("🗄️  migrate — adding created_at to user_memory")
            conn.execute("ALTER TABLE user_memory ADD COLUMN created_at TEXT")
            conn.execute(
                "UPDATE user_memory SET created_at = updated_at WHERE created_at IS NULL"
            )
            conn.commit()
            logger.info("🗄️  migrate — created_at back-filled ✓")
        if "updated_at" not in um_cols:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                f"ALTER TABLE user_memory ADD COLUMN updated_at TEXT DEFAULT '{now}'"
            )
            conn.commit()

        # ── reminders table migration ──────────────────────────────────────
        # v2.9.0 canonical schema uses cancelled/failed INTEGER booleans.
        # Any prior variant (no trigger_iso, status TEXT, user_name TEXT, etc.)
        # must be rebuilt. We check for the EXACT v2.9.0 columns; anything
        # missing triggers a rename+rebuild so queries never hit missing columns.
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "reminders" in tables:
            rem_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(reminders)").fetchall()
            }
            # v2.9.0 requires cancelled + failed boolean columns.
            # Any older schema (status TEXT, missing trigger_iso, etc.) is rebuilt.
            required = {"cancelled", "failed", "trigger_iso",
                        "chat_id", "whatsapp_id", "reminder_text", "created_at"}
            if not required.issubset(rem_cols):
                logger.info(
                    "🗄️  migrate — reminders schema outdated (missing: %s), rebuilding",
                    ", ".join(sorted(required - rem_cols)),
                )
                # Drop any partial index that references missing columns
                conn.execute("DROP INDEX IF EXISTS idx_reminders_trigger")
                # Rename old table so we can recreate with new schema
                conn.execute("ALTER TABLE reminders RENAME TO reminders_old")
                # Create the new schema (v2.9.0: cancelled/failed booleans)
                conn.execute("""
                    CREATE TABLE reminders (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        whatsapp_id    TEXT NOT NULL,
                        chat_id        TEXT NOT NULL,
                        reminder_text  TEXT NOT NULL,
                        trigger_iso    TEXT NOT NULL,
                        created_at     TEXT NOT NULL,
                        sent_at        TEXT,
                        cancelled      INTEGER NOT NULL DEFAULT 0,
                        failed         INTEGER NOT NULL DEFAULT 0
                    )
                """)
                # Copy rows that have the columns we need (best-effort).
                # If old table has trigger_iso we copy it; otherwise use a
                # sentinel past-date so the scheduler marks it stale+drops it.
                if "chat_id" in rem_cols and "reminder_text" in rem_cols:
                    now_iso     = datetime.now(UTC).isoformat()
                    wa_col      = "whatsapp_id" if "whatsapp_id" in rem_cols else "''"
                    ca_col      = "created_at"  if "created_at"  in rem_cols else f"'{now_iso}'"
                    trigger_col = "trigger_iso" if "trigger_iso" in rem_cols else "'2000-01-01T00:00:00+00:00'"
                    conn.execute(
                        f"""INSERT INTO reminders
                            (whatsapp_id, chat_id, reminder_text,
                             trigger_iso, created_at, failed)
                           SELECT {wa_col}, chat_id, reminder_text,
                                  {trigger_col}, {ca_col}, 1
                           FROM reminders_old
                           WHERE reminder_text IS NOT NULL"""
                    )
                    logger.info("🗄️  migrate — reminders rows copied (marked failed/stale)")
                conn.execute("DROP TABLE reminders_old")
                conn.commit()
                logger.info("🗄️  migrate — reminders table rebuilt ✓")

    # ── facts API ──────────────────────────────────────────────────────────

    async def get_all_facts(self, whatsapp_id: str) -> Dict[str, str]:
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_key, fact_value FROM user_memory "
                        "WHERE whatsapp_id=? ORDER BY updated_at DESC",
                        (whatsapp_id,),
                    )
                    out: Dict[str, str] = {}
                    for raw_key, val in cur.fetchall():
                        canon = normalize_key(raw_key)
                        if canon and canon not in out:
                            out[canon] = val
                    return out
            return await asyncio.to_thread(_do)

    async def upsert_fact(self, whatsapp_id: str, raw_key: str, value: str) -> str:
        key   = normalize_key(raw_key)
        value = (value or "").strip()
        if not key or not value:
            return "unchanged"
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_value FROM user_memory "
                        "WHERE whatsapp_id=? AND fact_key=?",
                        (whatsapp_id, key),
                    )
                    row = cur.fetchone()
                    if row is None:
                        conn.execute(
                            "INSERT INTO user_memory "
                            "(whatsapp_id, fact_key, fact_value, created_at, updated_at) "
                            "VALUES (?,?,?,?,?)",
                            (whatsapp_id, key, value, now, now),
                        )
                        conn.commit()
                        return "created"
                    if (row[0] or "").strip() == value:
                        return "unchanged"
                    conn.execute(
                        "UPDATE user_memory SET fact_value=?, updated_at=? "
                        "WHERE whatsapp_id=? AND fact_key=?",
                        (value, now, whatsapp_id, key),
                    )
                    conn.commit()
                    return "updated"
            return await asyncio.to_thread(_do)

    async def log_message(
        self, *, chat_id, whatsapp_id, direction, text, ts, event_id=None,
    ) -> None:
        if not chat_id or not (text or "").strip():
            return
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    try:
                        conn.execute(
                            "INSERT INTO message_log "
                            "(chat_id, whatsapp_id, direction, event_id, text, ts) "
                            "VALUES (?,?,?,?,?,?)",
                            (chat_id, whatsapp_id or "", direction, event_id or None, text, ts),
                        )
                        conn.commit()
                    except sqlite3.IntegrityError:
                        pass
            await asyncio.to_thread(_do)

    # ── reminders API ──────────────────────────────────────────────────────

    async def add_reminder(
        self,
        whatsapp_id: str,
        chat_id:     str,
        text:        str,
        trigger_iso: str,
    ) -> int:
        """Insert a new reminder. Returns its auto-increment id."""
        async with self._lock:
            def _do() -> int:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "INSERT INTO reminders "
                        "(whatsapp_id, chat_id, reminder_text, trigger_iso, created_at) "
                        "VALUES (?,?,?,?,?)",
                        (whatsapp_id, chat_id, text.strip(), trigger_iso.strip(), now),
                    )
                    conn.commit()
                    return cur.lastrowid
            return await asyncio.to_thread(_do)

    async def get_user_reminders(
        self, whatsapp_id: str, include_sent: bool = False,
    ) -> List[Reminder]:
        """Return reminders for a user (pending by default, or all)."""
        async with self._lock:
            def _do() -> List[Reminder]:
                with sqlite3.connect(self.path) as conn:
                    conn.row_factory = sqlite3.Row
                    base = (
                        "SELECT * FROM reminders WHERE whatsapp_id=? "
                        "ORDER BY trigger_iso ASC"
                    )
                    rows = conn.execute(base, (whatsapp_id,)).fetchall()
                    out = []
                    for row in rows:
                        r = Reminder(
                            id=row["id"], whatsapp_id=row["whatsapp_id"],
                            chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                            trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                            sent_at=row["sent_at"], cancelled=bool(row["cancelled"]),
                            failed=bool(row["failed"]),
                        )
                        if include_sent or r.status == "pending":
                            out.append(r)
                    return out
            return await asyncio.to_thread(_do)

    async def cancel_reminder(self, reminder_id: int) -> bool:
        """Mark a reminder as cancelled. Returns True if found."""
        async with self._lock:
            def _do() -> bool:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "UPDATE reminders SET cancelled=1 WHERE id=? AND sent_at IS NULL",
                        (reminder_id,),
                    )
                    conn.commit()
                    return cur.rowcount > 0
            return await asyncio.to_thread(_do)

    async def get_due_reminders(self) -> List[Reminder]:
        """Return all unfired, non-cancelled reminders whose trigger time <= now (UTC).

        IMPORTANT: Uses timezone-aware Python comparison, NOT SQL string comparison.
        SQL string comparison of ISO timestamps with different offsets is incorrect:
          '2026-03-09T13:00:00+05:30' > '2026-03-09T11:49:00+00:00'  ← string says NOT due
        but 13:00 IST = 07:30 UTC which IS before 11:49 UTC.
        """
        async with self._lock:
            def _do() -> List[Reminder]:
                now_utc = datetime.now(UTC)
                with sqlite3.connect(self.path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM reminders "
                        "WHERE sent_at IS NULL AND cancelled=0 AND failed=0 "
                        "ORDER BY trigger_iso ASC",
                    ).fetchall()
                    result = []
                    for row in rows:
                        try:
                            # Parse ISO with timezone offset (e.g. +05:30)
                            trigger_dt = datetime.fromisoformat(row["trigger_iso"])
                            if trigger_dt.tzinfo is None:
                                trigger_dt = trigger_dt.replace(tzinfo=UTC)
                            # Convert to UTC for correct comparison
                            trigger_utc = trigger_dt.astimezone(UTC)
                            if trigger_utc <= now_utc:
                                result.append(Reminder(
                                    id=row["id"], whatsapp_id=row["whatsapp_id"],
                                    chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                                    trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                                    sent_at=row["sent_at"],
                                ))
                        except (ValueError, TypeError):
                            # Bad trigger_iso — still return it so scheduler can mark failed
                            result.append(Reminder(
                                id=row["id"], whatsapp_id=row["whatsapp_id"],
                                chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                                trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                                sent_at=row["sent_at"],
                            ))
                    return result
            return await asyncio.to_thread(_do)

    async def mark_reminder_sent(self, reminder_id: int) -> None:
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "UPDATE reminders SET sent_at=? WHERE id=?", (now, reminder_id)
                    )
                    conn.commit()
            await asyncio.to_thread(_do)

    async def mark_reminder_failed(self, reminder_id: int) -> None:
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "UPDATE reminders SET failed=1 WHERE id=?", (reminder_id,)
                    )
                    conn.commit()
            await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self._model.encode(
            input, batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return emb.astype("float32").tolist()


@dataclass
class ContextSnippet:
    id:       str
    text:     str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class ChromaAmbient:
    def __init__(self, persist_dir, collection_name: str, embed_model: str):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=SentenceTransformerEmbedding(embed_model),
        )
        logger.info(
            "📚 chroma.ready  dir=%s  collection=%s",
            str(persist_dir), collection_name,
        )

    async def add_message(self, *, chat_id, whatsapp_id, direction, text, ts, message_id):
        if not (chat_id and (text or "").strip()):
            return
        doc_id = f"{chat_id}:{message_id}:{direction}"
        meta   = {"chat_id": chat_id, "whatsapp_id": whatsapp_id, "direction": direction, "ts": ts}
        await asyncio.to_thread(
            lambda: self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
        )

    async def search(self, *, chat_id, query, k):
        res = await asyncio.to_thread(
            lambda: self.collection.query(query_texts=[query], n_results=k, where={"chat_id": chat_id})
        )
        return [
            ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=dist)
            for _id, doc, meta, dist in zip(
                res.get("ids",[[]])[0], res.get("documents",[[]])[0],
                res.get("metadatas",[[]])[0], res.get("distances",[[None]*k])[0],
            )
        ]

    async def recent_window(self, *, chat_id, k):
        res = await asyncio.to_thread(
            lambda: self.collection.get(
                where={"chat_id": chat_id}, limit=max(50, k * 5),
                include=["documents", "metadatas"],
            )
        )
        items = [
            ContextSnippet(id=_id, text=doc, metadata=meta or {})
            for _id, doc, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            )
        ]
        items.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
        return items[:k]


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

sqlite_store: Optional[SQLiteMemory] = None
chroma_store: Optional[ChromaAmbient] = None


def init_stores() -> None:
    global sqlite_store, chroma_store
    sqlite_store = SQLiteMemory(settings.sqlite_path)
    if settings.chroma_enabled:
        chroma_store = ChromaAmbient(
            settings.chroma_dir, settings.chroma_collection, settings.chroma_embed_model,
        )
