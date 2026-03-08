"""
database.py — Shimmi v2.7.0

Critical fixes vs v2.5.0 (production):
  ① INSERT now includes both created_at AND updated_at.
     The live DB has `created_at TEXT NOT NULL` — the v2.5 INSERT only passed
     updated_at, causing `IntegrityError: NOT NULL constraint failed:
     user_memory.created_at` on every new fact write.

  ② _migrate() back-fills created_at on all existing rows without data loss.
     Runs at every startup — fully idempotent.

  ③ normalize_key() eliminates LLM key inconsistencies:
     user_name→name, user_city→city, user_favorite_drink→favorite_drink, etc.
     Applied on both read and write so duplicates are collapsed automatically.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from dataclasses import dataclass
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
    # name variants
    "user_name":          "name",
    "username":           "name",
    "first_name":         "name",
    "full_name":          "name",
    # city / location
    "user_city":          "city",
    "user_location":      "city",
    "location":           "city",
    "hometown":           "city",
    # country
    "user_country":       "country",
    # drink
    "user_favorite_drink":"favorite_drink",
    "preferred_drink":    "favorite_drink",
    "user_drink":         "favorite_drink",
    "drink":              "favorite_drink",
    # interests / hobbies
    "user_interests":     "interests",
    "user_interest":      "interests",
    "user_hobby":         "hobbies",
    "user_hobbies":       "hobbies",
    # occupation
    "user_occupation":    "occupation",
    "user_job":           "occupation",
    "job":                "occupation",
    # age
    "user_age":           "age",
    # language
    "user_language":      "preferred_language",
    "language":           "preferred_language",
}


def normalize_key(raw: str) -> str:
    """
    Produce a canonical fact key.

    Steps (in order):
      1. Strip + lowercase
      2. spaces/hyphens → underscore
      3. remove non-alphanumeric/underscore chars
      4. collapse repeated underscores; strip leading/trailing
      5. strip leading 'user_' prefix (common LLM artefact)
      6. apply explicit alias map
    """
    k = (raw or "").strip().lower()
    k = re.sub(r"[\s\-]+", "_", k)
    k = re.sub(r"[^\w]", "", k)
    k = re.sub(r"_+", "_", k)
    # Strip user_ prefix BEFORE stripping edge underscores
    if k.startswith("user_"):
        k = k[5:]
    k = k.strip("_")
    if not k or k == "user":
        return ""
    return _KEY_ALIASES.get(k, k)


# ---------------------------------------------------------------------------
# SQLite memory store
# ---------------------------------------------------------------------------

class SQLiteMemory:
    def __init__(self, path):
        self.path  = str(path)
        self._lock = asyncio.Lock()
        self._init()

    # ── schema + migration ──────────────────────────────────────────────────

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id  TEXT NOT NULL,
                    fact_key     TEXT NOT NULL,
                    fact_value   TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
                """
            )
            self._migrate(conn)

            conn.execute(
                """
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
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_log_chat_ts "
                "ON message_log(chat_id, ts DESC)"
            )
            conn.commit()

        logger.info("🗄️  sqlite.ready  path=%s", self.path)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Non-destructive, idempotent schema migrations."""
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_memory)").fetchall()
        }

        if "created_at" not in existing:
            logger.info("🗄️  migrate — adding created_at to user_memory")
            conn.execute("ALTER TABLE user_memory ADD COLUMN created_at TEXT")
            # Back-fill with updated_at so existing rows aren't NULL
            conn.execute(
                "UPDATE user_memory SET created_at = updated_at "
                "WHERE created_at IS NULL"
            )
            conn.commit()
            logger.info("🗄️  migrate — created_at back-filled ✓")

        if "updated_at" not in existing:
            logger.info("🗄️  migrate — adding updated_at to user_memory")
            now = datetime.now(UTC).isoformat()
            conn.execute(
                f"ALTER TABLE user_memory ADD COLUMN updated_at TEXT DEFAULT '{now}'"
            )
            conn.commit()

    # ── public API ──────────────────────────────────────────────────────────

    async def get_all_facts(self, whatsapp_id: str) -> Dict[str, str]:
        async with self._lock:
            def _do() -> Dict[str, str]:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_key, fact_value FROM user_memory "
                        "WHERE whatsapp_id=? ORDER BY updated_at DESC",
                        (whatsapp_id,),
                    )
                    # Normalise keys on read — first (most-recent) wins per canonical key
                    out: Dict[str, str] = {}
                    for raw_key, val in cur.fetchall():
                        canon = normalize_key(raw_key)
                        if canon and canon not in out:
                            out[canon] = val
                    return out
            return await asyncio.to_thread(_do)

    async def upsert_fact(self, whatsapp_id: str, raw_key: str, value: str) -> str:
        """
        Insert or update a single user fact.
        Always writes both created_at (on INSERT) and updated_at.
        Returns: "created" | "updated" | "unchanged"
        """
        key   = normalize_key(raw_key)
        value = (value or "").strip()
        if not key or not value:
            return "unchanged"

        async with self._lock:
            def _do() -> str:
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
        self,
        *,
        chat_id:     str,
        whatsapp_id: Optional[str],
        direction:   str,
        text:        str,
        ts:          str,
        event_id:    Optional[str] = None,
    ) -> None:
        if not chat_id or not (text or "").strip():
            return
        async with self._lock:
            def _do() -> None:
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
                        pass  # duplicate event_id — silently skip
            await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# ChromaDB ambient context store
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self._model.encode(
            input,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
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

    async def add_message(
        self, *, chat_id: str, whatsapp_id: str, direction: str,
        text: str, ts: str, message_id: str,
    ) -> None:
        if not (chat_id and (text or "").strip()):
            return
        doc_id = f"{chat_id}:{message_id}:{direction}"
        meta   = {"chat_id": chat_id, "whatsapp_id": whatsapp_id, "direction": direction, "ts": ts}
        await asyncio.to_thread(
            lambda: self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
        )

    async def search(self, *, chat_id: str, query: str, k: int) -> List[ContextSnippet]:
        res = await asyncio.to_thread(
            lambda: self.collection.query(query_texts=[query], n_results=k, where={"chat_id": chat_id})
        )
        out: List[ContextSnippet] = []
        ids   = res.get("ids",       [[]])[0]
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[None] * len(ids)])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=dist))
        return out

    async def recent_window(self, *, chat_id: str, k: int) -> List[ContextSnippet]:
        res = await asyncio.to_thread(
            lambda: self.collection.get(
                where={"chat_id": chat_id},
                limit=max(50, k * 5),
                include=["documents", "metadatas"],
            )
        )
        items: List[ContextSnippet] = []
        for _id, doc, meta in zip(
            res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
        ):
            items.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=None))
        items.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
        return items[:k]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

sqlite_store: Optional[SQLiteMemory] = None
chroma_store:  Optional[ChromaAmbient] = None


def init_stores() -> None:
    global sqlite_store, chroma_store
    sqlite_store = SQLiteMemory(settings.sqlite_path)
    if settings.chroma_enabled:
        chroma_store = ChromaAmbient(
            settings.chroma_dir, settings.chroma_collection, settings.chroma_embed_model,
        )
