"""
Database module with init_stores() function

Refactors:
- Fix Chroma 'where' filters using operator form {"$and":[...]} for compatibility.
- Make embedding function sync (Chroma expects sync callables) + thread-safe lock.
- Provide async embedding helper for app async flows.
- Add safe SQLite schema auto-migration for created_at / updated_at drift.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger("app.database")
UTC = timezone.utc

# Global instances
sqlite_store: Optional["SQLiteMemory"] = None
chroma_store: Optional["ChromaAmbient"] = None


class SQLiteMemory:
    """SQLite memory with anti-hallucination features"""

    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=3000")

            # Ensure base tables exist (forward-compatible schema)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id TEXT NOT NULL,
                    fact_key    TEXT NOT NULL,
                    fact_value  TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(whatsapp_id)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    whatsapp_id TEXT,
                    direction TEXT NOT NULL,
                    event_id TEXT,
                    text TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    UNIQUE(chat_id, event_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_log_chat_ts ON message_log(chat_id, ts DESC)")

            # --- Safe schema drift migration (no non-constant defaults) ---
            cols = [r[1] for r in conn.execute("PRAGMA table_info(user_memory)").fetchall()]
            if "created_at" not in cols:
                conn.execute("ALTER TABLE user_memory ADD COLUMN created_at TEXT")
                # Backfill to something sensible (use updated_at if present)
                try:
                    conn.execute("UPDATE user_memory SET created_at = updated_at WHERE created_at IS NULL")
                except Exception:
                    pass

            if "updated_at" not in cols:
                conn.execute("ALTER TABLE user_memory ADD COLUMN updated_at TEXT")
                try:
                    now = datetime.now(UTC).isoformat()
                    conn.execute("UPDATE user_memory SET updated_at = ? WHERE updated_at IS NULL", (now,))
                except Exception:
                    pass

            conn.commit()

        logger.info("🗄️ sqlite.ready path=%s", self.path)

    async def get_all_facts(self, whatsapp_id: str) -> Dict[str, str]:
        """Get all facts for a user"""
        async with self._lock:

            def _do() -> Dict[str, str]:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_key, fact_value FROM user_memory WHERE whatsapp_id=? ORDER BY updated_at DESC",
                        (whatsapp_id,),
                    )
                    return {k: v for (k, v) in cur.fetchall()}

            return await asyncio.to_thread(_do)

    async def upsert_fact(self, whatsapp_id: str, key: str, value: str) -> str:
        """Upsert a fact"""
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return "unchanged"

        async with self._lock:

            def _do() -> str:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_value FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                        (whatsapp_id, key),
                    )
                    row = cur.fetchone()
                    if row is None:
                        conn.execute(
                            "INSERT INTO user_memory (whatsapp_id, fact_key, fact_value, created_at, updated_at) "
                            "VALUES (?,?,?,?,?)",
                            (whatsapp_id, key, value, now, now),
                        )
                        conn.commit()
                        return "created"

                    if (row[0] or "").strip() == value:
                        return "unchanged"

                    conn.execute(
                        "UPDATE user_memory SET fact_value=?, updated_at=? WHERE whatsapp_id=? AND fact_key=?",
                        (value, now, whatsapp_id, key),
                    )
                    conn.commit()
                    return "updated"

            return await asyncio.to_thread(_do)

    async def log_message(
        self,
        *,
        chat_id: str,
        whatsapp_id: Optional[str],
        direction: str,
        text: str,
        ts: str,
        event_id: Optional[str] = None,
    ) -> None:
        """Log a message"""
        if not chat_id or not (text or "").strip():
            return

        async with self._lock:

            def _do() -> None:
                with sqlite3.connect(self.path) as conn:
                    try:
                        conn.execute(
                            "INSERT INTO message_log (chat_id, whatsapp_id, direction, event_id, text, ts) "
                            "VALUES (?,?,?,?,?,?)",
                            (chat_id, whatsapp_id or "", direction, event_id or None, text, ts),
                        )
                        conn.commit()
                    except sqlite3.IntegrityError:
                        pass

            await asyncio.to_thread(_do)


class SentenceTransformerEmbedding:
    """
    Thread-safe embedding function.

    Important:
    - Chroma expects embedding_function to be synchronous callable: __call__(List[str]) -> List[List[float]]
    - We also provide aembed() to use it safely from async contexts.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._lock = threading.Lock()

    def __call__(self, input: List[str]) -> List[List[float]]:
        with self._lock:
            emb = self._model.encode(
                input,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb.astype("float32").tolist()

    async def aembed(self, input: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self, input)


@dataclass
class ContextSnippet:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class ChromaAmbient:
    """ChromaDB with user isolation to prevent hallucinations"""

    def __init__(self, persist_dir: Path, collection_name: str, embed_model: str):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Single embedding instance used both by Chroma and by us.
        self.embed_fn = SentenceTransformerEmbedding(embed_model)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embed_fn,  # sync callable
        )

        logger.info("📚 chroma.ready dir=%s collection=%s", str(persist_dir), collection_name)

    @staticmethod
    def _and_where(*clauses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Chroma 'where' in a way compatible with versions that require exactly one
        top-level operator (e.g. {"$and":[...]}).
        """
        filtered = [c for c in clauses if c]  # remove empties
        if len(filtered) == 1:
            return filtered[0]
        return {"$and": filtered}

    def _build_where(
        self,
        *,
        chat_id: str,
        whatsapp_id: Optional[str] = None,
        direction: Optional[str] = None,
        is_ambient: Optional[bool] = None,
    ) -> Dict[str, Any]:
        clauses: List[Dict[str, Any]] = [{"chat_id": chat_id}]
        if whatsapp_id:
            clauses.append({"whatsapp_id": whatsapp_id})
        if direction:
            clauses.append({"direction": direction})
        if is_ambient is not None:
            clauses.append({"is_ambient": is_ambient})
        return self._and_where(*clauses)

    async def add_message(
        self,
        *,
        chat_id: str,
        whatsapp_id: str,
        direction: str,
        text: str,
        ts: str,
        message_id: str,
    ) -> None:
        """Add message with user attribution"""
        if not (chat_id and (text or "").strip() and whatsapp_id):
            return

        doc_id = f"{whatsapp_id}:{chat_id}:{message_id}:{direction}"
        meta = {
            "chat_id": chat_id,
            "whatsapp_id": whatsapp_id,
            "direction": direction,
            "ts": ts,
        }

        try:
            embeddings = await self.embed_fn.aembed([text])
            await asyncio.to_thread(
                lambda: self.collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                    embeddings=embeddings,
                )
            )
        except Exception as e:
            logger.warning("chroma.add_failed err=%s", str(e)[:200])

    async def search(
        self,
        *,
        chat_id: str,
        query: str,
        k: int,
        whatsapp_id: Optional[str] = None,
    ) -> List[ContextSnippet]:
        """Search with optional user filtering"""

        def _do() -> List[ContextSnippet]:
            where_clause = self._build_where(chat_id=chat_id, whatsapp_id=whatsapp_id)

            try:
                count = self.collection.count()
                actual_k = min(k, count) if count > 0 else 1
            except Exception:
                actual_k = k

            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=actual_k,
                    where=where_clause,
                )
            except Exception as e:
                logger.warning("chroma.search_failed err=%s", str(e)[:200])
                return []

            out: List[ContextSnippet] = []
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids)

            for did, doc, meta, dist in zip(ids, docs, metas, dists):
                out.append(ContextSnippet(id=did, text=doc, metadata=meta, distance=dist))
            return out

        return await asyncio.to_thread(_do)

    async def recent_window(
        self,
        *,
        chat_id: str,
        k: int,
        whatsapp_id: Optional[str] = None,
    ) -> List[ContextSnippet]:
        """Get recent inbound messages (direction=in)"""

        def _do() -> List[ContextSnippet]:
            where_clause = self._build_where(chat_id=chat_id, whatsapp_id=whatsapp_id, direction="in")

            try:
                res = self.collection.get(where=where_clause, include=["documents", "metadatas"])
            except Exception as e:
                logger.warning("chroma.recent_failed err=%s", str(e)[:200])
                return []

            items = []
            for did, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                items.append((did, doc, meta, (meta or {}).get("ts", "")))

            items.sort(key=lambda x: x[3], reverse=True)
            return [ContextSnippet(id=i[0], text=i[1], metadata=i[2]) for i in items[:k]]

        return await asyncio.to_thread(_do)


def init_stores() -> None:
    """Initialize database stores - REQUIRED by main.py"""
    global sqlite_store, chroma_store
    from .config import settings

    sqlite_store = SQLiteMemory(settings.sqlite_path)

    chroma_store = None
    if settings.chroma_enabled:
        chroma_store = ChromaAmbient(
            persist_dir=settings.chroma_dir,
            collection_name=getattr(settings, "chroma_collection", "shimmi_conversations"),
            embed_model=getattr(settings, "chroma_embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
