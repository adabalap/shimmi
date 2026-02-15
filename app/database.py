from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger("app.database")
UTC = timezone.utc


class SQLiteMemory:
    def __init__(self, path):
        self.path = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=3000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
                """
            )
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
            conn.commit()
        logger.info("🗄️ sqlite.ready path=%s", self.path)

    async def get_all_facts(self, whatsapp_id: str) -> Dict[str, str]:
        async with self._lock:
            def _do() -> Dict[str, str]:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute("SELECT fact_key, fact_value FROM user_memory WHERE whatsapp_id=?", (whatsapp_id,))
                    return {k: v for (k, v) in cur.fetchall()}
            return await asyncio.to_thread(_do)

    async def upsert_fact(self, whatsapp_id: str, key: str, value: str) -> str:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return "unchanged"
        async with self._lock:
            def _do() -> str:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute("SELECT fact_value FROM user_memory WHERE whatsapp_id=? AND fact_key=?", (whatsapp_id, key))
                    row = cur.fetchone()
                    if row is None:
                        conn.execute("INSERT INTO user_memory (whatsapp_id, fact_key, fact_value, updated_at) VALUES (?,?,?,?)", (whatsapp_id, key, value, now))
                        conn.commit()
                        return "created"
                    if (row[0] or "").strip() == value:
                        return "unchanged"
                    conn.execute("UPDATE user_memory SET fact_value=?, updated_at=? WHERE whatsapp_id=? AND fact_key=?", (value, now, whatsapp_id, key))
                    conn.commit()
                    return "updated"
            return await asyncio.to_thread(_do)

    async def log_message(self, *, chat_id: str, whatsapp_id: Optional[str], direction: str, text: str, ts: str, event_id: Optional[str] = None) -> None:
        if not chat_id or not (text or "").strip():
            return
        async with self._lock:
            def _do() -> None:
                with sqlite3.connect(self.path) as conn:
                    try:
                        conn.execute("INSERT INTO message_log (chat_id, whatsapp_id, direction, event_id, text, ts) VALUES (?,?,?,?,?,?)", (chat_id, whatsapp_id or "", direction, event_id or None, text, ts))
                        conn.commit()
                    except sqlite3.IntegrityError:
                        pass
            await asyncio.to_thread(_do)


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self._model.encode(input, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype("float32").tolist()


@dataclass
class ContextSnippet:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class ChromaAmbient:
    def __init__(self, persist_dir, collection_name: str, embed_model: str):
        self.client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=SentenceTransformerEmbedding(embed_model),
        )
        logger.info("📚 chroma.ready dir=%s collection=%s", str(persist_dir), collection_name)

    async def add_message(self, *, chat_id: str, whatsapp_id: str, direction: str, text: str, ts: str, message_id: str) -> None:
        if not (chat_id and (text or "").strip()):
            return
        doc_id = f"{chat_id}:{message_id}:{direction}"
        meta = {"chat_id": chat_id, "whatsapp_id": whatsapp_id, "direction": direction, "ts": ts}
        await asyncio.to_thread(lambda: self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta]))

    #async def search(self, *, chat_id: str, query: str, k: int) -> List[ContextSnippet]:
    #    res = await asyncio.to_thread(lambda: self.collection.query(query_texts=[query], n_results=k, where={"chat_id": chat_id}))
    #    out: List[ContextSnippet] = []
    #    ids = res.get("ids", [[]])[0]
    #    docs = res.get("documents", [[]])[0]
    #    metas = res.get("metadatas", [[]])[0]
    #    dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids)
    #    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
    #        out.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=dist))
    #    return out

    #async def recent_window(self, *, chat_id: str, k: int) -> List[ContextSnippet]:
    #    res = await asyncio.to_thread(lambda: self.collection.get(where={"chat_id": chat_id}, limit=max(50, k * 5), include=["documents", "metadatas"]))
    #    items: List[ContextSnippet] = []
    #    for _id, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
    #        items.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=None))
    #    items.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
    #    return items[:k]
    async def search(self, *, chat_id: str, query: str, k: int) -> List[ContextSnippet]:
        """Search with dynamic k to prevent warnings"""

        # Get actual count first
        try:
            count = await asyncio.to_thread(
                lambda: self.collection.count(where={"chat_id": chat_id})
            )
        except Exception:
            count = 0

        # Adjust k to actual available items
        actual_k = min(k, max(count, 1))

        if actual_k == 0:
            return []

        res = await asyncio.to_thread(
            lambda: self.collection.query(
                query_texts=[query],
                n_results=actual_k,
                where={"chat_id": chat_id}
            )
        )

        out: List[ContextSnippet] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(ids)

        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=dist))

        return out

    async def recent_window(self, *, chat_id: str, k: int) -> List[ContextSnippet]:
        """Get recent messages with dynamic sizing"""

        # Get count first
        try:
            count = await asyncio.to_thread(
                lambda: self.collection.count(where={"chat_id": chat_id})
            )
        except Exception:
            count = 0

        if count == 0:
            return []

        # Fetch more than needed for sorting
        fetch_limit = max(min(count, 50), k * 5)

        res = await asyncio.to_thread(
            lambda: self.collection.get(
                where={"chat_id": chat_id},
                limit=fetch_limit,
                include=["documents", "metadatas"]
            )
        )

        items: List[ContextSnippet] = []
        for _id, doc, meta in zip(
            res.get("ids", []),
            res.get("documents", []),
            res.get("metadatas", [])
        ):
            items.append(ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=None))

        items.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
        return items[:k]

sqlite_store: Optional[SQLiteMemory] = None
chroma_store: Optional[ChromaAmbient] = None


def init_stores() -> None:
    global sqlite_store, chroma_store
    sqlite_store = SQLiteMemory(settings.sqlite_path)
    chroma_store = ChromaAmbient(settings.chroma_dir, settings.chroma_collection, settings.chroma_embed_model) if settings.chroma_enabled else None
