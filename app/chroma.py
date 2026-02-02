
# app/chroma.py
from __future__ import annotations
import os, logging, time
from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
import aiosqlite
from . import config

logger = logging.getLogger("app.chroma")

DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")
EMBED_MODEL_NAME = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_ROWS = int(os.getenv("CHROMA_MAX_ROWS", "600"))
SNIPPET_CHARS = int(os.getenv("CHROMA_SNIPPET_CHARS", "220"))
NORMALIZE = True

from sentence_transformers import SentenceTransformer
import faiss

_model: SentenceTransformer | None = None
_dim: int = 0

def _load_model():
    global _model, _dim
    if _model is None:
        t0 = time.perf_counter()
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        _dim = _model.get_sentence_embedding_dimension()
        dt = int((time.perf_counter() - t0) * 1000)
        logger.info("model_loaded name=%s dim=%s load_ms=%s", EMBED_MODEL_NAME, _dim, dt)

INIT_SQL = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    sender_id TEXT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_chat ON rag_chunks(chat_id, id DESC);

CREATE TABLE IF NOT EXISTS rag_vecs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL,
    chat_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES rag_chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rag_vecs_chat ON rag_vecs(chat_id, id DESC);
"""

_init_done = False

async def _ensure_init():
    global _init_done
    if _init_done: return
    _load_model()
    async with aiosqlite.connect(DB_FILE) as db:
        for stmt in [s.strip() for s in INIT_SQL.split(";") if s.strip()]:
            await db.execute(stmt)
        await db.commit()
    _init_done = True

def _embed(texts: List[str]) -> np.ndarray:
    emb = _model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    return emb

def _to_blob(vec: np.ndarray) -> bytes:
    assert vec.dtype == np.float32 and vec.ndim == 1
    return vec.tobytes(order="C")

def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr if arr.size == dim else np.zeros(dim, dtype=np.float32)

def _cosine_index(dim: int):
    return faiss.IndexFlatIP(dim)  # cosine on normalized vecs

def _snippet(s: str) -> str:
    txt = (s or "").replace("\n", " ").strip()
    if len(txt) > SNIPPET_CHARS:
        txt = txt[:SNIPPET_CHARS] + "…"
    return f"• {txt}"

async def add_text(chat_id: str, sender_id: str, text: str):
    if not (chat_id and (text or "").strip()):
        return
    await _ensure_init()
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_FILE) as db:
        cur = await db.execute(
            "INSERT INTO rag_chunks (chat_id, sender_id, text, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, sender_id or "", text, now),
        )
        await db.commit()
        chunk_id = cur.lastrowid

    t0 = time.perf_counter()
    vec = _embed([text])[0]
    dt = int((time.perf_counter() - t0) * 1000)

    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO rag_vecs (chunk_id, chat_id, dim, vec) VALUES (?, ?, ?, ?)",
            (chunk_id, chat_id, _dim, _to_blob(vec)),
        )
        await db.commit()

    logger.info("📚 chroma_add id=%s sender_id=%s chat_id=%s embed_ms=%s", chunk_id, sender_id, chat_id, dt)

async def add_profile_snapshot(chat_id: str, sender_id: str, text: str):
    if not text: return
    await add_text(chat_id, sender_id, f"PROFILE_FACTS:\n{text}")

async def query(chat_id: str, text: str, k: int = 3, *, since_iso: Optional[str] = None, until_iso: Optional[str] = None) -> str:
    await _ensure_init()
    if not (chat_id and (text or "").strip()):
        logger.info("📚 chroma_query k=%s (empty)", k)
        return ""

    where = "WHERE v.chat_id=?"
    args = [chat_id]
    if since_iso and until_iso:
        where += " AND c.created_at BETWEEN ? AND ?"
        args.extend([since_iso, until_iso])

    async with aiosqlite.connect(DB_FILE) as db:
        rows = await db.execute(
            f"SELECT v.id, v.chunk_id, v.dim, v.vec, c.text "
            f"FROM rag_vecs v JOIN rag_chunks c ON c.id=v.chunk_id "
            f"{where} ORDER BY v.id DESC LIMIT ?",
            (*args, MAX_ROWS),
        )
        data = await rows.fetchall()

    if not data:
        logger.info("📚 chroma_query k=%s (no_data)", k)
        return ""

    dim0 = data[0][2]
    X = np.stack([_from_blob(row[3], dim0) for row in data], axis=0)
    texts = [row[4] for row in data]
    n = X.shape[0]

    index = _cosine_index(dim0); index.add(X)
    t0 = time.perf_counter()
    q = _embed([text])
    dt = int((time.perf_counter() - t0) * 1000)

    kk = max(1, min(k, n))
    D, I = index.search(q, kk)
    logger.info("📚 chroma_query k=%s rows=%s embed_ms=%s", k, n, dt)

    out: List[str] = []; seen = set()
    for idx in I[0]:
        if idx < 0 or idx >= n: continue
        t = texts[idx]
        if t in seen: continue
        seen.add(t)
        out.append(_snippet(t))

    return "\n".join(out)

# NEW: pre-warm on startup
async def warmup() -> int:
    await _ensure_init()
    return _dim

