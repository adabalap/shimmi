
# app/diag.py
from __future__ import annotations
import aiosqlite
from fastapi import APIRouter
from typing import Dict, Any
from .memory import build_profile_snapshot_text
from . import config

DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")
router = APIRouter(prefix="/diag", tags=["diag"])

@router.get("/rag")
async def rag_overview() -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_chat": [], "totals": {}, "peek": []}
    async with aiosqlite.connect(DB_FILE) as db:
        rows = await db.execute("SELECT chat_id, COUNT(*) FROM rag_vecs GROUP BY chat_id ORDER BY COUNT(*) DESC")
        per_chat = await rows.fetchall()
        rows2 = await db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM rag_vecs")
        tot = await rows2.fetchone()
        out["per_chat"] = [{"chat_id": r[0], "vecs": r[1]} for r in per_chat]
        out["totals"] = {"count": (tot[0] if tot else 0), "min_id": (tot[1] if tot else None), "max_id": (tot[2] if tot else None)}
        if per_chat:
            top_chat = per_chat[0][0]
            rows3 = await db.execute(
                "SELECT c.id, c.text, c.created_at FROM rag_chunks c JOIN rag_vecs v ON v.chunk_id=c.id "
                "WHERE v.chat_id=? ORDER BY v.id DESC LIMIT 5", (top_chat,)
            )
            out["peek"] = [{"chunk_id": r[0], "text": (r[1] or "")[:200], "created_at": r[2]} for r in await rows3.fetchall()]
    return out

@router.get("/profile")
async def profile_snapshot(sender_id: str) -> Dict[str, Any]:
    txt = await build_profile_snapshot_text(sender_id)
    return {"sender_id": sender_id, "profile_snapshot": txt or ""}

