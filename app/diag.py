from __future__ import annotations

import os
import aiosqlite
from fastapi import APIRouter, Header, HTTPException
from typing import Dict, Any
from .memory import build_profile_snapshot_text
from . import config

DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")
router = APIRouter(prefix="/diag", tags=["diag"])

DIAG_ENABLED = os.getenv("DIAG_ENABLED", "0").lower() in ("1","true","yes","on")
DIAG_API_KEY = os.getenv("DIAG_API_KEY", "").strip()


def _auth(x_diag_key: str | None):
    if not DIAG_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if DIAG_API_KEY and (x_diag_key or "") != DIAG_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.get("/rag")
async def rag_overview(x_diag_key: str | None = Header(default=None)) -> Dict[str, Any]:
    _auth(x_diag_key)
    out: Dict[str, Any] = {"per_chat": [], "totals": {}}
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        async with db.execute("SELECT chat_id, COUNT(*) FROM rag_vecs GROUP BY chat_id ORDER BY COUNT(*) DESC LIMIT 50") as cur:
            per_chat = await cur.fetchall()
        async with db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM rag_vecs") as cur:
            tot = await cur.fetchone()
    out["per_chat"] = [{"chat_id": r[0], "vecs": r[1]} for r in (per_chat or [])]
    out["totals"] = {"count": (tot[0] if tot else 0), "min_id": (tot[1] if tot else None), "max_id": (tot[2] if tot else None)}
    return out


@router.get("/profile")
async def profile_snapshot(sender_id: str, x_diag_key: str | None = Header(default=None)) -> Dict[str, Any]:
    _auth(x_diag_key)
    txt = await build_profile_snapshot_text(sender_id)
    return {"sender_id": sender_id, "profile_snapshot": txt or ""}
