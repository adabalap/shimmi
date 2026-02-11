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


@router.get("/profile")
async def profile_snapshot(sender_id: str, x_diag_key: str | None = Header(default=None)) -> Dict[str, Any]:
    _auth(x_diag_key)
    txt = await build_profile_snapshot_text(sender_id)
    return {"sender_id": sender_id, "profile_snapshot": txt or ""}
