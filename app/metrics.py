from __future__ import annotations

import aiosqlite
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging
from . import config

logger = logging.getLogger("app.metrics")
DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def ensure_metrics_tables():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        await db.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          provider TEXT,
          model TEXT,
          chat_id TEXT,
          sender_id TEXT,
          tokens_prompt INTEGER,
          tokens_completion INTEGER,
          cost REAL
        )""")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_ts ON usage_events(ts)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_pm_ts ON usage_events(provider, model, ts)")
        await db.commit()


async def record_usage(provider: str, model: str, chat_id: Optional[str], sender_id: Optional[str],
                       tokens_prompt: Optional[int] = None, tokens_completion: Optional[int] = None,
                       cost: Optional[float] = None):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        await db.execute(
            """INSERT INTO usage_events (ts, provider, model, chat_id, sender_id, tokens_prompt, tokens_completion, cost)
               VALUES (?,?,?,?,?,?,?,?)""",
            (_now_iso(), provider, model, chat_id, sender_id, tokens_prompt, tokens_completion, cost),
        )
        await db.commit()
    logger.debug("usage.record provider=%s model=%s", provider, model)


async def usage_summary_last(days: int = 7) -> List[Dict[str, Any]]:
    days = max(1, min(int(days), 365))
    modifier = f"-{days} day"
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        async with db.execute(
            """SELECT substr(ts,1,10) AS day, provider, model,
                      COUNT(*) AS calls,
                      COALESCE(SUM(tokens_prompt),0) AS tp,
                      COALESCE(SUM(tokens_completion),0) AS tc,
                      COALESCE(SUM(cost),0.0) AS cost
                 FROM usage_events
                WHERE substr(ts,1,10) >= date('now', ?)
                GROUP BY day, provider, model
                ORDER BY day DESC, provider, model""",
            (modifier,),
        ) as cur:
            rows = await cur.fetchall()

    return [
        {
            "day": r[0], "provider": r[1], "model": r[2],
            "calls": r[3], "tokens_prompt": r[4], "tokens_completion": r[5],
            "cost": float(r[6]) if r[6] is not None else 0.0,
        }
        for r in rows
    ]
