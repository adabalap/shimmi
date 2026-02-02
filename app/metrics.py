
# app/metrics.py
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
        await db.execute("""
        CREATE TABLE IF NOT EXISTS usage_ledger (
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
        await db.commit()

async def record_usage(
    provider: str,
    model: str,
    chat_id: Optional[str],
    sender_id: Optional[str],
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    cost: Optional[float] = None
):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
          INSERT INTO usage_ledger
          (ts, provider, model, chat_id, sender_id, tokens_prompt, tokens_completion, cost)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (_now_iso(), provider, model, chat_id, sender_id, tokens_prompt, tokens_completion, cost))
        await db.commit()
    logger.info("📈 usage.record provider=%s model=%s tp=%s tc=%s", provider, model, tokens_prompt, tokens_completion)

async def usage_summary_last(days: int = 7) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_FILE) as db:
        rows = await db.execute_fetchall(f"""
          SELECT substr(ts,1,10) AS day,
                 provider, model,
                 COUNT(*) AS calls,
                 COALESCE(SUM(tokens_prompt),0) AS tp,
                 COALESCE(SUM(tokens_completion),0) AS tc,
                 COALESCE(SUM(cost),0.0) AS cost
            FROM usage_ledger
           WHERE ts >= date('now','-{days} day')
           GROUP BY day, provider, model
           ORDER BY day DESC, provider, model
        """)
        out = []
        for r in rows:
            out.append({
                "day": r[0], "provider": r[1], "model": r[2],
                "calls": r[3], "tokens_prompt": r[4], "tokens_completion": r[5],
                "cost": float(r[6]) if r[6] is not None else 0.0
            })
        return out

