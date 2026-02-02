
# app/workers.py
from __future__ import annotations
import asyncio, os, logging, time
from typing import List, Tuple, Optional
from datetime import datetime, timezone

import aiosqlite

from . import config
from .clients_llm import groq_chat  # you can switch to Gemini in this function if you prefer

logger = logging.getLogger("app.workers")

DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")

# ---------- Config (read once) ----------
SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "1") == "1"
SUMMARY_INTERVAL_SEC = int(os.getenv("SUMMARY_INTERVAL_SEC", "3600"))
SUMMARY_WINDOW_MSGS = int(os.getenv("SUMMARY_WINDOW_MSGS", "10"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "120"))
SUMMARY_IDLE_ONLY = os.getenv("SUMMARY_IDLE_ONLY", "1") == "1"
SUMMARY_IDLE_THRESHOLD_SEC = int(os.getenv("SUMMARY_IDLE_THRESHOLD_SEC", "900"))

SALIENCE_ENABLED = os.getenv("SALIENCE_ENABLED", "0") == "1"
SALIENCE_INTERVAL_SEC = int(os.getenv("SALIENCE_INTERVAL_SEC", "1800"))
SALIENCE_DECAY_PER_DAY = float(os.getenv("SALIENCE_DECAY_PER_DAY", "0.02"))  # 2%/day
SALIENCE_CONF_MIN = float(os.getenv("SALIENCE_CONF_MIN", "0.30"))
SALIENCE_CONF_MAX = float(os.getenv("SALIENCE_CONF_MAX", "0.98"))

# ---------- Util ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def _get_distinct_chats() -> List[str]:
    async with aiosqlite.connect(DB_FILE) as db:
        rows = await db.execute_fetchall("SELECT DISTINCT chat_id FROM messages")
        return [r[0] for r in rows if r and r[0]]

async def _get_recent_msgs(chat_id: str, limit: int) -> List[Tuple[str, str]]:
    async with aiosqlite.connect(DB_FILE) as db:
        rows = await db.execute_fetchall(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit),
        )
        # newest first; reverse for temporal order
        return list(reversed(rows or []))

async def _get_last_msg_ts(chat_id: str) -> Optional[float]:
    async with aiosqlite.connect(DB_FILE) as db:
        r = await db.execute_fetchone(
            "SELECT MAX(timestamp) FROM messages WHERE chat_id=?", (chat_id,)
        )
        if r and r[0]:
            return float(r[0])
        return None

async def _ensure_summary_table():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        await db.commit()

async def _insert_summary(chat_id: str, summary: str):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO chat_summaries (chat_id, summary, created_at) VALUES (?, ?, ?)",
            (chat_id, summary, _now_iso()),
        )
        await db.commit()

# ---------- Summarizer Worker ----------
async def _summarize_chat(chat_id: str) -> bool:
    msgs = await _get_recent_msgs(chat_id, SUMMARY_WINDOW_MSGS)
    if not msgs:
        return False
    content = "\n".join(f"{r.upper()}: {c}" for r, c in msgs if c)
    sys = ("You are a conversation summarizer. Produce a succinct micro-summary "
           f"(<= {SUMMARY_MAX_TOKENS} tokens). Use plain text, no markdown.")
    reply, ok, _ = await groq_chat(chat_id, sys, content)
    if not ok or not reply:
        return False
    await _insert_summary(chat_id, reply.strip())
    logger.info("🧵 summary.done chat_id=%s len=%s", chat_id, len(reply or ""))
    return True

async def summary_worker(stop_event: asyncio.Event):
    await _ensure_summary_table()
    logger.info("🧵 summary_worker.start interval_sec=%s idle_only=%s", SUMMARY_INTERVAL_SEC, SUMMARY_IDLE_ONLY)
    try:
        while not stop_event.is_set():
            chats = await _get_distinct_chats()
            now = time.time()
            for cid in chats:
                if SUMMARY_IDLE_ONLY:
                    last_ts = await _get_last_msg_ts(cid)
                    if last_ts is None or (now - last_ts) < SUMMARY_IDLE_THRESHOLD_SEC:
                        continue
                try:
                    await _summarize_chat(cid)
                except Exception as e:
                    logger.warning("🧵 summary_worker.error chat_id=%s err=%s", cid, e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=SUMMARY_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        pass
    logger.info("🧵 summary_worker.stop")

# ---------- Salience Worker ----------
async def salience_worker(stop_event: asyncio.Event):
    logger.info("🧵 salience_worker.start interval_sec=%s decay_per_day=%s", SALIENCE_INTERVAL_SEC, SALIENCE_DECAY_PER_DAY)
    try:
        while not stop_event.is_set():
            factor = SALIENCE_DECAY_PER_DAY * (SALIENCE_INTERVAL_SEC / 86400.0)
            decay = max(0.0, min(factor, 0.5))  # cap
            async with aiosqlite.connect(DB_FILE) as db:
                # Multiplicative decay with clamping
                await db.execute(f"""
                    UPDATE user_facts
                       SET confidence = CASE
                           WHEN confidence IS NULL THEN NULL
                           ELSE MAX({SALIENCE_CONF_MIN}, MIN({SALIENCE_CONF_MAX}, confidence * (1.0 - ?)))
                       END,
                           updated_at = ?
                """, (decay, _now_iso()))
                await db.commit()
            logger.info("🧵 salience.decay batch_factor=%s", round(decay, 6))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=SALIENCE_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        pass
    logger.info("🧵 salience_worker.stop")

# ---------- Entrypoints called from main ----------
def start_workers() -> List[asyncio.Task]:
    tasks: List[asyncio.Task] = []
    if SUMMARY_ENABLED:
        s_stop = asyncio.Event()
        t = asyncio.create_task(summary_worker(s_stop))
        t._stop_evt = s_stop  # stash for shutdown
        tasks.append(t)
    if SALIENCE_ENABLED:
        sl_stop = asyncio.Event()
        t = asyncio.create_task(salience_worker(sl_stop))
        t._stop_evt = sl_stop
        tasks.append(t)
    return tasks

async def stop_workers(tasks: List[asyncio.Task]):
    for t in tasks:
        try:
            evt = getattr(t, "_stop_evt", None)
            if evt: evt.set()
            t.cancel()
        except Exception:
            pass

