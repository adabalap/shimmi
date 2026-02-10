from __future__ import annotations

import asyncio, os, logging, time
from typing import List, Tuple, Optional
from datetime import datetime

import aiosqlite

from .clients_llm import groq_chat
from . import config

logger = logging.getLogger("app.workers")
DB_FILE = getattr(config, "DB_FILE", "bot_memory.db")

SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "0") == "1"
SUMMARY_INTERVAL_SEC = int(os.getenv("SUMMARY_INTERVAL_SEC", "3600"))
SUMMARY_WINDOW_MSGS = int(os.getenv("SUMMARY_WINDOW_MSGS", "12"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "160"))
SUMMARY_IDLE_ONLY = os.getenv("SUMMARY_IDLE_ONLY", "1") == "1"
SUMMARY_IDLE_THRESHOLD_SEC = int(os.getenv("SUMMARY_IDLE_THRESHOLD_SEC", "900"))


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def _get_distinct_chats() -> List[str]:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        async with db.execute("SELECT DISTINCT chat_id FROM messages") as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows if r and r[0]]


async def _get_recent_msgs(chat_id: str, limit: int) -> List[Tuple[str, str]]:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        async with db.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return list(reversed(rows or []))


async def _get_last_msg_ts(chat_id: str) -> Optional[float]:
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        async with db.execute("SELECT MAX(timestamp) FROM messages WHERE chat_id=?", (chat_id,)) as cur:
            r = await cur.fetchone()
        if r and r[0]:
            try:
                return datetime.fromisoformat(str(r[0])).timestamp()
            except Exception:
                return None
        return None


async def _ensure_summary_table():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        await db.execute(
            """CREATE TABLE IF NOT EXISTS chat_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        await db.commit()


async def _insert_summary(chat_id: str, summary: str):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("PRAGMA busy_timeout=3000")
        await db.execute(
            "INSERT INTO chat_summaries (chat_id, summary, created_at) VALUES (?, ?, ?)",
            (chat_id, summary, _now_iso()),
        )
        await db.commit()


async def _summarize_chat(chat_id: str) -> bool:
    msgs = await _get_recent_msgs(chat_id, SUMMARY_WINDOW_MSGS)
    if not msgs:
        return False
    content = " ".join(f"{r.upper()}: {c}" for r, c in msgs if c)
    sys = (
        "You summarize ONLY the conversation lines provided. Plain text. No markdown. "
        "Do not invent details that are not present."
    )
    reply, ok, _ = await groq_chat(chat_id, sys, content, temperature=0.0, max_tokens=SUMMARY_MAX_TOKENS)
    if not ok or not reply:
        return False
    await _insert_summary(chat_id, reply.strip())
    logger.info("summary.done chat_id=%s len=%s", chat_id, len(reply or ""))
    return True


async def summary_worker(stop_event: asyncio.Event):
    await _ensure_summary_table()
    logger.info("summary_worker.start interval_sec=%s idle_only=%s", SUMMARY_INTERVAL_SEC, SUMMARY_IDLE_ONLY)
    try:
        while not stop_event.is_set():
            chats = await _get_distinct_chats()
            now = time.time()
            for cid in chats[:50]:
                if SUMMARY_IDLE_ONLY:
                    last_ts = await _get_last_msg_ts(cid)
                    if last_ts is None or (now - last_ts) < SUMMARY_IDLE_THRESHOLD_SEC:
                        continue
                try:
                    await _summarize_chat(cid)
                except Exception as e:
                    logger.warning("summary_worker.error chat_id=%s err=%s", cid, str(e)[:160])
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=SUMMARY_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
    except asyncio.CancelledError:
        pass
    logger.info("summary_worker.stop")


def start_workers() -> List[asyncio.Task]:
    tasks: List[asyncio.Task] = []
    if SUMMARY_ENABLED:
        evt = asyncio.Event()
        t = asyncio.create_task(summary_worker(evt))
        t._stop_evt = evt
        tasks.append(t)
    return tasks


async def stop_workers(tasks: List[asyncio.Task]):
    for t in tasks:
        try:
            evt = getattr(t, "_stop_evt", None)
            if evt:
                evt.set()
            t.cancel()
        except Exception:
            pass
