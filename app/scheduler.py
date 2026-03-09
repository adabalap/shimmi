"""
scheduler.py — Shimmi v2.8.0

Background reminder scheduler.
Runs as an asyncio task every 60 seconds. Checks reminders table for entries
whose trigger_iso <= now (UTC), sends a WhatsApp message, and marks as sent.

Missed reminders (e.g. bot was down) are fired immediately on startup
if they are less than 2 hours overdue — silently dropped if older.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from .prompts import render, REMINDER_MESSAGE_TEMPLATE

logger = logging.getLogger("app.scheduler")
UTC = timezone.utc

# Reminders more than this many hours overdue will NOT be sent (stale).
_MAX_OVERDUE_HOURS = 2


class ReminderScheduler:
    def __init__(
        self,
        db,                     # SQLiteMemory instance
        send_fn: Callable,      # async fn(chat_id, text) → dict
        interval_sec: float = 60.0,
    ):
        self._db          = db
        self._send        = send_fn
        self._interval    = interval_sec
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="reminder_scheduler")
        logger.info("🕐 scheduler.start  interval=%.0fs", self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("🕐 scheduler.stop")

    async def _loop(self) -> None:
        # Initial delay so the app is fully ready before first check
        await asyncio.sleep(10)
        while True:
            try:
                await self._check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("scheduler.error")
            await asyncio.sleep(self._interval)

    async def _check(self) -> None:
        due = await self._db.get_due_reminders()
        if not due:
            return

        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_MAX_OVERDUE_HOURS)

        for r in due:
            try:
                # Parse trigger time
                trigger_dt = self._parse_iso(r.trigger_iso)
            except Exception:
                logger.warning(
                    "scheduler.bad_trigger  id=%d  trigger_iso=%r",
                    r.id, r.trigger_iso,
                )
                await self._db.mark_reminder_failed(r.id)
                continue

            # Too old — drop silently
            if trigger_dt < cutoff:
                logger.info(
                    "scheduler.skip_stale  id=%d  trigger=%s  (overdue by >%dh)",
                    r.id, r.trigger_iso, _MAX_OVERDUE_HOURS,
                )
                await self._db.mark_reminder_failed(r.id)
                continue

            # Send notification
            name_str = f", {r.user_name}" if r.user_name else ""
            msg = render(
                REMINDER_MESSAGE_TEMPLATE,
                text=r.reminder_text,
            )
            # Add friendly personalisation if we have the user's name
            if r.user_name:
                msg = msg + f"\n\n_Just a nudge from your memory, {r.user_name}_ 👋"

            try:
                await self._send(r.chat_id, msg)
                await self._db.mark_reminder_sent(r.id)
                logger.info(
                    "🔔 reminder.sent  id=%d  chat=%s  text=%.60s",
                    r.id, r.chat_id, r.reminder_text,
                )
            except Exception as exc:
                logger.error(
                    "scheduler.send_failed  id=%d  chat=%s  err=%s",
                    r.id, r.chat_id, exc,
                )
                # Don't mark failed — retry on next tick

    @staticmethod
    def _parse_iso(s: str) -> datetime:
        """Parse an ISO 8601 datetime string with timezone to a UTC datetime."""
        s = (s or "").strip()
        # Python 3.11+ supports full ISO, but we need 3.9+ compat
        # Handle "+05:30" style offsets which fromisoformat supports in 3.11
        # Fallback: strip offset and add it manually
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # Try replacing space with T
            dt = datetime.fromisoformat(s.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
