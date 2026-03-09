"""
reminder.py — Shimmi v2.8.0

Background asyncio task that checks the reminders table every N seconds
and sends a WhatsApp message when a reminder is due.

Design:
  - Runs as a single long-lived asyncio task, started in main.py lifespan.
  - Polls SQLite for reminders WHERE trigger_at <= now AND fired_at IS NULL.
  - Sends a formatted WhatsApp message to the stored chat_id.
  - Marks reminder as fired so it never fires again.
  - Errors on individual reminders are isolated — one failure doesn't abort others.
  - Gracefully handles WAHA being unavailable at startup.

Reminder trigger_at format:
  ISO 8601 string with timezone offset, e.g. "2026-03-09T06:00:00+05:30".
  The check compares trigger_at (as stored) against UTC now by normalising both.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("app.reminder")
UTC = timezone.utc


def _parse_iso(s: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 datetime string that may include a timezone offset.
    Returns UTC datetime or None if unparseable.
    Handles: "2026-03-09T06:00:00+05:30", "2026-03-09T06:00:00Z", "2026-03-09T00:30:00"
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        # Python 3.11+ handles +05:30 natively
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)  # assume UTC if no offset
        return dt.astimezone(UTC)
    except ValueError:
        pass

    # Manual fallback — handle +HH:MM offset
    m = re.match(
        r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)"
        r"([+-]\d{2}:\d{2}|Z)?$",
        s,
    )
    if not m:
        logger.warning("reminder.parse_failed  trigger_at=%r", s[:80])
        return None
    date_part, time_part, tz_part = m.groups()
    base = f"{date_part}T{time_part}"
    if len(time_part) == 5:  # HH:MM only
        base += ":00"
    try:
        dt = datetime.fromisoformat(base).replace(tzinfo=UTC)
        if tz_part and tz_part != "Z":
            sign   = 1 if tz_part[0] == "+" else -1
            h, mn  = map(int, tz_part[1:].split(":"))
            offset = timedelta(hours=h, minutes=mn) * sign
            dt     = (dt - offset).replace(tzinfo=UTC)  # convert to UTC
        return dt
    except Exception as exc:
        logger.warning("reminder.parse_error  trigger_at=%r  err=%s", s[:80], exc)
        return None


def _format_reminder_message(text: str, trigger_at_raw: str) -> str:
    """Format the WhatsApp message that gets sent when a reminder fires."""
    # Try to show the time in human-readable form
    dt = _parse_iso(trigger_at_raw)
    if dt:
        # Show UTC time — ideally we'd convert to user's tz but we store UTC
        # For IST users trigger_at is already in IST (stored as +05:30)
        # Re-parse the original string to get the local time display
        try:
            dt_local = datetime.fromisoformat(trigger_at_raw.strip())
            time_str = dt_local.strftime("%-I:%M %p")
        except Exception:
            time_str = dt.strftime("%H:%M UTC")
    else:
        time_str = ""

    lines = [f"⏰ *Reminder*"]
    if time_str:
        lines.append(f"_{time_str}_")
    lines.append("")
    lines.append(text)
    return "\n".join(lines)


async def run_reminder_loop(
    check_interval_sec: int = 60,
) -> None:
    """
    Long-running asyncio task.  Import database and waha_provider lazily
    to avoid circular imports and to handle late initialisation gracefully.
    """
    # Small initial delay so WAHA and DB are fully initialised before first check
    await asyncio.sleep(10)
    logger.info("⏰ reminder.loop_started  interval=%ds", check_interval_sec)

    while True:
        try:
            await _check_reminders()
        except asyncio.CancelledError:
            logger.info("⏰ reminder.loop_cancelled")
            return
        except Exception:
            logger.exception("reminder.loop_error")
        await asyncio.sleep(check_interval_sec)


async def _check_reminders() -> None:
    from . import database
    from .waha_provider import send_text

    if not database.sqlite_store:
        return

    now_utc = datetime.now(UTC).isoformat()
    due     = await database.sqlite_store.get_due_reminders(now_utc)

    if not due:
        return

    logger.info("⏰ reminder.check  due=%d  now=%s", len(due), now_utc[:19])

    for reminder in due:
        rid        = reminder["id"]
        chat_id    = reminder["chat_id"]
        text       = reminder["text"]
        trigger_at = reminder["trigger_at"]

        try:
            msg = _format_reminder_message(text, trigger_at)
            await send_text(chat_id, msg)
            await database.sqlite_store.mark_reminder_fired(rid)
            logger.info(
                "⏰ reminder.fired  id=%d  chat=%s  text=%r",
                rid, chat_id, text[:80],
            )
        except Exception as exc:
            logger.error(
                "⏰ reminder.send_failed  id=%d  chat=%s  err=%s",
                rid, chat_id, exc,
            )
            # Don't mark fired — will retry on next cycle
