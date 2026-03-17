"""
scheduler.py — Shimmi v2.8.0

Background reminder scheduler.
Runs as an asyncio task every 60 seconds. Checks the reminders table for entries
whose trigger_iso <= now (UTC), sends a WhatsApp message, and marks as sent.

Missed reminders (e.g. bot was down) are fired if <= 2 hours overdue — silently
dropped if older (a user doesn't need an alarm from 3 hours ago).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("app.scheduler")
UTC = timezone.utc

_MAX_OVERDUE_HOURS = 2


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO 8601 string (with or without tz offset) → UTC datetime."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        pass
    # Manual fallback for Python < 3.11 with "+05:30" offsets
    m = re.match(
        r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)([+-]\d{2}:\d{2}|Z)?$", s
    )
    if not m:
        return None
    date_p, time_p, tz_p = m.groups()
    base = f"{date_p}T{time_p}" + (":00" if len(time_p) == 5 else "")
    try:
        dt = datetime.fromisoformat(base).replace(tzinfo=UTC)
        if tz_p and tz_p != "Z":
            sign = 1 if tz_p[0] == "+" else -1
            h, mn = map(int, tz_p[1:].split(":"))
            offset = timedelta(hours=h, minutes=mn) * sign
            dt = (dt - offset).replace(tzinfo=UTC)
        return dt
    except Exception:
        return None


def _friendly_time(trigger_iso: str) -> str:
    """Return e.g. '6:00 AM' from '2026-03-09T06:00+05:30'."""
    dt = _parse_iso(trigger_iso)
    if not dt:
        return ""
    # Try to show local time from the stored string (preserves offset)
    try:
        dt_local = datetime.fromisoformat(trigger_iso.strip())
        # 12-hour with AM/PM, e.g. "6:00 AM"
        return dt_local.strftime("%-I:%M %p").lstrip("0") or dt_local.strftime("%I:%M %p")
    except Exception:
        return dt.strftime("%H:%M UTC")


async def run_reminder_loop(check_interval_sec: int = 60) -> None:
    """
    Long-running asyncio task. Import DB and WAHA lazily to avoid circular
    imports and to handle late initialisation.
    """
    await asyncio.sleep(15)   # wait for WAHA + DB to fully initialise
    logger.info("🕐 scheduler.started  interval=%ds", check_interval_sec)

    while True:
        try:
            await _fire_due_reminders()
        except asyncio.CancelledError:
            logger.info("🕐 scheduler.cancelled")
            return
        except Exception:
            logger.exception("scheduler.error")
        await asyncio.sleep(check_interval_sec)


async def _fire_due_reminders() -> None:
    from . import database
    from .waha_provider import send_text

    if not database.sqlite_store:
        return

    due = await database.sqlite_store.get_due_reminders()
    if not due:
        return

    now      = datetime.now(UTC)
    cutoff   = now - timedelta(hours=_MAX_OVERDUE_HOURS)

    logger.info("🕐 scheduler.check  due=%d  now=%s", len(due), now.isoformat()[:19])

    for r in due:
        trigger_dt = _parse_iso(r.trigger_iso)
        if trigger_dt is None:
            logger.warning("scheduler.bad_trigger  id=%d  iso=%r", r.id, r.trigger_iso)
            await database.sqlite_store.mark_reminder_failed(r.id)
            continue

        # Stale — too far overdue
        if trigger_dt < cutoff:
            logger.info(
                "scheduler.stale  id=%d  trigger=%s  (>%dh overdue — dropped)",
                r.id, r.trigger_iso[:16], _MAX_OVERDUE_HOURS,
            )
            await database.sqlite_store.mark_reminder_failed(r.id)
            continue

        # Fetch user's name for personalisation
        user_name = ""
        try:
            facts = await database.sqlite_store.get_all_facts(r.whatsapp_id)
            user_name = facts.get("name", "")
        except Exception:
            pass

        time_str  = _friendly_time(r.trigger_iso)
        name_part = f", {user_name}" if user_name else ""
        msg_lines = [
            f"⏰ *Reminder{name_part}*",
        ]
        if time_str:
            msg_lines.append(f"_{time_str}_")
        msg_lines.append("")
        msg_lines.append(r.reminder_text)
        msg_lines.append("")
        msg_lines.append("_— Shimmi memory system_")
        msg = "\n".join(msg_lines)

        try:
            await send_text(r.chat_id, msg)
            await database.sqlite_store.mark_reminder_sent(r.id)
            logger.info(
                "🔔 reminder.sent  id=%d  chat=%s  text=%.60s",
                r.id, r.chat_id, r.reminder_text,
            )
        except Exception as exc:
            logger.error(
                "scheduler.send_failed  id=%d  chat=%s  err=%s", r.id, r.chat_id, exc,
            )
            # Don't mark failed — will retry on next cycle
