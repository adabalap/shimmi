"""
scheduler.py — Shimmi v3.4.0

Background reminder scheduler.
Runs as an asyncio task every 60 seconds.

Changes vs v2.8.0:
  IMPR-3 Rolling conversation summary refresh.
         Every ~6 minutes the scheduler checks for users whose conversation_summary
         fact is older than 2 days and regenerates it silently from ChromaDB history.
         This fixes the stale summary bug where the summary was frozen at March 17
         despite active use through late March. Checks the reminders table for entries
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


# Rolling summary: regenerate when stored summary is older than this many days.
_SUMMARY_STALE_DAYS: int = 2
# Track which users have had their summary refreshed this scheduler cycle
# so we don't hammer the LLM on every 60-second tick.
_SUMMARY_LAST_REFRESH: dict = {}   # whatsapp_id → date string (YYYY-MM-DD)


async def run_reminder_loop(check_interval_sec: int = 60) -> None:
    """
    Long-running asyncio task. Import DB and WAHA lazily to avoid circular
    imports and to handle late initialisation.

    Every cycle: fire due reminders.
    Every ~6 hours: refresh stale conversation summaries for active users.
    """
    await asyncio.sleep(15)   # wait for WAHA + DB to fully initialise
    logger.info("🕐 scheduler.started  interval=%ds", check_interval_sec)

    _summary_cycle = 0
    _SUMMARY_CHECK_EVERY = max(1, 360 // check_interval_sec)   # ~every 6 min (6 ticks of 60s)

    while True:
        try:
            await _fire_due_reminders()
        except asyncio.CancelledError:
            logger.info("🕐 scheduler.cancelled")
            return
        except Exception:
            logger.exception("scheduler.error")

        _summary_cycle += 1
        if _summary_cycle >= _SUMMARY_CHECK_EVERY:
            _summary_cycle = 0
            try:
                await _refresh_stale_summaries()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("scheduler.summary_refresh.error")

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


async def _refresh_stale_summaries() -> None:
    """
    IMPROVEMENT: Rolling conversation summary.

    Finds users whose conversation_summary fact is older than _SUMMARY_STALE_DAYS
    and regenerates it from recent ChromaDB messages.  Runs silently in the
    background every ~6 minutes — the user never sees it happening.

    Why this matters: the logs showed conversation_summary stuck at 2026-03-17
    even though the user was actively chatting until March 28+.  The orchestrator
    only updated the summary when the user explicitly asked.  This background
    refresh keeps the summary rolling automatically.
    """
    from . import database
    from datetime import date

    if not database.sqlite_store:
        return

    today_str = date.today().isoformat()

    # Fetch all users who have a conversation_summary fact
    try:
        users = await database.sqlite_store.get_users_with_key("conversation_summary")
    except AttributeError:
        # get_users_with_key may not exist on older DB versions — skip gracefully
        return
    except Exception as exc:
        logger.debug("summary_refresh.get_users_fail  err=%s", exc)
        return

    for whatsapp_id, summary_val, updated_at in (users or []):
        # Skip if already refreshed today
        if _SUMMARY_LAST_REFRESH.get(whatsapp_id) == today_str:
            continue

        # Check staleness
        try:
            from datetime import datetime as _dt, timezone as _tz
            updated_dt = _dt.fromisoformat(updated_at).replace(tzinfo=_tz.utc) if updated_at else None
            if updated_dt:
                age_days = (datetime.now(UTC) - updated_dt).days
                if age_days < _SUMMARY_STALE_DAYS:
                    continue
        except Exception:
            continue

        # Regenerate summary from recent Chroma messages
        try:
            await _regenerate_summary(whatsapp_id)
            _SUMMARY_LAST_REFRESH[whatsapp_id] = today_str
            logger.info("🧠 summary.refreshed  sender=%s", whatsapp_id)
        except Exception as exc:
            logger.debug("summary_refresh.regen_fail  sender=%s  err=%s", whatsapp_id, str(exc)[:120])


async def _regenerate_summary(whatsapp_id: str) -> None:
    """Pull recent messages and ask the LLM to generate an updated summary."""
    from . import database
    from .agent_engine import _groq_raw
    from .database import normalize_key
    import json

    if not database.chroma_store or not database.sqlite_store:
        return

    # Get all chat_ids this user appears in
    recent = await database.chroma_store.recent_window(
        chat_id=whatsapp_id,   # best-effort; chroma stores by chat_id
        k=40,
    )

    if not recent:
        return

    # Build a readable transcript
    lines = []
    for msg in recent:
        direction = (msg.metadata or {}).get("direction", "?")
        who = "You" if direction == "in" else "Shimmi"
        lines.append(f"{who}: {msg.text[:200]}")

    transcript = chr(10).join(lines[-30:])   # last 30 exchanges

    try:
        raw = await _groq_raw(
            [
                {
                    "role": "system",
                    "content": (
                        "Summarise the following WhatsApp conversation as a concise bullet list. "
                        "Focus on topics discussed, decisions made, and facts shared. "
                        "Use past tense. Max 8 bullets. No JSON — plain text only."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            max_tokens=300,
            chat_id=whatsapp_id,
            label="rolling_summary",
            role="extract",
            timeout=20.0,
        )
    except Exception as exc:
        logger.debug("summary_regen.llm_fail  sender=%s  err=%s", whatsapp_id, str(exc)[:100])
        return

    summary = (raw or "").strip()
    if not summary or len(summary) < 30:
        return

    await database.sqlite_store.upsert_fact(
        whatsapp_id,
        normalize_key("conversation_summary"),
        summary,
        source="bot_inferred",
    )
