"""
tests/unit/test_reminders.py — Shimmi v3.3.0

Zero-quota unit tests for the reminder subsystem.

Coverage:
  ① _is_reminder_duplicate() — dedup gate
  ② _fix_reminder_tz()        — UTC→IST offset correction
  ③ _save_reminders() logic   — blank text/iso skip, dedup skip
  ④ Scheduler ISO parsing     — _parse_iso handles offsets correctly
  ⑤ WAHA webhook dedup        — _inbound_seen_check TTL behaviour
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / stubs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Reminder:
    id: int
    whatsapp_id: str
    chat_id: str
    reminder_text: str
    trigger_iso: str
    status: str = "pending"


# ─────────────────────────────────────────────────────────────────────────────
# ① _is_reminder_duplicate()
# ─────────────────────────────────────────────────────────────────────────────

class TestReminderDedup:

    def _existing(self, text, iso, status="pending"):
        return _Reminder(1, "u1", "c1", text, iso, status)

    def test_exact_text_match_is_duplicate(self):
        from app.agent_engine import _is_reminder_duplicate
        existing = [self._existing("dentist appointment", "2026-03-20T10:00:00+05:30")]
        assert _is_reminder_duplicate("dentist appointment", "2026-03-25T10:00:00+05:30", existing)

    def test_case_insensitive_text_match(self):
        from app.agent_engine import _is_reminder_duplicate
        existing = [self._existing("Call Dentist", "2026-03-20T10:00:00+05:30")]
        assert _is_reminder_duplicate("call dentist", "2026-03-25T10:00:00+05:30", existing)

    def test_same_trigger_minute_is_duplicate(self):
        from app.agent_engine import _is_reminder_duplicate
        existing = [self._existing("anything", "2026-03-20T10:00:00+05:30")]
        # Same to-the-minute ISO prefix → duplicate regardless of text
        assert _is_reminder_duplicate("something else", "2026-03-20T10:00:45+05:30", existing)

    def test_different_text_and_time_not_duplicate(self):
        from app.agent_engine import _is_reminder_duplicate
        existing = [self._existing("dentist", "2026-03-20T10:00:00+05:30")]
        assert not _is_reminder_duplicate("haircut", "2026-03-21T14:00:00+05:30", existing)

    def test_sent_reminder_not_counted_as_duplicate(self):
        from app.agent_engine import _is_reminder_duplicate
        sent = [self._existing("dentist", "2026-03-20T10:00:00+05:30", status="sent")]
        # Same text, same time, but status=sent → not a pending duplicate
        assert not _is_reminder_duplicate("dentist", "2026-03-20T10:00:00+05:30", sent)

    def test_empty_existing_never_duplicate(self):
        from app.agent_engine import _is_reminder_duplicate
        assert not _is_reminder_duplicate("anything", "2026-03-20T10:00:00+05:30", [])


# ─────────────────────────────────────────────────────────────────────────────
# ② _fix_reminder_tz()
# ─────────────────────────────────────────────────────────────────────────────

class TestReminderTzFix:

    def test_utc_iso_corrected_to_ist(self, monkeypatch):
        """When APP_TIMEZONE=Asia/Kolkata, UTC+00:00 suffix → +05:30."""
        monkeypatch.setenv("APP_TIMEZONE", "Asia/Kolkata")
        # Re-import to pick up patched env
        from importlib import reload
        import app.config as cfg
        # Use the function directly without reloading config (settings is frozen dataclass)
        from app.agent_engine import _fix_reminder_tz, _utc_offset_str
        local = _utc_offset_str()  # should be +05:30 for Asia/Kolkata
        if local == "+05:30":
            fixed = _fix_reminder_tz("2026-03-20T10:00:00+00:00")
            assert fixed == "2026-03-20T10:00:00+05:30"
        else:
            pytest.skip(f"Server timezone not IST (got {local}), skipping tz correction test")

    def test_z_suffix_corrected(self, monkeypatch):
        monkeypatch.setenv("APP_TIMEZONE", "Asia/Kolkata")
        from app.agent_engine import _fix_reminder_tz, _utc_offset_str
        local = _utc_offset_str()
        if local == "+05:30":
            fixed = _fix_reminder_tz("2026-03-20T10:00:00Z")
            assert fixed == "2026-03-20T10:00:00+05:30"
        else:
            pytest.skip("Not IST")

    def test_already_offset_not_touched(self):
        from app.agent_engine import _fix_reminder_tz
        iso = "2026-03-20T10:00:00+05:30"
        assert _fix_reminder_tz(iso) == iso


# ─────────────────────────────────────────────────────────────────────────────
# ③ Reminder save logic — blank / dedup skip
# ─────────────────────────────────────────────────────────────────────────────

class TestReminderSaveLogic:

    def _stub_entry(self, text="call dentist", iso="2026-03-20T10:00:00+05:30"):
        r = MagicMock()
        r.text = text
        r.trigger_iso = iso
        return r

    @pytest.mark.asyncio
    async def test_blank_text_skipped(self):
        """Reminders with empty text must be silently skipped, not saved."""
        from app.main import _save_reminders
        blank = self._stub_entry(text="")
        with patch("app.main.database") as db:
            db.sqlite_store = AsyncMock()
            saved = await _save_reminders("u1", "c1", [blank], existing=[])
        assert saved == 0
        db.sqlite_store.add_reminder.assert_not_called()

    @pytest.mark.asyncio
    async def test_blank_iso_skipped(self):
        from app.main import _save_reminders
        blank = self._stub_entry(iso="")
        with patch("app.main.database") as db:
            db.sqlite_store = AsyncMock()
            saved = await _save_reminders("u1", "c1", [blank], existing=[])
        assert saved == 0

    @pytest.mark.asyncio
    async def test_dedup_skipped(self):
        """Reminder identical to existing pending one must not be re-saved."""
        from app.main import _save_reminders
        entry = self._stub_entry("call dentist", "2026-03-20T10:00:00+05:30")
        existing = [_Reminder(1, "u1", "c1", "call dentist", "2026-03-20T10:00:00+05:30")]
        with patch("app.main.database") as db:
            db.sqlite_store = AsyncMock()
            saved = await _save_reminders("u1", "c1", [entry], existing=existing)
        assert saved == 0

    @pytest.mark.asyncio
    async def test_unique_reminder_saved(self):
        from app.main import _save_reminders
        entry = self._stub_entry("call dentist", "2026-03-20T10:00:00+05:30")
        with patch("app.main.database") as db:
            db.sqlite_store = AsyncMock()
            db.sqlite_store.add_reminder = AsyncMock(return_value=42)
            saved = await _save_reminders("u1", "c1", [entry], existing=[])
        assert saved == 1
        db.sqlite_store.add_reminder.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# ④ Scheduler ISO parsing — _parse_iso
# ─────────────────────────────────────────────────────────────────────────────

class TestSchedulerIsoParsing:

    def test_ist_offset_parsed_to_utc(self):
        from app.scheduler import _parse_iso
        from datetime import timezone
        dt = _parse_iso("2026-03-20T10:00:00+05:30")
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        # 10:00 IST = 04:30 UTC
        assert dt.hour == 4
        assert dt.minute == 30

    def test_utc_z_parsed(self):
        from app.scheduler import _parse_iso
        dt = _parse_iso("2026-03-20T10:00:00Z")
        assert dt is not None
        assert dt.hour == 10

    def test_blank_returns_none(self):
        from app.scheduler import _parse_iso
        assert _parse_iso("") is None
        assert _parse_iso(None) is None

    def test_invalid_string_returns_none(self):
        from app.scheduler import _parse_iso
        assert _parse_iso("not-a-date") is None

    def test_no_tz_assumes_utc(self):
        from app.scheduler import _parse_iso
        dt = _parse_iso("2026-03-20T10:00:00")
        assert dt is not None
        assert dt.hour == 10


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Webhook dedup — _inbound_seen_check
# ─────────────────────────────────────────────────────────────────────────────

class TestWebhookDedup:

    def test_new_event_not_seen(self):
        """First time an event_id appears → not duplicate."""
        from app.main import _inbound_seen_check, _INBOUND_SEEN
        _INBOUND_SEEN.clear()
        assert _inbound_seen_check("evt-001") is False

    def test_same_event_seen_twice_is_duplicate(self):
        """Seeing the same event_id a second time → duplicate (WAHA retry)."""
        from app.main import _inbound_seen_check, _INBOUND_SEEN
        _INBOUND_SEEN.clear()
        _inbound_seen_check("evt-002")      # first → not duplicate
        assert _inbound_seen_check("evt-002") is True  # second → duplicate

    def test_different_events_independent(self):
        from app.main import _inbound_seen_check, _INBOUND_SEEN
        _INBOUND_SEEN.clear()
        _inbound_seen_check("evt-A")
        assert _inbound_seen_check("evt-B") is False

    def test_stale_entries_pruned(self):
        """Entries older than 60s are pruned so memory doesn't grow forever."""
        from app.main import _inbound_seen_check, _INBOUND_SEEN
        _INBOUND_SEEN.clear()
        # Insert a fake stale entry
        _INBOUND_SEEN["evt-stale"] = time.monotonic() - 70.0
        # Calling _inbound_seen_check for a new event triggers the prune
        _inbound_seen_check("evt-new")
        # Stale entry should now be gone
        assert "evt-stale" not in _INBOUND_SEEN
