"""
Reminder Manager - CRUD Operations
Handles all database operations for reminders
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger("app.reminder_manager")
UTC = timezone.utc


@dataclass
class Reminder:
    """Reminder model"""
    id: int
    whatsapp_id: str
    chat_id: str
    title: str
    due_datetime: datetime
    status: str
    recurrence_type: str = 'once'
    recurrence_interval: Optional[int] = None
    category: str = 'personal'
    emoji: str = '⏰'
    advance_notifications: List[int] = None
    notifications_sent: List[str] = None
    completion_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.advance_notifications is None:
            self.advance_notifications = [60, 5]
        if self.notifications_sent is None:
            self.notifications_sent = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'whatsapp_id': self.whatsapp_id,
            'chat_id': self.chat_id,
            'title': self.title,
            'due_datetime': self.due_datetime.isoformat() if isinstance(self.due_datetime, datetime) else self.due_datetime,
            'status': self.status,
            'recurrence_type': self.recurrence_type,
            'emoji': self.emoji,
            'category': self.category,
        }


class ReminderManager:
    """Manages reminder lifecycle with SQLite backend"""
    
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._lock = asyncio.Lock()
    
    async def create_reminder(
        self,
        *,
        whatsapp_id: str,
        chat_id: str,
        title: str,
        due_datetime: datetime,
        recurrence_type: str = 'once',
        recurrence_interval: Optional[int] = None,
        recurrence_days: Optional[List[str]] = None,
        advance_notifications: Optional[List[int]] = None,
        category: str = 'personal',
        emoji: str = '⏰',
        description: Optional[str] = None,
        allow_group_complete: bool = False,
    ) -> int:
        """
        Create a new reminder
        Returns reminder ID
        """
        async with self._lock:
            def _do() -> int:
                now = datetime.now(UTC).isoformat()
                due_iso = due_datetime.isoformat()
                
                adv_notif = json.dumps(advance_notifications or [60, 5])
                rec_days = json.dumps(recurrence_days) if recurrence_days else None
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO reminders (
                            whatsapp_id, chat_id, title, description,
                            due_datetime, timezone, original_due,
                            recurrence_type, recurrence_interval, recurrence_days,
                            advance_notifications, notifications_sent,
                            category, emoji,
                            status, allow_group_complete,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            whatsapp_id, chat_id, title, description,
                            due_iso, 'Asia/Kolkata', due_iso,
                            recurrence_type, recurrence_interval, rec_days,
                            adv_notif, '[]',
                            category, emoji,
                            'active', 1 if allow_group_complete else 0,
                            now, now
                        )
                    )
                    reminder_id = cursor.lastrowid
                    
                    # Log creation
                    conn.execute(
                        "INSERT INTO reminder_log (reminder_id, action, timestamp, whatsapp_id) VALUES (?, ?, ?, ?)",
                        (reminder_id, 'created', now, whatsapp_id)
                    )
                    
                    conn.commit()
                    
                    logger.info(
                        "reminder.created id=%d user=%s title=%s due=%s recurrence=%s",
                        reminder_id, whatsapp_id, title, due_iso, recurrence_type
                    )
                    
                    return reminder_id
            
            return await asyncio.to_thread(_do)
    
    async def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        """Get single reminder by ID"""
        async with self._lock:
            def _do() -> Optional[Reminder]:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM reminders WHERE id = ?",
                        (reminder_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_reminder(row)
                    return None
            
            return await asyncio.to_thread(_do)
    
    async def list_reminders(
        self,
        whatsapp_id: str,
        status: str = 'active',
        limit: int = 20
    ) -> List[Reminder]:
        """List reminders for a user"""
        async with self._lock:
            def _do() -> List[Reminder]:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        """
                        SELECT * FROM reminders 
                        WHERE whatsapp_id = ? AND status = ?
                        ORDER BY due_datetime ASC
                        LIMIT ?
                        """,
                        (whatsapp_id, status, limit)
                    )
                    
                    return [self._row_to_reminder(row) for row in cursor.fetchall()]
            
            return await asyncio.to_thread(_do)
    
    async def complete_reminder(
        self,
        reminder_id: int,
        whatsapp_id: str,
        mark_date: Optional[datetime] = None
    ) -> bool:
        """
        Mark reminder as complete
        
        For recurring reminders:
        - Updates to next occurrence
        - Increments completion count
        
        For one-time reminders:
        - Marks as completed
        """
        async with self._lock:
            def _do() -> bool:
                reminder = self.get_reminder_sync(reminder_id)
                if not reminder:
                    return False
                
                # Verify ownership
                if reminder.whatsapp_id != whatsapp_id:
                    logger.warning("reminder.complete_denied id=%d user=%s (not owner)", reminder_id, whatsapp_id)
                    return False
                
                now = datetime.now(UTC).isoformat()
                completed_at = mark_date.isoformat() if mark_date else now
                
                with sqlite3.connect(self.db_path) as conn:
                    if reminder.recurrence_type == 'once':
                        # One-time: mark as completed
                        conn.execute(
                            """
                            UPDATE reminders 
                            SET status = 'completed',
                                completed_at = ?,
                                completion_count = completion_count + 1,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (completed_at, now, reminder_id)
                        )
                        
                        logger.info("reminder.completed id=%d title=%s", reminder_id, reminder.title)
                    else:
                        # Recurring: schedule next occurrence
                        next_due = self._calculate_next_due(reminder)
                        
                        conn.execute(
                            """
                            UPDATE reminders
                            SET due_datetime = ?,
                                last_action_date = ?,
                                completion_count = completion_count + 1,
                                notifications_sent = '[]',
                                overdue_notified_at = NULL,
                                status = 'active',
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (next_due.isoformat(), completed_at, now, reminder_id)
                        )
                        
                        logger.info(
                            "reminder.completed_recurring id=%d next_due=%s",
                            reminder_id, next_due.isoformat()
                        )
                    
                    # Log action
                    conn.execute(
                        "INSERT INTO reminder_log (reminder_id, action, timestamp, whatsapp_id) VALUES (?, ?, ?, ?)",
                        (reminder_id, 'completed', now, whatsapp_id)
                    )
                    
                    conn.commit()
                    return True
            
            return await asyncio.to_thread(_do)
    
    async def snooze_reminder(
        self,
        reminder_id: int,
        whatsapp_id: str,
        minutes: int = 30
    ) -> Optional[datetime]:
        """Snooze reminder for X minutes"""
        async with self._lock:
            def _do() -> Optional[datetime]:
                reminder = self.get_reminder_sync(reminder_id)
                if not reminder or reminder.whatsapp_id != whatsapp_id:
                    return None
                
                now = datetime.now(UTC)
                snooze_until = now + timedelta(minutes=minutes)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE reminders
                        SET status = 'snoozed',
                            snoozed_until = ?,
                            snooze_count = snooze_count + 1,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (snooze_until.isoformat(), now.isoformat(), reminder_id)
                    )
                    
                    conn.execute(
                        "INSERT INTO reminder_log (reminder_id, action, timestamp, details, whatsapp_id) VALUES (?, ?, ?, ?, ?)",
                        (reminder_id, 'snoozed', now.isoformat(), json.dumps({'minutes': minutes}), whatsapp_id)
                    )
                    
                    conn.commit()
                    
                    logger.info("reminder.snoozed id=%d minutes=%d until=%s", reminder_id, minutes, snooze_until)
                    return snooze_until
            
            return await asyncio.to_thread(_do)
    
    async def cancel_reminder(self, reminder_id: int, whatsapp_id: str) -> bool:
        """Cancel/delete a reminder"""
        async with self._lock:
            def _do() -> bool:
                reminder = self.get_reminder_sync(reminder_id)
                if not reminder or reminder.whatsapp_id != whatsapp_id:
                    return False
                
                now = datetime.now(UTC).isoformat()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE reminders SET status = 'cancelled', updated_at = ? WHERE id = ?",
                        (now, reminder_id)
                    )
                    
                    conn.execute(
                        "INSERT INTO reminder_log (reminder_id, action, timestamp, whatsapp_id) VALUES (?, ?, ?, ?)",
                        (reminder_id, 'cancelled', now, whatsapp_id)
                    )
                    
                    conn.commit()
                    
                    logger.info("reminder.cancelled id=%d title=%s", reminder_id, reminder.title)
                    return True
            
            return await asyncio.to_thread(_do)
    
    async def get_due_reminders(self, check_time: Optional[datetime] = None) -> List[Reminder]:
        """Get reminders that are due now (for scheduler)"""
        if check_time is None:
            check_time = datetime.now(UTC)
        
        async with self._lock:
            def _do() -> List[Reminder]:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Get active reminders that are due
                    cursor = conn.execute(
                        """
                        SELECT * FROM reminders
                        WHERE status = 'active'
                          AND datetime(due_datetime) <= datetime(?)
                          AND NOT json_extract(notifications_sent, '$') LIKE '%"due"%'
                        ORDER BY due_datetime ASC
                        """,
                        (check_time.isoformat(),)
                    )
                    
                    return [self._row_to_reminder(row) for row in cursor.fetchall()]
            
            return await asyncio.to_thread(_do)
    
    async def get_advance_reminders(
        self,
        minutes_before: int,
        check_time: Optional[datetime] = None
    ) -> List[Reminder]:
        """Get reminders needing advance notification"""
        if check_time is None:
            check_time = datetime.now(UTC)
        
        target_time = check_time + timedelta(minutes=minutes_before)
        alert_key = f"advance_{minutes_before}"
        
        async with self._lock:
            def _do() -> List[Reminder]:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    cursor = conn.execute(
                        """
                        SELECT * FROM reminders
                        WHERE status = 'active'
                          AND datetime(due_datetime) <= datetime(?)
                          AND datetime(due_datetime) > datetime(?)
                          AND json_extract(advance_notifications, '$') LIKE ?
                          AND NOT json_extract(notifications_sent, '$') LIKE ?
                        """,
                        (
                            target_time.isoformat(),
                            check_time.isoformat(),
                            f'%{minutes_before}%',
                            f'%"{alert_key}"%'
                        )
                    )
                    
                    return [self._row_to_reminder(row) for row in cursor.fetchall()]
            
            return await asyncio.to_thread(_do)
    
    async def mark_notification_sent(self, reminder_id: int, notification_type: str):
        """Mark that a notification has been sent"""
        async with self._lock:
            def _do():
                with sqlite3.connect(self.db_path) as conn:
                    # Get current notifications_sent
                    cursor = conn.execute(
                        "SELECT notifications_sent FROM reminders WHERE id = ?",
                        (reminder_id,)
                    )
                    row = cursor.fetchone()
                    if not row:
                        return
                    
                    sent = json.loads(row[0] or '[]')
                    if notification_type not in sent:
                        sent.append(notification_type)
                    
                    conn.execute(
                        "UPDATE reminders SET notifications_sent = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(sent), datetime.now(UTC).isoformat(), reminder_id)
                    )
                    
                    conn.execute(
                        "INSERT INTO reminder_log (reminder_id, action, timestamp, details) VALUES (?, ?, ?, ?)",
                        (reminder_id, 'notified', datetime.now(UTC).isoformat(), json.dumps({'type': notification_type}))
                    )
                    
                    conn.commit()
            
            await asyncio.to_thread(_do)
    
    # Helper methods
    
    def get_reminder_sync(self, reminder_id: int) -> Optional[Reminder]:
        """Synchronous get (for use within transactions)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
            row = cursor.fetchone()
            return self._row_to_reminder(row) if row else None
    
    def _row_to_reminder(self, row: sqlite3.Row) -> Reminder:
        """Convert DB row to Reminder object"""
        return Reminder(
            id=row['id'],
            whatsapp_id=row['whatsapp_id'],
            chat_id=row['chat_id'],
            title=row['title'],
            description=row['description'],
            due_datetime=datetime.fromisoformat(row['due_datetime']),
            status=row['status'],
            recurrence_type=row['recurrence_type'] or 'once',
            recurrence_interval=row['recurrence_interval'],
            category=row['category'] or 'personal',
            emoji=row['emoji'] or '⏰',
            advance_notifications=json.loads(row['advance_notifications'] or '[60, 5]'),
            notifications_sent=json.loads(row['notifications_sent'] or '[]'),
            completion_count=row['completion_count'] or 0,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
        )
    
    def _calculate_next_due(self, reminder: Reminder) -> datetime:
        """Calculate next due date for recurring reminder"""
        current_due = reminder.due_datetime
        
        if reminder.recurrence_type == 'daily':
            return current_due + timedelta(days=1)
        elif reminder.recurrence_type == 'every_n_days':
            days = reminder.recurrence_interval or 1
            return current_due + timedelta(days=days)
        elif reminder.recurrence_type == 'weekly':
            return current_due + timedelta(weeks=1)
        elif reminder.recurrence_type == 'monthly':
            # Approximate: add 30 days
            return current_due + timedelta(days=30)
        else:
            # Default: same time tomorrow
            return current_due + timedelta(days=1)
