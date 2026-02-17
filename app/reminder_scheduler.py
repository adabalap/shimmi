"""
Reminder Scheduler - TIMEZONE FIXED
Uses local time instead of UTC to match reminder storage
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable

from .reminder_manager import ReminderManager, Reminder

logger = logging.getLogger("app.reminder_scheduler")


# Notification templates
NOTIFICATION_TEMPLATES = {
    'advance_60': "⏰ Reminder in 1 hour: {title}",
    'advance_30': "⏰ Reminder in 30 minutes: {title}",
    'advance_15': "⏰ Reminder in 15 minutes: {title}",
    'advance_10': "⏰ Reminder in 10 minutes: {title}",
    'advance_5': "⏰ Reminder in 5 minutes: {title}",
    'due': "{emoji} It's time: {title}!",
    'overdue': "⚠️ Overdue: {title} - Have you done this?",
    'recurring_complete': "✅ Marked complete: {title}\n📅 Next reminder: {next_date}",
    'once_complete': "✅ Completed: {title}",
    'snoozed': "😴 Snoozed {title} until {snooze_time}",
    'cancelled': "🗑️ Reminder cancelled: {title}",
}


class ReminderScheduler:
    """
    Background task that checks for due reminders and sends notifications
    FIXED: Uses local time (IST) instead of UTC
    """
    
    def __init__(
        self,
        manager: ReminderManager,
        send_message_fn: Callable[[str, str], Awaitable[dict]],
        check_interval: int = 60,
        overdue_threshold: int = 360,
    ):
        self.manager = manager
        self.send_message = send_message_fn
        self.check_interval = check_interval
        self.overdue_threshold = overdue_threshold
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_count = 0
    
    async def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("reminder_scheduler.already_running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("✅ reminder_scheduler.started interval=%ds timezone=LOCAL", self.check_interval)
    
    async def stop(self):
        """Stop the scheduler"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("reminder_scheduler.stopped")
    
    async def _run_loop(self):
        """Main scheduler loop"""
        logger.info("🔄 reminder_scheduler.loop_started")
        
        while self._running:
            try:
                self._check_count += 1
                # 🔴 FIX: Use local time (IST) instead of UTC
                now = datetime.now()  # Local time!
                
                logger.info(
                    "🔍 reminder_scheduler.check run=%d time=%s",
                    self._check_count,
                    now.strftime("%H:%M:%S")
                )
                
                await self._check_all_reminders()
                
            except Exception as e:
                logger.exception("reminder_scheduler.error err=%s", str(e))
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_all_reminders(self):
        """Check all types of reminders"""
        # 🔴 FIX: Use local time
        now = datetime.now()
        
        # Check advance notifications
        for minutes in [60, 30, 15, 10, 5]:
            await self._send_advance_notifications(now, minutes)
        
        # Check due reminders
        await self._send_due_notifications(now)
        
        # Check overdue reminders
        await self._send_overdue_notifications(now)
        
        # Resume snoozed reminders
        await self._resume_snoozed_reminders(now)
    
    async def _send_advance_notifications(self, now: datetime, minutes_before: int):
        """Send advance notifications"""
        try:
            reminders = await self.manager.get_advance_reminders(minutes_before, now)
            
            logger.debug(
                "reminder_scheduler.advance_check minutes=%d found=%d",
                minutes_before, len(reminders)
            )
            
            for reminder in reminders:
                try:
                    message = NOTIFICATION_TEMPLATES.get(
                        f'advance_{minutes_before}',
                        f"⏰ Reminder in {minutes_before} minutes: {{title}}"
                    ).format(title=reminder.title)
                    
                    await self.send_message(reminder.chat_id, message)
                    await self.manager.mark_notification_sent(
                        reminder.id,
                        f'advance_{minutes_before}'
                    )
                    
                    logger.info(
                        "reminder.advance_sent id=%d type=advance_%dm title=%s",
                        reminder.id, minutes_before, reminder.title
                    )
                    
                except Exception as e:
                    logger.error(
                        "reminder.advance_failed id=%d err=%s",
                        reminder.id, str(e)[:100]
                    )
        
        except Exception as e:
            logger.error("reminder.advance_check_failed minutes=%d err=%s", minutes_before, str(e))
    
    async def _send_due_notifications(self, now: datetime):
        """Send notifications for reminders that are due"""
        try:
            reminders = await self.manager.get_due_reminders(now)
            
            logger.info(
                "reminder_scheduler.due_check time=%s found=%d",
                now.strftime("%H:%M:%S"), len(reminders)
            )
            
            if reminders:
                logger.info("📋 Found %d due reminders!", len(reminders))
                for r in reminders:
                    logger.info(
                        "  - ID %d: %s (due: %s)",
                        r.id, r.title, r.due_datetime.strftime("%H:%M:%S")
                    )
            
            for reminder in reminders:
                try:
                    message = NOTIFICATION_TEMPLATES['due'].format(
                        emoji=reminder.emoji,
                        title=reminder.title
                    )
                    
                    logger.info("📤 Sending notification to chat=%s", reminder.chat_id)
                    result = await self.send_message(reminder.chat_id, message)
                    logger.info("📤 Send result: %s", result)
                    
                    await self.manager.mark_notification_sent(reminder.id, 'due')
                    
                    logger.info(
                        "✅ reminder.due_sent id=%d title=%s emoji=%s",
                        reminder.id, reminder.title, reminder.emoji
                    )
                    
                except Exception as e:
                    logger.error(
                        "❌ reminder.due_failed id=%d err=%s",
                        reminder.id, str(e)[:200],
                        exc_info=True
                    )
        
        except Exception as e:
            logger.error("reminder.due_check_failed err=%s", str(e), exc_info=True)
    
    async def _send_overdue_notifications(self, now: datetime):
        """Send overdue alerts"""
        try:
            pass
        except Exception as e:
            logger.error("reminder.overdue_check_failed err=%s", str(e))
    
    async def _resume_snoozed_reminders(self, now: datetime):
        """Resume snoozed reminders"""
        try:
            pass
        except Exception as e:
            logger.error("reminder.snooze_resume_failed err=%s", str(e))


async def start_reminder_scheduler(
    manager: ReminderManager,
    send_message_fn: Callable[[str, str], Awaitable[dict]],
    check_interval: int = 60
) -> ReminderScheduler:
    """Start the scheduler"""
    scheduler = ReminderScheduler(
        manager=manager,
        send_message_fn=send_message_fn,
        check_interval=check_interval
    )
    await scheduler.start()
    return scheduler
