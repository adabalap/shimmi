"""
Reminder Scheduler - v3 PRODUCTION
With monitoring, metrics, health checks, alerting
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass, field

from .reminder_manager import ReminderManager
from .reminder_commands import (
    format_due_notification,
    format_advance_notification,
    format_overdue_notification,
)

logger = logging.getLogger("app.reminder_scheduler")


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------

@dataclass
class SchedulerMetrics:
    """Production metrics for monitoring"""
    # Counters
    total_checks: int = 0
    advance_sent: int = 0
    due_sent: int = 0
    overdue_sent: int = 0
    failed_notifications: int = 0
    
    # Performance
    avg_check_duration_ms: float = 0.0
    last_check_time: Optional[datetime] = None
    
    # Status
    is_running: bool = False
    started_at: Optional[datetime] = None
    
    # Recent failures
    recent_errors: list = field(default_factory=list)
    
    def record_check(self, duration_ms: float):
        """Record a check cycle"""
        self.total_checks += 1
        self.last_check_time = datetime.now()
        
        # Rolling average
        if self.avg_check_duration_ms == 0:
            self.avg_check_duration_ms = duration_ms
        else:
            # Exponential moving average
            self.avg_check_duration_ms = 0.9 * self.avg_check_duration_ms + 0.1 * duration_ms
    
    def record_notification(self, notification_type: str, success: bool = True):
        """Record notification attempt"""
        if success:
            if notification_type == 'advance':
                self.advance_sent += 1
            elif notification_type == 'due':
                self.due_sent += 1
            elif notification_type == 'overdue':
                self.overdue_sent += 1
        else:
            self.failed_notifications += 1
    
    def record_error(self, error: str):
        """Record an error"""
        self.recent_errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error[:200]
        })
        # Keep only last 10 errors
        if len(self.recent_errors) > 10:
            self.recent_errors = self.recent_errors[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dict"""
        return {
            'total_checks': self.total_checks,
            'notifications': {
                'advance': self.advance_sent,
                'due': self.due_sent,
                'overdue': self.overdue_sent,
                'failed': self.failed_notifications,
            },
            'performance': {
                'avg_check_duration_ms': round(self.avg_check_duration_ms, 2),
                'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            },
            'status': {
                'running': self.is_running,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'uptime_seconds': (datetime.now() - self.started_at).total_seconds() if self.started_at else 0,
            },
            'recent_errors': self.recent_errors,
        }


# ---------------------------------------------------------------------------
# Production Scheduler
# ---------------------------------------------------------------------------

class ReminderScheduler:
    """
    Production-grade scheduler with:
    - Monitoring & metrics
    - Health checks
    - Error recovery
    - Performance tracking
    """
    
    def __init__(
        self,
        manager: ReminderManager,
        send_message_fn: Callable[[str, str], Awaitable[dict]],
        check_interval: int = 60,
        overdue_threshold_hours: int = 6,
    ):
        self.manager = manager
        self.send_message = send_message_fn
        self.check_interval = check_interval
        self.overdue_threshold_hours = overdue_threshold_hours
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_count = 0
        
        # Production features
        self.metrics = SchedulerMetrics()
        self._last_health_check = datetime.now()
    
    async def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("reminder_scheduler.already_running")
            return
        
        self._running = True
        self.metrics.is_running = True
        self.metrics.started_at = datetime.now()
        
        self._task = asyncio.create_task(self._run_loop())
        logger.info("✅ reminder_scheduler.started interval=%ds", self.check_interval)
    
    async def stop(self):
        """Stop the scheduler gracefully"""
        if not self._running:
            return
        
        self._running = False
        self.metrics.is_running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("reminder_scheduler.stopped checks=%d", self.metrics.total_checks)
    
    async def _run_loop(self):
        """Main scheduler loop with error recovery"""
        logger.info("🔄 reminder_scheduler.loop_started")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self._running:
            start_time = time.time()
            
            try:
                self._check_count += 1
                now = datetime.now()
                
                logger.info(
                    "🔍 reminder_scheduler.check run=%d time=%s",
                    self._check_count, now.strftime("%H:%M:%S")
                )
                
                await self._check_all(now)
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_check(duration_ms)
                
                # Periodic health check (every 10 checks)
                if self._check_count % 10 == 0:
                    await self._perform_health_check()
            
            except Exception as e:
                consecutive_errors += 1
                error_msg = f"{type(e).__name__}: {str(e)[:150]}"
                
                logger.exception("reminder_scheduler.error count=%d err=%s", consecutive_errors, error_msg)
                self.metrics.record_error(error_msg)
                
                # Circuit breaker: stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "ALERT: Scheduler stopping after %d consecutive errors!",
                        consecutive_errors
                    )
                    self._running = False
                    break
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_all(self, now: datetime):
        """Complete check cycle"""
        await self._check_advance(now)
        await self._check_due(now)
        await self._check_overdue(now)
        await self._check_snooze_resume(now)
    
    # -------------------------------------------------------------------------
    # Notification Checks
    # -------------------------------------------------------------------------
    
    async def _check_advance(self, now: datetime):
        """Send advance alerts"""
        for mins in [60, 30, 15, 5]:
            try:
                reminders = await self.manager.get_advance_reminders(mins, now)
                
                for reminder in reminders:
                    try:
                        msg = format_advance_notification(reminder, mins)
                        await self.send_message(reminder.chat_id, msg)
                        await self.manager.mark_notification_sent(reminder.id, f'advance_{mins}')
                        
                        self.metrics.record_notification('advance', success=True)
                        logger.info(
                            "reminder.advance_sent id=%d mins=%d",
                            reminder.id, mins
                        )
                    except Exception as e:
                        self.metrics.record_notification('advance', success=False)
                        logger.error(
                            "reminder.advance_failed id=%d err=%s",
                            reminder.id, str(e)[:150]
                        )
            except Exception as e:
                logger.error("reminder.advance_check_error mins=%d err=%s", mins, str(e))
    
    async def _check_due(self, now: datetime):
        """Send due notifications"""
        try:
            reminders = await self.manager.get_due_reminders(now)
            
            logger.info(
                "reminder_scheduler.due_check found=%d",
                len(reminders)
            )
            
            for reminder in reminders:
                try:
                    msg = format_due_notification(reminder)
                    await self.send_message(reminder.chat_id, msg)
                    await self.manager.mark_notification_sent(reminder.id, 'due')
                    
                    self.metrics.record_notification('due', success=True)
                    logger.info("✅ reminder.due_sent id=%d", reminder.id)
                
                except Exception as e:
                    self.metrics.record_notification('due', success=False)
                    logger.error(
                        "❌ reminder.due_failed id=%d err=%s",
                        reminder.id, str(e)[:200]
                    )
        except Exception as e:
            logger.error("reminder.due_check_error err=%s", str(e))
    
    async def _check_overdue(self, now: datetime):
        """Send overdue alerts"""
        try:
            overdue_cutoff = now - timedelta(hours=self.overdue_threshold_hours)
            
            import sqlite3
            def _query():
                with sqlite3.connect(self.manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.execute(
                        """
                        SELECT * FROM reminders
                        WHERE status = 'active'
                          AND datetime(due_datetime) <= datetime(?)
                          AND datetime(due_datetime) >= datetime(?, '-1 day')
                          AND (overdue_notified_at IS NULL
                               OR datetime(overdue_notified_at) < datetime(?, '-3 hours'))
                        """,
                        (
                            overdue_cutoff.isoformat(),
                            now.isoformat(),
                            now.isoformat(),
                        )
                    )
                    return cur.fetchall()
            
            rows = await asyncio.to_thread(_query)
            
            for row in rows:
                try:
                    reminder = self.manager._row_to_reminder(row)
                    msg = format_overdue_notification(reminder)
                    await self.send_message(reminder.chat_id, msg)
                    
                    # Update overdue_notified_at
                    import sqlite3
                    def _update():
                        with sqlite3.connect(self.manager.db_path) as conn:
                            conn.execute(
                                "UPDATE reminders SET overdue_notified_at = ?, status = 'overdue' WHERE id = ?",
                                (now.isoformat(), reminder.id)
                            )
                            conn.commit()
                    await asyncio.to_thread(_update)
                    
                    self.metrics.record_notification('overdue', success=True)
                    logger.info("reminder.overdue_sent id=%d", reminder.id)
                
                except Exception as e:
                    self.metrics.record_notification('overdue', success=False)
                    logger.error("reminder.overdue_failed id=%d err=%s", row['id'], str(e)[:150])
        
        except Exception as e:
            logger.error("reminder.overdue_check_error err=%s", str(e))
    
    async def _check_snooze_resume(self, now: datetime):
        """Re-activate snoozed reminders"""
        try:
            import sqlite3
            def _query():
                with sqlite3.connect(self.manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.execute(
                        """
                        SELECT * FROM reminders
                        WHERE status = 'snoozed'
                          AND datetime(snoozed_until) <= datetime(?)
                        """,
                        (now.isoformat(),)
                    )
                    return cur.fetchall()
            
            rows = await asyncio.to_thread(_query)
            
            for row in rows:
                try:
                    reminder = self.manager._row_to_reminder(row)
                    
                    def _reactivate():
                        with sqlite3.connect(self.manager.db_path) as conn:
                            conn.execute(
                                """
                                UPDATE reminders
                                SET status = 'active',
                                    due_datetime = ?,
                                    snoozed_until = NULL,
                                    notifications_sent = '[]',
                                    updated_at = ?
                                WHERE id = ?
                                """,
                                (now.isoformat(), now.isoformat(), reminder.id)
                            )
                            conn.commit()
                    await asyncio.to_thread(_reactivate)
                    
                    logger.info("reminder.snooze_resumed id=%d", reminder.id)
                
                except Exception as e:
                    logger.error("reminder.snooze_resume_failed id=%d err=%s", row['id'], str(e)[:150])
        
        except Exception as e:
            logger.error("reminder.snooze_resume_error err=%s", str(e))
    
    # -------------------------------------------------------------------------
    # Monitoring & Health
    # -------------------------------------------------------------------------
    
    async def _perform_health_check(self):
        """Periodic health check"""
        try:
            now = datetime.now()
            
            # Check database accessibility
            try:
                active_count = await self.manager.count_active_reminders("health_check")
                db_ok = True
            except:
                db_ok = False
                logger.error("ALERT: Database not accessible!")
            
            # Check last successful check
            if self.metrics.last_check_time:
                time_since_check = (now - self.metrics.last_check_time).total_seconds()
                if time_since_check > 300:  # 5 minutes
                    logger.warning("ALERT: No successful checks in %ds", time_since_check)
            
            # Check failure rate
            if self.metrics.total_checks > 0:
                failure_rate = self.metrics.failed_notifications / max(1, self.metrics.due_sent + self.metrics.advance_sent)
                if failure_rate > 0.1:  # >10% failures
                    logger.warning("ALERT: High failure rate: %.1f%%", failure_rate * 100)
            
            self._last_health_check = now
            logger.debug("health_check.ok db=%s", db_ok)
        
        except Exception as e:
            logger.error("health_check.failed err=%s", str(e))
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        now = datetime.now()
        
        # Calculate uptime
        uptime = 0
        if self.metrics.started_at:
            uptime = (now - self.metrics.started_at).total_seconds()
        
        # Time since last check
        last_check_age = 0
        if self.metrics.last_check_time:
            last_check_age = (now - self.metrics.last_check_time).total_seconds()
        
        # Determine overall health
        is_healthy = (
            self.metrics.is_running and
            last_check_age < 120 and  # Checked within 2 minutes
            self.metrics.failed_notifications < 10
        )
        
        return {
            'status': 'healthy' if is_healthy else 'degraded',
            'running': self.metrics.is_running,
            'uptime_seconds': int(uptime),
            'last_check_age_seconds': int(last_check_age),
            'metrics': self.metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

async def start_reminder_scheduler(
    manager: ReminderManager,
    send_message_fn: Callable[[str, str], Awaitable[dict]],
    check_interval: int = 60,
) -> ReminderScheduler:
    """Start the production scheduler"""
    scheduler = ReminderScheduler(
        manager=manager,
        send_message_fn=send_message_fn,
        check_interval=check_interval,
    )
    await scheduler.start()
    return scheduler
