"""
Reminder Commands - v3 PRODUCTION
Full validation, rate limiting, duplicate detection, rich UX
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from difflib import SequenceMatcher

from .reminder_manager import ReminderManager, Reminder
from .reminder_parser import ReminderParser

logger = logging.getLogger("app.reminder_commands")


# ---------------------------------------------------------------------------
# Production Configuration
# ---------------------------------------------------------------------------

class ReminderConfig:
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_REMINDERS_PER_USER = 100
    MAX_CREATE_PER_MINUTE = 10
    MAX_LIST_PER_MINUTE = 20
    MAX_DELETE_PER_MINUTE = 10
    DUPLICATE_THRESHOLD_MINUTES = 5
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85
    MAX_FUTURE_DAYS = 365


# ---------------------------------------------------------------------------
# Validation Layer
# ---------------------------------------------------------------------------

class ReminderValidator:
    """Input validation for all reminder operations"""
    
    @staticmethod
    def validate_title(title: str) -> Tuple[bool, str]:
        if not title or not title.strip():
            return False, "❌ Reminder title cannot be empty"
        
        title = title.strip()
        if len(title) > ReminderConfig.MAX_TITLE_LENGTH:
            return False, f"❌ Title too long (max {ReminderConfig.MAX_TITLE_LENGTH} characters)"
        
        # Check for suspicious patterns
        if title.count('\n') > 5:
            return False, "❌ Title cannot contain multiple line breaks"
        
        return True, ""
    
    @staticmethod
    def validate_datetime(dt: datetime) -> Tuple[bool, str]:
        now = datetime.now()
        
        if dt < now - timedelta(minutes=5):  # Allow 5 min grace period
            return False, "❌ Reminder time must be in the future"
        
        if dt > now + timedelta(days=ReminderConfig.MAX_FUTURE_DAYS):
            return False, f"❌ Reminder cannot be more than {ReminderConfig.MAX_FUTURE_DAYS} days ahead"
        
        return True, ""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Clean up user input"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        return text.strip()


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-user rate limiting for spam prevention"""
    
    def __init__(self):
        self._operations = {}  # {user_id: {operation: [(timestamp, count)]}}
    
    def check_limit(
        self,
        user_id: str,
        operation: str,
        max_ops: int,
        window_seconds: int = 60
    ) -> Tuple[bool, str]:
        """Check if operation is allowed. Returns (allowed, error_message)"""
        now = time.time()
        
        # Initialize user if needed
        if user_id not in self._operations:
            self._operations[user_id] = {}
        
        if operation not in self._operations[user_id]:
            self._operations[user_id][operation] = []
        
        # Clean old entries
        self._operations[user_id][operation] = [
            (ts, cnt) for ts, cnt in self._operations[user_id][operation]
            if now - ts < window_seconds
        ]
        
        # Count recent operations
        recent_count = sum(cnt for ts, cnt in self._operations[user_id][operation])
        
        if recent_count >= max_ops:
            wait_time = int(window_seconds - (now - self._operations[user_id][operation][0][0]))
            return False, f"⏱️ Too many {operation} operations. Please wait {wait_time}s"
        
        # Record this operation
        self._operations[user_id][operation].append((now, 1))
        return True, ""
    
    def cleanup(self):
        """Periodic cleanup of old data"""
        now = time.time()
        for user_id in list(self._operations.keys()):
            for operation in list(self._operations[user_id].keys()):
                self._operations[user_id][operation] = [
                    (ts, cnt) for ts, cnt in self._operations[user_id][operation]
                    if now - ts < 120  # Keep 2 minutes of history
                ]
                if not self._operations[user_id][operation]:
                    del self._operations[user_id][operation]
            if not self._operations[user_id]:
                del self._operations[user_id]


# ---------------------------------------------------------------------------
# Duplicate Detector
# ---------------------------------------------------------------------------

class DuplicateDetector:
    """Detect duplicate reminder creation attempts"""
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """Calculate similarity between two strings (0.0 to 1.0)"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    async def check_duplicate(
        manager: ReminderManager,
        user_id: str,
        title: str,
        due_datetime: datetime
    ) -> Optional[int]:
        """
        Check if similar reminder already exists.
        Returns reminder_id if duplicate found, None otherwise.
        """
        # Get recent reminders (last 7 days)
        recent = await manager.list_reminders(user_id, status='active', limit=50)
        
        for reminder in recent:
            # Check title similarity
            similarity = DuplicateDetector.similarity_ratio(title, reminder.title)
            
            # Check time proximity
            time_diff = abs((reminder.due_datetime - due_datetime).total_seconds() / 60)
            
            # Consider it a duplicate if:
            # - Titles are very similar (>85%)
            # - Due times are within 5 minutes
            if (similarity >= ReminderConfig.DUPLICATE_SIMILARITY_THRESHOLD and
                time_diff <= ReminderConfig.DUPLICATE_THRESHOLD_MINUTES):
                return reminder.id
        
        return None


# ---------------------------------------------------------------------------
# Rich Formatting (from v2, enhanced)
# ---------------------------------------------------------------------------

def format_reminder_card(reminder: Reminder, header: str = "🔔 Reminder") -> str:
    """Format a reminder as a rich WhatsApp card"""
    due = reminder.due_datetime
    date_str = due.strftime("%-d %B %Y")
    time_str = due.strftime("%I:%M %p").lstrip("0")
    
    lines = [f"📍 *{header}:* {reminder.title}"]
    
    if reminder.description:
        lines.append(f"📝 *Note:* {reminder.description}")
    
    lines += [
        f"📅 *Date:* {date_str}",
        f"⏰ *Time:* {time_str}",
    ]
    
    if reminder.recurrence_type != 'once':
        lines.append(f"🔄 *Repeats:* {_format_recurrence(reminder)}")
    
    return "\n".join(lines)


def format_reminder_list(reminders: List[Reminder], total_count: int = None) -> str:
    """Format numbered list with pagination info"""
    if not reminders:
        return (
            "You have no active reminders.\n\n"
            "_Try: 'Remind me to water plants every 3 days'_"
        )
    
    count_str = f"{total_count}" if total_count else f"{len(reminders)}"
    lines = [f"📋 *Your Reminders* ({count_str} active)\n"]
    
    for i, r in enumerate(reminders, 1):
        due = r.due_datetime
        date_str = due.strftime("%-d %b")
        time_str = due.strftime("%I:%M %p").lstrip("0")
        
        recurrence = ""
        if r.recurrence_type != 'once':
            recurrence = f" · 🔄 {_format_recurrence(r)}"
        
        lines.append(
            f"*{i}.* {r.emoji} {r.title}\n"
            f"    📅 {date_str} · ⏰ {time_str}{recurrence}"
        )
    
    lines.append("\n_Reply 'delete <number>' to remove_")
    return "\n".join(lines)


def format_due_notification(reminder: Reminder) -> str:
    """Due time notification"""
    due = reminder.due_datetime
    date_str = due.strftime("%-d %B %Y")
    time_str = due.strftime("%I:%M %p").lstrip("0")
    
    lines = [
        f"🔔 *It's time!*\n",
        f"📍 *Reminder:* {reminder.title}",
    ]
    
    if reminder.description:
        lines.append(f"📝 *Note:* {reminder.description}")
    
    lines += [
        f"📅 *Date:* {date_str}",
        f"⏰ *Time:* {time_str}",
    ]
    
    if reminder.recurrence_type != 'once':
        lines.append(f"\n🔄 _{_format_recurrence(reminder)}_")
        lines.append("_Reply 'done' to mark complete_")
    else:
        lines.append("\n_Reply 'done' to complete_")
    
    return "\n".join(lines)


def format_advance_notification(reminder: Reminder, minutes_before: int) -> str:
    """Advance alert"""
    due = reminder.due_datetime
    time_str = due.strftime("%I:%M %p").lstrip("0")
    
    if minutes_before >= 60:
        when = f"{minutes_before // 60} hour{'s' if minutes_before > 60 else ''}"
    else:
        when = f"{minutes_before} minute{'s' if minutes_before > 1 else ''}"
    
    lines = [
        f"⏰ *Coming up in {when}*\n",
        f"📍 *Reminder:* {reminder.title}",
    ]
    
    if reminder.description:
        lines.append(f"📝 *Note:* {reminder.description}")
    
    lines.append(f"⏰ *Scheduled:* {time_str}")
    return "\n".join(lines)


def format_overdue_notification(reminder: Reminder) -> str:
    """Overdue alert"""
    due = reminder.due_datetime
    date_str = due.strftime("%-d %B %Y")
    time_str = due.strftime("%I:%M %p").lstrip("0")
    
    return (
        f"⚠️ *Overdue Reminder*\n\n"
        f"📍 *Reminder:* {reminder.title}\n"
        f"📅 *Was due:* {date_str} at {time_str}\n\n"
        f"_Reply 'done' if completed, 'snooze 30' to delay, or 'delete {reminder.id}' to remove_"
    )


# ---------------------------------------------------------------------------
# Command Detection
# ---------------------------------------------------------------------------

COMMAND_PATTERNS = {
    'create': [
        r'remind me (?:to |about )?(.+)',
        r'reminder (?:for |to |about )?(.+)',
        r'set (?:a )?reminder (?:for |to |about )?(.+)',
        r'don\'t (?:let me )?forget (?:to )?(.+)',
    ],
    'list': [
        r'^(?:list |show |view )?(?:my )?reminders?$',
        r'^what(?:\'s| are) (?:my )?reminders?',
        r'^reminders?$',
    ],
    'complete': [
        r'^done$',
        r'^completed?$',
        r'^finished?$',
    ],
    'snooze': [
        r'^snooze(?:\s+(?:for\s+)?(\d+))?',
        r'^remind (?:me )?later',
    ],
    'delete': [
        r'^(?:delete|remove|cancel|del)\s+(\d+)$',
        r'^(?:delete|remove|cancel) reminder',
    ],
}


def detect_command(text: str) -> Optional[str]:
    """Detect reminder command type"""
    text_lower = text.lower().strip()
    for command, patterns in COMMAND_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return command
    return None


# ---------------------------------------------------------------------------
# Main Command Handler
# ---------------------------------------------------------------------------

class ReminderCommands:
    """Production-grade command handler with all features"""
    
    def __init__(self, manager: ReminderManager):
        self.manager = manager
        self.parser = ReminderParser()
        self.validator = ReminderValidator()
        self.rate_limiter = RateLimiter()
        self.duplicate_detector = DuplicateDetector()
    
    async def handle(
        self,
        command: str,
        text: str,
        whatsapp_id: str,
        chat_id: str
    ) -> Optional[str]:
        """Main entry point - routes to specific handlers"""
        
        if command == 'create':
            return await self._create(text, whatsapp_id, chat_id)
        elif command == 'list':
            return await self._list(whatsapp_id)
        elif command == 'complete':
            return await self._complete(whatsapp_id)
        elif command == 'snooze':
            return await self._snooze(text, whatsapp_id)
        elif command == 'delete':
            return await self._delete(text, whatsapp_id)
        
        return None
    
    async def _create(self, text: str, whatsapp_id: str, chat_id: str) -> str:
        """Create reminder with full validation"""
        try:
            # Rate limiting
            allowed, error = self.rate_limiter.check_limit(
                whatsapp_id,
                'create',
                ReminderConfig.MAX_CREATE_PER_MINUTE
            )
            if not allowed:
                return error
            
            # Check user limit
            active_count = await self.manager.count_active_reminders(whatsapp_id)
            if active_count >= ReminderConfig.MAX_REMINDERS_PER_USER:
                return (
                    f"❌ You've reached the limit of {ReminderConfig.MAX_REMINDERS_PER_USER} active reminders.\n\n"
                    f"_Delete some reminders first: 'list my reminders'_"
                )
            
            # Parse input
            data = self.parser.parse(text)
            
            # Validate title
            data.title = self.validator.sanitize_text(data.title)
            valid, error = self.validator.validate_title(data.title)
            if not valid:
                return error
            
            # Validate datetime
            valid, error = self.validator.validate_datetime(data.due_datetime)
            if not valid:
                return error
            
            # Check for duplicates
            duplicate_id = await self.duplicate_detector.check_duplicate(
                self.manager,
                whatsapp_id,
                data.title,
                data.due_datetime
            )
            
            if duplicate_id:
                return (
                    f"⚠️ *Duplicate detected!*\n\n"
                    f"You already have a similar reminder:\n"
                    f"_{data.title}_\n\n"
                    f"_Reply 'yes' to create anyway, or 'no' to cancel_"
                )
            
            # Smart advance notifications
            now = datetime.now()
            mins_until = (data.due_datetime - now).total_seconds() / 60
            
            if mins_until <= 30:
                advance = []
            elif mins_until <= 120:
                advance = [15]
            elif mins_until <= 480:
                advance = [30]
            else:
                advance = [60]
            
            # Create reminder
            reminder_id = await self.manager.create_reminder(
                whatsapp_id=whatsapp_id,
                chat_id=chat_id,
                title=data.title,
                due_datetime=data.due_datetime,
                recurrence_type=data.recurrence_type,
                recurrence_interval=data.recurrence_interval,
                recurrence_days=data.recurrence_days,
                advance_notifications=advance,
                category=data.category,
                emoji=data.emoji,
                description=data.description,
            )
            
            # Confirmation
            reminder = await self.manager.get_reminder(reminder_id)
            card = format_reminder_card(reminder, header="Reminder Set ✅")
            
            if advance:
                alert_str = f"{advance[0] // 60}h" if advance[0] >= 60 else f"{advance[0]}min"
                card += f"\n🔔 *Alert:* {alert_str} before"
            
            logger.info(
                "reminder.created id=%d user=%s title=%s",
                reminder_id, whatsapp_id, data.title
            )
            return card
        
        except Exception as e:
            logger.error("reminder.create_failed user=%s err=%s", whatsapp_id, str(e), exc_info=True)
            return (
                "❌ I couldn't set that reminder. Try:\n\n"
                "_'Remind me to water plants every 3 days'_\n"
                "_'Remind me to call mom at 5 PM today'_"
            )
    
    async def _list(self, whatsapp_id: str) -> str:
        """List reminders with rate limiting"""
        try:
            # Rate limiting
            allowed, error = self.rate_limiter.check_limit(
                whatsapp_id,
                'list',
                ReminderConfig.MAX_LIST_PER_MINUTE
            )
            if not allowed:
                return error
            
            reminders = await self.manager.list_reminders(
                whatsapp_id,
                status='active',
                limit=20
            )
            
            total_count = await self.manager.count_active_reminders(whatsapp_id)
            return format_reminder_list(reminders, total_count)
        
        except Exception as e:
            logger.error("reminder.list_failed user=%s err=%s", whatsapp_id, str(e))
            return "❌ Couldn't fetch your reminders. Please try again."
    
    async def _complete(self, whatsapp_id: str) -> str:
        """Complete most recent reminder"""
        try:
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=1)
            if not reminders:
                return "No active reminders to complete."
            
            reminder = reminders[0]
            await self.manager.complete_reminder(reminder.id, whatsapp_id)
            
            if reminder.recurrence_type == 'once':
                return f"✅ *Done!* _{reminder.title}_ marked as complete."
            else:
                updated = await self.manager.get_reminder(reminder.id)
                next_due = updated.due_datetime
                date_str = next_due.strftime("%-d %B")
                time_str = next_due.strftime("%I:%M %p").lstrip("0")
                return (
                    f"✅ *Done!* _{reminder.title}_ completed.\n\n"
                    f"📅 *Next:* {date_str} at {time_str}"
                )
        
        except Exception as e:
            logger.error("reminder.complete_failed user=%s err=%s", whatsapp_id, str(e))
            return "❌ Couldn't mark as complete. Try again."
    
    async def _snooze(self, text: str, whatsapp_id: str) -> str:
        """Snooze reminder"""
        try:
            match = re.search(r'(\d+)', text)
            minutes = int(match.group(1)) if match else 30
            
            if minutes > 1440:  # Max 24 hours
                return "❌ Snooze duration cannot exceed 24 hours"
            
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=1)
            if not reminders:
                return "No active reminders to snooze."
            
            reminder = reminders[0]
            snooze_until = await self.manager.snooze_reminder(reminder.id, whatsapp_id, minutes)
            
            if not snooze_until:
                return "❌ Couldn't snooze that reminder."
            
            time_str = snooze_until.strftime("%I:%M %p").lstrip("0")
            return f"😴 *Snoozed!* _{reminder.title}_ → *{time_str}*"
        
        except Exception as e:
            logger.error("reminder.snooze_failed user=%s err=%s", whatsapp_id, str(e))
            return "❌ Couldn't snooze. Try again."
    
    async def _delete(self, text: str, whatsapp_id: str) -> str:
        """Delete reminder with rate limiting"""
        try:
            # Rate limiting
            allowed, error = self.rate_limiter.check_limit(
                whatsapp_id,
                'delete',
                ReminderConfig.MAX_DELETE_PER_MINUTE
            )
            if not allowed:
                return error
            
            # Parse number
            match = re.search(r'(\d+)', text)
            if match:
                index = int(match.group(1))
                reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=50)
                
                if not reminders:
                    return "You have no active reminders to delete."
                
                if index < 1 or index > len(reminders):
                    return f"❌ Number must be between 1 and {len(reminders)}"
                
                reminder = reminders[index - 1]
                await self.manager.cancel_reminder(reminder.id, whatsapp_id)
                return f"🗑️ *Deleted:* _{reminder.title}_"
            
            # Show list if no number provided
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=20)
            return "Which reminder would you like to delete?\n\n" + format_reminder_list(reminders)
        
        except Exception as e:
            logger.error("reminder.delete_failed user=%s err=%s", whatsapp_id, str(e))
            return "❌ Couldn't delete. Try again."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_recurrence(reminder: Reminder) -> str:
    """Format recurrence pattern"""
    t = reminder.recurrence_type
    if t == 'daily':
        return "Every day"
    elif t == 'every_n_days':
        n = reminder.recurrence_interval or 1
        return f"Every {n} day{'s' if n > 1 else ''}"
    elif t == 'weekly':
        return "Every week"
    elif t == 'monthly':
        return "Every month"
    return "Once"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def handle_reminder_command(
    text: str,
    whatsapp_id: str,
    chat_id: str,
    manager: ReminderManager,
) -> Optional[str]:
    """
    Main entry point called from main.py
    Returns response if this was a reminder command, else None
    """
    command = detect_command(text)
    if not command:
        return None
    
    handler = ReminderCommands(manager)
    return await handler.handle(command, text, whatsapp_id, chat_id)
