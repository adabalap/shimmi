"""
Agent Integration for Reminder Commands
Handles user interaction with the reminder system
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

from .reminder_manager import ReminderManager, Reminder
from .reminder_parser import ReminderParser, ReminderData
from .reminder_scheduler import NOTIFICATION_TEMPLATES

logger = logging.getLogger("app.reminder_commands")


class ReminderCommands:
    """Handle reminder-related user commands"""
    
    # Command detection patterns
    COMMAND_PATTERNS = {
        'create': [
            r'remind me (?:to |about )?(.+)',
            r'reminder (?:to |for |about )?(.+)',
            r'set reminder (?:to |for )?(.+)',
            r'i need to remember (?:to )?(.+)',
            r'don\'t let me forget (?:to )?(.+)',
        ],
        'list': [
            r'(?:list |show |what are )?my reminders',
            r'what reminders (?:do i have|are there)',
            r'show (?:me )?reminders',
            r'reminders?$',
        ],
        'complete': [
            r'^(?:done|finished|completed?)$',
            r'(?:mark |i )?(?:done|finished|completed) (?:the )?(.+)',
            r'i (?:did|finished) (?:the )?(.+)',
        ],
        'snooze': [
            r'snooze (?:for )?(\d+)?\s*(?:min|minutes?|hours?)?',
            r'remind (?:me )?later',
            r'postpone',
        ],
        'cancel': [
            r'cancel (?:the )?reminder (?:for |about )?(.+)',
            r'delete (?:the )?reminder (?:for |about )?(.+)',
            r'remove (?:the )?reminder (?:for |about )?(.+)',
            r'forget (?:about )?(?:the )?(.+)',
        ]
    }
    
    def __init__(self, manager: ReminderManager):
        self.manager = manager
        self.parser = ReminderParser()
    
    def detect_command(self, text: str) -> Optional[str]:
        """Detect which reminder command is being used"""
        text_lower = text.lower().strip()
        
        for command, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return command
        
        return None
    
    async def handle_command(
        self,
        command: str,
        text: str,
        whatsapp_id: str,
        chat_id: str
    ) -> Optional[str]:
        """
        Route command to appropriate handler
        Returns response text if handled, None if not a reminder command
        """
        if command == 'create':
            return await self.create_reminder(text, whatsapp_id, chat_id)
        elif command == 'list':
            return await self.list_reminders(whatsapp_id)
        elif command == 'complete':
            return await self.complete_reminder(text, whatsapp_id)
        elif command == 'snooze':
            return await self.snooze_reminder(text, whatsapp_id)
        elif command == 'cancel':
            return await self.cancel_reminder(text, whatsapp_id)
        
        return None
    
    async def create_reminder(
        self,
        text: str,
        whatsapp_id: str,
        chat_id: str
    ) -> str:
        """Create a new reminder from natural language"""
        try:
            # Parse the reminder
            reminder_data = self.parser.parse(text)
            
            # Create in database
            reminder_id = await self.manager.create_reminder(
                whatsapp_id=whatsapp_id,
                chat_id=chat_id,
                title=reminder_data.title,
                due_datetime=reminder_data.due_datetime,
                recurrence_type=reminder_data.recurrence_type,
                recurrence_interval=reminder_data.recurrence_interval,
                recurrence_days=reminder_data.recurrence_days,
                advance_notifications=reminder_data.advance_notifications,
                category=reminder_data.category,
                emoji=reminder_data.emoji,
                description=reminder_data.description,
            )
            
            # Format confirmation message
            due_str = reminder_data.due_datetime.strftime("%b %d at %I:%M %p")
            
            if reminder_data.recurrence_type == 'once':
                response = f"✅ Reminder set!\n\n{reminder_data.emoji} {reminder_data.title}\n📅 {due_str}"
                
                # Mention advance alerts
                if reminder_data.advance_notifications:
                    alerts = []
                    for mins in sorted(reminder_data.advance_notifications, reverse=True):
                        if mins >= 60:
                            alerts.append(f"{mins//60}h")
                        else:
                            alerts.append(f"{mins}m")
                    response += f"\n🔔 Alerts: {', '.join(alerts)} before"
            
            else:
                # Recurring reminder
                recurrence_text = self._format_recurrence(reminder_data)
                response = f"✅ Recurring reminder set!\n\n{reminder_data.emoji} {reminder_data.title}\n📅 First: {due_str}\n🔄 {recurrence_text}"
            
            logger.info(
                "reminder.created_via_command id=%d user=%s title=%s",
                reminder_id, whatsapp_id, reminder_data.title
            )
            
            return response
            
        except Exception as e:
            logger.error("reminder.create_failed user=%s err=%s", whatsapp_id, str(e))
            return "I had trouble creating that reminder. Could you rephrase it?\n\nExample: 'Remind me to water plants every 3 days'"
    
    async def list_reminders(self, whatsapp_id: str) -> str:
        """List active reminders for user"""
        try:
            reminders = await self.manager.list_reminders(
                whatsapp_id=whatsapp_id,
                status='active',
                limit=10
            )
            
            if not reminders:
                return "You don't have any active reminders.\n\nTry: 'Remind me to water plants in 3 days'"
            
            # Format list
            lines = ["📋 Your active reminders:\n"]
            
            for i, reminder in enumerate(reminders, 1):
                due_str = reminder.due_datetime.strftime("%b %d, %I:%M %p")
                
                if reminder.recurrence_type == 'once':
                    lines.append(f"{i}. {reminder.emoji} {reminder.title}")
                    lines.append(f"   📅 {due_str}")
                else:
                    rec_text = self._format_recurrence_from_reminder(reminder)
                    lines.append(f"{i}. {reminder.emoji} {reminder.title}")
                    lines.append(f"   📅 Next: {due_str}")
                    lines.append(f"   🔄 {rec_text}")
                
                lines.append("")  # Blank line between reminders
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error("reminder.list_failed user=%s err=%s", whatsapp_id, str(e))
            return "I had trouble fetching your reminders. Please try again."
    
    async def complete_reminder(self, text: str, whatsapp_id: str) -> str:
        """Mark a reminder as complete"""
        try:
            # Get active reminders
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=5)
            
            if not reminders:
                return "You don't have any active reminders to complete."
            
            # If just "done" or "completed", complete the most recent
            if re.match(r'^(done|finished|completed?)$', text.lower().strip()):
                reminder = reminders[0]  # Most recent
            else:
                # Try to match reminder by title
                text_lower = text.lower()
                reminder = None
                for r in reminders:
                    if r.title.lower() in text_lower or text_lower in r.title.lower():
                        reminder = r
                        break
                
                if not reminder:
                    return "I couldn't find that reminder. Try: 'list my reminders'"
            
            # Mark as complete
            success = await self.manager.complete_reminder(reminder.id, whatsapp_id)
            
            if not success:
                return "I couldn't complete that reminder. Please try again."
            
            # Format response
            if reminder.recurrence_type == 'once':
                return NOTIFICATION_TEMPLATES['once_complete'].format(title=reminder.title)
            else:
                # Calculate next due
                # (This would need to fetch the updated reminder)
                reminder_updated = await self.manager.get_reminder(reminder.id)
                next_date = reminder_updated.due_datetime.strftime("%b %d at %I:%M %p")
                
                return NOTIFICATION_TEMPLATES['recurring_complete'].format(
                    title=reminder.title,
                    next_date=next_date
                )
            
        except Exception as e:
            logger.error("reminder.complete_failed user=%s err=%s", whatsapp_id, str(e))
            return "I had trouble marking that as complete. Please try again."
    
    async def snooze_reminder(self, text: str, whatsapp_id: str) -> str:
        """Snooze a reminder"""
        try:
            # Extract duration
            match = re.search(r'(\d+)\s*(min|minutes?|hours?|hr|hrs)?', text.lower())
            
            if match:
                number = int(match.group(1))
                unit = match.group(2) or 'minutes'
                
                if 'hour' in unit or 'hr' in unit:
                    minutes = number * 60
                else:
                    minutes = number
            else:
                minutes = 30  # Default: 30 minutes
            
            # Get most recent active reminder
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=1)
            
            if not reminders:
                return "You don't have any active reminders to snooze."
            
            reminder = reminders[0]
            
            # Snooze it
            snooze_until = await self.manager.snooze_reminder(
                reminder.id,
                whatsapp_id,
                minutes
            )
            
            if not snooze_until:
                return "I couldn't snooze that reminder. Please try again."
            
            snooze_str = snooze_until.strftime("%I:%M %p")
            
            return NOTIFICATION_TEMPLATES['snoozed'].format(
                title=reminder.title,
                snooze_time=snooze_str
            )
            
        except Exception as e:
            logger.error("reminder.snooze_failed user=%s err=%s", whatsapp_id, str(e))
            return "I had trouble snoozing that reminder. Please try again."
    
    async def cancel_reminder(self, text: str, whatsapp_id: str) -> str:
        """Cancel a reminder"""
        try:
            # Get active reminders
            reminders = await self.manager.list_reminders(whatsapp_id, 'active', limit=5)
            
            if not reminders:
                return "You don't have any active reminders to cancel."
            
            # Try to match by title
            text_lower = text.lower()
            reminder = None
            for r in reminders:
                if r.title.lower() in text_lower or text_lower in r.title.lower():
                    reminder = r
                    break
            
            if not reminder:
                # If no match, show list
                return await self.list_reminders(whatsapp_id) + "\n\nWhich one would you like to cancel?"
            
            # Cancel it
            success = await self.manager.cancel_reminder(reminder.id, whatsapp_id)
            
            if not success:
                return "I couldn't cancel that reminder. Please try again."
            
            return NOTIFICATION_TEMPLATES['cancelled'].format(title=reminder.title)
            
        except Exception as e:
            logger.error("reminder.cancel_failed user=%s err=%s", whatsapp_id, str(e))
            return "I had trouble cancelling that reminder. Please try again."
    
    # Helper methods
    
    def _format_recurrence(self, data: ReminderData) -> str:
        """Format recurrence pattern for display"""
        if data.recurrence_type == 'daily':
            return "Every day"
        elif data.recurrence_type == 'every_n_days':
            days = data.recurrence_interval
            return f"Every {days} days"
        elif data.recurrence_type == 'weekly':
            if data.recurrence_days:
                days_str = ', '.join(d.capitalize() for d in data.recurrence_days)
                return f"Every {days_str}"
            return "Weekly"
        elif data.recurrence_type == 'monthly':
            return "Monthly"
        else:
            return "Once"
    
    def _format_recurrence_from_reminder(self, reminder: Reminder) -> str:
        """Format recurrence from Reminder object"""
        if reminder.recurrence_type == 'daily':
            return "Every day"
        elif reminder.recurrence_type == 'every_n_days':
            return f"Every {reminder.recurrence_interval} days"
        elif reminder.recurrence_type == 'weekly':
            return "Weekly"
        elif reminder.recurrence_type == 'monthly':
            return "Monthly"
        else:
            return "Once"


# Convenience function for easy integration
async def handle_reminder_command(
    text: str,
    whatsapp_id: str,
    chat_id: str,
    manager: ReminderManager
) -> Optional[str]:
    """
    Quick check if message is a reminder command and handle it
    
    Returns response text if it was a reminder command, None otherwise
    
    Usage in agent_engine.py:
        from .reminder_commands import handle_reminder_command
        
        # Before running normal agent:
        reminder_response = await handle_reminder_command(
            text=user_text,
            whatsapp_id=sender_key,
            chat_id=chat_id,
            manager=reminder_manager
        )
        
        if reminder_response:
            return AgentResult(reply=ReplyPayload(text=reminder_response), memory_updates=[])
    """
    commands = ReminderCommands(manager)
    
    command = commands.detect_command(text)
    if command:
        return await commands.handle_command(command, text, whatsapp_id, chat_id)
    
    return None
