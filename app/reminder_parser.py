"""
Natural Language Parser for Reminders - FIXED VERSION
Converts user input into structured reminder data

FIX: Properly handle "in X minutes/hours" relative times
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("app.reminder_parser")

# Timezone constant
IST = "Asia/Kolkata"


@dataclass
class ReminderData:
    """Structured reminder information"""
    title: str
    due_datetime: datetime
    recurrence_type: str = 'once'
    recurrence_interval: Optional[int] = None
    recurrence_days: Optional[List[str]] = None
    advance_notifications: List[int] = None  # Minutes before
    category: str = 'personal'
    emoji: str = '⏰'
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.advance_notifications is None:
            self.advance_notifications = [60, 5]  # Default: 1h, 5m


class ReminderParser:
    """Parse natural language into structured reminder data"""
    
    # Category detection with emojis
    CATEGORY_PATTERNS = {
        'home': {
            'emoji': '🏠',
            'keywords': ['plant', 'water', 'garden', 'clean', 'vacuum', 'laundry', 'dishes', 'trash', 'garbage']
        },
        'pets': {
            'emoji': '🐕',
            'keywords': ['dog', 'cat', 'pet', 'feed', 'walk dog', 'vet']
        },
        'health': {
            'emoji': '💊',
            'keywords': ['medicine', 'pill', 'medication', 'doctor', 'dentist', 'appointment', 'gym', 'workout', 'exercise']
        },
        'work': {
            'emoji': '💼',
            'keywords': ['meeting', 'call', 'presentation', 'deadline', 'project', 'report']
        },
        'food': {
            'emoji': '🍽️',
            'keywords': ['cook', 'dinner', 'lunch', 'breakfast', 'meal', 'grocery', 'shopping']
        }
    }
    
    # Recurrence patterns
    RECURRENCE_PATTERNS = [
        (r'every\s+(\d+)\s+days?', 'every_n_days', lambda m: int(m.group(1))),
        (r'daily|every\s+day', 'daily', None),
        (r'weekly|every\s+week', 'weekly', None),
        (r'monthly|every\s+month', 'monthly', None),
        (r'every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'weekly_day', lambda m: m.group(1)),
    ]
    
    # Advance notification patterns
    ADVANCE_PATTERNS = [
        (r'(\d+)\s+(hour|hours|hr|hrs)\s+before', lambda m: int(m.group(1)) * 60),
        (r'(\d+)\s+(minute|minutes|min|mins)\s+before', lambda m: int(m.group(1))),
        (r'remind\s+(\d+)\s+(hour|hours|hr|hrs)\s+and\s+(\d+)\s+(minute|minutes|min|mins)\s+before', 
         lambda m: [int(m.group(1)) * 60, int(m.group(3))]),
    ]
    
    def __init__(self, default_timezone: str = IST):
        self.default_timezone = default_timezone
    
    def parse(self, text: str, context_date: Optional[datetime] = None) -> ReminderData:
        """
        Parse reminder from natural language
        
        Examples:
            "water plants every 3 days"
            "feed dog at 6 PM today"
            "dentist tomorrow at 2:30 PM"
            "gym every Monday at 7 PM"
            "test in 2 minutes"  ← FIXED!
        """
        if context_date is None:
            context_date = datetime.now()
        
        text_lower = text.lower()
        
        # Extract components
        title = self._extract_title(text)
        category, emoji = self._detect_category(text_lower)
        recurrence_type, recurrence_interval, recurrence_days = self._extract_recurrence(text_lower)
        due_datetime = self._extract_datetime(text_lower, context_date)
        advance_notifications = self._extract_advance_alerts(text_lower)
        
        return ReminderData(
            title=title,
            due_datetime=due_datetime,
            recurrence_type=recurrence_type,
            recurrence_interval=recurrence_interval,
            recurrence_days=recurrence_days,
            advance_notifications=advance_notifications,
            category=category,
            emoji=emoji
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract clean title from text"""
        # Remove common prefixes
        text = re.sub(r'^(remind me to|reminder to|remind to|set reminder for|i need to)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(spock,?|hey spock,?|@spock)\s+', '', text, flags=re.IGNORECASE)
        
        # Remove timing/recurrence suffixes
        text = re.sub(r'\s+(every\s+\d+\s+days?|daily|weekly|monthly).*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+(at\s+\d+:?\d*\s*(?:am|pm)?).*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+(in\s+\d+\s+(?:hours?|minutes?|mins?|days?)).*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+(today|tomorrow|tonight).*$', '', text, flags=re.IGNORECASE)
        
        # Remove "remind X before" patterns
        text = re.sub(r',?\s+remind.*before.*$', '', text, flags=re.IGNORECASE)
        
        return text.strip().capitalize()
    
    def _detect_category(self, text: str) -> Tuple[str, str]:
        """Auto-detect category and emoji from keywords"""
        for category, config in self.CATEGORY_PATTERNS.items():
            for keyword in config['keywords']:
                if re.search(r'\b' + keyword + r'\b', text):
                    return category, config['emoji']
        return 'personal', '⏰'
    
    def _extract_recurrence(self, text: str) -> Tuple[str, Optional[int], Optional[List[str]]]:
        """Extract recurrence pattern"""
        for pattern, rec_type, extractor in self.RECURRENCE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if rec_type == 'every_n_days':
                    interval = extractor(match)
                    return 'every_n_days', interval, None
                elif rec_type == 'weekly_day':
                    day = extractor(match)
                    return 'weekly', None, [day]
                else:
                    return rec_type, None, None
        
        return 'once', None, None
    
    def _extract_datetime(self, text: str, base_date: datetime) -> datetime:
        """
        Extract due datetime from text
        
        FIXED: Properly handle relative times like "in 2 minutes", "in 3 hours"
        """
        
        # 🔴 FIX #1: Handle "in X minutes" FIRST (highest priority)
        match = re.search(r'in\s+(\d+)\s+(minute|minutes|min|mins)', text)
        if match:
            minutes = int(match.group(1))
            due = base_date + timedelta(minutes=minutes)
            logger.info("parser.relative_minutes minutes=%d due=%s", minutes, due.isoformat())
            return due
        
        # 🔴 FIX #2: Handle "in X hours"
        match = re.search(r'in\s+(\d+)\s+(hour|hours|hr|hrs)', text)
        if match:
            hours = int(match.group(1))
            due = base_date + timedelta(hours=hours)
            logger.info("parser.relative_hours hours=%d due=%s", hours, due.isoformat())
            return due
        
        # 🔴 FIX #3: Handle "in X days"
        match = re.search(r'in\s+(\d+)\s+(day|days)', text)
        if match:
            days = int(match.group(1))
            # For "in X days", keep the time or use default
            target_date = (base_date + timedelta(days=days)).date()
            
            # Check if there's a time specified
            time_match = re.search(r'at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?', text)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                am_pm = time_match.group(3)
                
                if am_pm:
                    if am_pm.lower() == 'pm' and hour < 12:
                        hour += 12
                    elif am_pm.lower() == 'am' and hour == 12:
                        hour = 0
                
                due = datetime.combine(target_date, base_date.replace(hour=hour, minute=minute, second=0, microsecond=0).time())
            else:
                # Default to 9 AM if no time specified
                due = datetime.combine(target_date, base_date.replace(hour=9, minute=0, second=0, microsecond=0).time())
            
            logger.info("parser.relative_days days=%d due=%s", days, due.isoformat())
            return due
        
        # Handle "today", "tomorrow", "tonight"
        if 'today' in text or 'tonight' in text:
            target_date = base_date.date()
            default_hour = 20 if 'tonight' in text else 9
        elif 'tomorrow' in text:
            target_date = (base_date + timedelta(days=1)).date()
            default_hour = 9
        else:
            # No date keyword found, default to today
            target_date = base_date.date()
            default_hour = 9
        
        # Extract time component
        time_match = re.search(r'at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3)
            
            if am_pm:
                if am_pm.lower() == 'pm' and hour < 12:
                    hour += 12
                elif am_pm.lower() == 'am' and hour == 12:
                    hour = 0
        else:
            # Use default hour
            hour = default_hour
            minute = 0
        
        final_datetime = datetime.combine(target_date, base_date.replace(hour=hour, minute=minute, second=0, microsecond=0).time())
        
        # If datetime is in the past and no explicit date given, assume next occurrence
        if final_datetime < base_date and 'today' not in text and 'tomorrow' not in text and 'in' not in text:
            final_datetime += timedelta(days=1)
        
        logger.info("parser.datetime due=%s", final_datetime.isoformat())
        return final_datetime
    
    def _extract_advance_alerts(self, text: str) -> List[int]:
        """Extract advance notification times"""
        alerts = []
        
        for pattern, extractor in self.ADVANCE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                result = extractor(match)
                if isinstance(result, list):
                    alerts.extend(result)
                else:
                    alerts.append(result)
        
        # If no explicit alerts mentioned, use defaults
        if not alerts:
            alerts = [60, 5]  # 1 hour, 5 minutes
        
        # Sort in descending order (longest first)
        return sorted(set(alerts), reverse=True)


# Convenience function
def parse_reminder(text: str) -> ReminderData:
    """Quick parse function"""
    parser = ReminderParser()
    return parser.parse(text)
