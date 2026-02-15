"""
Structured Actions Module
Handles lists, reminders, todos, notes
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("app.actions")
UTC = timezone.utc


class StructuredActionsStore:
    """Manages structured data like lists, reminders, todos"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._init_tables()
    
    def _init_tables(self):
        """Create tables for structured data"""
        with sqlite3.connect(self.db_path) as conn:
            # Lists (shopping lists, todo lists, etc.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_lists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id TEXT NOT NULL,
                    list_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(whatsapp_id, list_name)
                )
            """)
            
            # List items
            conn.execute("""
                CREATE TABLE IF NOT EXISTS list_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    list_id INTEGER NOT NULL,
                    item_text TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    added_at TEXT NOT NULL,
                    FOREIGN KEY(list_id) REFERENCES user_lists(id) ON DELETE CASCADE
                )
            """)
            
            # Reminders
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id TEXT NOT NULL,
                    reminder_text TEXT NOT NULL,
                    remind_at TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL
                )
            """)
            
            # Todos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id TEXT NOT NULL,
                    todo_text TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'normal',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            
            conn.commit()
        logger.info("✅ structured_actions.init tables_created")
    
    # Lists methods
    async def create_list(self, whatsapp_id: str, list_name: str) -> Dict:
        """Create a new list"""
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.db_path) as conn:
                    try:
                        conn.execute(
                            "INSERT INTO user_lists (whatsapp_id, list_name, created_at) VALUES (?,?,?)",
                            (whatsapp_id, list_name, now)
                        )
                        conn.commit()
                        return {"status": "created", "list_name": list_name}
                    except sqlite3.IntegrityError:
                        return {"status": "exists", "list_name": list_name}
            
            return await asyncio.to_thread(_do)
    
    async def add_to_list(self, whatsapp_id: str, list_name: str, items: List[str]) -> Dict:
        """Add items to a list"""
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.db_path) as conn:
                    # Get or create list
                    cur = conn.execute(
                        "SELECT id FROM user_lists WHERE whatsapp_id=? AND list_name=?",
                        (whatsapp_id, list_name)
                    )
                    row = cur.fetchone()
                    
                    if not row:
                        conn.execute(
                            "INSERT INTO user_lists (whatsapp_id, list_name, created_at) VALUES (?,?,?)",
                            (whatsapp_id, list_name, now)
                        )
                        list_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    else:
                        list_id = row[0]
                    
                    # Add items
                    for item in items:
                        conn.execute(
                            "INSERT INTO list_items (list_id, item_text, added_at) VALUES (?,?,?)",
                            (list_id, item, now)
                        )
                    
                    conn.commit()
                    return {"status": "added", "count": len(items), "list_name": list_name}
            
            return await asyncio.to_thread(_do)
    
    async def get_lists(self, whatsapp_id: str) -> List[Dict]:
        """Get all lists for a user"""
        async with self._lock:
            def _do():
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute("""
                        SELECT ul.list_name, COUNT(li.id) as item_count
                        FROM user_lists ul
                        LEFT JOIN list_items li ON ul.id = li.list_id AND li.status='active'
                        WHERE ul.whatsapp_id=?
                        GROUP BY ul.id, ul.list_name
                    """, (whatsapp_id,))
                    
                    return [{"name": row[0], "count": row[1]} for row in cur.fetchall()]
            
            return await asyncio.to_thread(_do)
    
    async def get_list_items(self, whatsapp_id: str, list_name: str) -> List[str]:
        """Get items in a specific list"""
        async with self._lock:
            def _do():
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute("""
                        SELECT li.item_text
                        FROM list_items li
                        JOIN user_lists ul ON li.list_id = ul.id
                        WHERE ul.whatsapp_id=? AND ul.list_name=? AND li.status='active'
                        ORDER BY li.added_at DESC
                    """, (whatsapp_id, list_name))
                    
                    return [row[0] for row in cur.fetchall()]
            
            return await asyncio.to_thread(_do)


# Global instance (initialized in main.py)
actions_store: Optional[StructuredActionsStore] = None


def init_actions_store(db_path: str):
    """Initialize the actions store"""
    global actions_store
    actions_store = StructuredActionsStore(db_path)
    logger.info("✅ actions.init ready")
