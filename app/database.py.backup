"""
Smart Long-Term Memory Management
Addresses: Unlimited growth, fact importance, consolidation, archiving
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("app.database")
UTC = timezone.utc


# ============================================================================
# SMART MEMORY STORE
# ============================================================================

@dataclass
class FactMetadata:
    """Enhanced fact with importance scoring"""
    key: str
    value: str
    importance: float  # 0.0 to 1.0
    last_accessed: str
    access_count: int
    created_at: str
    updated_at: str
    category: str  # 'profile', 'preference', 'context', 'temporary'
    

class SmartSQLiteMemory:
    """
    Enhanced SQLite memory with:
    - Fact importance scoring
    - Automatic cleanup of stale facts
    - Consolidation of similar facts
    - Archiving of old data
    """
    
    def __init__(self, path):
        self.path = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=3000")
            
            # Enhanced user_memory table with importance tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    category TEXT DEFAULT 'context',
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
                """
            )
            
            # Archive table for old facts
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    archive_reason TEXT
                )
                """
            )
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON user_memory(whatsapp_id, importance DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_accessed ON user_memory(whatsapp_id, last_accessed DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_category ON user_memory(whatsapp_id, category)")
            
            # Message log (keep existing)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    whatsapp_id TEXT,
                    direction TEXT NOT NULL,
                    event_id TEXT,
                    text TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    UNIQUE(chat_id, event_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_log_chat_ts ON message_log(chat_id, ts DESC)")
            
            conn.commit()
        logger.info("🗄️ smart_sqlite.ready path=%s", self.path)

    # ========================================================================
    # CORE FACT OPERATIONS (with importance tracking)
    # ========================================================================
    
    async def get_all_facts(self, whatsapp_id: str, min_importance: float = 0.0) -> Dict[str, str]:
        """Get all facts above importance threshold"""
        async with self._lock:
            def _do() -> Dict[str, str]:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    # Get facts and update access time
                    cur = conn.execute(
                        """
                        SELECT fact_key, fact_value, importance 
                        FROM user_memory 
                        WHERE whatsapp_id=? AND importance >= ?
                        ORDER BY importance DESC
                        """,
                        (whatsapp_id, min_importance)
                    )
                    facts = {k: v for (k, v, _) in cur.fetchall()}
                    
                    # Update access tracking for retrieved facts
                    if facts:
                        conn.execute(
                            """
                            UPDATE user_memory 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE whatsapp_id = ? AND fact_key IN ({})
                            """.format(','.join('?' * len(facts))),
                            (now, whatsapp_id, *facts.keys())
                        )
                        conn.commit()
                    
                    return facts
            return await asyncio.to_thread(_do)
    
    async def upsert_fact(
        self,
        whatsapp_id: str,
        key: str,
        value: str,
        importance: float = 0.5,
        category: str = "context"
    ) -> str:
        """Upsert fact with importance and category"""
        key = (key or "").strip()
        value = (value or "").strip()
        
        if not key or not value:
            return "unchanged"
        
        # Categorize and score importance automatically
        importance, category = self._calculate_importance(key, value, category)
        
        async with self._lock:
            def _do() -> str:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_value, importance FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                        (whatsapp_id, key)
                    )
                    row = cur.fetchone()
                    
                    if row is None:
                        # New fact
                        conn.execute(
                            """
                            INSERT INTO user_memory 
                            (whatsapp_id, fact_key, fact_value, importance, category, access_count, last_accessed, created_at, updated_at)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (whatsapp_id, key, value, importance, category, 1, now, now, now)
                        )
                        conn.commit()
                        return "created"
                    
                    if (row[0] or "").strip() == value:
                        # Same value, just boost importance slightly
                        new_importance = min(1.0, row[1] * 1.1)
                        conn.execute(
                            "UPDATE user_memory SET importance=?, access_count=access_count+1, last_accessed=? WHERE whatsapp_id=? AND fact_key=?",
                            (new_importance, now, whatsapp_id, key)
                        )
                        conn.commit()
                        return "unchanged"
                    
                    # Value changed - this is important!
                    new_importance = min(1.0, importance * 1.2)
                    conn.execute(
                        """
                        UPDATE user_memory 
                        SET fact_value=?, importance=?, category=?, updated_at=?, last_accessed=?, access_count=access_count+1
                        WHERE whatsapp_id=? AND fact_key=?
                        """,
                        (value, new_importance, category, now, now, whatsapp_id, key)
                    )
                    conn.commit()
                    return "updated"
            
            return await asyncio.to_thread(_do)
    
    def _calculate_importance(self, key: str, value: str, category: str) -> Tuple[float, str]:
        """
        Calculate fact importance and category.
        Returns: (importance_score, category)
        """
        # Profile facts are most important
        profile_keys = {'name', 'email', 'phone', 'age', 'birthday', 'city', 'country', 'location', 'postal_code'}
        if key.lower() in profile_keys:
            return 1.0, 'profile'
        
        # Preferences are quite important
        pref_keys = {'favorite', 'prefer', 'like', 'dislike', 'language', 'timezone', 'currency'}
        if any(k in key.lower() for k in pref_keys):
            return 0.8, 'preference'
        
        # Temporary/session facts are less important
        temp_keys = {'last_query', 'last_search', 'current', 'temp', 'session'}
        if any(k in key.lower() for k in temp_keys):
            return 0.3, 'temporary'
        
        # Default: contextual information
        return 0.5, category or 'context'
    
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    async def cleanup_stale_facts(self, whatsapp_id: str, days_threshold: int = 90) -> int:
        """
        Archive facts not accessed in X days with low importance.
        Returns: number of facts archived
        """
        async with self._lock:
            def _do() -> int:
                cutoff = (datetime.now(UTC) - timedelta(days=days_threshold)).isoformat()
                now = datetime.now(UTC).isoformat()
                
                with sqlite3.connect(self.path) as conn:
                    # Find stale facts (low importance + old access)
                    cur = conn.execute(
                        """
                        SELECT fact_key, fact_value 
                        FROM user_memory
                        WHERE whatsapp_id=? 
                        AND importance < 0.5 
                        AND category IN ('context', 'temporary')
                        AND (last_accessed < ? OR last_accessed IS NULL)
                        """,
                        (whatsapp_id, cutoff)
                    )
                    stale_facts = cur.fetchall()
                    
                    if not stale_facts:
                        return 0
                    
                    # Archive them
                    for key, value in stale_facts:
                        conn.execute(
                            "INSERT INTO user_memory_archive (whatsapp_id, fact_key, fact_value, archived_at, archive_reason) VALUES (?,?,?,?,?)",
                            (whatsapp_id, key, value, now, f"stale_{days_threshold}d")
                        )
                    
                    # Delete from active memory
                    conn.execute(
                        """
                        DELETE FROM user_memory
                        WHERE whatsapp_id=? 
                        AND importance < 0.5 
                        AND category IN ('context', 'temporary')
                        AND (last_accessed < ? OR last_accessed IS NULL)
                        """,
                        (whatsapp_id, cutoff)
                    )
                    
                    conn.commit()
                    return len(stale_facts)
            
            count = await asyncio.to_thread(_do)
            if count > 0:
                logger.info("cleanup.archived whatsapp_id=%s count=%d", whatsapp_id, count)
            return count
    
    async def decay_importance(self, whatsapp_id: str, decay_rate: float = 0.02) -> int:
        """
        Decay importance of facts that haven't been accessed recently.
        Returns: number of facts updated
        """
        async with self._lock:
            def _do() -> int:
                cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
                
                with sqlite3.connect(self.path) as conn:
                    # Decay facts not accessed in last 7 days
                    result = conn.execute(
                        """
                        UPDATE user_memory
                        SET importance = MAX(0.1, importance * (1.0 - ?))
                        WHERE whatsapp_id=?
                        AND (last_accessed < ? OR last_accessed IS NULL)
                        AND category NOT IN ('profile')
                        """,
                        (decay_rate, whatsapp_id, cutoff)
                    )
                    conn.commit()
                    return result.rowcount
            
            return await asyncio.to_thread(_do)
    
    async def consolidate_facts(self, whatsapp_id: str) -> List[str]:
        """
        Identify and merge similar/redundant facts.
        Returns: list of consolidation messages
        """
        # This is complex - you could use embeddings to find similar facts
        # For now, just log what could be consolidated
        
        async with self._lock:
            def _do() -> List[str]:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_key, fact_value FROM user_memory WHERE whatsapp_id=?",
                        (whatsapp_id,)
                    )
                    facts = cur.fetchall()
                    
                    # Simple consolidation: find facts with similar keys
                    consolidations = []
                    seen_bases = set()
                    
                    for key, value in facts:
                        # Extract base key (e.g., "city" from "current_city", "home_city")
                        base = key.split('_')[-1] if '_' in key else key
                        
                        if base in seen_bases and base not in {'id', 'key', 'value'}:
                            consolidations.append(f"Similar keys found for '{base}': {key}")
                        seen_bases.add(base)
                    
                    return consolidations
            
            return await asyncio.to_thread(_do)
    
    async def get_memory_stats(self, whatsapp_id: str) -> Dict[str, any]:
        """Get memory statistics for a user"""
        async with self._lock:
            def _do() -> Dict:
                with sqlite3.connect(self.path) as conn:
                    # Total facts
                    total = conn.execute(
                        "SELECT COUNT(*) FROM user_memory WHERE whatsapp_id=?",
                        (whatsapp_id,)
                    ).fetchone()[0]
                    
                    # By category
                    by_category = {}
                    cur = conn.execute(
                        "SELECT category, COUNT(*) FROM user_memory WHERE whatsapp_id=? GROUP BY category",
                        (whatsapp_id,)
                    )
                    for cat, count in cur.fetchall():
                        by_category[cat] = count
                    
                    # Average importance
                    avg_importance = conn.execute(
                        "SELECT AVG(importance) FROM user_memory WHERE whatsapp_id=?",
                        (whatsapp_id,)
                    ).fetchone()[0] or 0.0
                    
                    # Most important facts
                    top_facts = conn.execute(
                        "SELECT fact_key, importance FROM user_memory WHERE whatsapp_id=? ORDER BY importance DESC LIMIT 5",
                        (whatsapp_id,)
                    ).fetchall()
                    
                    return {
                        'total_facts': total,
                        'by_category': by_category,
                        'avg_importance': round(avg_importance, 2),
                        'top_facts': [{'key': k, 'importance': round(i, 2)} for k, i in top_facts]
                    }
            
            return await asyncio.to_thread(_do)
    
    # ========================================================================
    # EXISTING MESSAGE LOG (unchanged)
    # ========================================================================
    
    async def log_message(self, *, chat_id: str, whatsapp_id: Optional[str], direction: str, text: str, ts: str, event_id: Optional[str] = None) -> None:
        """Log message (unchanged from original)"""
        if not chat_id or not (text or "").strip():
            return
        async with self._lock:
            def _do() -> None:
                with sqlite3.connect(self.path) as conn:
                    try:
                        conn.execute(
                            "INSERT INTO message_log (chat_id, whatsapp_id, direction, event_id, text, ts) VALUES (?,?,?,?,?,?)",
                            (chat_id, whatsapp_id or "", direction, event_id or None, text, ts)
                        )
                        conn.commit()
                    except sqlite3.IntegrityError:
                        pass
            await asyncio.to_thread(_do)


# ============================================================================
# BACKGROUND MAINTENANCE TASK
# ============================================================================

async def memory_maintenance_loop(store: SmartSQLiteMemory):
    """
    Background task to maintain memory health.
    Run this as a scheduled task or background worker.
    """
    while True:
        try:
            # Wait 6 hours between maintenance runs
            await asyncio.sleep(21600)
            
            logger.info("memory_maintenance.start")
            
            # Get all users with memory
            async with store._lock:
                def get_users():
                    with sqlite3.connect(store.path) as conn:
                        cur = conn.execute("SELECT DISTINCT whatsapp_id FROM user_memory")
                        return [row[0] for row in cur.fetchall()]
                users = await asyncio.to_thread(get_users)
            
            for user_id in users:
                # Decay importance
                decayed = await store.decay_importance(user_id, decay_rate=0.02)
                
                # Cleanup stale facts
                archived = await store.cleanup_stale_facts(user_id, days_threshold=90)
                
                if decayed > 0 or archived > 0:
                    logger.info(
                        "memory_maintenance.user user=%s decayed=%d archived=%d",
                        user_id, decayed, archived
                    )
            
            logger.info("memory_maintenance.complete users=%d", len(users))
            
        except Exception:
            logger.exception("memory_maintenance.error")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of how to use the smart memory"""
    
    store = SmartSQLiteMemory("/opt/shimmi/data/shimmi.sqlite")
    
    # Store facts with auto-categorization
    await store.upsert_fact(
        whatsapp_id="user123",
        key="city",
        value="Hyderabad",
        # importance and category calculated automatically
    )
    
    # Get important facts only
    facts = await store.get_all_facts("user123", min_importance=0.7)
    
    # Get memory statistics
    stats = await store.get_memory_stats("user123")
    print(f"Total facts: {stats['total_facts']}")
    print(f"Top facts: {stats['top_facts']}")
    
    # Run maintenance
    archived_count = await store.cleanup_stale_facts("user123", days_threshold=60)
    print(f"Archived {archived_count} stale facts")
    
    # Start background maintenance (in your app startup)
    # asyncio.create_task(memory_maintenance_loop(store))
