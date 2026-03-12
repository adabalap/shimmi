"""
database.py — Shimmi v2.9.2

Changes vs v2.7.0:
  ① reminders table + ReminderStore methods
     - add_reminder(), get_due_reminders(), mark_reminder_sent()
     - get_user_reminders(), cancel_reminder()
  ② Dataclass Reminder for typed reminder records
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger("app.database")
UTC    = timezone.utc


# ---------------------------------------------------------------------------
# DeleteOutcome — structured return value replacing fragile string-matching
# ---------------------------------------------------------------------------

class DeleteOutcome(str, Enum):
    """
    Machine-readable outcome from delete_fact() / delete_facts_batch().
    Using str-Enum so it serialises cleanly in logs and JSON.
    """
    DELETED        = "deleted"       # row found and removed ✓
    NOT_FOUND      = "not_found"     # key not in DB (no-op, not an error)
    NEEDS_CONFIRM  = "needs_confirm" # high-stakes key (shopping/todo/grocery list) — must get yes first
    BLOCKED        = "blocked"       # key not in _DELETABLE_KEYS allowlist
    EMPTY_KEY      = "empty_key"     # normalize_key() returned ""

# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

# ── Canonical key map ──────────────────────────────────────────────────────
# Every variant the LLM might hallucinate → single canonical key.
# normalize_key() applies this AFTER stripping user_ prefix, so entries
# here should NOT have the user_ prefix (it's already stripped).
_KEY_ALIASES: Dict[str, str] = {
    # ── name ──────────────────────────────────────────────────────────────
    "username": "name", "first_name": "name", "full_name": "name",
    "user_name": "name", "display_name": "name",

    # ── location ──────────────────────────────────────────────────────────
    "user_city": "city", "user_location": "city",
    "location": "city", "hometown": "city", "current_city": "city",
    "user_country": "country",
    "zip": "postal_code", "zipcode": "postal_code", "pin": "postal_code",
    "pincode": "postal_code",

    # ── color (canonical: favorite_color) ─────────────────────────────────
    "colour": "favorite_color", "favorite_colour": "favorite_color",
    "favourite_color": "favorite_color", "favourite_colour": "favorite_color",
    "preferred_color": "favorite_color", "preferred_colour": "favorite_color",

    # ── drink ─────────────────────────────────────────────────────────────
    "user_favorite_drink": "favorite_drink", "preferred_drink": "favorite_drink",
    "user_drink": "favorite_drink", "drink": "favorite_drink",
    "favourite_drink": "favorite_drink", "fav_drink": "favorite_drink",

    # ── food ──────────────────────────────────────────────────────────────
    "favourite_food": "favorite_food", "fav_food": "favorite_food",
    "preferred_food": "favorite_food",
    "favourite_cuisine": "favorite_cuisine", "fav_cuisine": "favorite_cuisine",
    "preferred_cuisine": "favorite_cuisine",

    # ── interests / hobbies ───────────────────────────────────────────────
    "user_interests": "interests", "user_interest": "interests",
    "interest": "interests", "passion": "interests", "passions": "interests",
    "technical_interests": "interests",
    "user_hobby": "hobbies", "user_hobbies": "hobbies",
    "hobby": "hobbies",

    # ── occupation / work ─────────────────────────────────────────────────
    "user_occupation": "occupation", "user_job": "occupation",
    "job": "occupation", "profession": "occupation", "role": "occupation",
    "job_title": "occupation", "current_job_title": "occupation",
    "work": "occupation",
    "employer": "company", "current_company": "company",
    "workplace": "company", "work_place": "company",

    # ── education ─────────────────────────────────────────────────────────
    "educational_background": "education",
    "degree_background": "education",
    "school": "education", "college": "education",

    # ── fitness / health ──────────────────────────────────────────────────
    "fitness_goal": "fitness_goals", "fitness_target": "fitness_goals",
    "health_goal": "fitness_goals", "health_goals": "fitness_goals",
    "marathon_goal": "fitness_goals",

    # ── travel ────────────────────────────────────────────────────────────
    "travel_plan": "travel_plans", "next_trip": "travel_plans",
    "upcoming_trip": "travel_plans",

    # ── pets ──────────────────────────────────────────────────────────────
    "pet": "pets", "pet_name": "pets",

    # ── vehicle ───────────────────────────────────────────────────────────
    "vehicle": "car",

    # ── books / reading ───────────────────────────────────────────────────
    "book": "recent_book", "books": "recent_book",
    "books_read": "recent_book", "current_book": "recent_book",
    "reading": "recent_book", "last_book": "recent_book",

    # ── lists ─────────────────────────────────────────────────────────────
    "grocery": "grocery_list", "groceries": "grocery_list",
    "shopping": "shopping_list",
    "todo": "todo_list", "todos": "todo_list", "task": "todo_list",

    # ── language ──────────────────────────────────────────────────────────
    "user_language": "preferred_language", "language": "preferred_language",
    "lang": "preferred_language",

    # ── age ───────────────────────────────────────────────────────────────
    "user_age": "age",

    # ── career / goals ────────────────────────────────────────────────────
    "career_goal": "career_goals", "career_aspiration": "career_goals",
    "career_aspirations": "career_goals",
    "goal": "personal_goals", "life_goal": "personal_goals",
}

_SPECIAL_PREFIXES = ("_reminder", "_cancel_reminder")

# ---------------------------------------------------------------------------
# Deletion guardrails (P1-GUARD)
# ---------------------------------------------------------------------------

# Only keys in this set may be deleted by the agent.
# System keys, context keys, and anything not explicitly listed are blocked.
_DELETABLE_KEYS: frozenset[str] = frozenset({
    # Identity
    "name", "age",
    # Location
    "city", "country", "postal_code",
    # Occupation / personal
    "occupation",
    # Preferences
    "favorite_drink", "favorite_food", "favorite_cuisine",
    "favorite_color", "favorite_trail",
    "hobbies", "interests",
    "dietary_restriction", "allergies",
    # Vehicles
    "car", "bike", "vehicle",
    # Pets
    "pets",
    # Lists — allowed but require confirmation (see _CONFIRM_BEFORE_DELETE)
    "shopping_list", "grocery_list", "todo_list",
    # Misc personal
    "motivational_quote", "preferred_language",
    # reminder_notes is personal but excluded — deleting it wouldn't cancel
    # the actual reminder rows; handled separately via cancel_reminder()
})

# Subset of _DELETABLE_KEYS that are destructive enough to require the agent
# to include a confirm=True flag (set by orchestrator) before deletion fires.
# Without confirm=True these keys are BLOCKED even if in _DELETABLE_KEYS.
_CONFIRM_BEFORE_DELETE: frozenset[str] = frozenset({
    "shopping_list",
    "grocery_list",
    "todo_list",
})

# Keys that are structurally protected — can never be deleted via agent.
_PROTECTED_KEYS: frozenset[str] = frozenset({
    "whatsapp_id", "chat_id",   # should never be stored as facts, but guard anyway
})


def is_key_deletable(key: str, *, confirmed: bool = False) -> tuple[bool, str]:
    """
    Check whether a normalized fact key may be deleted.

    Args:
        key:       Normalized fact key (output of normalize_key()).
        confirmed: True when the user has explicitly confirmed a destructive
                   delete (e.g. "yes, clear my shopping list").

    Returns:
        (allowed: bool, reason: str)
        reason is a human-readable explanation used for logging and the
        agent reply when a delete is blocked.
    """
    if not key:
        return False, "empty key"
    if key in _PROTECTED_KEYS:
        return False, f"key '{key}' is system-protected and cannot be deleted"
    if key not in _DELETABLE_KEYS:
        return False, (
            f"key '{key}' is not in the deletable-keys allowlist; "
            "only personal facts can be deleted"
        )
    if key in _CONFIRM_BEFORE_DELETE and not confirmed:
        return False, (
            f"deleting '{key}' requires explicit confirmation — "
            "set confirm=true after the user says yes"
        )
    return True, "ok"


def normalize_key(raw: str) -> str:
    k = (raw or "").strip().lower()
    k = re.sub(r"[\s\-]+", "_", k)
    k = re.sub(r"[^\w]", "", k)
    k = re.sub(r"_+", "_", k)
    if k.startswith("user_"):
        k = k[5:]
    k = k.strip("_")
    if not k or k == "user":
        return ""
    return _KEY_ALIASES.get(k, k)


# ---------------------------------------------------------------------------
# Reminder dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reminder:
    id:            int
    whatsapp_id:   str
    chat_id:       str
    reminder_text: str
    trigger_iso:   str          # ISO 8601 with tz offset, e.g. "2026-03-09T06:00+05:30"
    created_at:    str
    sent_at:       Optional[str] = None
    cancelled:     bool          = False
    failed:        bool          = False
    user_name:     str           = ""  # populated from facts at send time

    @property
    def status(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.failed:
            return "failed"
        if self.sent_at:
            return "sent"
        return "pending"


# ---------------------------------------------------------------------------
# SQLite memory + reminder store
# ---------------------------------------------------------------------------

class SQLiteMemory:
    def __init__(self, path):
        self.path  = str(path)
        self._lock = asyncio.Lock()
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")

            # ── user facts KV ──────────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    whatsapp_id  TEXT NOT NULL,
                    fact_key     TEXT NOT NULL,
                    fact_value   TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    PRIMARY KEY (whatsapp_id, fact_key)
                )
            """)
            self._migrate(conn)

            # ── message log ───────────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id     TEXT NOT NULL,
                    whatsapp_id TEXT,
                    direction   TEXT NOT NULL,
                    event_id    TEXT,
                    text        TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    UNIQUE(chat_id, event_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_message_log_chat_ts "
                "ON message_log(chat_id, ts DESC)"
            )

            # ── reminders ─────────────────────────────────────────────────
            # NOTE: _migrate() may have already rebuilt this table from an older
            # schema. CREATE TABLE IF NOT EXISTS is a no-op in that case — the
            # migrated table with the correct schema is already in place.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    whatsapp_id    TEXT NOT NULL,
                    chat_id        TEXT NOT NULL,
                    reminder_text  TEXT NOT NULL,
                    trigger_iso    TEXT NOT NULL,
                    created_at     TEXT NOT NULL,
                    sent_at        TEXT,
                    cancelled      INTEGER NOT NULL DEFAULT 0,
                    failed         INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_trigger "
                "ON reminders(trigger_iso) WHERE sent_at IS NULL AND cancelled=0 AND failed=0"
            )
            conn.commit()

        logger.info("🗄️  sqlite.ready  path=%s", self.path)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        # ── user_memory migrations ─────────────────────────────────────────
        um_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_memory)").fetchall()
        }
        if "created_at" not in um_cols:
            logger.info("🗄️  migrate — adding created_at to user_memory")
            conn.execute("ALTER TABLE user_memory ADD COLUMN created_at TEXT")
            conn.execute(
                "UPDATE user_memory SET created_at = updated_at WHERE created_at IS NULL"
            )
            conn.commit()
            logger.info("🗄️  migrate — created_at back-filled ✓")
        if "updated_at" not in um_cols:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                f"ALTER TABLE user_memory ADD COLUMN updated_at TEXT DEFAULT '{now}'"
            )
            conn.commit()

        # ── FIX-P0-4: delete junk fact values persisted by early bot versions ──
        _JUNK = ("unknown", "none", "null", "n/a", "-", "")
        placeholders = ",".join("?" * len(_JUNK))
        conn.execute(
            f"DELETE FROM user_memory WHERE LOWER(TRIM(fact_value)) IN ({placeholders})",
            _JUNK,
        )
        n_junk = conn.execute("SELECT changes()").fetchone()[0]
        if n_junk:
            conn.commit()
            logger.info("🗄️  migrate — deleted %d junk fact rows (unknown/none/null)", n_junk)

        # ── FIX-P2: consolidate duplicate keys caused by LLM key sprawl ──────
        # For each user, identify rows whose fact_key normalizes to the same
        # canonical key.  Keep the most-recently-updated row, delete the rest.
        rows = conn.execute(
            "SELECT whatsapp_id, fact_key, fact_value, updated_at FROM user_memory"
        ).fetchall()
        # Group by (whatsapp_id, canonical_key); keep latest
        from collections import defaultdict
        best: dict = {}   # (whatsapp_id, canon) → (raw_key, value, updated_at)
        for wid, raw_key, val, upd in rows:
            canon = normalize_key(raw_key)
            if not canon:
                continue
            k = (wid, canon)
            if k not in best or (upd or "") > (best[k][2] or ""):
                best[k] = (raw_key, val, upd)
        # Delete rows that are not the keeper for their canonical key
        deleted_dupes = 0
        for wid, raw_key, val, upd in rows:
            canon = normalize_key(raw_key)
            if not canon:
                continue
            keeper_raw = best[(wid, canon)][0]
            if raw_key != keeper_raw:
                conn.execute(
                    "DELETE FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                    (wid, raw_key),
                )
                deleted_dupes += 1
        # Rename keeper rows to their canonical key if different
        renamed = 0
        for (wid, canon), (raw_key, val, upd) in best.items():
            if raw_key != canon:
                # Check if canonical key row already exists (shouldn't after dedup but be safe)
                existing = conn.execute(
                    "SELECT 1 FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                    (wid, canon),
                ).fetchone()
                if existing:
                    conn.execute(
                        "DELETE FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                        (wid, raw_key),
                    )
                else:
                    conn.execute(
                        "UPDATE user_memory SET fact_key=? WHERE whatsapp_id=? AND fact_key=?",
                        (canon, wid, raw_key),
                    )
                renamed += 1
        if deleted_dupes or renamed:
            conn.commit()
            logger.info(
                "🗄️  migrate — key_dedup: removed %d duplicate rows, canonicalized %d keys",
                deleted_dupes, renamed,
            )

        # ── reminders table migration ──────────────────────────────────────
        # v2.9.0 canonical schema uses cancelled/failed INTEGER booleans.
        # Any prior variant (no trigger_iso, status TEXT, user_name TEXT, etc.)
        # must be rebuilt. We check for the EXACT v2.9.0 columns; anything
        # missing triggers a rename+rebuild so queries never hit missing columns.
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "reminders" in tables:
            rem_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(reminders)").fetchall()
            }
            # v2.9.0 requires cancelled + failed boolean columns.
            # Any older schema (status TEXT, missing trigger_iso, etc.) is rebuilt.
            required = {"cancelled", "failed", "trigger_iso",
                        "chat_id", "whatsapp_id", "reminder_text", "created_at"}
            if not required.issubset(rem_cols):
                logger.info(
                    "🗄️  migrate — reminders schema outdated (missing: %s), rebuilding",
                    ", ".join(sorted(required - rem_cols)),
                )
                # Drop any partial index that references missing columns
                conn.execute("DROP INDEX IF EXISTS idx_reminders_trigger")
                # Rename old table so we can recreate with new schema
                conn.execute("ALTER TABLE reminders RENAME TO reminders_old")
                # Create the new schema (v2.9.0: cancelled/failed booleans)
                conn.execute("""
                    CREATE TABLE reminders (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        whatsapp_id    TEXT NOT NULL,
                        chat_id        TEXT NOT NULL,
                        reminder_text  TEXT NOT NULL,
                        trigger_iso    TEXT NOT NULL,
                        created_at     TEXT NOT NULL,
                        sent_at        TEXT,
                        cancelled      INTEGER NOT NULL DEFAULT 0,
                        failed         INTEGER NOT NULL DEFAULT 0
                    )
                """)
                # Copy rows that have the columns we need (best-effort).
                # If old table has trigger_iso we copy it; otherwise use a
                # sentinel past-date so the scheduler marks it stale+drops it.
                if "chat_id" in rem_cols and "reminder_text" in rem_cols:
                    now_iso     = datetime.now(UTC).isoformat()
                    wa_col      = "whatsapp_id" if "whatsapp_id" in rem_cols else "''"
                    ca_col      = "created_at"  if "created_at"  in rem_cols else f"'{now_iso}'"
                    trigger_col = "trigger_iso" if "trigger_iso" in rem_cols else "'2000-01-01T00:00:00+00:00'"
                    conn.execute(
                        f"""INSERT INTO reminders
                            (whatsapp_id, chat_id, reminder_text,
                             trigger_iso, created_at, failed)
                           SELECT {wa_col}, chat_id, reminder_text,
                                  {trigger_col}, {ca_col}, 1
                           FROM reminders_old
                           WHERE reminder_text IS NOT NULL"""
                    )
                    logger.info("🗄️  migrate — reminders rows copied (marked failed/stale)")
                conn.execute("DROP TABLE reminders_old")
                conn.commit()
                logger.info("🗄️  migrate — reminders table rebuilt ✓")

    # ── facts API ──────────────────────────────────────────────────────────

    async def get_all_facts(self, whatsapp_id: str) -> Dict[str, str]:
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_key, fact_value FROM user_memory "
                        "WHERE whatsapp_id=? ORDER BY updated_at DESC",
                        (whatsapp_id,),
                    )
                    _JUNK = frozenset({"unknown", "none", "null", "n/a", "-", ""})
                    out: Dict[str, str] = {}
                    for raw_key, val in cur.fetchall():
                        if (val or "").strip().lower() in _JUNK:  # FIX-P0-4
                            continue
                        canon = normalize_key(raw_key)
                        if canon and canon not in out:
                            out[canon] = val
                    return out
            return await asyncio.to_thread(_do)

    async def upsert_fact(self, whatsapp_id: str, raw_key: str, value: str) -> str:
        key   = normalize_key(raw_key)
        value = (value or "").strip()
        if not key or not value:
            return "unchanged"
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "SELECT fact_value FROM user_memory "
                        "WHERE whatsapp_id=? AND fact_key=?",
                        (whatsapp_id, key),
                    )
                    row = cur.fetchone()
                    if row is None:
                        conn.execute(
                            "INSERT INTO user_memory "
                            "(whatsapp_id, fact_key, fact_value, created_at, updated_at) "
                            "VALUES (?,?,?,?,?)",
                            (whatsapp_id, key, value, now, now),
                        )
                        conn.commit()
                        return "created"
                    if (row[0] or "").strip() == value:
                        return "unchanged"
                    conn.execute(
                        "UPDATE user_memory SET fact_value=?, updated_at=? "
                        "WHERE whatsapp_id=? AND fact_key=?",
                        (value, now, whatsapp_id, key),
                    )
                    conn.commit()
                    return "updated"
            return await asyncio.to_thread(_do)

    async def log_message(
        self, *, chat_id, whatsapp_id, direction, text, ts, event_id=None,
    ) -> None:
        if not chat_id or not (text or "").strip():
            return
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    try:
                        conn.execute(
                            "INSERT INTO message_log "
                            "(chat_id, whatsapp_id, direction, event_id, text, ts) "
                            "VALUES (?,?,?,?,?,?)",
                            (chat_id, whatsapp_id or "", direction, event_id or None, text, ts),
                        )
                        conn.commit()
                    except sqlite3.IntegrityError:
                        pass
            await asyncio.to_thread(_do)

    # ── reminders API ──────────────────────────────────────────────────────

    async def add_reminder(
        self,
        whatsapp_id: str,
        chat_id:     str,
        text:        str,
        trigger_iso: str,
    ) -> int:
        """Insert a new reminder. Returns its auto-increment id."""
        async with self._lock:
            def _do() -> int:
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "INSERT INTO reminders "
                        "(whatsapp_id, chat_id, reminder_text, trigger_iso, created_at) "
                        "VALUES (?,?,?,?,?)",
                        (whatsapp_id, chat_id, text.strip(), trigger_iso.strip(), now),
                    )
                    conn.commit()
                    return cur.lastrowid
            return await asyncio.to_thread(_do)

    async def get_user_reminders(
        self, whatsapp_id: str, include_sent: bool = False,
    ) -> List[Reminder]:
        """Return reminders for a user (pending by default, or all)."""
        async with self._lock:
            def _do() -> List[Reminder]:
                with sqlite3.connect(self.path) as conn:
                    conn.row_factory = sqlite3.Row
                    base = (
                        "SELECT * FROM reminders WHERE whatsapp_id=? "
                        "ORDER BY trigger_iso ASC"
                    )
                    rows = conn.execute(base, (whatsapp_id,)).fetchall()
                    out = []
                    for row in rows:
                        r = Reminder(
                            id=row["id"], whatsapp_id=row["whatsapp_id"],
                            chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                            trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                            sent_at=row["sent_at"], cancelled=bool(row["cancelled"]),
                            failed=bool(row["failed"]),
                        )
                        if include_sent or r.status == "pending":
                            out.append(r)
                    return out
            return await asyncio.to_thread(_do)

    # ── P1-FEAT-2 + P1-GUARD: memory deletion with guardrails ────────────

    async def delete_fact(
        self,
        whatsapp_id: str,
        raw_key: str,
        *,
        confirmed: bool = False,
    ) -> DeleteOutcome:
        """
        P1-FEAT-2 + P1-GUARD: Delete a single fact from long-term memory.

        Enforces three layers of protection before touching the database:
          1. Key must normalize to a non-empty string.
          2. Key must be in _DELETABLE_KEYS allowlist (only personal facts).
          3. High-stakes list keys (shopping_list / grocery_list / todo_list)
             require confirmed=True — the caller must obtain the user's yes first.

        ISOLATION GUARANTEE:
            whatsapp_id is the authenticated sender key from the WAHA webhook
            payload — it is set in main.py BEFORE the LLM is called and is never
            taken from LLM output.  This means:
              • Alice can only delete her own facts.
              • Bob messaging "delete Alice's shopping list" produces a delete
                scoped to BOB's row, not Alice's.
              • No user in a group chat can delete another member's facts.

        Args:
            whatsapp_id: Authenticated sender key (from WAHA webhook, never from LLM).
            raw_key:     Fact key (normalized internally via normalize_key()).
            confirmed:   Must be True for _CONFIRM_BEFORE_DELETE keys.
                         Pass True only after the user has explicitly said yes.

        Returns:
            DeleteOutcome enum value:
              DELETED       — row found and removed successfully
              NOT_FOUND     — key not in DB (no-op, not an error)
              NEEDS_CONFIRM — list key, needs user confirmation first
              BLOCKED       — key not in allowlist or otherwise protected
              EMPTY_KEY     — raw_key normalized to ""
        """
        key = normalize_key(raw_key)
        if not key:
            logger.warning(
                "delete_fact.empty_key  sender=%s  raw_key=%r", whatsapp_id, raw_key
            )
            return DeleteOutcome.EMPTY_KEY

        allowed, reason = is_key_deletable(key, confirmed=confirmed)
        if not allowed:
            # Distinguish "needs confirmation" from a hard block so callers
            # can give the user a confirmation prompt vs. a "can't do that" reply.
            if key in _CONFIRM_BEFORE_DELETE and not confirmed:
                logger.info(
                    "⏳ delete_fact.needs_confirm  sender=%s  key=%s", whatsapp_id, key
                )
                return DeleteOutcome.NEEDS_CONFIRM
            logger.warning(
                "🚫 delete_fact.blocked  sender=%s  key=%s  reason=%s",
                whatsapp_id, key, reason,
            )
            return DeleteOutcome.BLOCKED

        async with self._lock:
            def _do() -> bool:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "DELETE FROM user_memory WHERE whatsapp_id=? AND fact_key=?",
                        (whatsapp_id, key),
                    )
                    conn.commit()
                    return cur.rowcount > 0

            deleted = await asyncio.to_thread(_do)

        if deleted:
            logger.info(
                "🗑️  fact.deleted  sender=%s  key=%s  confirmed=%s",
                whatsapp_id, key, confirmed,
            )
            return DeleteOutcome.DELETED
        else:
            logger.debug(
                "🗑️  fact.delete_noop  sender=%s  key=%s  (key not in DB)",
                whatsapp_id, key,
            )
            return DeleteOutcome.NOT_FOUND

    async def delete_facts_batch(
        self,
        whatsapp_id: str,
        raw_keys: List[str],
        *,
        confirmed: bool = False,
    ) -> Tuple[int, Dict[str, DeleteOutcome]]:
        """
        P1-FEAT-2 + P1-GUARD: Delete multiple facts in a single transaction.

        Each key is individually checked against the allowlist + confirmation
        requirement before any DB write occurs.

        Args:
            whatsapp_id: Authenticated sender key (from WAHA webhook, never from LLM).
            raw_keys:    List of raw fact keys to delete.
            confirmed:   Pass True for _CONFIRM_BEFORE_DELETE keys (list keys).

        Returns:
            (deleted_count: int,  outcomes: Dict[normalized_key, DeleteOutcome])
            outcomes contains an entry for every input key — callers can inspect
            individual outcomes to surface precise messages to the user.
        """
        outcomes: Dict[str, DeleteOutcome] = {}
        allowed_keys: List[str] = []

        for rk in raw_keys:
            key = normalize_key(rk)
            if not key:
                continue
            ok, reason = is_key_deletable(key, confirmed=confirmed)
            if ok:
                allowed_keys.append(key)
            else:
                outcome = (
                    DeleteOutcome.NEEDS_CONFIRM
                    if (key in _CONFIRM_BEFORE_DELETE and not confirmed)
                    else DeleteOutcome.BLOCKED
                )
                outcomes[key] = outcome
                logger.warning(
                    "🚫 delete_facts_batch.blocked  sender=%s  key=%s  outcome=%s",
                    whatsapp_id, key, outcome,
                )

        if not allowed_keys:
            return 0, outcomes

        async with self._lock:
            def _do() -> int:
                with sqlite3.connect(self.path) as conn:
                    placeholders = ",".join("?" * len(allowed_keys))
                    cur = conn.execute(
                        f"DELETE FROM user_memory "
                        f"WHERE whatsapp_id=? AND fact_key IN ({placeholders})",
                        (whatsapp_id, *allowed_keys),
                    )
                    conn.commit()
                    n = cur.rowcount
                    logger.info(
                        "🗑️  facts.batch_deleted  sender=%s  keys=%s  count=%d",
                        whatsapp_id, allowed_keys, n,
                    )
                    return n

        n = await asyncio.to_thread(_do)
        for key in allowed_keys:
            outcomes[key] = DeleteOutcome.DELETED
        return n, outcomes

    async def cancel_reminder(self, reminder_id: int) -> bool:
        """Mark a reminder as cancelled. Returns True if found."""
        async with self._lock:
            def _do() -> bool:
                with sqlite3.connect(self.path) as conn:
                    cur = conn.execute(
                        "UPDATE reminders SET cancelled=1 WHERE id=? AND sent_at IS NULL",
                        (reminder_id,),
                    )
                    conn.commit()
                    return cur.rowcount > 0
            return await asyncio.to_thread(_do)

    async def get_due_reminders(self) -> List[Reminder]:
        """Return all unfired, non-cancelled reminders whose trigger time <= now (UTC).

        IMPORTANT: Uses timezone-aware Python comparison, NOT SQL string comparison.
        SQL string comparison of ISO timestamps with different offsets is incorrect:
          '2026-03-09T13:00:00+05:30' > '2026-03-09T11:49:00+00:00'  ← string says NOT due
        but 13:00 IST = 07:30 UTC which IS before 11:49 UTC.
        """
        async with self._lock:
            def _do() -> List[Reminder]:
                now_utc = datetime.now(UTC)
                with sqlite3.connect(self.path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM reminders "
                        "WHERE sent_at IS NULL AND cancelled=0 AND failed=0 "
                        "ORDER BY trigger_iso ASC",
                    ).fetchall()
                    result = []
                    for row in rows:
                        try:
                            # Parse ISO with timezone offset (e.g. +05:30)
                            trigger_dt = datetime.fromisoformat(row["trigger_iso"])
                            if trigger_dt.tzinfo is None:
                                trigger_dt = trigger_dt.replace(tzinfo=UTC)
                            # Convert to UTC for correct comparison
                            trigger_utc = trigger_dt.astimezone(UTC)
                            if trigger_utc <= now_utc:
                                result.append(Reminder(
                                    id=row["id"], whatsapp_id=row["whatsapp_id"],
                                    chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                                    trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                                    sent_at=row["sent_at"],
                                ))
                        except (ValueError, TypeError):
                            # Bad trigger_iso — still return it so scheduler can mark failed
                            result.append(Reminder(
                                id=row["id"], whatsapp_id=row["whatsapp_id"],
                                chat_id=row["chat_id"], reminder_text=row["reminder_text"],
                                trigger_iso=row["trigger_iso"], created_at=row["created_at"],
                                sent_at=row["sent_at"],
                            ))
                    return result
            return await asyncio.to_thread(_do)

    async def mark_reminder_sent(self, reminder_id: int) -> None:
        async with self._lock:
            def _do():
                now = datetime.now(UTC).isoformat()
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "UPDATE reminders SET sent_at=? WHERE id=?", (now, reminder_id)
                    )
                    conn.commit()
            await asyncio.to_thread(_do)

    async def mark_reminder_failed(self, reminder_id: int) -> None:
        async with self._lock:
            def _do():
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "UPDATE reminders SET failed=1 WHERE id=?", (reminder_id,)
                    )
                    conn.commit()
            await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self._model.encode(
            input, batch_size=32, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return emb.astype("float32").tolist()


@dataclass
class ContextSnippet:
    id:       str
    text:     str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class ChromaAmbient:
    def __init__(self, persist_dir, collection_name: str, embed_model: str):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=SentenceTransformerEmbedding(embed_model),
        )
        # FIX-P0-1: SentenceTransformer's Rust tokenizer is NOT re-entrant.
        # Concurrent asyncio.to_thread() calls → "RuntimeError: Already borrowed".
        # This lock serialises all embedding operations (upsert + query + recent).
        self._embed_lock = asyncio.Lock()
        logger.info(
            "📚 chroma.ready  dir=%s  collection=%s",
            str(persist_dir), collection_name,
        )

    async def add_message(self, *, chat_id, whatsapp_id, direction, text, ts, message_id):
        if not (chat_id and (text or "").strip()):
            return
        doc_id = f"{chat_id}:{message_id}:{direction}"
        meta   = {"chat_id": chat_id, "whatsapp_id": whatsapp_id, "direction": direction, "ts": ts}
        async with self._embed_lock:  # FIX-P0-1: serialise embed calls
            await asyncio.to_thread(
                lambda: self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
            )

    async def search(self, *, chat_id, query, k):
        async with self._embed_lock:  # FIX-P0-1: serialise embed calls
            res = await asyncio.to_thread(
                lambda: self.collection.query(query_texts=[query], n_results=k, where={"chat_id": chat_id})
            )
        return [
            ContextSnippet(id=_id, text=doc, metadata=meta or {}, distance=dist)
            for _id, doc, meta, dist in zip(
                res.get("ids",[[]])[0], res.get("documents",[[]])[0],
                res.get("metadatas",[[]])[0], res.get("distances",[[None]*k])[0],
            )
        ]

    async def recent_window(self, *, chat_id, k):
        async with self._embed_lock:  # FIX-P0-1: serialise embed calls
            res = await asyncio.to_thread(
                lambda: self.collection.get(
                    where={"chat_id": chat_id}, limit=max(50, k * 5),
                    include=["documents", "metadatas"],
                )
            )
        items = [
            ContextSnippet(id=_id, text=doc, metadata=meta or {})
            for _id, doc, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            )
        ]
        items.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
        return items[:k]


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

sqlite_store: Optional[SQLiteMemory] = None
chroma_store: Optional[ChromaAmbient] = None


def init_stores() -> None:
    global sqlite_store, chroma_store
    sqlite_store = SQLiteMemory(settings.sqlite_path)
    if settings.chroma_enabled:
        chroma_store = ChromaAmbient(
            settings.chroma_dir, settings.chroma_collection, settings.chroma_embed_model,
        )


# ---------------------------------------------------------------------------
# Module-level convenience helpers (P1-FEAT-2)
# ---------------------------------------------------------------------------

async def upsert_fact(whatsapp_id: str, raw_key: str, value: str) -> str:
    """Module-level shorthand that forwards to the global sqlite_store."""
    if sqlite_store is None:
        raise RuntimeError("sqlite_store not initialised — call init_stores() first")
    return await sqlite_store.upsert_fact(whatsapp_id, raw_key, value)


async def delete_fact(
    whatsapp_id: str,
    raw_key: str,
    *,
    confirmed: bool = False,
) -> "DeleteOutcome":
    """
    P1-FEAT-2 + P1-GUARD: Module-level shorthand for guarded fact deletion.

    Forwards to sqlite_store.delete_fact(). Returns a DeleteOutcome enum so
    callers can branch on the result without string-matching.
    """
    if sqlite_store is None:
        raise RuntimeError("sqlite_store not initialised — call init_stores() first")
    return await sqlite_store.delete_fact(whatsapp_id, raw_key, confirmed=confirmed)
