"""
tests/unit/test_database_p1.py

Phase 1 additions to the database unit test suite.
Tests for delete_fact(), delete_facts_batch(), and verify that
P1-FEAT-2 (memory deletion) + P1-GUARD (guardrails) work end-to-end
at the DB layer.

All tests use a tmp SQLite file — no real data affected.
"""
from __future__ import annotations

import asyncio
import pytest

from app.database import SQLiteMemory, DeleteOutcome, normalize_key


SENDER = "919876543210@s.whatsapp.net"


# ─────────────────────────────────────────────────────────────────────────────
# delete_fact — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteFact:
    @pytest.mark.asyncio
    async def test_delete_existing_fact_returns_deleted(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda Civic")
        outcome = await tmp_db.delete_fact(SENDER, "car")
        assert outcome == DeleteOutcome.DELETED

    @pytest.mark.asyncio
    async def test_deleted_fact_is_gone(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda Civic")
        await tmp_db.delete_fact(SENDER, "car")
        facts = await tmp_db.get_all_facts(SENDER)
        assert "car" not in facts

    @pytest.mark.asyncio
    async def test_delete_nonallowlisted_key_returns_blocked(self, tmp_db):
        outcome = await tmp_db.delete_fact(SENDER, "nonexistent_key")
        assert outcome == DeleteOutcome.BLOCKED

    @pytest.mark.asyncio
    async def test_delete_allowlisted_key_not_in_db_returns_not_found(self, tmp_db):
        # "car" is in the allowlist but not stored → NOT_FOUND
        outcome = await tmp_db.delete_fact(SENDER, "car")
        assert outcome == DeleteOutcome.NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_only_affects_target_key(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        await tmp_db.upsert_fact(SENDER, "city", "Hyderabad")
        await tmp_db.delete_fact(SENDER, "car")
        facts = await tmp_db.get_all_facts(SENDER)
        assert "car" not in facts
        assert facts.get("city") == "Hyderabad"

    @pytest.mark.asyncio
    async def test_delete_only_affects_target_sender(self, tmp_db):
        other_sender = "911234567890@s.whatsapp.net"
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        await tmp_db.upsert_fact(other_sender, "car", "Toyota")
        await tmp_db.delete_fact(SENDER, "car")
        other_facts = await tmp_db.get_all_facts(other_sender)
        assert other_facts.get("car") == "Toyota"

    @pytest.mark.asyncio
    async def test_delete_normalizes_key(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "name", "Phani")
        outcome = await tmp_db.delete_fact(SENDER, "user_name")  # alias → "name"
        assert outcome == DeleteOutcome.DELETED
        facts = await tmp_db.get_all_facts(SENDER)
        assert "name" not in facts

    @pytest.mark.asyncio
    async def test_delete_empty_key_returns_empty_key(self, tmp_db):
        outcome = await tmp_db.delete_fact(SENDER, "")
        assert outcome == DeleteOutcome.EMPTY_KEY

    @pytest.mark.asyncio
    async def test_delete_returns_not_found_after_already_deleted(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "bike", "Royal Enfield")
        await tmp_db.delete_fact(SENDER, "bike")
        outcome_again = await tmp_db.delete_fact(SENDER, "bike")
        assert outcome_again == DeleteOutcome.NOT_FOUND

    @pytest.mark.asyncio
    async def test_list_key_returns_needs_confirm_without_confirmed(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "shopping_list", "milk, bread")
        outcome = await tmp_db.delete_fact(SENDER, "shopping_list", confirmed=False)
        assert outcome == DeleteOutcome.NEEDS_CONFIRM
        facts = await tmp_db.get_all_facts(SENDER)
        assert "shopping_list" in facts  # row must still be present

    @pytest.mark.asyncio
    async def test_list_key_deleted_with_confirmed_true(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "shopping_list", "milk, bread")
        outcome = await tmp_db.delete_fact(SENDER, "shopping_list", confirmed=True)
        assert outcome == DeleteOutcome.DELETED
        facts = await tmp_db.get_all_facts(SENDER)
        assert "shopping_list" not in facts

    @pytest.mark.asyncio
    async def test_re_create_fact_after_delete(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        await tmp_db.delete_fact(SENDER, "car")
        status = await tmp_db.upsert_fact(SENDER, "car", "BMW")
        assert status == "created"
        facts = await tmp_db.get_all_facts(SENDER)
        assert facts.get("car") == "BMW"

    @pytest.mark.asyncio
    async def test_return_type_is_delete_outcome(self, tmp_db):
        outcome = await tmp_db.delete_fact(SENDER, "car")
        assert isinstance(outcome, DeleteOutcome)


# ─────────────────────────────────────────────────────────────────────────────
# delete_facts_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteFactsBatch:
    @pytest.mark.asyncio
    async def test_batch_delete_multiple_keys(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        await tmp_db.upsert_fact(SENDER, "bike", "Yamaha")
        await tmp_db.upsert_fact(SENDER, "city", "Hyderabad")

        count, blocked = await tmp_db.delete_facts_batch(SENDER, ["car", "bike"])
        assert count == 2
        assert blocked == []    # no blocked keys

        facts = await tmp_db.get_all_facts(SENDER)
        assert "car" not in facts
        assert "bike" not in facts
        assert "city" in facts  # untouched

    @pytest.mark.asyncio
    async def test_batch_delete_empty_list_returns_zero(self, tmp_db):
        count, blocked = await tmp_db.delete_facts_batch(SENDER, [])
        assert count == 0
        assert blocked == []

    @pytest.mark.asyncio
    async def test_batch_delete_nonallowlisted_keys_blocked(self, tmp_db):
        count, blocked = await tmp_db.delete_facts_batch(SENDER, ["ghost_key1", "ghost_key2"])
        assert count == 0
        assert len(blocked) == 2   # both reported as blocked

    @pytest.mark.asyncio
    async def test_batch_delete_mixed_existing_nonexisting(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        # "bike" is allowlisted but not in DB — still deleted count 1, no blocked
        count, blocked = await tmp_db.delete_facts_batch(SENDER, ["car", "bike"])
        assert count == 1
        assert blocked == []

    @pytest.mark.asyncio
    async def test_batch_delete_normalizes_keys(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "name", "Phani")
        count, blocked = await tmp_db.delete_facts_batch(SENDER, ["user_name"])  # → "name"
        assert count == 1
        assert blocked == []
        assert "name" not in await tmp_db.get_all_facts(SENDER)

    @pytest.mark.asyncio
    async def test_batch_delete_filters_empty_keys(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "city", "Delhi")
        count, blocked = await tmp_db.delete_facts_batch(SENDER, ["", "city", "  "])
        assert count == 1
        assert blocked == []

    @pytest.mark.asyncio
    async def test_batch_list_keys_blocked_without_confirmed(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "shopping_list", "milk")
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        count, blocked = await tmp_db.delete_facts_batch(
            SENDER, ["shopping_list", "car"], confirmed=False
        )
        # car deleted (1), shopping_list blocked (needs confirm)
        assert count == 1
        assert len(blocked) == 1
        assert "shopping_list" in " ".join(blocked)
        facts = await tmp_db.get_all_facts(SENDER)
        assert "shopping_list" in facts  # still there

    @pytest.mark.asyncio
    async def test_batch_list_keys_allowed_with_confirmed(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "shopping_list", "milk")
        await tmp_db.upsert_fact(SENDER, "todo_list", "buy groceries")
        count, blocked = await tmp_db.delete_facts_batch(
            SENDER, ["shopping_list", "todo_list"], confirmed=True
        )
        assert count == 2
        assert blocked == []
        facts = await tmp_db.get_all_facts(SENDER)
        assert "shopping_list" not in facts
        assert "todo_list" not in facts


# ─────────────────────────────────────────────────────────────────────────────
# delete + upsert interaction
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteUpsertInteraction:
    @pytest.mark.asyncio
    async def test_upsert_after_batch_delete(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "car", "Honda")
        await tmp_db.upsert_fact(SENDER, "bike", "Royal Enfield")
        await tmp_db.delete_facts_batch(SENDER, ["car", "bike"])
        status = await tmp_db.upsert_fact(SENDER, "car", "BMW")
        assert status == "created"
        facts = await tmp_db.get_all_facts(SENDER)
        assert facts["car"] == "BMW"
        assert "bike" not in facts

    @pytest.mark.asyncio
    async def test_junk_values_not_returned_after_migrate(self, tmp_db):
        """Junk-value migration should filter 'unknown' etc. at read time."""
        import sqlite3
        import datetime
        from app.database import UTC

        now = datetime.datetime.now(UTC).isoformat()
        with sqlite3.connect(str(tmp_db.path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_memory "
                "(whatsapp_id, fact_key, fact_value, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (SENDER, "junk_key", "unknown", now, now),
            )
            conn.commit()

        facts = await tmp_db.get_all_facts(SENDER)
        assert "junk_key" not in facts
