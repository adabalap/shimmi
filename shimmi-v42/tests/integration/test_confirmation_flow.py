"""
tests/integration/test_confirmation_flow.py

Integration tests for the full confirmation flow for high-stakes list deletions:

  Turn 1: User says "clear my shopping list"
          → Orchestrator emits delete=true, confirm=false
          → Backend detects NEEDS_CONFIRM outcome, queues pending delete
          → Bot replies: "Are you sure? Reply yes/no"

  Turn 2a: User says "yes"
           → _check_pending_delete() returns ("confirm", "shopping_list")
           → run_agent() intercepts BEFORE the LLM (zero tokens)
           → Returns MemoryUpdate(delete=True, confirm=True)
           → main.py calls delete_fact(confirmed=True) → DeleteOutcome.DELETED

  Turn 2b: User says "no"
           → _check_pending_delete() returns ("cancel", "shopping_list")
           → Bot confirms the list is kept, no DB change

All LLMs mocked — zero quota.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.database import DeleteOutcome

SENDER   = "919876543210@s.whatsapp.net"
CHAT_ID  = "919876543210-1234567890@g.us"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db_with_list(tmp_db):
    """SQLiteMemory with shopping_list pre-seeded."""
    asyncio.get_event_loop().run_until_complete(
        tmp_db.upsert_fact(SENDER, "shopping_list", "milk, bread, eggs")
    )
    return tmp_db


# ─────────────────────────────────────────────────────────────────────────────
# Turn 1 — list delete queues confirmation
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmationFlowConfirm:
    @pytest.mark.asyncio
    async def test_turn1_list_delete_queues_confirmation(self, db_with_list):
        """
        Turn 1: LLM returns delete shopping_list.
        main.py detects NEEDS_CONFIRM, queues pending-delete,
        rewrites the reply to a confirmation prompt.
        """
        from app.agent_engine import MemoryUpdate, _PENDING_DELETES
        _PENDING_DELETES.clear()

        updates = [
            MemoryUpdate(key="shopping_list", value="", delete=True, confirm=False)
        ]
        reply_text = ["Your shopping list has been cleared."]

        for mu in updates:
            if mu.delete:
                confirmed = mu.confirm  # False for Turn 1
                outcome = await db_with_list.delete_fact(
                    SENDER, mu.key, confirmed=confirmed
                )
                if outcome == DeleteOutcome.NEEDS_CONFIRM:
                    from app.agent_engine import _register_pending_delete
                    current_val = "milk, bread, eggs"
                    _register_pending_delete(SENDER, mu.key, current_val)
                    reply_text[0] = (
                        "⚠️ Are you sure you want to clear your "
                        "*shopping list*? Reply *yes* to confirm or *no* to keep it."
                    )

        # DB unchanged
        facts = await db_with_list.get_all_facts(SENDER)
        assert "shopping_list" in facts
        assert facts["shopping_list"] == "milk, bread, eggs"

        # Pending delete registered
        key = f"{SENDER}:shopping_list"
        assert key in _PENDING_DELETES

        # Reply rewritten to confirmation prompt
        assert "Are you sure" in reply_text[0]

    @pytest.mark.asyncio
    async def test_turn2a_yes_executes_delete(self, db_with_list):
        """
        Turn 2a: User says yes.
        run_agent intercepts before LLM and returns confirmed delete.
        """
        from app.agent_engine import (
            run_agent, _PENDING_DELETES, _register_pending_delete
        )
        _PENDING_DELETES.clear()
        _register_pending_delete(SENDER, "shopping_list", "milk, bread, eggs")

        with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
            result = await run_agent(
                chat_id=CHAT_ID,
                sender_key=SENDER,
                user_text="yes",
                facts={"shopping_list": "milk, bread, eggs"},
                context=[],
                reminders=[],
            )

        assert result.provider_used == "pending_delete"
        assert len(result.memory_updates) == 1
        mu = result.memory_updates[0]
        assert mu.key == "shopping_list"
        assert mu.delete is True
        assert mu.confirm is True  # system-set, not LLM

    @pytest.mark.asyncio
    async def test_turn2a_confirm_delete_fact_executes(self, db_with_list):
        """
        After run_agent returns confirm=True, main.py calls
        delete_fact(confirmed=True) → DeleteOutcome.DELETED.
        """
        from app.agent_engine import MemoryUpdate

        mu = MemoryUpdate(key="shopping_list", value="", delete=True, confirm=True)
        outcome = await db_with_list.delete_fact(
            SENDER, mu.key, confirmed=mu.confirm
        )
        assert outcome == DeleteOutcome.DELETED
        facts = await db_with_list.get_all_facts(SENDER)
        assert "shopping_list" not in facts

    @pytest.mark.asyncio
    async def test_full_two_turn_confirm_flow(self, db_with_list):
        """
        Full end-to-end: Turn 1 queues pending delete.
        Turn 2 (yes) returns MemoryUpdate(confirm=True).
        DB row is deleted.
        """
        from app.agent_engine import (
            run_agent, MemoryUpdate, _PENDING_DELETES, _register_pending_delete
        )
        _PENDING_DELETES.clear()

        # Turn 1: queue the pending delete
        _register_pending_delete(SENDER, "shopping_list", "milk, bread, eggs")

        # Turn 2: user says yes
        with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
            result = await run_agent(
                chat_id=CHAT_ID,
                sender_key=SENDER,
                user_text="yes please",
                facts={"shopping_list": "milk, bread, eggs"},
                context=[],
                reminders=[],
            )

        assert result.provider_used == "pending_delete"
        mu = result.memory_updates[0]
        assert mu.confirm is True

        # Apply the delete (as main.py would)
        outcome = await db_with_list.delete_fact(SENDER, mu.key, confirmed=mu.confirm)
        assert outcome == DeleteOutcome.DELETED

        facts = await db_with_list.get_all_facts(SENDER)
        assert "shopping_list" not in facts


# ─────────────────────────────────────────────────────────────────────────────
# Turn 2b — cancel
# ─────────────────────────────────────────────────────────────────────────────

class TestConfirmationFlowCancel:
    @pytest.mark.asyncio
    async def test_turn2b_no_keeps_list(self, db_with_list):
        """
        Turn 2b: User says no.
        run_agent returns a cancel reply — NO memory_updates, DB unchanged.
        """
        from app.agent_engine import (
            run_agent, _PENDING_DELETES, _register_pending_delete
        )
        _PENDING_DELETES.clear()
        _register_pending_delete(SENDER, "shopping_list", "milk, bread, eggs")

        with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
            result = await run_agent(
                chat_id=CHAT_ID,
                sender_key=SENDER,
                user_text="no",
                facts={"shopping_list": "milk, bread, eggs"},
                context=[],
                reminders=[],
            )

        assert result.provider_used == "pending_delete_cancel"
        assert len(result.memory_updates) == 0

        facts = await db_with_list.get_all_facts(SENDER)
        assert "shopping_list" in facts

    @pytest.mark.asyncio
    async def test_turn2b_nevermind_cancels(self, db_with_list):
        from app.agent_engine import (
            run_agent, _PENDING_DELETES, _register_pending_delete
        )
        _PENDING_DELETES.clear()
        _register_pending_delete(SENDER, "grocery_list", "tomatoes, onions")

        with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
            result = await run_agent(
                chat_id=CHAT_ID,
                sender_key=SENDER,
                user_text="never mind keep it",
                facts={"grocery_list": "tomatoes, onions"},
                context=[],
                reminders=[],
            )

        assert result.provider_used == "pending_delete_cancel"
        assert len(result.memory_updates) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Non-list keys: no confirmation needed
# ─────────────────────────────────────────────────────────────────────────────

class TestNonListDeletion:
    @pytest.mark.asyncio
    async def test_car_deleted_without_confirmation(self, tmp_db):
        """For non-list keys, deletion fires immediately — no confirmation round-trip."""
        await tmp_db.upsert_fact(SENDER, "car", "Honda Civic")

        from app.agent_engine import MemoryUpdate
        mu = MemoryUpdate(key="car", value="", delete=True, confirm=False)
        outcome = await tmp_db.delete_fact(SENDER, mu.key, confirmed=mu.confirm)

        assert outcome == DeleteOutcome.DELETED
        facts = await tmp_db.get_all_facts(SENDER)
        assert "car" not in facts

    @pytest.mark.asyncio
    async def test_city_deleted_without_confirmation(self, tmp_db):
        await tmp_db.upsert_fact(SENDER, "city", "Hyderabad")
        outcome = await tmp_db.delete_fact(SENDER, "city")
        assert outcome == DeleteOutcome.DELETED
        facts = await tmp_db.get_all_facts(SENDER)
        assert "city" not in facts


# ─────────────────────────────────────────────────────────────────────────────
# Group chat isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupChatIsolation:
    @pytest.mark.asyncio
    async def test_each_user_has_independent_shopping_list(self, tmp_db):
        """
        In a group chat, Alice and Bob each have their own shopping_list.
        When Alice clears hers, Bob's is untouched.
        """
        ALICE = "91111@s.whatsapp.net"
        BOB   = "92222@s.whatsapp.net"

        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk, bread")
        await tmp_db.upsert_fact(BOB,   "shopping_list", "eggs, butter")

        # Alice confirms deletion of her list
        outcome = await tmp_db.delete_fact(ALICE, "shopping_list", confirmed=True)
        assert outcome == DeleteOutcome.DELETED

        # Alice's list gone
        alice_facts = await tmp_db.get_all_facts(ALICE)
        assert "shopping_list" not in alice_facts

        # Bob's list untouched
        bob_facts = await tmp_db.get_all_facts(BOB)
        assert bob_facts.get("shopping_list") == "eggs, butter"

    @pytest.mark.asyncio
    async def test_pending_delete_scoped_to_sender(self):
        """
        Pending-delete entries are keyed by sender_key.
        Bob's "yes" cannot confirm Alice's pending delete.
        """
        ALICE = "91111@s.whatsapp.net"
        BOB   = "92222@s.whatsapp.net"

        from app.agent_engine import (
            _register_pending_delete, _check_pending_delete, _PENDING_DELETES
        )
        _PENDING_DELETES.clear()

        _register_pending_delete(ALICE, "shopping_list", "milk")

        bob_result = _check_pending_delete(BOB, "yes")
        assert bob_result is None  # Bob's yes doesn't affect Alice

        alice_result = _check_pending_delete(ALICE, "yes")
        assert alice_result is not None
        assert alice_result[0] == "confirm"
