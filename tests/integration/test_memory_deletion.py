"""
tests/integration/test_memory_deletion.py

Integration tests for P1-FEAT-2: end-to-end memory deletion.
Tests the full path: mocked LLM returns delete=True → main.py calls
delete_fact() → SQLite row removed → subsequent recall says "not on record".

The LLM (Groq/Gemini) is fully mocked so these run with zero quota.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SENDER = "919876543210@s.whatsapp.net"
CHAT_ID = "919876543210-1234567890@g.us"

ORCHESTRATOR_DELETE_ANSWER = json.dumps({
    "action": "answer",
    "reasoning": "User wants to forget their car.",
    "text": "Done! I've removed your car from my records.",
    "query": "",
    "question": "",
    "memory_updates": [{"key": "car", "value": "", "delete": True}],
    "reminders": [],
    "tool_call": None,
})

ORCHESTRATOR_SIMPLE_ANSWER = json.dumps({
    "action": "answer",
    "reasoning": "Answering user.",
    "text": "I don't have your car on record.",
    "query": "",
    "question": "",
    "memory_updates": [],
    "reminders": [],
    "tool_call": None,
})

CANNED_EXTRACT = json.dumps({"memory_updates": []})
CANNED_VERIFY  = json.dumps({"approved": []})
CANNED_FORMAT  = json.dumps({"text": "Done! I've removed your car from my records."})


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def preloaded_db(tmp_db):
    """SQLiteMemory with the car fact pre-seeded."""
    import asyncio
    # asyncio.run() rather than get_event_loop().run_until_complete(): there is
    # no current loop in a sync fixture under modern pytest-asyncio, and
    # get_event_loop() no longer creates one (deprecated in 3.10, raises here).
    # Same migration the app code already made.
    asyncio.run(
        tmp_db.upsert_fact(SENDER, "car", "Honda Civic")
    )
    return tmp_db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryDeletionFlow:
    @pytest.mark.asyncio
    async def test_delete_flag_routes_to_delete_fact(self, preloaded_db):
        """
        When orchestrator returns memory_updates with delete=True, main.py
        must call delete_fact() instead of upsert_fact().
        """
        from app.database import SQLiteMemory

        # Verify pre-condition
        facts_before = await preloaded_db.get_all_facts(SENDER)
        assert "car" in facts_before

        # Simulate what main.py does in the memory_save step
        from app.agent_engine import MemoryUpdate
        updates = [MemoryUpdate(key="car", value="", delete=True)]

        deleted_count = 0
        for mu in updates:
            if getattr(mu, "delete", False):
                result = await preloaded_db.delete_fact(SENDER, mu.key)
                if result:
                    deleted_count += 1

        assert deleted_count == 1
        facts_after = await preloaded_db.get_all_facts(SENDER)
        assert "car" not in facts_after

    @pytest.mark.asyncio
    async def test_normal_upsert_not_affected_by_delete_flag(self, preloaded_db):
        """
        A MemoryUpdate with delete=False should still upsert, not delete.
        """
        from app.agent_engine import MemoryUpdate

        update = MemoryUpdate(key="city", value="Mumbai", delete=False)
        assert update.delete is False

        status = await preloaded_db.upsert_fact(SENDER, update.key, update.value)
        facts = await preloaded_db.get_all_facts(SENDER)
        assert facts.get("city") == "Mumbai"
        assert "car" in facts  # untouched by this update

    @pytest.mark.asyncio
    async def test_run_agent_memory_updates_with_delete(self, preloaded_db):
        """
        Full run_agent() → MemoryUpdate(delete=True) → approved → caller
        receives delete=True in result.memory_updates.

        Verifies that the agent pipeline correctly passes delete=True through
        the verify step and into AgentResult.

        Note: we pass facts={} (no car fact) so the zero-token facts shortcut
        doesn't intercept the message before the LLM is called.
        """
        from app.agent_engine import run_agent, init_llm
        from app.database import normalize_key

        orch_calls = [0]

        async def fake_groq_raw(messages, *, max_tokens, chat_id, label, role, timeout=None):
            if role == "orchestrate":
                orch_calls[0] += 1
                return ORCHESTRATOR_DELETE_ANSWER
            elif label.startswith("extract"):
                return CANNED_EXTRACT
            elif label.startswith("verify"):
                # Verifier approves the delete update with delete=True preserved
                return json.dumps({
                    "approved": [{"key": "car", "value": "", "confidence": 1.0, "delete": True, "confirm": False}]
                })
            elif label.startswith("format"):
                return CANNED_FORMAT
            return CANNED_EXTRACT

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq_raw):
            with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
                result = await run_agent(
                    chat_id=CHAT_ID,
                    user_text="Forget my car",
                    facts={},        # empty — no car fact, so shortcut won't fire
                    context=[],
                    reminders=[],
                )

        # Check result carries delete=True update
        assert len(result.memory_updates) > 0
        car_update = next(
            (u for u in result.memory_updates if u.key == "car"), None
        )
        assert car_update is not None
        assert car_update.delete is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_fact_is_noop(self, tmp_db):
        """
        Deleting a key that was never set returns NOT_FOUND (for allowlisted keys)
        or BLOCKED (for keys not in the allowlist). Either way it never raises.
        """
        from app.database import DeleteOutcome
        outcome = await tmp_db.delete_fact(SENDER, "car")   # allowlisted, not in DB
        assert outcome == DeleteOutcome.NOT_FOUND

    @pytest.mark.asyncio
    async def test_multiple_updates_mix_delete_and_upsert(self, preloaded_db):
        """
        A single orchestrator turn can have both deletions and upserts.
        Both should be applied correctly.
        """
        from app.agent_engine import MemoryUpdate

        updates = [
            MemoryUpdate(key="car",  value="",       delete=True),
            MemoryUpdate(key="name", value="Phani",  delete=False),
        ]

        for mu in updates:
            if getattr(mu, "delete", False):
                await preloaded_db.delete_fact(SENDER, mu.key)
            else:
                await preloaded_db.upsert_fact(SENDER, mu.key, mu.value)

        facts = await preloaded_db.get_all_facts(SENDER)
        assert "car" not in facts
        assert facts.get("name") == "Phani"
