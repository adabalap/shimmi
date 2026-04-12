"""
tests/unit/test_delete_guardrails.py  — P1-GUARD comprehensive tests

Covers:
  ① is_key_deletable()        allowlist + confirmation requirement
  ② DeleteOutcome enum        all five states
  ③ delete_fact() guardrails  blocked / needs_confirm / not_found / deleted
  ④ delete_facts_batch()      mixed outcomes per key
  ⑤ Sender isolation          Alice cannot affect Bob's facts
  ⑥ Pending-delete cache      register / confirm / cancel / expire / isolate
  ⑦ run_agent() intercept     yes/no bypass LLM; unrelated messages don't
  ⑧ main.py branch logic      NEEDS_CONFIRM rewrites reply; BLOCKED logs error
  ⑨ Prompt completeness       LLM told about deletable keys + confirm=false rule
"""
from __future__ import annotations

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.database import (
    DeleteOutcome,
    SQLiteMemory,
    normalize_key,
    is_key_deletable,
    _DELETABLE_KEYS,
    _CONFIRM_BEFORE_DELETE,
    _PROTECTED_KEYS,
)

ALICE = "911111111111@s.whatsapp.net"
BOB   = "922222222222@s.whatsapp.net"
CHAT  = "919876543210@s.whatsapp.net"  # DM chat — no prefix required for yes/no


# ─────────────────────────────────────────────────────────────────────────────
# ① is_key_deletable()
# ─────────────────────────────────────────────────────────────────────────────

class TestIsKeyDeletable:

    def test_car_is_deletable(self):
        ok, reason = is_key_deletable("car")
        assert ok is True and reason == "ok"

    def test_name_is_deletable(self):
        ok, _ = is_key_deletable("name")
        assert ok is True

    def test_all_deletable_keys_pass(self):
        for key in _DELETABLE_KEYS:
            confirmed = key in _CONFIRM_BEFORE_DELETE
            ok, reason = is_key_deletable(key, confirmed=confirmed)
            assert ok is True, f"Expected {key!r} deletable; reason={reason!r}"

    def test_shopping_list_blocked_without_confirm(self):
        ok, reason = is_key_deletable("shopping_list", confirmed=False)
        assert ok is False
        assert "confirm" in reason.lower()

    def test_grocery_list_blocked_without_confirm(self):
        ok, _ = is_key_deletable("grocery_list", confirmed=False)
        assert ok is False

    def test_todo_list_blocked_without_confirm(self):
        ok, _ = is_key_deletable("todo_list", confirmed=False)
        assert ok is False

    def test_shopping_list_allowed_with_confirm(self):
        ok, _ = is_key_deletable("shopping_list", confirmed=True)
        assert ok is True

    def test_grocery_list_allowed_with_confirm(self):
        ok, _ = is_key_deletable("grocery_list", confirmed=True)
        assert ok is True

    def test_todo_list_allowed_with_confirm(self):
        ok, _ = is_key_deletable("todo_list", confirmed=True)
        assert ok is True

    def test_unknown_freeform_key_blocked(self):
        ok, reason = is_key_deletable("next_meeting_topic")
        assert ok is False
        assert "allowlist" in reason.lower() or "not in" in reason.lower()

    def test_reminder_notes_blocked(self):
        ok, _ = is_key_deletable("reminder_notes")
        assert ok is False

    def test_empty_key_blocked(self):
        ok, reason = is_key_deletable("")
        assert ok is False and "empty" in reason.lower()

    def test_protected_keys_always_blocked(self):
        for key in _PROTECTED_KEYS:
            ok, reason = is_key_deletable(key)
            assert ok is False, f"Protected key {key!r} must always be blocked"

    def test_injection_style_keys_blocked(self):
        for bad in ("whatsapp_id", "chat_id", "admin", "__proto__", "system"):
            ok, _ = is_key_deletable(bad)
            assert ok is False, f"{bad!r} must be blocked"


# ─────────────────────────────────────────────────────────────────────────────
# ② DeleteOutcome enum
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteOutcome:

    def test_string_values(self):
        assert DeleteOutcome.DELETED       == "deleted"
        assert DeleteOutcome.NOT_FOUND     == "not_found"
        assert DeleteOutcome.NEEDS_CONFIRM == "needs_confirm"
        assert DeleteOutcome.BLOCKED       == "blocked"
        assert DeleteOutcome.EMPTY_KEY     == "empty_key"

    def test_equality_check(self):
        outcome = DeleteOutcome.DELETED
        assert outcome == DeleteOutcome.DELETED
        assert outcome != DeleteOutcome.BLOCKED

    def test_groupable_in_tuple(self):
        outcome = DeleteOutcome.BLOCKED
        assert outcome in (DeleteOutcome.BLOCKED, DeleteOutcome.EMPTY_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# ③ SQLiteMemory.delete_fact() guardrails
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteFactGuardrails:

    @pytest.mark.asyncio
    async def test_returns_deleted(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car", "Honda")
        assert await tmp_db.delete_fact(ALICE, "car") == DeleteOutcome.DELETED

    @pytest.mark.asyncio
    async def test_returns_not_found(self, tmp_db):
        assert await tmp_db.delete_fact(ALICE, "car") == DeleteOutcome.NOT_FOUND

    @pytest.mark.asyncio
    async def test_returns_blocked_for_non_allowlisted_key(self, tmp_db):
        import sqlite3, datetime
        from app.database import UTC
        now = datetime.datetime.now(UTC).isoformat()
        with sqlite3.connect(str(tmp_db.path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_memory "
                "(whatsapp_id, fact_key, fact_value, created_at, updated_at) VALUES (?,?,?,?,?)",
                (ALICE, "next_meeting_topic", "Q2 Planning", now, now),
            )
            conn.commit()
        outcome = await tmp_db.delete_fact(ALICE, "next_meeting_topic")
        assert outcome == DeleteOutcome.BLOCKED
        facts = await tmp_db.get_all_facts(ALICE)
        assert "next_meeting_topic" in facts  # NOT deleted

    @pytest.mark.asyncio
    async def test_shopping_list_needs_confirm_without_flag(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk, bread")
        outcome = await tmp_db.delete_fact(ALICE, "shopping_list", confirmed=False)
        assert outcome == DeleteOutcome.NEEDS_CONFIRM
        facts = await tmp_db.get_all_facts(ALICE)
        assert facts.get("shopping_list") == "milk, bread"  # NOT deleted

    @pytest.mark.asyncio
    async def test_shopping_list_deleted_with_confirmed(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk, bread")
        outcome = await tmp_db.delete_fact(ALICE, "shopping_list", confirmed=True)
        assert outcome == DeleteOutcome.DELETED
        assert "shopping_list" not in await tmp_db.get_all_facts(ALICE)

    @pytest.mark.asyncio
    async def test_grocery_list_needs_confirm(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "grocery_list", "eggs")
        assert await tmp_db.delete_fact(ALICE, "grocery_list") == DeleteOutcome.NEEDS_CONFIRM

    @pytest.mark.asyncio
    async def test_todo_list_needs_confirm(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "todo_list", "gym")
        assert await tmp_db.delete_fact(ALICE, "todo_list") == DeleteOutcome.NEEDS_CONFIRM

    @pytest.mark.asyncio
    async def test_empty_key_returns_empty_key_outcome(self, tmp_db):
        assert await tmp_db.delete_fact(ALICE, "") == DeleteOutcome.EMPTY_KEY

    @pytest.mark.asyncio
    async def test_key_normalization_applied(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car", "Honda")
        outcome = await tmp_db.delete_fact(ALICE, "user_car")  # normalizes to "car"
        assert outcome == DeleteOutcome.DELETED
        assert "car" not in await tmp_db.get_all_facts(ALICE)

    @pytest.mark.asyncio
    async def test_protected_key_blocked(self, tmp_db):
        assert await tmp_db.delete_fact(ALICE, "whatsapp_id") == DeleteOutcome.BLOCKED

    @pytest.mark.asyncio
    async def test_reminder_notes_blocked(self, tmp_db):
        assert await tmp_db.delete_fact(ALICE, "reminder_notes") == DeleteOutcome.BLOCKED

    @pytest.mark.asyncio
    async def test_re_create_after_delete(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car", "Honda")
        await tmp_db.delete_fact(ALICE, "car")
        status = await tmp_db.upsert_fact(ALICE, "car", "BMW")
        assert status == "created"
        assert (await tmp_db.get_all_facts(ALICE)).get("car") == "BMW"


# ─────────────────────────────────────────────────────────────────────────────
# ④ Sender isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestSenderIsolation:

    @pytest.mark.asyncio
    async def test_delete_scoped_to_sender(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car", "Honda Civic")
        await tmp_db.upsert_fact(BOB,   "car", "Toyota Camry")
        await tmp_db.delete_fact(ALICE, "car")
        bob_facts = await tmp_db.get_all_facts(BOB)
        assert bob_facts.get("car") == "Toyota Camry"

    @pytest.mark.asyncio
    async def test_wrong_sender_cannot_delete_others_list(self, tmp_db):
        await tmp_db.upsert_fact(BOB, "shopping_list", "milk, eggs, bread")
        # Alice "tries" to delete Bob's list but uses wrong sender_key
        outcome = await tmp_db.delete_fact(ALICE, "shopping_list", confirmed=True)
        assert outcome == DeleteOutcome.NOT_FOUND  # Alice has no shopping_list
        bob_facts = await tmp_db.get_all_facts(BOB)
        assert bob_facts.get("shopping_list") == "milk, eggs, bread"

    @pytest.mark.asyncio
    async def test_three_users_independent(self, tmp_db):
        charlie = "933333333333@s.whatsapp.net"
        for user, city in [(ALICE, "Hyderabad"), (BOB, "Mumbai"), (charlie, "Chennai")]:
            await tmp_db.upsert_fact(user, "city", city)
        await tmp_db.delete_fact(BOB, "city")
        assert (await tmp_db.get_all_facts(ALICE)).get("city") == "Hyderabad"
        assert (await tmp_db.get_all_facts(charlie)).get("city") == "Chennai"
        assert "city" not in await tmp_db.get_all_facts(BOB)


# ─────────────────────────────────────────────────────────────────────────────
# ④ delete_facts_batch() — mixed outcomes
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteFactsBatch:
    # Note: delete_facts_batch returns (int, List[str]) where the list
    # contains blocked-reason strings for any keys that could not be deleted.

    @pytest.mark.asyncio
    async def test_returns_per_key_outcomes(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car",  "Honda")
        await tmp_db.upsert_fact(ALICE, "bike", "Yamaha")
        count, blocked = await tmp_db.delete_facts_batch(
            ALICE, ["car", "bike", "shopping_list", "next_meeting_topic"]
        )
        assert count == 2
        assert len(blocked) == 2   # shopping_list + next_meeting_topic blocked
        blocked_str = " ".join(blocked)
        assert "shopping_list"      in blocked_str
        assert "next_meeting_topic" in blocked_str

    @pytest.mark.asyncio
    async def test_batch_with_confirmed_deletes_list(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk")
        count, blocked = await tmp_db.delete_facts_batch(ALICE, ["shopping_list"], confirmed=True)
        assert count == 1
        assert blocked == []
        assert "shopping_list" not in await tmp_db.get_all_facts(ALICE)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, tmp_db):
        count, blocked = await tmp_db.delete_facts_batch(ALICE, [])
        assert count == 0
        assert blocked == []

    @pytest.mark.asyncio
    async def test_batch_scoped_to_sender(self, tmp_db):
        await tmp_db.upsert_fact(ALICE, "car", "Honda")
        await tmp_db.upsert_fact(BOB,   "car", "Toyota")
        count, _ = await tmp_db.delete_facts_batch(ALICE, ["car"])
        assert count == 1
        assert (await tmp_db.get_all_facts(BOB)).get("car") == "Toyota"


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Pending-delete cache
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingDeleteCache:

    def setup_method(self):
        from app.agent_engine import _PENDING_DELETES
        _PENDING_DELETES.clear()

    def test_register_and_confirm(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk, bread")
        assert _check_pending_delete(ALICE, "yes") == ("confirm", "shopping_list")

    def test_register_and_cancel(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete
        _register_pending_delete(ALICE, "grocery_list", "eggs")
        assert _check_pending_delete(ALICE, "no") == ("cancel", "grocery_list")

    def test_all_confirm_words(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete, _PENDING_DELETES
        for phrase in ["yeah", "sure", "ok", "go ahead", "do it", "yep", "definitely"]:
            _PENDING_DELETES.clear()
            _register_pending_delete(ALICE, "shopping_list", "x")
            result = _check_pending_delete(ALICE, phrase)
            assert result is not None and result[0] == "confirm", f"'{phrase}' should confirm"

    def test_all_cancel_words(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete, _PENDING_DELETES
        for phrase in ["no", "nope", "cancel", "keep it", "never mind"]:
            _PENDING_DELETES.clear()
            _register_pending_delete(ALICE, "shopping_list", "x")
            result = _check_pending_delete(ALICE, phrase)
            assert result is not None and result[0] == "cancel", f"'{phrase}' should cancel"

    def test_unrelated_message_returns_none(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk")
        assert _check_pending_delete(ALICE, "What time is it?") is None

    def test_isolated_by_sender(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk")
        assert _check_pending_delete(BOB, "yes") is None

    def test_consumed_after_confirm(self):
        from app.agent_engine import _register_pending_delete, _check_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk")
        _check_pending_delete(ALICE, "yes")
        assert _check_pending_delete(ALICE, "yes") is None

    def test_expires_after_ttl(self):
        from app.agent_engine import (
            _register_pending_delete, _check_pending_delete,
            _PENDING_DELETES, _PENDING_DELETE_TTL,
        )
        _register_pending_delete(ALICE, "shopping_list", "milk")
        key = f"{ALICE}:shopping_list"
        val, _ = _PENDING_DELETES[key]
        _PENDING_DELETES[key] = (val, time.monotonic() - _PENDING_DELETE_TTL - 1)
        assert _check_pending_delete(ALICE, "yes") is None


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ run_agent() pending-delete intercept
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgentPendingDeleteIntercept:

    def setup_method(self):
        from app.agent_engine import _PENDING_DELETES
        _PENDING_DELETES.clear()

    @pytest.mark.asyncio
    async def test_yes_returns_delete_update_without_llm(self):
        from app.agent_engine import run_agent, _register_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk, bread")

        llm_called = []
        async def fake_groq(*a, **kw):
            llm_called.append(True)
            return '{"action":"answer","text":"x","memory_updates":[],"reminders":[]}'

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq):
            result = await run_agent(
                chat_id=CHAT, user_text="yes",
                facts={"shopping_list": "milk, bread"},
                context=[], reminders=[], sender_key=ALICE,
            )

        assert not llm_called, "LLM must NOT be called for pending-delete yes"
        assert len(result.memory_updates) == 1
        mu = result.memory_updates[0]
        assert mu.key == "shopping_list" and mu.delete is True and mu.confirm is True
        assert result.provider_used == "pending_delete"

    @pytest.mark.asyncio
    async def test_no_returns_cancel_without_llm(self):
        from app.agent_engine import run_agent, _register_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk, bread")

        with patch("app.agent_engine._groq_raw", new_callable=AsyncMock) as mock:
            result = await run_agent(
                chat_id=CHAT, user_text="no",
                facts={"shopping_list": "milk, bread"},
                context=[], reminders=[], sender_key=ALICE,
            )
        mock.assert_not_called()
        assert result.memory_updates == []
        assert result.provider_used == "pending_delete_cancel"

    @pytest.mark.asyncio
    async def test_unrelated_message_still_calls_llm(self):
        from app.agent_engine import run_agent, _register_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk")

        llm_called = []
        async def fake_groq(msgs, *, max_tokens, chat_id, label, role, timeout=None):
            llm_called.append(label)
            return '{"action":"answer","text":"ok","memory_updates":[],"reminders":[]}'

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq):
            await run_agent(
                chat_id=CHAT, user_text="What's the weather?",
                facts={}, context=[], reminders=[], sender_key=ALICE,
            )

        assert any("orchestrate" in c for c in llm_called)

    @pytest.mark.asyncio
    async def test_bob_yes_does_not_trigger_alice_pending(self):
        from app.agent_engine import run_agent, _register_pending_delete
        _register_pending_delete(ALICE, "shopping_list", "milk")

        llm_called = []
        async def fake_groq(msgs, *, max_tokens, chat_id, label, role, timeout=None):
            llm_called.append(True)
            return '{"action":"answer","text":"ok","memory_updates":[],"reminders":[]}'

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq):
            result = await run_agent(
                chat_id=CHAT, user_text="yes",
                facts={}, context=[], reminders=[], sender_key=BOB,
            )
        assert llm_called, "LLM should be called for BOB who has no pending delete"
        assert result.provider_used != "pending_delete"


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ main.py branch logic
# ─────────────────────────────────────────────────────────────────────────────

class TestMainBranchLogic:

    @pytest.mark.asyncio
    async def test_needs_confirm_rewrites_reply_and_registers_pending(self, tmp_db):
        from app.agent_engine import AgentResult, ReplyPayload, MemoryUpdate, _PENDING_DELETES, _register_pending_delete
        from app.database import DeleteOutcome
        _PENDING_DELETES.clear()

        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk, bread")
        fake_result = AgentResult(
            reply=ReplyPayload(type="text", text="Clearing now..."),
            memory_updates=[MemoryUpdate(key="shopping_list", value="", delete=True, confirm=False)],
        )
        facts = {"shopping_list": "milk, bread"}

        outcome = await tmp_db.delete_fact(ALICE, "shopping_list", confirmed=False)
        assert outcome == DeleteOutcome.NEEDS_CONFIRM

        # Simulate main.py NEEDS_CONFIRM branch
        if outcome == DeleteOutcome.NEEDS_CONFIRM:
            _register_pending_delete(ALICE, "shopping_list", facts.get("shopping_list", ""))
            key_label = "shopping_list".replace("_", " ")
            confirm_text = (
                f"⚠️ Are you sure you want to clear your "
                f"*{key_label}*? Reply *yes* to confirm or *no* to keep it."
            )
            object.__setattr__(fake_result.reply, "text", confirm_text)

        assert "Are you sure" in fake_result.reply.text
        assert f"{ALICE}:shopping_list" in _PENDING_DELETES

    @pytest.mark.asyncio
    async def test_blocked_key_adds_to_save_errors(self, tmp_db):
        from app.database import DeleteOutcome
        save_errors = []
        outcome = await tmp_db.delete_fact(ALICE, "next_meeting_topic")
        assert outcome == DeleteOutcome.BLOCKED
        if outcome in (DeleteOutcome.BLOCKED, DeleteOutcome.EMPTY_KEY):
            save_errors.append(f"next_meeting_topic: {outcome}")
        assert len(save_errors) == 1 and "next_meeting_topic" in save_errors[0]

    @pytest.mark.asyncio
    async def test_not_found_is_silent(self, tmp_db):
        from app.database import DeleteOutcome
        save_errors = []
        outcome = await tmp_db.delete_fact(ALICE, "car")
        assert outcome == DeleteOutcome.NOT_FOUND
        if outcome == DeleteOutcome.NOT_FOUND:
            pass  # intentional no-op
        assert save_errors == []

    @pytest.mark.asyncio
    async def test_confirmed_delete_succeeds(self, tmp_db):
        from app.agent_engine import MemoryUpdate
        from app.database import DeleteOutcome
        await tmp_db.upsert_fact(ALICE, "shopping_list", "milk, bread")
        mu = MemoryUpdate(key="shopping_list", value="", delete=True, confirm=True)
        outcome = await tmp_db.delete_fact(ALICE, mu.key, confirmed=mu.confirm)
        assert outcome == DeleteOutcome.DELETED
        assert "shopping_list" not in await tmp_db.get_all_facts(ALICE)


# ─────────────────────────────────────────────────────────────────────────────
# ⑧ Prompt completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptCompleteness:

    def test_orchestrator_has_key_examples(self):
        from app.prompts import ORCHESTRATOR_PROMPT
        for key in ("car", "bike", "city", "name", "shopping_list"):
            assert key in ORCHESTRATOR_PROMPT, f"Missing key example: {key!r}"

    def test_orchestrator_has_delete_true_syntax(self):
        from app.prompts import ORCHESTRATOR_PROMPT
        assert ('"delete":true' in ORCHESTRATOR_PROMPT or
                '"delete": true' in ORCHESTRATOR_PROMPT), \
            "Prompt must show delete=true syntax"

    def test_orchestrator_confirm_false_rule(self):
        from app.prompts import ORCHESTRATOR_PROMPT
        assert ("confirm=false" in ORCHESTRATOR_PROMPT.lower() or
                '"confirm": false' in ORCHESTRATOR_PROMPT or
                '"confirm":false' in ORCHESTRATOR_PROMPT), \
            "Prompt must tell LLM to emit confirm=false"

    def test_verifier_acknowledges_delete_flag(self):
        from app.prompts import VERIFIER_PROMPT
        assert "delete" in VERIFIER_PROMPT.lower()

    def test_orchestrator_has_deletion_example(self):
        from app.prompts import ORCHESTRATOR_PROMPT
        assert ("forget" in ORCHESTRATOR_PROMPT.lower() or
                "delete" in ORCHESTRATOR_PROMPT.lower()), \
            "Prompt must include a deletion example"
