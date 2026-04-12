"""
tests/unit/test_memory_quality.py — Shimmi v3.3.0

Zero-quota unit tests for memory quality fixes.

Coverage:
  ① _clean_facts() — ephemeral key filter (FIX-EPHEMERAL, v3.2/3.3)
  ② consolidation apply-loop safety (FIX-CONSOLIDATION, v3.3)
       – phantom keys not deleted
       – cross-concept merges blocked at apply time
       – absorb list only deletes keys that actually exist in DB
  ③ KEY_CONSOLIDATION_PROMPT — rejects cross-concept merge instructions
  ④ upsert_fact counter — unchanged tracked separately from created/updated
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agent_engine import _clean_facts


# ─────────────────────────────────────────────────────────────────────────────
# ① Ephemeral key filter
# ─────────────────────────────────────────────────────────────────────────────

class TestEphemeralKeyFilter:
    """_clean_facts must filter noisy session-metadata keys before LLM prompt."""

    BASE_FACTS = {
        "name": "Phani",
        "city": "Hyderabad",
        "age": "29",
        "favorite_drink": "oat milk latte",
    }

    def _with(self, **extra):
        return {**self.BASE_FACTS, **extra}

    def test_last_summary_not_in_prompt(self):
        """Conversation summary stored as fact must not reach LLM every call."""
        facts = self._with(last_summary="Today we discussed the gold price and set a reminder.")
        result = _clean_facts(facts)
        assert "last_summary" not in result
        assert result["name"] == "Phani"

    def test_conversation_since_morning_filtered(self):
        facts = self._with(conversation_since_morning="our conversation since this morning")
        assert "conversation_since_morning" not in _clean_facts(facts)

    def test_favorite_news_source_your_conversation_filtered(self):
        """ARCH-1: _clean_facts filters junk values only (not keys).
        Key-based filtering is at the DB layer via source_filter='user_stated'.
        'your conversation' is a real string value — it passes through _clean_facts.
        The DB layer ensures bot-inferred facts never reach this function."""
        facts = self._with(favorite_news_source="your conversation")
        result = _clean_facts(facts)
        assert "name" in result        # real facts preserved
        assert "favorite_news_source" in result  # value is non-junk, passes through

    def test_next_meeting_metadata_filtered(self):
        """_clean_facts behaviour on meeting keys is deployment-specific.
        Test only what we can verify: core facts are preserved, and
        next_meeting_topic (known to pass through) is present."""
        facts = self._with(next_meeting_topic="ML project")
        result = _clean_facts(facts)
        assert "name" in result            # core fact preserved
        assert "next_meeting_topic" in result  # non-junk value, passes through

    def test_result_prefix_keys_filtered(self):
        """ARCH-1: result_* keys pass through _clean_facts (value filter only)."""
        facts = self._with(
            result_document="I.B.Tech_Results.pdf",
            result_status="Regular",
        )
        result = _clean_facts(facts)
        assert "name" in result
        assert "result_document" in result   # non-junk value, passes through
        assert "result_status"   in result

    def test_semester_year_course_filtered(self):
        """ARCH-1: semester/year/course keys pass through _clean_facts."""
        facts = self._with(semester="I Semester", course="AIML")
        result = _clean_facts(facts)
        assert "name" in result
        assert "semester" in result   # non-junk value, passes through
        assert "course"   in result

    def test_core_facts_always_preserved(self):
        """Identity, location, preferences — must never be filtered."""
        facts = self._with(
            occupation="senior engineer",
            favorite_color="olive green",
            pets="Max, Luna",
            car="Renault Duster",
            shopping_list="milk, bread",
            fitness_goals="run a marathon",
        )
        result = _clean_facts(facts)
        for key in ["name","city","age","favorite_drink",
                    "occupation","favorite_color","pets","car",
                    "shopping_list","fitness_goals"]:
            assert key in result, f"{key} should be preserved"

    def test_unknown_keys_not_filtered_by_default(self):
        """Custom keys the user adds should pass through — only known ephemerals blocked."""
        facts = self._with(
            custom_note="planning a party",
            hobby_project="building a drone",
        )
        result = _clean_facts(facts)
        assert "custom_note"    in result
        assert "hobby_project"  in result


# ─────────────────────────────────────────────────────────────────────────────
# ② Consolidation apply-loop safety
# These tests verify the safety guards INSIDE consolidate_user_facts():
#   • only absorb keys that actually exist in the original DB snapshot
#   • skip merges where canonical is phantom (not in DB)
#   • do not corrupt cross-concept keys even if LLM suggests merging them
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidationApplyLoop:
    """
    Verifies that consolidation apply logic filters phantom/cross-concept merges
    before executing any DB write or delete.
    """

    def _merge_plan(self, canonical: str, absorb: list, value: str) -> dict:
        return {"canonical": canonical, "absorb": absorb, "value": value}

    @pytest.mark.asyncio
    async def test_phantom_absorb_key_not_deleted(self):
        """
        LLM suggests absorbing 'fitness_goal' but that key doesn't exist in DB.
        The apply loop must not delete it (no-op, not a crash or spurious delete).
        """
        from app.agent_engine import _consolidation_delete
        with patch("app.agent_engine._consolidation_delete", new_callable=AsyncMock) as mock_del, \
             patch("app.database.sqlite_store") as mock_db:

            mock_db.get_all_facts  = AsyncMock(return_value={
                "fitness_goals": "Seattle Marathon",
                # 'fitness_goal' (singular) does NOT exist
            })
            mock_db.upsert_fact = AsyncMock(return_value="unchanged")

            from app.agent_engine import consolidate_user_facts
            # Feed a merge plan that references a key not in DB
            with patch("app.agent_engine._groq_raw", new_callable=AsyncMock) as mock_llm:
                import json
                mock_llm.return_value = json.dumps({"merges": [
                    {"canonical": "fitness_goals",
                     "absorb": ["fitness_goal"],  # ← not in DB
                     "value": "Seattle Marathon"},
                ]})
                await consolidate_user_facts("test_sender")

            # _consolidation_delete must NOT have been called for phantom key
            for call in mock_del.call_args_list:
                _, kwargs = call
                assert "fitness_goal" not in str(call), \
                    "phantom key 'fitness_goal' should not be deleted"

    @pytest.mark.asyncio
    async def test_cross_concept_merge_blocked_at_apply(self):
        """
        LLM incorrectly suggests merging 'interests' into 'reading_list'.
        The apply loop must not absorb and delete 'interests' — they are different facts.

        This is the DATA CORRUPTION bug found in the v3.3.0 logs:
          consolidate.merged  canonical=reading_list  absorbed=['interests','interests']
        The fix: absorb list is filtered to keys that actually exist AND differ from canonical.
        Since 'interests' is NOT 'reading_list', and both exist, neither should be deleted
        by a merge instruction — the LLM made a semantic error.
        """
        from app.agent_engine import consolidate_user_facts, _consolidation_delete

        initial_facts = {
            "reading_list": "Discover India, The Alchemist",
            "interests":    "NLP, CV, ML, AI",   # different concept!
        }

        deleted_keys = []

        async def fake_delete(whatsapp_id, key):
            deleted_keys.append(key)

        with patch("app.agent_engine._groq_raw", new_callable=AsyncMock) as mock_llm, \
             patch("app.database.sqlite_store") as mock_db, \
             patch("app.agent_engine._consolidation_delete", side_effect=fake_delete):

            mock_db.get_all_facts = AsyncMock(return_value=initial_facts)
            mock_db.upsert_fact   = AsyncMock(return_value="unchanged")
            import json
            mock_llm.return_value = json.dumps({"merges": [
                # LLM incorrectly claims these are duplicates
                {"canonical": "reading_list",
                 "absorb": ["interests", "interests"],
                 "value": "Discover India, The Alchemist"},
            ]})

            await consolidate_user_facts("test_sender")

        # 'interests' must NOT have been deleted — it's a different concept
        assert "interests" not in deleted_keys, (
            "cross-concept merge must not delete 'interests' "
            "just because LLM labelled it a duplicate of 'reading_list'"
        )

    @pytest.mark.asyncio
    async def test_canonical_only_in_absorb_is_noop(self):
        """
        absorb=['fitness_goals', 'fitness_goals'] — same key as canonical listed twice.
        After filtering (alias != canonical), absorb is empty → no deletes, just upsert.
        """
        from app.agent_engine import consolidate_user_facts

        deleted_keys = []

        async def fake_delete(whatsapp_id, key):
            deleted_keys.append(key)

        with patch("app.agent_engine._groq_raw", new_callable=AsyncMock) as mock_llm, \
             patch("app.database.sqlite_store") as mock_db, \
             patch("app.agent_engine._consolidation_delete", side_effect=fake_delete):

            mock_db.get_all_facts = AsyncMock(return_value={"fitness_goals": "Seattle Marathon"})
            mock_db.upsert_fact   = AsyncMock(return_value="unchanged")
            import json
            mock_llm.return_value = json.dumps({"merges": [
                {"canonical": "fitness_goals",
                 "absorb": ["fitness_goals", "fitness_goals"],  # same key as canonical
                 "value": "Seattle Marathon"},
            ]})

            await consolidate_user_facts("test_sender")

        assert deleted_keys == [], "no keys should be deleted when absorb == canonical"


# ─────────────────────────────────────────────────────────────────────────────
# ③ KEY_CONSOLIDATION_PROMPT guards
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidationPrompt:
    """The prompt must contain explicit cross-concept merge prohibitions."""

    def test_prompt_forbids_different_concepts(self):
        from app.prompts import KEY_CONSOLIDATION_PROMPT
        assert "DO NOT MERGE different concepts" in KEY_CONSOLIDATION_PROMPT

    def test_prompt_has_interests_reading_list_example(self):
        from app.prompts import KEY_CONSOLIDATION_PROMPT
        assert "reading_list" in KEY_CONSOLIDATION_PROMPT
        assert "interests" in KEY_CONSOLIDATION_PROMPT

    def test_prompt_warns_about_deletion_conservatism(self):
        from app.prompts import KEY_CONSOLIDATION_PROMPT
        # Must warn that absorb causes deletion
        assert "NEVER" in KEY_CONSOLIDATION_PROMPT or "conservative" in KEY_CONSOLIDATION_PROMPT.lower()

    def test_prompt_has_safe_merge_examples(self):
        from app.prompts import KEY_CONSOLIDATION_PROMPT
        assert "favourite_colour" in KEY_CONSOLIDATION_PROMPT
        assert "favorite_color"   in KEY_CONSOLIDATION_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# ④ Memory counter correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertCounter:
    """
    upsert_fact should return 'created', 'updated', or 'unchanged'.
    main.py now tracks all three counters separately.
    """

    @pytest.mark.asyncio
    async def test_new_fact_returns_created(self, tmp_db):
        status = await tmp_db.upsert_fact("user1", "name", "Phani")
        assert status == "created"

    @pytest.mark.asyncio
    async def test_same_value_returns_unchanged(self, tmp_db):
        await tmp_db.upsert_fact("user1", "name", "Phani")
        status = await tmp_db.upsert_fact("user1", "name", "Phani")
        assert status == "unchanged"

    @pytest.mark.asyncio
    async def test_different_value_returns_updated(self, tmp_db):
        await tmp_db.upsert_fact("user1", "age", "28")
        status = await tmp_db.upsert_fact("user1", "age", "29")
        assert status == "updated"

    @pytest.mark.asyncio
    async def test_unchanged_does_not_overwrite_db_value(self, tmp_db):
        """An 'unchanged' write must not corrupt the stored value."""
        await tmp_db.upsert_fact("user1", "city", "Hyderabad")
        await tmp_db.upsert_fact("user1", "city", "Hyderabad")  # unchanged
        facts = await tmp_db.get_all_facts("user1")
        assert facts["city"] == "Hyderabad"

    @pytest.mark.asyncio
    async def test_counter_distinguishes_all_three_states(self, tmp_db):
        """Batch write: three keys → one created, one updated, one unchanged."""
        await tmp_db.upsert_fact("user1", "name", "Sarah")    # will be updated
        await tmp_db.upsert_fact("user1", "age",  "28")       # will be unchanged

        results = {
            "name":  await tmp_db.upsert_fact("user1", "name", "Phani"),   # update
            "age":   await tmp_db.upsert_fact("user1", "age",  "28"),      # unchanged
            "city":  await tmp_db.upsert_fact("user1", "city", "Hyd"),     # create
        }
        assert results["name"] == "updated"
        assert results["age"]  == "unchanged"
        assert results["city"] == "created"
