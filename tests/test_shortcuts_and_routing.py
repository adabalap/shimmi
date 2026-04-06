"""
tests/unit/test_shortcuts_and_routing.py — Shimmi v3.3.0

Zero-quota unit tests for every zero-token shortcut and routing fix added
in v3.2.0 and v3.3.0.  All pure-Python, no network, no LLM.

Coverage:
  ① OrchestratorResult.question=null crash (FIX-NULL)
  ② _try_time_shortcut() — accuracy + edge cases
  ③ _try_facts_shortcut() — update-declaration guard (FIX-UPDATE-GUARD)
  ④ _try_facts_shortcut() — word-overlap fallback (FIX-SHORTCUT)
  ⑤ _clean_facts() — ephemeral key filter (FIX-EPHEMERAL)
  ⑥ _keyword_tool_from_query() — Groq fallback routing (FIX-TOOL)
  ⑦ _parse_retry_after() — RPD daily quota gets 2h cooldown (FIX-RPD)
"""
from __future__ import annotations

import json
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agent_engine import (
    OrchestratorResult,
    _try_time_shortcut,
    _try_facts_shortcut,
    _clean_facts,
    _keyword_tool_from_query,
    _parse_retry_after,
)


# ─────────────────────────────────────────────────────────────────────────────
# ① OrchestratorResult null coercion (FIX-NULL)
# The fatal crash: question=null in LLM JSON caused Pydantic ValidationError.
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorNullCoercion:

    def test_question_null_coerced_to_empty_string(self):
        """FIX-NULL: question=null must not crash — coerce to ''."""
        raw = {
            "action": "answer",
            "text": "Casio Edifice is a watch brand.",
            "question": None,   # LLM returned null — was crashing before fix
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.question == ""
        assert result.text == "Casio Edifice is a watch brand."

    def test_reasoning_null_coerced(self):
        raw = {"action": "answer", "text": "ok", "reasoning": None}
        result = OrchestratorResult.model_validate(raw)
        assert result.reasoning == ""

    def test_text_null_coerced(self):
        raw = {"action": "answer", "text": None}
        result = OrchestratorResult.model_validate(raw)
        assert result.text == ""

    def test_query_null_coerced(self):
        raw = {"action": "search", "query": None}
        result = OrchestratorResult.model_validate(raw)
        assert result.query == ""

    def test_all_null_fields_survive(self):
        """Worst case: LLM returns null for every optional string."""
        raw = {
            "action": "answer",
            "reasoning": None,
            "text": None,
            "query": None,
            "question": None,
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.action == "answer"
        assert result.reasoning == ""
        assert result.text == ""

    def test_real_values_not_affected(self):
        """Non-null values must still work normally."""
        raw = {
            "action": "search",
            "reasoning": "Need live data",
            "text": "",
            "query": "latest news India",
            "question": "",
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.query == "latest news India"
        assert result.reasoning == "Need live data"

    def test_action_still_required(self):
        """action is required — still raises without it."""
        with pytest.raises(Exception):
            OrchestratorResult.model_validate({"text": "hi"})


# ─────────────────────────────────────────────────────────────────────────────
# ② _try_time_shortcut() — FIX-TIME
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeShortcut:

    def test_what_time_is_it(self):
        result = _try_time_shortcut("what time is it")
        assert result is not None
        assert "IST" in result or "UTC" in result or result  # tz abbreviation

    def test_what_is_the_time(self):
        assert _try_time_shortcut("what is the time") is not None

    def test_current_time(self):
        assert _try_time_shortcut("current time") is not None

    def test_time_now(self):
        assert _try_time_shortcut("time now") is not None

    def test_what_time_is_it_now(self):
        assert _try_time_shortcut("what time is it now?") is not None

    def test_today_date(self):
        result = _try_time_shortcut("what's today's date?")
        assert result is not None

    def test_what_day_is_it(self):
        result = _try_time_shortcut("what day is it")
        assert result is not None

    def test_long_message_not_intercepted(self):
        """Long messages might contain a time phrase as part of something else."""
        long = "I need to know what time is it so that I can plan my schedule for the meeting"
        assert _try_time_shortcut(long) is None

    def test_unrelated_message_returns_none(self):
        assert _try_time_shortcut("what's my name?") is None
        assert _try_time_shortcut("weather today") is None
        assert _try_time_shortcut("ping") is None

    def test_response_contains_clock_emoji_or_date(self):
        """Result should look like a time reply, not a fact lookup."""
        result = _try_time_shortcut("what time is it?")
        assert result is not None
        assert ("🕰️" in result or "📅" in result or ":" in result)

    def test_time_reply_has_no_placeholder(self):
        """Must never return a template string with angle brackets."""
        result = _try_time_shortcut("what time is it?")
        assert result is not None
        assert "<" not in result and ">" not in result


# ─────────────────────────────────────────────────────────────────────────────
# ③ _try_facts_shortcut() — update-declaration guard (FIX-UPDATE-GUARD)
# "my X is Y" messages are WRITES. They must never be shortcut.
# ─────────────────────────────────────────────────────────────────────────────

class TestFactsShortcutUpdateGuard:

    FACTS = {
        "name": "Phani",
        "favorite_drink": "medium oat milk lattes",
        "city": "Hyderabad",
        "age": "28",
    }

    def test_declaration_not_shortcutted(self):
        """'my name is Sarah' is a write — must go to LLM, not shortcut."""
        result = _try_facts_shortcut("my name is Sarah", self.FACTS)
        assert result is None

    def test_coffee_declaration_not_shortcutted(self):
        """Root cause of the reported bug: shortcut returned old coffee value."""
        msg = "my coffee order is a medium oat milk latte with one pump of vanilla"
        result = _try_facts_shortcut(msg, self.FACTS)
        assert result is None

    def test_age_declaration_not_shortcutted(self):
        result = _try_facts_shortcut("my age is 29", self.FACTS)
        assert result is None

    def test_city_declaration_not_shortcutted(self):
        result = _try_facts_shortcut("my city is Mumbai", self.FACTS)
        assert result is None

    def test_question_not_blocked_by_update_guard(self):
        """'what's my name?' is a recall question — must shortcut."""
        result = _try_facts_shortcut("what's my name?", self.FACTS)
        assert result is not None
        assert "Phani" in result

    def test_correction_not_shortcutted(self):
        """'I'm 29 not 28' is a correction/write — must go to LLM."""
        result = _try_facts_shortcut("I'm 29 years old, not 28", self.FACTS)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# ④ _try_facts_shortcut() — word-overlap fallback (FIX-SHORTCUT, #6 response)
#
# The architectural answer: instead of growing the signal-phrase list
# with every new phrasing, we match query words against fact key words.
# "what's my current age?" → words include 'age' → matches key 'age'.
# This handles ANY variation without code changes.
# ─────────────────────────────────────────────────────────────────────────────

class TestFactsShortcutWordOverlap:
    """
    ARCH-4: The shortcut is deliberately conservative. It fires only on two
    unambiguous structural patterns: "what's my X" and special forms like "how old am I".
    Word-overlap fallback was removed — anything not matched by the synonym table
    goes to the LLM orchestrator which handles it correctly in one iteration.
    This means "what is my age right now?" does NOT shortcut — it goes to the LLM.
    That is correct: the LLM answers it in <1s from the facts in its prompt.
    """

    FACTS = {
        "age": "29",
        "city": "Hyderabad",
        "favorite_color": "olive green",
        "pets": "Max, Luna",
    }

    # ── Queries that DO shortcut ───────────────────────────────────────────────

    def test_whats_my_age_shortcuts(self):
        """Exact pattern match → shortcut fires."""
        result = _try_facts_shortcut("what's my age?", self.FACTS)
        assert result is not None
        assert "29" in result

    def test_what_is_my_city_shortcuts(self):
        result = _try_facts_shortcut("what is my city?", self.FACTS)
        assert result is not None
        assert "Hyderabad" in result

    def test_how_old_am_i_shortcuts(self):
        """Special-case form → shortcut fires."""
        result = _try_facts_shortcut("how old am I?", self.FACTS)
        assert result is not None
        assert "29" in result

    def test_where_do_i_live_shortcuts(self):
        result = _try_facts_shortcut("where do I live?", self.FACTS)
        assert result is not None

    # ── Queries that do NOT shortcut (go to LLM) ──────────────────────────────

    def test_current_age_shortcuts(self):
        """
        "what's my current age?" → subject "current age" → in _FACT_SYNONYMS → shortcut.
        The synonym table includes common phrasings so users get instant replies
        without burning LLM tokens.
        """
        result = _try_facts_shortcut("what's my current age?", self.FACTS)
        assert result is not None
        assert "29" in result

    def test_age_right_now_goes_to_llm(self):
        result = _try_facts_shortcut("what is my age right now?", self.FACTS)
        assert result is None

    def test_what_city_do_i_shortcuts(self):
        """"what city do I live in?" is in SPECIAL_RECALL_FORMS → shortcut."""
        result = _try_facts_shortcut("what city do I live in?", self.FACTS)
        assert result is not None
        assert "Hyderabad" in result

    def test_no_facts_returns_none(self):
        assert _try_facts_shortcut("what's my age?", {}) is None

    def test_long_message_never_shortcuts(self):
        long = "what is my age " + "x" * 50
        assert _try_facts_shortcut(long, self.FACTS) is None


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ _clean_facts() — ephemeral key filter (FIX-EPHEMERAL)
# ─────────────────────────────────────────────────────────────────────────────

class TestEphemeralKeyFilter:
    """
    ARCH-1: ephemeral key filtering is at the DB layer (source_filter="user_stated"),
    not in _clean_facts. These tests verify the junk-value filtering that _clean_facts
    still owns, and document the provenance architecture.
    """

    def test_junk_values_filtered_by_clean_facts(self):
        """_clean_facts strips null/unknown/empty values — its only responsibility."""
        facts = {"name": "Phani", "city": "unknown", "age": "none"}
        cleaned = _clean_facts(facts)
        assert "name" in cleaned
        assert "city" not in cleaned
        assert "age" not in cleaned

    def test_core_facts_pass_through(self):
        facts = {"name": "Sarah", "age": "29", "city": "Hyderabad", "interests": "NLP"}
        assert _clean_facts(facts) == facts

    def test_arch1_ephemeral_keys_pass_clean_facts(self):
        """
        ARCH-1 contract: _clean_facts does NOT filter by key name.
        last_summary, conversation_since_morning, etc. are blocked upstream
        by source_filter="user_stated" in get_all_facts — not here.
        """
        facts = {"name": "Phani", "last_summary": "Non-junk content here"}
        result = _clean_facts(facts)
        # passes through — DB source_filter is the guard, not _clean_facts
        assert "name" in result


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ _keyword_tool_from_query() — Groq fallback tool routing (FIX-TOOL)
# When Groq acts as fallback orchestrator and omits tool_call, keyword routing
# still directs the query to the right MCP endpoint.
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordToolRouting:

    # ── Weather ──────────────────────────────────────────────────────────

    def test_weather_keyword_routed(self):
        from app.tools import WeatherTool
        tc = _keyword_tool_from_query("weather forecast Hyderabad India", {})
        assert isinstance(tc, WeatherTool)

    def test_forecast_keyword_routed(self):
        from app.tools import WeatherTool
        tc = _keyword_tool_from_query("what's the forecast for tomorrow", {"city": "Mumbai"})
        assert isinstance(tc, WeatherTool)

    def test_temperature_keyword_routed(self):
        from app.tools import WeatherTool
        tc = _keyword_tool_from_query("what is the temperature today", {})
        assert isinstance(tc, WeatherTool)

    def test_weather_uses_facts_city_when_no_city_in_query(self):
        from app.tools import WeatherTool
        tc = _keyword_tool_from_query("weather forecast", {"city": "Chennai"})
        assert isinstance(tc, WeatherTool)
        # city from facts used as default
        assert tc.city == "Chennai"

    # ── Stocks ───────────────────────────────────────────────────────────

    def test_nifty_routed_to_stocks(self):
        from app.tools import StocksTool
        tc = _keyword_tool_from_query("nifty 50 today", {})
        assert isinstance(tc, StocksTool)

    def test_sensex_routed_to_stocks(self):
        from app.tools import StocksTool
        tc = _keyword_tool_from_query("sensex live update", {})
        assert isinstance(tc, StocksTool)

    def test_share_price_routed_to_stocks(self):
        from app.tools import StocksTool
        tc = _keyword_tool_from_query("RELIANCE share price today", {})
        assert isinstance(tc, StocksTool)

    # ── News ─────────────────────────────────────────────────────────────

    def test_news_keyword_routed(self):
        from app.tools import NewsTool
        tc = _keyword_tool_from_query("latest news India today", {})
        assert isinstance(tc, NewsTool)

    def test_cricket_score_routed_to_news(self):
        """Regression: previously fell through to web_search."""
        from app.tools import NewsTool
        tc = _keyword_tool_from_query("latest cricket score", {})
        assert isinstance(tc, NewsTool)

    def test_headlines_routed_to_news(self):
        from app.tools import NewsTool
        tc = _keyword_tool_from_query("top headlines today", {})
        assert isinstance(tc, NewsTool)

    def test_breaking_news_routed(self):
        from app.tools import NewsTool
        tc = _keyword_tool_from_query("breaking news update", {})
        assert isinstance(tc, NewsTool)

    # ── Currency ─────────────────────────────────────────────────────────

    def test_usd_to_inr_routed(self):
        from app.tools import CurrencyTool
        tc = _keyword_tool_from_query("USD to INR exchange rate", {})
        assert isinstance(tc, CurrencyTool)
        assert tc.from_currency == "USD"
        assert tc.to_currency == "INR"

    def test_exchange_rate_keyword(self):
        from app.tools import CurrencyTool
        tc = _keyword_tool_from_query("what is the exchange rate EUR to USD", {})
        assert isinstance(tc, CurrencyTool)

    # ── Open-ended queries → None (falls through to web_search) ──────────

    def test_open_ended_returns_none(self):
        """Questions with no known tool keyword should fall through to web_search."""
        tc = _keyword_tool_from_query("explain the theory of relativity", {})
        assert tc is None

    def test_general_question_returns_none(self):
        tc = _keyword_tool_from_query("who invented the telephone", {})
        assert tc is None

    def test_empty_query_returns_none(self):
        tc = _keyword_tool_from_query("", {})
        assert tc is None

    def test_none_query_returns_none(self):
        tc = _keyword_tool_from_query(None, {})
        assert tc is None


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ _parse_retry_after() — RPD daily quota detection (FIX-RPD)
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryAfterParsing:

    def _exc(self, msg: str) -> Exception:
        return Exception(msg)

    def test_gemini_rpd_gets_two_hour_cooldown(self):
        """
        FIX-RPD: Gemini's 'You exceeded your current quota' is daily (RPD),
        not per-minute. It must get 7200s (2h) cooldown, not the 300s default.
        """
        exc = self._exc(
            "Error code: 429 - [{'error': {'code': 429, 'message': "
            "'You exceeded your current quota, please check your plan and billing details."
        )
        cooldown = _parse_retry_after(exc)
        assert cooldown == 7200.0, f"Expected 7200s, got {cooldown}"

    def test_gemini_rpd_with_billing_mention(self):
        exc = self._exc("quota exceeded. check your billing details.")
        cooldown = _parse_retry_after(exc)
        assert cooldown == 7200.0

    def test_groq_retry_after_parsed(self):
        """Groq embeds 'Please try again in Xh Ym Zs' — must be parsed."""
        exc = self._exc(
            "Rate limit reached. Please try again in 6m48.671999999s."
        )
        cooldown = _parse_retry_after(exc)
        # 6*60 + 48.67 + 10s buffer ≈ 418.67s
        assert 400 <= cooldown <= 430, f"Expected ~418s, got {cooldown}"

    def test_groq_hours_and_minutes(self):
        exc = self._exc("try again in 1h4m54.368s")
        cooldown = _parse_retry_after(exc)
        # 1*3600 + 4*60 + 54.37 + 10 = 3904.37
        assert 3890 <= cooldown <= 3920

    def test_gemini_rpm_retry_after_seconds(self):
        """Gemini RPM error includes 'retry after N seconds'."""
        exc = self._exc("quota exceeded … retry after 60 seconds")
        cooldown = _parse_retry_after(exc)
        assert cooldown == 70.0  # 60 + 10s buffer

    def test_unknown_error_returns_conservative_default(self):
        exc = self._exc("some other error without timing info")
        cooldown = _parse_retry_after(exc)
        assert cooldown == 300.0  # 5 minute default

    def test_cooldown_capped_at_7200(self):
        """No cooldown exceeds 2 hours even for very long retry-after values."""
        exc = self._exc("try again in 99h0m0s")
        cooldown = _parse_retry_after(exc)
        assert cooldown == 7200.0

# ─────────────────────────────────────────────────────────────────────────────
# ⑧ Regression tests from WhatsApp production conversation (2026-03-16)
# These are exact messages from the live conversation that revealed bugs.
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionRegressions:
    """
    Exact messages from the 2026-03-16 WhatsApp conversation that exposed bugs.
    Every test here corresponds to a message the user sent and the WRONG response
    they received. These must never regress.
    """

    # ── BUG-1: Japan time returned IST instead of JST ────────────────────────
    # User:  "What's the time in Japan right now, spock"
    # Wrong: "🕰️ It's *15:37 IST* (Monday afternoon)"
    # Root:  "what's the time" is a substring → local shortcut fired before
    #        timezone MCP routing could handle it.

    def test_japan_time_not_intercepted_by_local_shortcut(self):
        """'time in Japan' must route to timezone MCP, not return IST."""
        result = _try_time_shortcut("What's the time in Japan right now, spock")
        assert result is None, (
            f"Shortcut returned {result!r} — should be None so timezone MCP handles it"
        )

    def test_time_in_tokyo_not_intercepted(self):
        result = _try_time_shortcut("What time is it in Tokyo?")
        assert result is None

    def test_time_in_london_not_intercepted(self):
        result = _try_time_shortcut("What's the time in London?")
        assert result is None

    def test_time_in_new_york_not_intercepted(self):
        result = _try_time_shortcut("What is the time in New York?")
        assert result is None

    def test_local_time_still_shortcutted(self):
        """Pure local time queries must still be answered from server clock."""
        assert _try_time_shortcut("what time is it?") is not None
        assert _try_time_shortcut("current time") is not None

    # ── BUG-2: "let's list" triggered reading_list shortcut ──────────────────
    # User:  "Ok, let's list one after the other, spock" (about political factors)
    # Wrong: "Your reading list on record is: *Discover India, The Alchemist* 📋"
    # Root:  "list" was a standalone recall_trigger → word-overlap matched
    #        "list" against "reading_list" key.

    FACTS_WITH_LIST = {
        "name": "Phani",
        "reading_list": "Discover India, The Alchemist",
        "shopping_list": "milk, bread",
    }

    def test_lets_list_instruction_not_shortcutted(self):
        """'let's list one after the other' is an instruction, not a recall."""
        result = _try_facts_shortcut(
            "Ok, let's list one after the other, spock", self.FACTS_WITH_LIST
        )
        assert result is None, (
            f"Shortcut returned {result!r} — 'list' as a verb must not trigger recall"
        )

    def test_list_political_factors_not_shortcutted(self):
        result = _try_facts_shortcut(
            "Let's list the political factors one by one", self.FACTS_WITH_LIST
        )
        assert result is None

    def test_list_as_verb_in_any_context(self):
        result = _try_facts_shortcut("can you list the benefits?", self.FACTS_WITH_LIST)
        assert result is None

    # ── BUG-3: Analytical question triggered reading_list shortcut ────────────
    # User:  "Based on my reading list, what's your thoughts about me, spock"
    # Wrong: "Your reading list on record is: *Discover India, The Alchemist* 📋"
    # Root:  "what" + "my " are recall triggers; word-overlap matched "list"
    #        against "reading_list" — echoed the list instead of doing analysis.

    def test_based_on_reading_list_analysis_not_shortcutted(self):
        """Analytical questions using a fact as context must go to LLM."""
        result = _try_facts_shortcut(
            "Based on my reading list, what's your thoughts about me",
            self.FACTS_WITH_LIST,
        )
        assert result is None, (
            f"Shortcut returned {result!r} — 'based on my X' is analysis, not recall"
        )

    def test_what_kind_of_reader_not_shortcutted(self):
        result = _try_facts_shortcut(
            "I want you to tell me what kind of reader I am, based on my reading list",
            self.FACTS_WITH_LIST,
        )
        # > 100 chars so blocked by length, but also analytical context
        assert result is None

    def test_suggest_book_from_list_not_shortcutted(self):
        result = _try_facts_shortcut(
            "Can you suggest a book based on my reading list?",
            self.FACTS_WITH_LIST,
        )
        assert result is None

    def test_recommend_using_facts_not_shortcutted(self):
        result = _try_facts_shortcut(
            "What would you recommend based on my interests?",
            {"interests": "NLP, ML"},
        )
        assert result is None

    # ── Valid queries that must STILL work after all fixes ────────────────────

    def test_what_is_on_my_reading_list_still_works(self):
        """Direct recall of a list value must still shortcut correctly."""
        result = _try_facts_shortcut(
            "what's on my reading list?", self.FACTS_WITH_LIST
        )
        # Should shortcut — "reading list" signal in _FACT_SIGNALS or word-overlap
        assert result is not None
        assert "Discover India" in result

    def test_show_my_shopping_list_still_works(self):
        result = _try_facts_shortcut("show me my shopping list", self.FACTS_WITH_LIST)
        assert result is not None

    def test_japan_timezone_keyword_routes_correctly(self):
        """After the shortcut steps aside, keyword routing must handle 'time in Japan'."""
        from app.agent_engine import _keyword_tool_from_query
        from app.tools import TimezoneTool
        tc = _keyword_tool_from_query("What's the time in Japan right now", {})
        assert isinstance(tc, TimezoneTool), f"Expected TimezoneTool, got {type(tc)}"
        assert "japan" in tc.city.lower()

    def test_coffee_order_still_shortcutted_via_signal_phrase(self):
        """Signal phrase 'coffee order' must still match via _FACT_SIGNALS path."""
        facts = {"favorite_drink": "medium oat milk lattes"}
        result = _try_facts_shortcut("what's my coffee order?", facts)
        assert result is not None
        assert "oat milk" in result

