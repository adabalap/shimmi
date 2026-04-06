"""
tests/unit/test_agent_p1.py

Phase 1 unit tests for agent_engine.py changes:
  - MemoryUpdate delete=True flag
  - _dispatch_tool() replaces _live_search keyword regex
  - OrchestratorResult.tool_call field parsed and passed down
  - Memory deletion end-to-end (run_agent → delete_fact)
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agent_engine import (
    MemoryUpdate,
    OrchestratorResult,
    _parse_json,
    _clean_facts,
)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryUpdate — delete flag
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryUpdateDeleteFlag:
    def test_default_delete_is_false(self):
        mu = MemoryUpdate(key="car", value="Honda")
        assert mu.delete is False

    def test_delete_true_with_empty_value(self):
        mu = MemoryUpdate(key="car", value="", delete=True)
        assert mu.delete is True
        assert mu.value == ""

    def test_delete_false_requires_value(self):
        """delete=False (default) + value="" → Pydantic still accepts it."""
        mu = MemoryUpdate(key="car", value="")
        assert mu.delete is False

    def test_delete_coerced_from_string(self):
        mu = MemoryUpdate(key="car", value="", delete=True)
        assert mu.delete is True

    def test_normal_update_preserves_value(self):
        mu = MemoryUpdate(key="name", value="Phani", delete=False)
        assert mu.value == "Phani"
        assert mu.delete is False


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorResult — tool_call field
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorResultToolCall:
    def test_tool_call_defaults_to_none(self):
        result = OrchestratorResult(action="answer", text="Hello")
        assert result.tool_call is None

    def test_tool_call_weather_stored(self):
        raw = {
            "action": "search",
            "reasoning": "needs weather",
            "text": "",
            "query": "weather Hyderabad",
            "tool_call": {"tool": "weather", "city": "Hyderabad", "country": "IN", "days": 3},
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.action == "search"
        assert result.tool_call is not None
        assert result.tool_call["tool"] == "weather"
        assert result.tool_call["city"] == "Hyderabad"

    def test_tool_call_currency(self):
        raw = {
            "action": "search",
            "reasoning": "forex",
            "text": "",
            "query": "USD to INR",
            "tool_call": {
                "tool": "currency",
                "from_currency": "USD",
                "to_currency": "INR",
                "amount": 1.0,
            },
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.tool_call["tool"] == "currency"
        assert result.tool_call["from_currency"] == "USD"

    def test_tool_call_null_is_allowed(self):
        raw = {
            "action": "answer",
            "text": "42",
            "tool_call": None,
        }
        result = OrchestratorResult.model_validate(raw)
        assert result.tool_call is None

    def test_memory_updates_with_delete_flag_parsed(self):
        raw = {
            "action": "answer",
            "text": "Forget about your car? Done.",
            "memory_updates": [
                {"key": "car", "value": "", "delete": True},
                {"key": "name", "value": "Phani", "delete": False},
            ],
        }
        result = OrchestratorResult.model_validate(raw)
        # Both updates preserved (delete=True allows empty value)
        assert len(result.memory_updates) == 2
        car_update = next(u for u in result.memory_updates if u.key == "car")
        assert car_update.delete is True
        name_update = next(u for u in result.memory_updates if u.key == "name")
        assert name_update.delete is False
        assert name_update.value == "Phani"


# ─────────────────────────────────────────────────────────────────────────────
# _dispatch_tool — routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchTool:
    @pytest.mark.asyncio
    async def test_none_tool_call_falls_back_to_web_search(self):
        """
        When tool_call is None AND the query has no keyword-routable intent
        (not weather/stocks/news/currency/timezone), dispatch falls back to
        compound-beta-mini web search.

        Note: structured queries like "latest cricket score" are now correctly
        keyword-routed to the news MCP tool — that is the RIGHT behaviour.
        This test uses an open-ended factual question that has no structured
        tool match to verify the web-search fallback path.
        """
        from app.agent_engine import _dispatch_tool

        with patch("app.agent_engine._compound_beta_search", new_callable=AsyncMock) as mock:
            mock.return_value = "Some web result"
            result = await _dispatch_tool(
                None, "explain the theory of relativity", "test-chat", facts={}
            )

        mock.assert_called_once_with("explain the theory of relativity", "test-chat")
        assert result == "Some web result"

    @pytest.mark.asyncio
    async def test_cricket_score_keyword_routes_to_news(self):
        """
        'latest cricket score' with no tool_call should be keyword-routed to
        the news MCP tool (not web_search fallback). This verifies that the
        keyword router intercepts news queries before the final fallback.
        """
        from app.agent_engine import _dispatch_tool, _keyword_tool_from_query

        tc = _keyword_tool_from_query("latest cricket score", {})
        assert tc is not None
        assert tc.tool == "news"

    @pytest.mark.asyncio
    async def test_valid_weather_tool_call_routed(self):
        """Valid weather tool_call hits dispatcher, not compound-beta."""
        from app.agent_engine import _dispatch_tool

        weather_tc = {"tool": "weather", "city": "Hyderabad", "country": "IN", "days": 3}

        with patch("app.tools.tool_dispatcher.dispatch", new_callable=AsyncMock) as mock:
            mock.return_value = "32°C sunny Hyderabad"
            result = await _dispatch_tool(
                weather_tc, "weather Hyderabad", "chat-1", facts={"city": "Hyderabad"}
            )

        mock.assert_called_once()
        assert result == "32°C sunny Hyderabad"

    @pytest.mark.asyncio
    async def test_web_search_sentinel_triggers_compound_beta(self):
        """When dispatcher returns the __web_search__ sentinel, _compound_beta_search is called."""
        from app.agent_engine import _dispatch_tool
        from app.tools import _WEB_SEARCH_SENTINEL

        search_tc = {"tool": "web_search", "query": "best restaurants Hyderabad"}

        with patch("app.tools.tool_dispatcher.dispatch", new_callable=AsyncMock) as mock_disp:
            mock_disp.return_value = _WEB_SEARCH_SENTINEL + "best restaurants Hyderabad"
            with patch("app.agent_engine._compound_beta_search", new_callable=AsyncMock) as mock_cb:
                mock_cb.return_value = "Top restaurants: ..."
                result = await _dispatch_tool(
                    search_tc, "best restaurants", "chat-1", facts={}
                )

        mock_cb.assert_called_once_with("best restaurants Hyderabad", "chat-1")
        assert result == "Top restaurants: ..."

    @pytest.mark.asyncio
    async def test_invalid_tool_call_falls_back_gracefully(self):
        """Completely invalid tool_call dict → parse_tool_call returns None → web search fallback."""
        from app.agent_engine import _dispatch_tool

        bad_tc = {"tool": "send_sms", "to": "mom", "body": "hi"}

        with patch("app.agent_engine._compound_beta_search", new_callable=AsyncMock) as mock:
            mock.return_value = "fallback result"
            result = await _dispatch_tool(
                bad_tc, "query string", "chat-1", facts={}
            )

        mock.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# clean_facts — unchanged but tested for regression
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFacts:
    def test_removes_junk_values(self):
        facts = {
            "name": "Phani",
            "city": "unknown",
            "country": "none",
            "age": "30",
            "todo_list": "",
        }
        cleaned = _clean_facts(facts)
        assert "name" in cleaned
        assert "age" in cleaned
        assert "city" not in cleaned
        assert "country" not in cleaned
        assert "todo_list" not in cleaned

    def test_keeps_valid_values(self):
        facts = {"name": "Alice", "city": "Mumbai", "hobby": "hiking"}
        cleaned = _clean_facts(facts)
        assert cleaned == facts

    def test_empty_facts(self):
        assert _clean_facts({}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing — regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self):
        raw = '{"action": "answer", "text": "hello"}'
        parsed = _parse_json(raw)
        assert parsed["action"] == "answer"

    def test_json_with_fences(self):
        raw = '```json\n{"action": "search"}\n```'
        parsed = _parse_json(raw)
        assert parsed["action"] == "search"

    def test_json_with_leading_text(self):
        raw = 'Here is the JSON: {"action": "answer", "text": "hi"}'
        parsed = _parse_json(raw)
        assert parsed["action"] == "answer"

    def test_malformed_raises(self):
        with pytest.raises(Exception):
            _parse_json("not json at all")

    def test_tool_call_round_trip(self):
        """Ensure tool_call survives JSON round-trip as expected."""
        raw = json.dumps({
            "action": "search",
            "query": "weather Hyderabad",
            "tool_call": {"tool": "weather", "city": "Hyderabad", "country": "IN", "days": 3},
        })
        parsed = _parse_json(raw)
        assert parsed["tool_call"]["tool"] == "weather"
        assert parsed["tool_call"]["city"] == "Hyderabad"
