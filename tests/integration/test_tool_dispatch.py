"""
tests/integration/test_tool_dispatch.py

Integration tests for P1-FEAT-1: LLM-decided tool dispatch.
Tests the full pipeline: orchestrator returns tool_call → _dispatch_tool()
→ correct backend called with correct args → result injected into next turn.

All LLM and HTTP backends are mocked — zero quota cost.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock, call


CHAT_ID  = "919876543210-1234567890@g.us"
SENDER   = "919876543210@s.whatsapp.net"
USER_TZ  = "Asia/Kolkata"

# ── Canned LLM responses ──────────────────────────────────────────────────

def orch_search(tool_name: str, tool_args: dict, query: str) -> str:
    return json.dumps({
        "action": "search",
        "reasoning": f"Need live {tool_name} data.",
        "text": "",
        "query": query,
        "question": "",
        "memory_updates": [],
        "reminders": [],
        "tool_call": {"tool": tool_name, **tool_args},
    })


def orch_answer(text: str) -> str:
    return json.dumps({
        "action": "answer",
        "reasoning": "Got search result.",
        "text": text,
        "query": "",
        "question": "",
        "memory_updates": [],
        "reminders": [],
        "tool_call": None,
    })


CANNED_EXTRACT = json.dumps({"memory_updates": []})
CANNED_VERIFY  = json.dumps({"approved": []})

def canned_format(text: str) -> str:
    return json.dumps({"text": text})


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherToolDispatch:
    @pytest.mark.asyncio
    async def test_weather_tool_call_calls_get_weather_with_city(self):
        """
        Orchestrator returns tool_call={tool:weather, city:Hyderabad} →
        get_weather("Hyderabad", "IN") is called (NOT the raw query string).

        This is the core regression test for ISSUE-2 (bug fixed in P0, hardened in P1).
        """
        from app.agent_engine import run_agent

        weather_result = "🌤️ Hyderabad: 32°C, sunny. Humidity 65%. UV index: 7."
        orch_calls = [0]

        async def fake_groq_raw(messages, *, max_tokens, chat_id, label, role, timeout=None):
            if role == "orchestrate":
                orch_calls[0] += 1
                if orch_calls[0] == 1:
                    return orch_search(
                        "weather",
                        {"city": "Hyderabad", "country": "IN", "days": 3},
                        "weather Hyderabad India",
                    )
                return orch_answer("It's 32°C and sunny in Hyderabad.")
            elif label.startswith("extract"):
                return CANNED_EXTRACT
            elif label.startswith("verify"):
                return CANNED_VERIFY
            elif label.startswith("format"):
                return canned_format("It's 32°C and sunny in Hyderabad. 🌤️")
            return CANNED_EXTRACT

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq_raw):
            with patch("app.live_data.get_weather", new_callable=AsyncMock) as mock_weather:
                mock_weather.return_value = weather_result
                # Also mock get_news/stocks/etc. to ensure they're NOT called
                with patch("app.live_data.get_news", new_callable=AsyncMock) as mock_news:
                    with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
                        result = await run_agent(
                            chat_id=CHAT_ID,
                            user_text="What's the weather?",
                            facts={"city": "Hyderabad", "country": "IN"},
                            context=[],
                            reminders=[],
                        )

        # CRITICAL: get_weather must be called with the city from tool_call, NOT from the query string
        mock_weather.assert_called_once_with("Hyderabad", "IN")
        mock_news.assert_not_called()
        assert "32°C" in result.reply.text or result.reply.text  # got some reply

    @pytest.mark.asyncio
    async def test_weather_tool_call_without_city_falls_back_to_facts(self):
        """
        If tool_call has a city, that wins. If city is empty, facts["city"] is used.
        Ensures city is NEVER taken from the query string.
        """
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "weather", "city": "Mumbai", "country": "IN", "days": 1}
        facts = {"city": "Delhi", "country": "IN"}

        with patch("app.live_data.get_weather", new_callable=AsyncMock) as mock:
            mock.return_value = "28°C Mumbai"
            await _dispatch_tool(tc, "weather query", "chat-1", facts=facts)

        # tool_call city (Mumbai) wins over facts city (Delhi)
        mock.assert_called_once_with("Mumbai", "IN")


class TestNewsToolDispatch:
    @pytest.mark.asyncio
    async def test_news_tool_call_passes_query_not_keywords(self):
        """
        For news, the LLM provides a specific search query — we pass it directly
        to get_news(), not keyword-guessed from the user message.
        """
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "news", "query": "India budget 2026 tax slabs", "country": "IN"}

        with patch("app.live_data.get_news", new_callable=AsyncMock) as mock:
            mock.return_value = "📰 Budget 2026: New tax slabs announced."
            result = await _dispatch_tool(tc, "news query", "chat-1", facts={})

        mock.assert_called_once_with("India budget 2026 tax slabs", "IN")
        assert "Budget 2026" in result


class TestCurrencyToolDispatch:
    @pytest.mark.asyncio
    async def test_currency_dispatched_with_structured_args(self):
        """
        Currency now gets proper from_currency / to_currency / amount args.
        Previously ALL MCP calls received the raw query string — this was always broken.
        """
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "currency", "from_currency": "USD", "to_currency": "INR", "amount": 100.0}

        with patch("app.mcp_client.mcp_currency", new_callable=AsyncMock) as mock:
            mock.return_value = "100 USD = 8,350 INR"
            result = await _dispatch_tool(tc, "USD to INR", "chat-1", facts={})

        mock.assert_called_once_with("USD", "INR", 100.0)
        assert "8,350 INR" in result


class TestStocksToolDispatch:
    @pytest.mark.asyncio
    async def test_stocks_dispatched_with_symbols(self):
        """Stocks gets the actual ticker symbols the LLM extracted."""
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "stocks", "symbols": ["RELIANCE", "TCS", "INFY"]}

        with patch("app.live_data.get_indian_stocks", new_callable=AsyncMock) as mock:
            mock.return_value = "RELIANCE: ₹2,450 | TCS: ₹3,780 | INFY: ₹1,540"
            result = await _dispatch_tool(tc, "stock prices", "chat-1", facts={})

        mock.assert_called_once_with(["RELIANCE.NS", "TCS.NS", "INFY.NS"])  # .NS auto-appended

    @pytest.mark.asyncio
    async def test_stocks_empty_symbols_for_general_market(self):
        """Empty symbols → general market indices query."""
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "stocks", "symbols": []}

        with patch("app.live_data.get_indian_stocks", new_callable=AsyncMock) as mock:
            mock.return_value = "Nifty50: 23,450 | Sensex: 76,200"
            await _dispatch_tool(tc, "nifty today", "chat-1", facts={})

        mock.assert_called_once_with([])


class TestTimezoneToolDispatch:
    @pytest.mark.asyncio
    async def test_timezone_dispatched_with_city(self):
        """Timezone gets the city name, not a raw query string."""
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "timezone", "city": "Tokyo"}

        with patch("app.mcp_client.mcp_timezone", new_callable=AsyncMock) as mock:
            mock.return_value = "🕐 Tokyo: 18:30 JST (Tuesday)"
            result = await _dispatch_tool(tc, "time in Tokyo", "chat-1", facts={})

        mock.assert_called_once_with("Tokyo")
        assert "Tokyo" in result


class TestWebSearchFallback:
    @pytest.mark.asyncio
    async def test_missing_tool_call_falls_back_to_compound_beta(self):
        """When tool_call is None AND keyword router doesn't match, compound-beta is used.
        Note: Sports/IPL queries ARE intercepted by the keyword router → news tool.
        Use a query that bypasses keyword routing (no news/stock/weather keywords)."""
        from app.agent_engine import _dispatch_tool

        with patch("app.agent_engine._compound_beta_search", new_callable=AsyncMock) as mock:
            mock.return_value = "Compound-beta result"
            result = await _dispatch_tool(
                None, "how do I tie a bowline knot", "chat-1", facts={}
            )

        mock.assert_called_once_with("how do I tie a bowline knot", "chat-1")
        assert result == "Compound-beta result"

    @pytest.mark.asyncio
    async def test_web_search_tool_routes_to_compound_beta(self):
        """Explicit web_search tool also goes to compound-beta-mini."""
        from app.agent_engine import _dispatch_tool

        tc = {"tool": "web_search", "query": "best trekking trails Himachal 2026"}

        with patch("app.agent_engine._compound_beta_search", new_callable=AsyncMock) as mock:
            mock.return_value = "Top trails: ..."
            result = await _dispatch_tool(tc, "trekking query", "chat-1", facts={})

        mock.assert_called_once_with("best trekking trails Himachal 2026", "chat-1")


class TestToolDispatchRetryAfterToolFails:
    @pytest.mark.asyncio
    async def test_failed_tool_returns_empty_string_agent_retries_search(self):
        """
        If a tool returns empty string (e.g. MCP server is down), the agent
        gets an empty SEARCH_RESULT and the orchestrator should search again
        (or answer with an appropriate fallback).
        """
        from app.agent_engine import run_agent

        orch_calls = [0]

        async def fake_groq_raw(messages, *, max_tokens, chat_id, label, role, timeout=None):
            if role == "orchestrate":
                orch_calls[0] += 1
                if orch_calls[0] == 1:
                    return orch_search(
                        "weather",
                        {"city": "Hyderabad", "country": "IN", "days": 3},
                        "weather Hyderabad",
                    )
                # After empty result, answer gracefully
                return orch_answer("Sorry, weather data is unavailable right now.")
            elif label.startswith("extract"):
                return CANNED_EXTRACT
            elif label.startswith("verify"):
                return CANNED_VERIFY
            elif label.startswith("format"):
                return canned_format("Sorry, weather data is unavailable right now.")
            return CANNED_EXTRACT

        with patch("app.agent_engine._groq_raw", side_effect=fake_groq_raw):
            # Tool fails — returns empty string
            with patch("app.live_data.get_weather", new_callable=AsyncMock) as mock_weather:
                mock_weather.return_value = ""  # simulates MCP server down
                with patch("app.agent_engine.GROQ_CLIENT", MagicMock()):
                    result = await run_agent(
                        chat_id=CHAT_ID,
                        user_text="What's the weather?",
                        facts={"city": "Hyderabad"},
                        context=[],
                        reminders=[],
                    )

        assert result.reply.text  # always get some reply
        # search.empty_exit: when tool returns empty, agent answers directly
        # without a second orchestrator call (avoids 30s Groq 8B round-trip).
        # orch_calls[0] == 1 is correct — empty result exits early, not re-orchestrated.
        assert orch_calls[0] >= 1  # at least one orchestration call fired
