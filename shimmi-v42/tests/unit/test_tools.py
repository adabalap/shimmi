"""
tests/unit/test_tools.py

Unit tests for app/tools.py — tool schema validation, parse_tool_call(),
and ToolDispatcher routing.  ALL tests are offline (no network, no LLM).
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.tools import (
    WeatherTool, NewsTool, StocksTool, CurrencyTool, TimezoneTool, WebSearchTool,
    parse_tool_call, ToolDispatcher, tool_dispatcher, _WEB_SEARCH_SENTINEL,
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_tool_call — input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestParseToolCall:
    def test_none_returns_none(self):
        assert parse_tool_call(None) is None

    def test_empty_dict_returns_none(self):
        assert parse_tool_call({}) is None

    def test_non_dict_returns_none(self):
        assert parse_tool_call("weather") is None
        assert parse_tool_call(42) is None
        assert parse_tool_call([]) is None

    def test_unknown_tool_returns_none(self):
        assert parse_tool_call({"tool": "send_email"}) is None

    # ── Weather ──────────────────────────────────────────────────────────

    def test_weather_basic(self):
        tc = parse_tool_call({"tool": "weather", "city": "Hyderabad", "country": "IN"})
        assert isinstance(tc, WeatherTool)
        assert tc.city == "Hyderabad"
        assert tc.country == "IN"
        assert tc.days == 3  # default

    def test_weather_alias_get_weather(self):
        tc = parse_tool_call({"tool": "get_weather", "city": "Mumbai", "country": "IN"})
        assert isinstance(tc, WeatherTool)
        assert tc.city == "Mumbai"

    def test_weather_country_uppercased(self):
        tc = parse_tool_call({"tool": "weather", "city": "Paris", "country": "fr"})
        assert tc.country == "FR"

    def test_weather_days_clamped(self):
        # days > 7 should fail validation (ge=1, le=7)
        tc = parse_tool_call({"tool": "weather", "city": "X", "country": "IN", "days": 10})
        assert tc is None  # validation error → None

    def test_weather_missing_city(self):
        # city has min_length=1 so empty string is invalid
        tc = parse_tool_call({"tool": "weather", "city": "", "country": "IN"})
        assert tc is None

    # ── News ─────────────────────────────────────────────────────────────

    def test_news_basic(self):
        tc = parse_tool_call({"tool": "news", "query": "India elections 2026", "country": "IN"})
        assert isinstance(tc, NewsTool)
        assert tc.query == "India elections 2026"
        assert tc.country == "IN"

    def test_news_default_country(self):
        tc = parse_tool_call({"tool": "news", "query": "tech news"})
        assert isinstance(tc, NewsTool)
        assert tc.country == "IN"

    # ── Stocks ───────────────────────────────────────────────────────────

    def test_stocks_with_symbols_list(self):
        tc = parse_tool_call({"tool": "stocks", "symbols": ["RELIANCE", "TCS"]})
        assert isinstance(tc, StocksTool)
        assert tc.symbols == ["RELIANCE", "TCS"]

    def test_stocks_with_symbols_string(self):
        # LLM might pass a comma-separated string
        tc = parse_tool_call({"tool": "stocks", "symbols": "INFY, WIPRO"})
        assert isinstance(tc, StocksTool)
        assert "INFY" in tc.symbols
        assert "WIPRO" in tc.symbols

    def test_stocks_empty_symbols(self):
        tc = parse_tool_call({"tool": "stocks", "symbols": []})
        assert isinstance(tc, StocksTool)
        assert tc.symbols == []

    def test_stocks_alias(self):
        tc = parse_tool_call({"tool": "stock", "symbols": []})
        assert isinstance(tc, StocksTool)

    # ── Currency ─────────────────────────────────────────────────────────

    def test_currency_basic(self):
        tc = parse_tool_call({
            "tool": "currency",
            "from_currency": "USD",
            "to_currency": "INR",
            "amount": 100.0,
        })
        assert isinstance(tc, CurrencyTool)
        assert tc.from_currency == "USD"
        assert tc.to_currency == "INR"
        assert tc.amount == 100.0

    def test_currency_lowercased_codes(self):
        tc = parse_tool_call({
            "tool": "currency", "from_currency": "usd", "to_currency": "eur", "amount": 1,
        })
        assert tc.from_currency == "USD"
        assert tc.to_currency == "EUR"

    def test_currency_alias_forex(self):
        tc = parse_tool_call({
            "tool": "forex", "from_currency": "GBP", "to_currency": "JPY", "amount": 1,
        })
        assert isinstance(tc, CurrencyTool)

    # ── Timezone ─────────────────────────────────────────────────────────

    def test_timezone_basic(self):
        tc = parse_tool_call({"tool": "timezone", "city": "Tokyo"})
        assert isinstance(tc, TimezoneTool)
        assert tc.city == "Tokyo"

    def test_timezone_alias_time(self):
        tc = parse_tool_call({"tool": "time", "city": "London"})
        assert isinstance(tc, TimezoneTool)

    # ── Web search ───────────────────────────────────────────────────────

    def test_web_search_basic(self):
        tc = parse_tool_call({"tool": "web_search", "query": "best Python libraries 2026"})
        assert isinstance(tc, WebSearchTool)
        assert tc.query == "best Python libraries 2026"

    def test_web_search_alias_search(self):
        tc = parse_tool_call({"tool": "search", "query": "trekking Himachal"})
        assert isinstance(tc, WebSearchTool)

    def test_web_search_alias_google(self):
        tc = parse_tool_call({"tool": "google", "query": "how to make biryani"})
        assert isinstance(tc, WebSearchTool)


# ─────────────────────────────────────────────────────────────────────────────
# ToolDispatcher.dispatch — routing
# ─────────────────────────────────────────────────────────────────────────────

class TestToolDispatcher:
    @pytest.fixture
    def dispatcher(self):
        return ToolDispatcher()

    @pytest.mark.asyncio
    async def test_weather_routes_to_get_weather(self, dispatcher):
        tc = WeatherTool(tool="weather", city="Hyderabad", country="IN", days=3)
        with patch("app.tools.ToolDispatcher._weather", new_callable=AsyncMock) as mock:
            mock.return_value = "32°C sunny"
            result = await dispatcher.dispatch(tc, facts={})
        mock.assert_called_once()
        assert result == "32°C sunny"

    @pytest.mark.asyncio
    async def test_news_routes_to_get_news(self, dispatcher):
        tc = NewsTool(tool="news", query="tech news", country="IN")
        with patch("app.tools.ToolDispatcher._news", new_callable=AsyncMock) as mock:
            mock.return_value = "📰 Tech headlines..."
            result = await dispatcher.dispatch(tc, facts={})
        mock.assert_called_once()
        assert result == "📰 Tech headlines..."

    @pytest.mark.asyncio
    async def test_stocks_routes_to_get_stocks(self, dispatcher):
        tc = StocksTool(tool="stocks", symbols=["RELIANCE"])
        with patch("app.tools.ToolDispatcher._stocks", new_callable=AsyncMock) as mock:
            mock.return_value = "RELIANCE: ₹2,450"
            result = await dispatcher.dispatch(tc, facts={})
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_currency_routes_correctly(self, dispatcher):
        tc = CurrencyTool(tool="currency", from_currency="USD", to_currency="INR", amount=1.0)
        with patch("app.tools.ToolDispatcher._currency", new_callable=AsyncMock) as mock:
            mock.return_value = "1 USD = 83.5 INR"
            result = await dispatcher.dispatch(tc, facts={})
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_returns_sentinel(self, dispatcher):
        """web_search tool should return the __web_search__: sentinel for compound-beta routing."""
        tc = WebSearchTool(tool="web_search", query="trekking Himachal")
        result = await dispatcher.dispatch(tc, facts={})
        assert result.startswith(_WEB_SEARCH_SENTINEL)
        assert "trekking Himachal" in result

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self, dispatcher):
        """If a tool backend raises, dispatch() should return '' not raise."""
        tc = WeatherTool(tool="weather", city="Hyderabad", country="IN")
        with patch("app.tools.ToolDispatcher._weather", new_callable=AsyncMock) as mock:
            mock.side_effect = ConnectionError("MCP server down")
            result = await dispatcher.dispatch(tc, facts={})
        assert result == ""

    @pytest.mark.asyncio
    async def test_weather_city_fallback_from_facts(self, dispatcher):
        """If tool_call has no city, dispatcher fills from facts["city"]."""
        # This tests the actual _weather method's fallback logic
        tc = WeatherTool(tool="weather", city="Hyderabad", country="IN")
        facts = {"city": "Chennai", "country": "IN"}
        # city in tool_call takes precedence
        with patch("app.live_data.get_weather", new_callable=AsyncMock) as mock:
            mock.return_value = "34°C Chennai"
            result = await dispatcher._weather(tc, facts)
        # city from tool_call (Hyderabad) wins over facts (Chennai)
        mock.assert_called_once_with("Hyderabad", "IN")

    @pytest.mark.asyncio
    async def test_weather_uses_facts_city_when_tool_call_empty(self, dispatcher):
        """If city is somehow empty despite validation, falls back to facts."""
        # Force city = "" bypassing validator
        tc = WeatherTool.model_construct(tool="weather", city="", country="IN", days=3)
        facts = {"city": "Pune", "country": "IN"}
        with patch("app.live_data.get_weather", new_callable=AsyncMock) as mock:
            mock.return_value = "28°C Pune"
            await dispatcher._weather(tc, facts)
        # Should use facts["city"] = "Pune"
        mock.assert_called_once_with("Pune", "IN")
