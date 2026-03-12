"""
tools.py — Shimmi Phase 1

P1-FEAT-1: Structured LLM tool dispatch.

Replaces the brittle keyword-regex routing in _live_search() with structured,
LLM-decided tool invocation.  The orchestrator now outputs a `tool_call` JSON
object alongside `action=search`, and ToolDispatcher routes it to the correct
live-data function with properly-typed parameters — no more passing a raw query
string as a city name.

Module layout:
  • Tool schema models (Pydantic)  — one per tool
  • ToolCall discriminated union
  • tool_schemas_json()            — serialises schemas for prompt injection
  • ToolDispatcher.dispatch()      — async, calls the right live-data fn
  • parse_tool_call()              — parses LLM JSON into a ToolCall model
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("app.tools")


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema models
# ─────────────────────────────────────────────────────────────────────────────

class WeatherTool(BaseModel):
    """Fetch current weather and short forecast for a city."""
    tool:    Literal["weather"]
    city:    str   = Field(..., description="City name, e.g. 'Hyderabad'")
    country: str   = Field("IN", description="ISO-3166-1 alpha-2 country code")
    days:    int   = Field(3, ge=1, le=7, description="Forecast days (1–7)")

    @field_validator("city", mode="before")
    @classmethod
    def _clean_city(cls, v):
        return (v or "").strip()

    @field_validator("country", mode="before")
    @classmethod
    def _clean_country(cls, v):
        return (str(v or "IN").strip().upper() or "IN")[:2]


class NewsTool(BaseModel):
    """Fetch latest news headlines matching a query."""
    tool:    Literal["news"]
    query:   str  = Field(..., description="Search topic, e.g. 'India cricket'")
    country: str  = Field("IN", description="ISO-3166-1 alpha-2 country code")

    @field_validator("query", mode="before")
    @classmethod
    def _clean_query(cls, v):
        return (v or "").strip()

    @field_validator("country", mode="before")
    @classmethod
    def _clean_country(cls, v):
        return (str(v or "IN").strip().upper() or "IN")[:2]


class StocksTool(BaseModel):
    """Fetch Indian stock / index prices from NSE/BSE."""
    tool:    Literal["stocks"]
    symbols: List[str] = Field(
        default_factory=list,
        description="Ticker symbols, e.g. ['RELIANCE', 'NIFTY50']. "
                    "Empty list fetches broad market summary.",
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def _clean_symbols(cls, v):
        if not isinstance(v, list):
            return []
        return [str(s).strip().upper() for s in v if s]


class CurrencyTool(BaseModel):
    """Convert an amount between two currencies."""
    tool:          Literal["currency"]
    from_currency: str   = Field(..., description="Source currency code, e.g. 'USD'")
    to_currency:   str   = Field(..., description="Target currency code, e.g. 'INR'")
    amount:        float = Field(1.0, ge=0, description="Amount to convert")

    @field_validator("from_currency", "to_currency", mode="before")
    @classmethod
    def _clean_cur(cls, v):
        return (str(v or "").strip().upper() or "USD")[:3]


class WebSearchTool(BaseModel):
    """General-purpose web search via compound-beta-mini.

    Use only when no structured tool (weather/stocks/news/currency) applies.
    """
    tool:  Literal["web_search"]
    query: str = Field(..., description="Freeform search query")

    @field_validator("query", mode="before")
    @classmethod
    def _clean_query(cls, v):
        return (v or "").strip()


# Discriminated union — Pydantic uses the `tool` field to pick the model.
ToolCall = Union[WeatherTool, NewsTool, StocksTool, CurrencyTool, WebSearchTool]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helper — embed tool schemas in the orchestrator system prompt
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_DESCRIPTIONS: List[Dict[str, Any]] = [
    {
        "name":        "weather",
        "description": "Current weather + short forecast for a city.",
        "parameters":  {
            "city":    "string — city name (use facts.city when available)",
            "country": "string — ISO alpha-2, default 'IN'",
            "days":    "integer 1–7, default 3",
        },
    },
    {
        "name":        "news",
        "description": "Latest news headlines on a topic.",
        "parameters":  {
            "query":   "string — topic / keywords",
            "country": "string — ISO alpha-2, default 'IN'",
        },
    },
    {
        "name":        "stocks",
        "description": "Indian stock / index prices (NSE/BSE).",
        "parameters":  {
            "symbols": "list[string] — ticker list, empty = broad summary",
        },
    },
    {
        "name":        "currency",
        "description": "Currency conversion between two codes.",
        "parameters":  {
            "from_currency": "string — 3-letter code, e.g. 'USD'",
            "to_currency":   "string — 3-letter code, e.g. 'INR'",
            "amount":        "number, default 1.0",
        },
    },
    {
        "name":        "web_search",
        "description": "General web search. Use ONLY when no structured tool applies.",
        "parameters":  {
            "query": "string — freeform search query",
        },
    },
]


def tool_schemas_json() -> str:
    """Return a compact JSON string of tool descriptions for prompt injection."""
    return json.dumps(_TOOL_DESCRIPTIONS, ensure_ascii=False, separators=(",", ":"))


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM output → ToolCall
# ─────────────────────────────────────────────────────────────────────────────

def parse_tool_call(
    raw: Any,
    *,
    fallback_query: str = "",
    facts: Optional[Dict[str, str]] = None,
) -> ToolCall:
    """
    Parse the LLM's `tool_call` field into a typed ToolCall model.

    The LLM output may be:
      • a dict already (parsed from JSON): {"tool": "weather", "city": "Hyderabad"}
      • a JSON string
      • None / missing (use fallback heuristics)

    Falls back to WebSearchTool(query=fallback_query) on any parse failure.
    """
    _facts = facts or {}

    # 1. Normalise to dict
    if raw is None:
        return _fallback_tool(fallback_query, _facts)

    if isinstance(raw, str):
        raw = raw.strip()
        # Strip ```json fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            raw = json.loads(raw)
        except Exception:
            logger.warning("⚠️  tool_call.json_parse_failed  raw=%r", raw[:120])
            return _fallback_tool(fallback_query, _facts)

    if not isinstance(raw, dict):
        return _fallback_tool(fallback_query, _facts)

    tool_name = (raw.get("tool") or "").strip().lower()
    if not tool_name:
        return _fallback_tool(fallback_query, _facts)

    # 2. Fill in city from facts when LLM omits it for weather
    if tool_name == "weather" and not raw.get("city"):
        city = (
            _facts.get("city")
            or _facts.get("user_city")
            or ""
        ).strip()
        if city:
            raw = {**raw, "city": city}
        else:
            # Cannot route to weather tool without a city — fall back
            logger.info(
                "⚠️  tool_call.weather_no_city  falling_back  query=%r", fallback_query
            )
            return WebSearchTool(tool="web_search", query=fallback_query or "weather today")

    # 3. Dispatch to the right Pydantic model
    _TOOL_MAP = {
        "weather":    WeatherTool,
        "news":       NewsTool,
        "stocks":     StocksTool,
        "currency":   CurrencyTool,
        "web_search": WebSearchTool,
    }
    model_cls = _TOOL_MAP.get(tool_name)
    if model_cls is None:
        logger.warning("⚠️  tool_call.unknown_tool  tool=%r", tool_name)
        return _fallback_tool(fallback_query, _facts)

    try:
        return model_cls.model_validate(raw)
    except Exception as exc:
        logger.warning("⚠️  tool_call.validation_failed  tool=%r  err=%s", tool_name, exc)
        return _fallback_tool(fallback_query, _facts)


def _fallback_tool(query: str, facts: Dict[str, str]) -> WebSearchTool:
    return WebSearchTool(tool="web_search", query=query or "")


# ─────────────────────────────────────────────────────────────────────────────
# ToolDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Routes a ToolCall to the appropriate live-data function.

    All methods are async and return a plain string suitable for injecting into
    the orchestrator as SEARCH_RESULT.  Empty string means no data retrieved.
    """

    async def dispatch(
        self,
        tool_call: ToolCall,
        *,
        groq_client: Any = None,           # passed through for web_search
        live_search_model: str = "compound-beta-mini",
        live_search_enabled: bool = True,
    ) -> str:
        """
        Dispatch a ToolCall and return the result string.

        Raises nothing — all exceptions are caught and logged; empty string
        is returned so the orchestrator can fall back gracefully.
        """
        try:
            if isinstance(tool_call, WeatherTool):
                return await self._weather(tool_call)
            elif isinstance(tool_call, NewsTool):
                return await self._news(tool_call)
            elif isinstance(tool_call, StocksTool):
                return await self._stocks(tool_call)
            elif isinstance(tool_call, CurrencyTool):
                return await self._currency(tool_call)
            elif isinstance(tool_call, WebSearchTool):
                return await self._web_search(
                    tool_call,
                    groq_client=groq_client,
                    model=live_search_model,
                    enabled=live_search_enabled,
                )
        except Exception as exc:
            logger.error(
                "❌ tool.dispatch_error  tool=%s  err=%s",
                getattr(tool_call, "tool", "?"),
                str(exc)[:200],
            )
        return ""

    # ── individual tool handlers ──────────────────────────────────────────

    async def _weather(self, tc: WeatherTool) -> str:
        from .live_data import get_weather
        logger.info("🌤️  tool.weather  city=%r  country=%r  days=%d",
                    tc.city, tc.country, tc.days)
        result = await get_weather(tc.city, tc.country) or ""
        if not result:
            logger.warning("⚠️  tool.weather  no_result  city=%r", tc.city)
        return result

    async def _news(self, tc: NewsTool) -> str:
        from .live_data import get_news
        logger.info("📰 tool.news  query=%r  country=%r", tc.query, tc.country)
        return await get_news(tc.query, tc.country) or ""

    async def _stocks(self, tc: StocksTool) -> str:
        from .live_data import get_indian_stocks
        symbols = tc.symbols or None
        logger.info("📈 tool.stocks  symbols=%r", symbols)
        return await get_indian_stocks(symbols) or ""

    async def _currency(self, tc: CurrencyTool) -> str:
        from .mcp_client import mcp_currency
        logger.info("💱 tool.currency  %r→%r  amount=%s",
                    tc.from_currency, tc.to_currency, tc.amount)
        result = await mcp_currency(tc.from_currency, tc.to_currency, tc.amount) or ""
        if not result:
            # MCP currency endpoint may not be live — fall back to web search
            logger.info("⚠️  tool.currency  mcp_empty  falling_back_to_websearch")
            return ""
        return result

    async def _web_search(
        self,
        tc: WebSearchTool,
        *,
        groq_client: Any,
        model: str,
        enabled: bool,
    ) -> str:
        """
        compound-beta-mini web search (unchanged from Phase 0 _live_search fallback).
        """
        import asyncio
        from .prompts import LIVE_SEARCH_PROMPT

        if not enabled:
            return "Live search is disabled."
        if groq_client is None:
            return "LLM not initialised."

        query = tc.query
        if not query:
            return ""

        logger.info("🔍 tool.web_search  query=%r", query[:80])
        messages = [
            {"role": "system", "content": LIVE_SEARCH_PROMPT},
            {"role": "user",   "content": query},
        ]

        for attempt, q_text in enumerate([query, query[:200], query[:80]], 1):
            messages[-1]["content"] = q_text
            try:
                resp = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.1,
                    ),
                    timeout=50.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                if len(text) < 60 and attempt < 3:
                    continue
                logger.info("🔍 tool.web_search.done  result_len=%d", len(text))
                return text[:1200] if len(text) > 1200 else text
            except Exception as exc:
                s = str(exc)
                if "413" in s and attempt < 3:
                    logger.warning("⚠️  tool.web_search.413  retrying_shorter")
                    continue
                if "429" in s or "rate_limit" in s.lower():
                    return "Live search quota exhausted. Please try again later."
                logger.error("❌ tool.web_search.error  %s", s[:200])
                return f"Search failed: {s[:80]}"

        return "Search returned no result."


# Module-level singleton — import and use this everywhere.
dispatcher = ToolDispatcher()
