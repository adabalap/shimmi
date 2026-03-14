"""
tools.py — Shimmi Phase 1

Replaces the brittle keyword-regex tool routing in agent_engine.py._live_search()
with a proper LLM-decided, Pydantic-validated tool dispatch system.

Why this matters (from the analysis report, ISSUE-5):
  - _live_search() passed raw LLM query strings as city/symbol/currency args
  - ALL MCP structured calls were silently failing and falling through to
    compound-beta-mini every single time
  - Phase 1 fix: LLM embeds a tool_call JSON block in its orchestrator output;
    ToolDispatcher validates + routes it with correct arguments.

Architecture:
  OrchestratorResult.tool_call (new field)
      ↓
  ToolDispatcher.dispatch(tool_call, facts)
      ├── WeatherTool   → live_data.get_weather(city, country)
      ├── NewsTool      → live_data.get_news(query, country)
      ├── StocksTool    → live_data.get_indian_stocks(symbols)
      ├── CurrencyTool  → mcp_client.mcp_currency(from, to, amount)
      ├── TimezoneTool  → mcp_client.mcp_timezone(city)
      └── WebSearchTool → GROQ compound-beta-mini (fallback for open-ended queries)
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("app.tools")


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas — one Pydantic model per tool
# ─────────────────────────────────────────────────────────────────────────────

class WeatherTool(BaseModel):
    tool: Literal["weather"]
    city: str = Field(..., min_length=1, description="City name to fetch weather for")
    country: str = Field("IN", description="ISO-2 country code, default IN")
    days: int = Field(3, ge=1, le=7, description="Forecast days")

    @field_validator("city", "country", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()

    @field_validator("country", mode="after")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()[:2] if v else "IN"


class NewsTool(BaseModel):
    tool: Literal["news"]
    query: str = Field(..., min_length=1, description="Search query for news")
    country: str = Field("IN", description="ISO-2 country code for news locale")

    @field_validator("query", "country", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()

    @field_validator("country", mode="after")
    @classmethod
    def _upper(cls, v: str) -> str:
        return (v.upper()[:2] if v else "IN")


class StocksTool(BaseModel):
    tool: Literal["stocks"]
    symbols: List[str] = Field(
        default_factory=list,
        description="List of NSE/BSE ticker symbols. Empty = top Nifty indices."
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            # Accept comma-separated string from LLM
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip().upper() for s in v if str(s).strip()]
        return []


class CurrencyTool(BaseModel):
    tool: Literal["currency"]
    from_currency: str = Field(..., min_length=1, description="Source ISO currency code, e.g. USD")
    to_currency: str = Field(..., min_length=1, description="Target ISO currency code, e.g. INR")
    amount: float = Field(1.0, ge=0.0, description="Amount to convert")

    @field_validator("from_currency", "to_currency", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> str:
        return str(v or "").strip().upper()


class TimezoneTool(BaseModel):
    tool: Literal["timezone"]
    city: str = Field(..., min_length=1, description="City to get current local time for")

    @field_validator("city", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()


class WebSearchTool(BaseModel):
    """Fallback: open-ended query routed to compound-beta-mini web search."""
    tool: Literal["web_search"]
    query: str = Field(..., min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()


# Union type — used as the type annotation for OrchestratorResult.tool_call
ToolCall = Annotated[
    Union[WeatherTool, NewsTool, StocksTool, CurrencyTool, TimezoneTool, WebSearchTool],
    Field(discriminator="tool"),
]


# ─────────────────────────────────────────────────────────────────────────────
# ToolDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Routes a validated ToolCall to the correct live-data backend.

    Usage::

        dispatcher = ToolDispatcher()
        result = await dispatcher.dispatch(tool_call, facts=user_facts)

    The dispatcher is stateless — instantiate once and reuse.
    """

    async def dispatch(
        self,
        tool_call: ToolCall,
        facts: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Dispatch a ToolCall to its backend.

        Args:
            tool_call: Validated Pydantic ToolCall instance.
            facts:     User fact dict (used to fill defaults like city/country).

        Returns:
            String result to inject as SEARCH_RESULT into the orchestrator prompt.
            Empty string if the tool returned no data.
        """
        _facts = facts or {}
        tool_name = tool_call.tool

        try:
            if tool_name == "weather":
                return await self._weather(tool_call, _facts)        # type: ignore[arg-type]
            elif tool_name == "news":
                return await self._news(tool_call, _facts)           # type: ignore[arg-type]
            elif tool_name == "stocks":
                return await self._stocks(tool_call)                  # type: ignore[arg-type]
            elif tool_name == "currency":
                return await self._currency(tool_call)                # type: ignore[arg-type]
            elif tool_name == "timezone":
                return await self._timezone(tool_call)                # type: ignore[arg-type]
            elif tool_name == "web_search":
                return await self._web_search(tool_call)              # type: ignore[arg-type]
            else:
                logger.warning("tools.dispatch — unknown tool=%r", tool_name)
                return ""
        except Exception as exc:
            logger.error(
                "tools.dispatch.error  tool=%s  err=%s",
                tool_name, str(exc)[:200],
            )
            return ""

    # ── Individual tool handlers ──────────────────────────────────────────

    async def _weather(self, tc: WeatherTool, facts: Dict[str, str]) -> str:
        from .live_data import get_weather  # local import — avoids circular deps

        # Fallback chain for city: tool_call → facts["city"] → facts["user_city"] → "Hyderabad"
        city = tc.city or facts.get("city") or facts.get("user_city") or ""
        if not city:
            city = "Hyderabad"
            logger.warning("tools.weather — no city in tool_call or facts; defaulting to Hyderabad")

        country = tc.country or facts.get("country") or "IN"
        logger.info("tools.weather  city=%r  country=%r  days=%d", city, country, tc.days)
        result = await get_weather(city, country)
        return result or ""

    async def _news(self, tc: NewsTool, facts: Dict[str, str]) -> str:
        from .live_data import get_news

        country = tc.country or facts.get("country") or "IN"
        logger.info("tools.news  query=%r  country=%r", tc.query, country)
        result = await get_news(tc.query, country[:2].upper())
        return result or ""

    async def _stocks(self, tc: StocksTool) -> str:
        from .live_data import get_indian_stocks

        symbols = tc.symbols or []
        logger.info("tools.stocks  symbols=%r", symbols)
        result = await get_indian_stocks(symbols)

        # FIX-B7: If result signals unavailability and at least one symbol lacks
        # an exchange suffix, retry with .NS appended (NSE India).
        # e.g. "PAYTM" → "PAYTM.NS". Skips symbols that already have a dot.
        if result and ("unavailable" in result.lower() or "not recognised" in result.lower()):
            ns_symbols = [
                s if "." in s else f"{s}.NS"
                for s in symbols
            ]
            if ns_symbols != symbols:
                logger.info("tools.stocks.retry_ns  symbols=%r", ns_symbols)
                retry = await get_indian_stocks(ns_symbols)
                if retry and "unavailable" not in retry.lower():
                    return retry or ""

        return result or ""

    async def _currency(self, tc: CurrencyTool) -> str:
        from .mcp_client import mcp_currency

        logger.info(
            "tools.currency  from=%s  to=%s  amount=%.2f",
            tc.from_currency, tc.to_currency, tc.amount,
        )
        result = await mcp_currency(tc.from_currency, tc.to_currency, tc.amount)
        return result or ""

    async def _timezone(self, tc: TimezoneTool) -> str:
        from .mcp_client import mcp_timezone

        logger.info("tools.timezone  city=%r", tc.city)
        result = await mcp_timezone(tc.city)
        return result or ""

    async def _web_search(self, tc: WebSearchTool) -> str:
        # Web search is handled upstream in agent_engine._live_search_fallback()
        # We return a sentinel that signals the caller to use compound-beta-mini.
        # This keeps the Groq client out of tools.py (no circular dep).
        logger.info("tools.web_search  query=%r  → delegating to compound-beta-mini", tc.query)
        return _WEB_SEARCH_SENTINEL + tc.query


# Sentinel prefix — agent_engine inspects this to route to compound-beta-mini
_WEB_SEARCH_SENTINEL = "__web_search__:"


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser — parses raw LLM dict into a validated ToolCall
# ─────────────────────────────────────────────────────────────────────────────

def parse_tool_call(raw: Any) -> Optional[ToolCall]:
    """
    Parse a raw dict (from the LLM JSON output) into a validated ToolCall.

    Returns None if the input is missing, malformed, or describes an unknown tool.
    Validation errors are logged at WARNING level (not raised) so the caller
    can gracefully fall back to compound-beta-mini.

    Args:
        raw: The value of orchestrator JSON key ``tool_call``. May be None,
             a dict, or garbage — this function handles all cases.
    """
    if not raw or not isinstance(raw, dict):
        return None

    tool_name = str(raw.get("tool", "")).strip().lower()
    if not tool_name:
        return None

    # Normalise common LLM typos / synonyms
    _ALIASES = {
        "get_weather":  "weather",
        "fetch_weather": "weather",
        "get_news":     "news",
        "fetch_news":   "news",
        "get_stocks":   "stocks",
        "stock":        "stocks",
        "stock_prices": "stocks",
        "exchange":     "currency",
        "forex":        "currency",
        "get_currency": "currency",
        "time":         "timezone",
        "get_timezone": "timezone",
        "search":       "web_search",
        "web":          "web_search",
        "google":       "web_search",
    }
    tool_name = _ALIASES.get(tool_name, tool_name)

    _TOOL_MAP = {
        "weather":    WeatherTool,
        "news":       NewsTool,
        "stocks":     StocksTool,
        "currency":   CurrencyTool,
        "timezone":   TimezoneTool,
        "web_search": WebSearchTool,
    }

    cls = _TOOL_MAP.get(tool_name)
    if cls is None:
        logger.warning("tools.parse — unknown tool=%r in LLM output", tool_name)
        return None

    try:
        payload = {**raw, "tool": tool_name}

        # FIX-D: LLM sometimes emits {"tool":"web_search"} with no "query" field,
        # causing a Pydantic "Field required" validation error and falling back to
        # the no_tool_call path (which then also has no query, so compound-beta
        # gets an empty string). Pull the query from sibling fields if missing.
        if tool_name == "web_search" and not payload.get("query"):
            payload["query"] = (
                str(raw.get("query") or raw.get("q") or raw.get("search_query") or "").strip()
                or None  # leave None so Pydantic still rejects genuinely empty calls
            )

        return cls.model_validate(payload)
    except Exception as exc:
        logger.warning(
            "tools.parse — validation failed  tool=%r  err=%s",
            tool_name, str(exc)[:200],
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Singleton dispatcher — import and use directly
# ─────────────────────────────────────────────────────────────────────────────

tool_dispatcher = ToolDispatcher()
