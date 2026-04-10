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
            raw = [s.strip().upper() for s in v.split(",") if s.strip()]
        elif isinstance(v, list):
            raw = [str(s).strip().upper() for s in v if str(s).strip()]
        else:
            return []
        # Normalise Indian equity tickers: bare symbols → .NS
        # LLMs often omit the exchange suffix; .NS (NSE) is the correct default.
        # Commodity/index tickers (GC=F, ^NSEI) and already-qualified ones pass through.
        _COMMODITY_PASS = {"GC=F", "SI=F", "CL=F", "NG=F", "BZ=F"}
        normalised = []
        for sym in raw:
            if sym in _COMMODITY_PASS:
                normalised.append(sym)
            elif sym.startswith("^"):          # index like ^NSEI
                normalised.append(sym)
            elif "." in sym:                   # already has exchange suffix
                normalised.append(sym)
            else:
                normalised.append(sym + ".NS") # bare Indian equity → NSE default
        return normalised


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


class FetchUrlTool(BaseModel):
    """Fetch and extract plain text from a URL the user shared."""
    tool: Literal["fetch_url"]
    url:  str = Field(..., min_length=4, description="Full URL to fetch.")

    @field_validator("url", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v or "").strip()


# Union type — used as the type annotation for OrchestratorResult.tool_call
ToolCall = Annotated[
    Union[WeatherTool, NewsTool, StocksTool, CurrencyTool, TimezoneTool,
          WebSearchTool, FetchUrlTool],
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
                return await self._stocks(tool_call, facts=_facts)    # type: ignore[arg-type]
            elif tool_name == "currency":
                return await self._currency(tool_call)                # type: ignore[arg-type]
            elif tool_name == "timezone":
                return await self._timezone(tool_call)                # type: ignore[arg-type]
            elif tool_name == "web_search":
                return await self._web_search(tool_call)              # type: ignore[arg-type]
            elif tool_name == "fetch_url":
                return await self._fetch_url(tool_call)               # type: ignore[arg-type]
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
        from .live_data import get_news, _normalize_news_query

        country = tc.country or facts.get("country") or "IN"
        # Normalize meta-phrase queries (e.g. "morning news round up" → "India news today")
        # BEFORE sending to live_data so the normalization fires even when the query
        # arrives via tool_call JSON rather than through the live_data.get_news path.
        effective_query = _normalize_news_query(tc.query or "India top news")
        if effective_query != tc.query:
            logger.info("tools.news.query_normalised  %r → %r", tc.query, effective_query)
        logger.info("tools.news  query=%r  country=%r", effective_query, country)
        result = await get_news(effective_query, country[:2].upper())
        return result or ""

    async def _stocks(self, tc: StocksTool, facts: Optional[Dict[str, str]] = None) -> str:
        from .live_data import get_indian_stocks, get_portfolio_review

        symbols = tc.symbols or []

        # Portfolio P&L review path — triggered by __PORTFOLIO_REVIEW__ sentinel
        if symbols == ["__PORTFOLIO_REVIEW__"]:
            _facts = facts or {}
            holdings_json = _facts.get("portfolio_holdings", "")
            if holdings_json:
                logger.info("tools.stocks  portfolio_review  from holdings_json")
                result = await get_portfolio_review(holdings_json)
                if result:
                    return result
            # Holdings JSON missing or unparseable — fall through to flat list
            symbols = []
            portfolio_str = _facts.get("portfolio_stocks", "")
            if portfolio_str:
                symbols = [t.strip() if ("." in t.strip() or t.strip().startswith("^"))
                           else t.strip() + ".NS"
                           for t in portfolio_str.split(",") if t.strip()]

        logger.info("tools.stocks  symbols=%r", symbols)
        result = await get_indian_stocks(symbols)
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

    async def _fetch_url(self, tc: FetchUrlTool) -> str:
        """
        Fetch and extract article content via MCP /fetch endpoint.

        MCP handles: HTTP fetch → trafilatura extraction (F1=0.958) →
        sumy LexRank compaction (reads whole article) → TTL caching.

        Returns a structured prompt block:
          ARTICLE_META: title / author / date / word_count
          ARTICLE_ABSTRACT: 5 key sentences by LexRank (whole-article aware)
          ARTICLE_FULL: clean text, sentence-boundary capped at ~3000 chars
        """
        from .mcp_client import mcp_fetch_url

        url = tc.url
        if not url.startswith(("http://", "https://")):
            return f"Could not fetch: invalid URL {url!r}"

        logger.info("tools.fetch_url  url=%r", url[:120])
        result = await mcp_fetch_url(url)

        if not result:
            # MCP unreachable (network error calling MCP itself)
            logger.warning("tools.fetch_url.mcp_unavailable  url=%r", url[:80])
            return f"Could not fetch {url} — the page may be unavailable or behind a paywall."

        if result.get("error"):
            # MCP reached the endpoint but the fetch failed (DNS, timeout, HTTP error etc.)
            err_detail = result["error"]
            logger.warning("tools.fetch_url.fetch_failed  url=%r  err=%s", url[:80], err_detail[:120])
            return f"Could not read that page — {err_detail}"

        # Build a structured prompt block so the LLM gets both the map
        # (LexRank abstract) and the territory (full clean text)
        parts = []

        meta_parts = []
        if result.get("title"):
            meta_parts.append(f"title: {result['title']}")
        if result.get("author"):
            meta_parts.append(f"author: {result['author']}")
        if result.get("date"):
            meta_parts.append(f"published: {result['date']}")
        if result.get("word_count"):
            meta_parts.append(f"word_count: {result['word_count']}")
        if meta_parts:
            parts.append("ARTICLE_META:\n  " + "\n  ".join(meta_parts))

        if result.get("abstract"):
            parts.append(f"ARTICLE_ABSTRACT (key sentences, whole-article LexRank):\n{result['abstract']}")

        if result.get("text"):
            truncation_note = " [truncated]" if result.get("truncated") else ""
            parts.append(f"ARTICLE_FULL{truncation_note}:\n{result['text']}")

        if not parts:
            return f"Could not extract readable content from {url}"

        logger.info(
            "tools.fetch_url.ok  url=%r  words=%d  abstract_chars=%d",
            url[:80], result.get("word_count", 0), len(result.get("abstract", "")),
        )
        return "\n\n".join(parts)

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
        "fetch_url":  FetchUrlTool,
    }

    cls = _TOOL_MAP.get(tool_name)
    if cls is None:
        logger.warning("tools.parse — unknown tool=%r in LLM output", tool_name)
        return None

    try:
        payload = {**raw, "tool": tool_name}
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
