"""
mcp_client.py — Shimmi v3.17.0

Changes vs v3.2.0:
  FIX-STOCKS  Per-endpoint timeouts: stocks=20s, news=10s, weather=10s (was 8s flat).
              The /stocks ReadTimeout error was causing PAYTM and all stock queries
              to fail — yfinance cold-starts take 15-20s on first call.
  FIX         Explicit httpx.Timeout object (connect=5s, read=default) for cleaner
              per-call overrides. timeout param added to _get().

Changes vs v3.0.3:
  FIX   Duplicate mcp_format() definition removed (was defined twice).
  FIX   _get() now logs repr(exc) so httpx.HTTPStatusError — which has an
        empty str() — produces a meaningful error message instead of
        "mcp.error  path=/stocks  err=".
  NEW   mcp_format(text) — offloads WhatsApp formatting to the deterministic
        MCP /format endpoint, saving a Groq 8B LLM call on every long reply.
"""
from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Optional

import httpx

logger = logging.getLogger("app.mcp")

# Context variable: set once per message, read by every _get() call.
# Attaches X-Shimmi-Trace-ID to MCP requests for cross-log correlation.
_current_trace_id: ContextVar[str] = ContextVar("shimmi_trace_id", default="")

def set_trace_id(trace_id: str) -> None:
    _current_trace_id.set(trace_id[:16] if trace_id else "")

_MCP_BASE = os.getenv("MCP_SERVER_URL", "http://localhost:7000")
_TIMEOUT  = float(os.getenv("MCP_TIMEOUT", "12"))

# Per-endpoint timeout overrides — stocks/yfinance is slow on cold start
_TIMEOUT_STOCKS  = float(os.getenv("MCP_TIMEOUT_STOCKS",  "20"))
_TIMEOUT_NEWS    = float(os.getenv("MCP_TIMEOUT_NEWS",    "10"))
_TIMEOUT_WEATHER = float(os.getenv("MCP_TIMEOUT_WEATHER", "10"))
_TIMEOUT_FETCH   = float(os.getenv("MCP_TIMEOUT_FETCH",   "25"))  # articles can be slow

_CLIENT: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _CLIENT
    # Use connect=5s but no read timeout at client level — per-call timeouts handle it
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(
            base_url=_MCP_BASE,
            timeout=httpx.Timeout(connect=5.0, read=_TIMEOUT, write=5.0, pool=2.0),
        )
    return _CLIENT


async def _get(
    path: str,
    timeout: Optional[float] = None,
    trace_id: Optional[str] = None,
    **params,
) -> Optional[dict]:
    """
    HTTP GET to MCP server.
    trace_id: if provided, sent as X-Shimmi-Trace-ID header so MCP logs
              can be correlated with bot logs by grepping one ID.
    """
    _tid = (trace_id or _current_trace_id.get() or "").strip()
    headers = {"X-Shimmi-Trace-ID": _tid} if _tid else {}
    try:
        resp = await _client().get(
            path,
            params={k: v for k, v in params.items() if v is not None},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        detail = str(exc).strip() or repr(exc)
        logger.warning(
            "mcp.error  path=%s  type=%s  err=%s",
            path, type(exc).__name__, detail[:300],
        )
        return None


# ── Structured-data endpoints ─────────────────────────────────────────────────

async def mcp_news(query: str = "top headlines", country: str = "in") -> Optional[dict]:
    """Fetch news headlines. Returns dict with 'articles' list or None."""
    return await _get("/news", timeout=_TIMEOUT_NEWS, q=query, country=country)


async def mcp_news_briefing(city: str = "Hyderabad") -> Optional[dict]:
    """
    Fetch structured multi-category news briefing (5 sections in parallel).
    Returns dict with 'sections' list, each section having category/emoji/articles.
    Uses /news/briefing endpoint — 30-min TTL cache on MCP side.
    """
    timeout = float(os.getenv("MCP_TIMEOUT_BRIEFING", "12"))
    return await _get("/news/briefing", timeout=timeout, city=city)


async def mcp_stocks(symbols: str = "^NSEI,^BSESN,RELIANCE.NS,TCS.NS,INFY.NS") -> Optional[dict]:
    """Fetch Indian stock prices. Returns dict with 'stocks' list or None."""
    # FIX-STOCKS-2: yfinance cold-starts can take 15s+ — use dedicated longer timeout
    return await _get("/stocks", timeout=_TIMEOUT_STOCKS, symbols=symbols)


async def mcp_weather(city: str, country: str = "IN", days: int = 3) -> Optional[dict]:
    """Fetch weather + 3-day forecast. Returns weather dict or None."""
    return await _get("/weather", timeout=_TIMEOUT_WEATHER, city=city, country=country, days=days)


async def mcp_currency(
    from_cur: str,
    to_cur:   str,
    amount:   float = 1.0,
) -> Optional[dict]:
    """
    Live currency conversion via Frankfurter (free ECB rates, no API key).
    Returns {from, to, rate, amount, converted, as_of} or None.
    """
    return await _get("/currency", **{"from": from_cur, "to": to_cur, "amount": amount})


async def mcp_timezone(city: str) -> Optional[dict]:
    """
    Current local time for any city.
    Returns {city, timezone, local_time, utc_offset, formatted} or None.
    """
    return await _get("/timezone", city=city)


async def mcp_fetch_url(url: str) -> Optional[dict]:
    """
    Fetch and extract article content from a URL via MCP /fetch.
    MCP handles: HTTP fetch, trafilatura extraction, LexRank compaction, caching.

    Returns structured dict with: url, title, author, date, abstract, text,
    word_count, truncated. Returns None if the URL cannot be fetched.
    """
    return await _get("/fetch", timeout=_TIMEOUT_FETCH, url=url)


# ── Utility endpoints ─────────────────────────────────────────────────────────

async def mcp_format(text: str) -> Optional[str]:
    """
    Deterministic WhatsApp formatting via MCP /format — zero LLM tokens.

    Replaces the Groq 8B _format_whatsapp() call for every reply longer
    than _MIN_FORMAT_LEN. The MCP endpoint applies a fixed rule set (bullet
    normalisation, bold conversion, length cap) with no inference cost.

    Returns the formatted string, or None if MCP is unavailable (caller
    falls back to the LLM formatter automatically).
    """
    try:
        resp = await _client().post("/format", json={"text": text}, timeout=3.0)
        resp.raise_for_status()
        formatted = resp.json().get("text")
        return formatted if formatted else None
    except Exception as exc:
        detail = str(exc).strip() or repr(exc)
        logger.debug("mcp.format_skip  err=%s", detail[:120])
        return None


async def mcp_health() -> Optional[dict]:
    """Ping the MCP server health endpoint."""
    return await _get("/health")


def mcp_available() -> bool:
    """Quick check if MCP_SERVER_URL is configured (no network call)."""
    return bool(_MCP_BASE)
