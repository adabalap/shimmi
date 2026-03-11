"""
mcp_client.py — Shimmi v3.0.3

Changes vs v3.0.1:
  NEW  mcp_currency(from_cur, to_cur, amount) — exchange rates via Frankfurter
  NEW  mcp_timezone(city) — world clock lookup
  ARCH live_data.py now calls this client instead of hitting external APIs directly

Thin async client for the local MCP server (mcp_server.py on :7000).
Falls back gracefully (returns None) if MCP server is not running.
"""
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("app.mcp")

_MCP_BASE = os.getenv("MCP_SERVER_URL", "http://localhost:7000")
_TIMEOUT  = float(os.getenv("MCP_TIMEOUT", "8"))

_CLIENT: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(base_url=_MCP_BASE, timeout=_TIMEOUT)
    return _CLIENT


async def _get(path: str, **params) -> Optional[dict]:
    try:
        resp = await _client().get(path, params={k: v for k, v in params.items() if v is not None})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("mcp.error  path=%s  err=%s", path, exc)
        return None


# ── Tool functions ────────────────────────────────────────────────────────────

async def mcp_news(query: str = "top headlines", country: str = "in") -> Optional[dict]:
    """Fetch news via MCP server. Returns dict with 'articles' list or None."""
    return await _get("/news", q=query, country=country)


async def mcp_stocks(symbols: str = "^NSEI,^BSESN,RELIANCE.NS,TCS.NS,INFY.NS") -> Optional[dict]:
    """Fetch Indian stock prices via MCP server. Returns dict with 'stocks' list or None."""
    return await _get("/stocks", symbols=symbols)


async def mcp_weather(city: str, country: str = "IN", days: int = 3) -> Optional[dict]:
    """Fetch weather for a city via MCP server. Returns weather dict or None."""
    return await _get("/weather", city=city, country=country, days=days)


async def mcp_currency(
    from_cur: str,
    to_cur:   str,
    amount:   float = 1.0,
) -> Optional[dict]:
    """
    Live currency conversion via MCP server.
    Returns dict with {from, to, rate, amount, converted, as_of} or None.
    Example: mcp_currency("USD", "INR", 100) → {"converted": 8350.0, "rate": 83.5, ...}
    """
    return await _get("/currency", **{"from": from_cur, "to": to_cur, "amount": amount})


async def mcp_timezone(city: str) -> Optional[dict]:
    """
    World clock for a city via MCP server.
    Returns dict with {city, timezone, local_time, utc_offset} or None.
    Example: mcp_timezone("Tokyo") → {"local_time": "2026-03-10T15:45:00", ...}
    """
    return await _get("/timezone", city=city)


async def mcp_health() -> Optional[dict]:
    """Ping the MCP server health endpoint."""
    return await _get("/health")


def mcp_available() -> bool:
    """Quick check if MCP_SERVER_URL is configured (doesn't ping the server)."""
    return bool(_MCP_BASE)
