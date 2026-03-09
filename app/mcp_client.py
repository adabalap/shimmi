"""
mcp_client.py — Shimmi v2.9.2

Thin async client for the local MCP server (mcp_server.py).
Falls back gracefully if MCP server is not running.

Usage in agent_engine:
    from .mcp_client import mcp_news, mcp_stocks, mcp_weather
    result = await mcp_weather(city="Hyderabad", country="IN")
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("app.mcp")

_MCP_BASE = os.getenv("MCP_SERVER_URL", "http://localhost:7000")
_TIMEOUT  = float(os.getenv("MCP_TIMEOUT", "8"))

# Shared client (initialised lazily)
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


async def mcp_news(query: str = "top headlines", country: str = "in") -> Optional[dict]:
    """Fetch news via MCP server. Returns dict with 'articles' list or None on failure."""
    return await _get("/news", q=query, country=country)


async def mcp_stocks(symbols: str = "^NSEI,^BSESN,RELIANCE.NS,TCS.NS,INFY.NS") -> Optional[dict]:
    """Fetch Indian stock prices via MCP server. Returns dict with 'stocks' list or None."""
    return await _get("/stocks", symbols=symbols)


async def mcp_weather(city: str, country: str = "IN", days: int = 3) -> Optional[dict]:
    """Fetch weather for a city via MCP server. Returns weather dict or None on failure."""
    return await _get("/weather", city=city, country=country, days=days)


def mcp_available() -> bool:
    """Quick check if MCP_SERVER_URL is configured (doesn't ping the server)."""
    return bool(_MCP_BASE)
