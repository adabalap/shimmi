"""
mcp_client.py — Shimmi v3.2.0

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
        resp = await _client().get(
            path,
            params={k: v for k, v in params.items() if v is not None},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        # httpx.HTTPStatusError.str() is often empty — repr() always has context
        detail = str(exc).strip() or repr(exc)
        logger.warning(
            "mcp.error  path=%s  type=%s  err=%s",
            path, type(exc).__name__, detail[:300],
        )
        return None


# ── Structured-data endpoints ─────────────────────────────────────────────────

async def mcp_news(query: str = "top headlines", country: str = "in") -> Optional[dict]:
    """Fetch news headlines. Returns dict with 'articles' list or None."""
    return await _get("/news", q=query, country=country)


async def mcp_stocks(symbols: str = "^NSEI,^BSESN,RELIANCE.NS,TCS.NS,INFY.NS") -> Optional[dict]:
    """Fetch Indian stock prices. Returns dict with 'stocks' list or None."""
    return await _get("/stocks", symbols=symbols)


async def mcp_weather(city: str, country: str = "IN", days: int = 3) -> Optional[dict]:
    """Fetch weather + 3-day forecast. Returns weather dict or None."""
    return await _get("/weather", city=city, country=country, days=days)


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
