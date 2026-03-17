#!/usr/bin/env python3
"""
mcp_server.py — Shimmi MCP Server v2.0.0

Changes vs v1.0.0:

  CACHE-1   TTL response cache for all external-API endpoints.
            Weather:  10-minute TTL (data changes slowly)
            Stocks:   3-minute TTL  (15-min delayed anyway, no point hammering)
            News:     5-minute TTL
            Currency: 1-hour TTL    (ECB rates update once daily)
            Timezone: 24-hour TTL   (city → tz mapping is static)
            Eliminates redundant external calls when multiple users ask about
            the same city/stock within the TTL window.

  CACHE-2   Cache key includes query parameters so city=Hyderabad and
            city=Mumbai never collide.

  FORMAT-1  New POST /format endpoint — deterministic WhatsApp formatting
            rules implemented in pure Python (zero LLM tokens).
            Replaces the Groq 8B _format_whatsapp() LLM call for routine
            markdown → WhatsApp conversion. Saves ~50-100K tokens/day.
            Rules: ** → *, bullet normalisation, table → bullets,
            code-fence removal, filler phrase stripping, length cap.

  STOCKS-2  Added per-ticker timeout guard in _fetch_sync() — a single
            slow/hung ticker no longer stalls the entire stocks call.

  HTTP-1    _HTTP client timeout reduced to 12s (was 30s) — stocks calls
            were occasionally hanging for 25+ seconds per ticker.

Endpoints:
  GET  /health
  GET  /news?q=&country=
  GET  /stocks?symbols=
  GET  /weather?city=&country=&days=
  GET  /currency?from=&to=&amount=
  GET  /timezone?city=
  GET  /datetime?tz=
  POST /format   {"text": "..."}  →  {"text": "...", "changed": bool}
"""

import os
import asyncio
import hashlib
import html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s mcp:%(message)s")

app = FastAPI(title="Shimmi MCP Server", version="2.0.0")

UTC  = timezone.utc
_HTTP: Optional[httpx.AsyncClient] = None

# ─────────────────────────────────────────────────────────────────────────────
# TTL Cache
# ─────────────────────────────────────────────────────────────────────────────

# Maps cache_key → (payload: Any, expires_at: float)
_CACHE: Dict[str, Tuple[Any, float]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _CACHE.get(key)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    _CACHE.pop(key, None)
    return None


def _cache_set(key: str, value: Any, ttl: float) -> None:
    _CACHE[key] = (value, time.monotonic() + ttl)


def _cache_key(*parts: str) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


# TTLs (seconds)
_TTL_WEATHER  = 600    # 10 min
_TTL_STOCKS   = 180    # 3 min
_TTL_NEWS     = 300    # 5 min
_TTL_CURRENCY = 3600   # 1 hour
_TTL_TIMEZONE = 86400  # 24 hours (tz mapping is static)


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global _HTTP
    _HTTP = httpx.AsyncClient(timeout=12.0, follow_redirects=True)
    logger.info("🚀 MCP server v2.0.0 ready on :7000")

@app.on_event("shutdown")
async def _shutdown():
    if _HTTP:
        await _HTTP.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ts": datetime.now(UTC).isoformat(),
        "cache_entries": len(_CACHE),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /news
# ─────────────────────────────────────────────────────────────────────────────

GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")


@app.get("/news")
async def get_news(
    q: str           = Query("top headlines"),
    country: str     = Query("in"),
    lang: str        = Query("en"),
    max_results: int = Query(6, ge=1, le=10),
):
    ck = _cache_key("news", q, country, lang)
    cached = _cache_get(ck)
    if cached:
        logger.info("💾 cache.hit  endpoint=news  q=%r", q[:40])
        return cached

    result = None

    if GNEWS_KEY:
        try:
            resp = await _HTTP.get(
                "https://gnews.io/api/v4/search",
                params={"q": q, "lang": lang, "country": country,
                        "max": max_results, "apikey": GNEWS_KEY},
            )
            data = resp.json()
            articles = [
                {
                    "title":       a.get("title", ""),
                    "description": a.get("description", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "url":         a.get("url", ""),
                    "published":   a.get("publishedAt", ""),
                }
                for a in data.get("articles", [])
            ]
            result = {"source": "gnews", "query": q, "count": len(articles), "articles": articles}
        except Exception as e:
            logger.warning("gnews.error  q=%r  err=%s", q, e)

    if not result:
        # Fallback: Google News RSS via rss2json
        try:
            rss_q   = q.replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={rss_q}&hl=en-IN&gl=IN&ceid=IN:en"
            r2j_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&count={max_results}"
            resp    = await _HTTP.get(r2j_url)
            items   = resp.json().get("items", [])
            articles = [
                {
                    "title":       i.get("title", ""),
                    "description": i.get("description", "")[:200],
                    "source":      i.get("author", ""),
                    "url":         i.get("link", ""),
                    "published":   i.get("pubDate", ""),
                }
                for i in items
            ]
            result = {"source": "rss2json+google_news", "query": q,
                      "count": len(articles), "articles": articles}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"news fetch failed: {e}")

    _cache_set(ck, result, _TTL_NEWS)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# /stocks
# ─────────────────────────────────────────────────────────────────────────────

_NSE_DEFAULTS = [
    "^NSEI", "^BSESN", "^NSEBANK",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS",
]

_STOCKS_PER_TICKER_TIMEOUT = 8.0   # seconds per ticker before giving up


@app.get("/stocks")
async def get_stocks(
    symbols: str = Query(
        ",".join(_NSE_DEFAULTS[:6]),
        description="Comma-separated Yahoo Finance tickers.",
    ),
):
    ck = _cache_key("stocks", symbols)
    cached = _cache_get(ck)
    if cached:
        logger.info("💾 cache.hit  endpoint=stocks  symbols=%r", symbols[:40])
        return cached

    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(status_code=503, detail="yfinance not installed")

    ticker_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    def _fetch_one(sym: str) -> dict:
        """Fetch a single ticker — called via asyncio.wait_for for timeout."""
        try:
            info       = yf.Ticker(sym).fast_info
            price      = getattr(info, "last_price",     None)
            prev_close = getattr(info, "previous_close", None)
            currency   = getattr(info, "currency",       "INR")
            name       = getattr(info, "display_name",   None) or sym
            change = change_pct = None
            if price and prev_close:
                change     = round(price - prev_close, 2)
                change_pct = round((change / prev_close) * 100, 2)
            return {
                "symbol":     sym,
                "name":       name,
                "price":      round(price, 2)      if price      else None,
                "prev_close": round(prev_close, 2) if prev_close else None,
                "change":     change,
                "change_pct": change_pct,
                "currency":   currency,
                "as_of":      datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {"symbol": sym, "error": str(e)[:120]}

    async def _fetch_with_timeout(sym: str) -> dict:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_fetch_one, sym),
                timeout=_STOCKS_PER_TICKER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("stocks.timeout  sym=%s  (>%.0fs)", sym, _STOCKS_PER_TICKER_TIMEOUT)
            return {"symbol": sym, "error": "timeout"}

    results = await asyncio.gather(*[_fetch_with_timeout(s) for s in ticker_list])

    payload = {
        "source": "yfinance (Yahoo Finance, ~15min delay)",
        "count":  len(results),
        "stocks": list(results),
    }
    _cache_set(ck, payload, _TTL_STOCKS)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# /weather
# ─────────────────────────────────────────────────────────────────────────────

def _wmo(code: int) -> str:
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Heavy thunderstorm + hail",
    }
    return codes.get(code, f"Code {code}")


@app.get("/weather")
async def get_weather(
    city: str    = Query(...),
    country: str = Query(""),
    days: int    = Query(3, ge=1, le=7),
):
    ck = _cache_key("weather", city.lower(), country.lower(), str(days))
    cached = _cache_get(ck)
    if cached:
        logger.info("💾 cache.hit  endpoint=weather  city=%r", city)
        return cached

    geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country:
        geo_params["country_code"] = country.upper()

    geo_resp = await _HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search", params=geo_params,
    )
    results_geo = geo_resp.json().get("results", [])
    if not results_geo:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    loc       = results_geo[0]
    lat, lon  = loc["latitude"], loc["longitude"]
    tz        = loc.get("timezone", "Asia/Kolkata")
    city_full = f"{loc.get('name', city)}, {loc.get('country', '')}"

    wx_resp = await _HTTP.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "timezone": tz,
            "forecast_days": days,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "weather_code,wind_speed_10m,precipitation",
            "daily":   "weather_code,temperature_2m_max,temperature_2m_min,"
                       "precipitation_sum,uv_index_max",
        },
    )
    wx      = wx_resp.json()
    current = wx.get("current", {})
    daily   = wx.get("daily", {})

    forecast_days_list = []
    for i, day in enumerate(daily.get("time", [])):
        forecast_days_list.append({
            "date":      day,
            "condition": _wmo(daily.get("weather_code", [0]*10)[i] or 0),
            "temp_max_c": daily.get("temperature_2m_max",  [None]*10)[i],
            "temp_min_c": daily.get("temperature_2m_min",  [None]*10)[i],
            "rain_mm":    daily.get("precipitation_sum",   [None]*10)[i],
            "uv_index":   daily.get("uv_index_max",        [None]*10)[i],
        })

    payload = {
        "source":   "Open-Meteo (free, no API key)",
        "city":     city_full,
        "lat":      lat, "lon": lon, "timezone": tz,
        "current": {
            "temp_c":       current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kph":     current.get("wind_speed_10m"),
            "rain_mm":      current.get("precipitation"),
            "condition":    _wmo(current.get("weather_code", 0) or 0),
            "as_of":        current.get("time"),
        },
        "forecast": forecast_days_list,
    }
    _cache_set(ck, payload, _TTL_WEATHER)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# /currency
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/currency")
async def get_currency(
    from_cur: str   = Query("USD", alias="from"),
    to_cur:   str   = Query("INR", alias="to"),
    amount:   float = Query(1.0),
):
    from_cur = from_cur.upper()
    to_cur   = to_cur.upper()

    ck = _cache_key("currency", from_cur, to_cur)
    cached = _cache_get(ck)
    if cached:
        # Recompute converted for the requested amount (rate is cached)
        result = dict(cached)
        result["amount"]    = amount
        result["converted"] = round(cached["rate"] * amount, 4)
        logger.info("💾 cache.hit  endpoint=currency  %s→%s", from_cur, to_cur)
        return result

    try:
        resp = await _HTTP.get(
            "https://api.frankfurter.app/latest",
            params={"from": from_cur, "to": to_cur},
        )
        data = resp.json()
        rate = data.get("rates", {}).get(to_cur)
        if rate is None:
            raise HTTPException(status_code=502, detail=f"Rate not found for {from_cur}→{to_cur}")
        result = {
            "from": from_cur, "to": to_cur, "rate": rate,
            "amount": amount, "converted": round(rate * amount, 4),
            "as_of": data.get("date", datetime.now(UTC).date().isoformat()),
        }
        _cache_set(ck, result, _TTL_CURRENCY)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Currency fetch failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /timezone
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/timezone")
async def get_timezone(city: str = Query(...)):
    from zoneinfo import ZoneInfo

    ck = _cache_key("timezone", city.lower())
    cached = _cache_get(ck)
    if cached:
        # Return fresh local_time but cached tz_name
        try:
            tz       = ZoneInfo(cached["timezone"])
            local_dt = datetime.now(tz)
            offset   = local_dt.strftime("%z")
            fmt_off  = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
            return {
                "city":       cached["city"],
                "timezone":   cached["timezone"],
                "local_time": local_dt.isoformat(timespec="seconds"),
                "utc_offset": fmt_off,
                "formatted":  local_dt.strftime("%H:%M %Z, %A %d %b %Y"),
            }
        except Exception:
            pass

    geo_resp = await _HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
    )
    results = geo_resp.json().get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    loc       = results[0]
    tz_name   = loc.get("timezone", "UTC")
    city_full = f"{loc.get('name', city)}, {loc.get('country', '')}"

    try:
        tz       = ZoneInfo(tz_name)
        local_dt = datetime.now(tz)
        offset   = local_dt.strftime("%z")
        fmt_off  = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
        result   = {
            "city": city_full, "timezone": tz_name,
            "local_time": local_dt.isoformat(timespec="seconds"),
            "utc_offset": fmt_off,
            "formatted":  local_dt.strftime("%H:%M %Z, %A %d %b %Y"),
        }
        # Cache the geo lookup, not the time (we recompute time on hit)
        _cache_set(ck, {"city": city_full, "timezone": tz_name}, _TTL_TIMEZONE)
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Timezone error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /datetime  (server clock for current time queries)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/datetime")
async def get_datetime(tz: str = Query("Asia/Kolkata")):
    from zoneinfo import ZoneInfo
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")
    now = datetime.now(zone)
    return {
        "timezone":      tz,
        "iso":           now.isoformat(timespec="seconds"),
        "date":          now.strftime("%Y-%m-%d"),
        "time":          now.strftime("%H:%M:%S"),
        "formatted":     now.strftime("%H:%M %Z, %A %d %b %Y"),
        "day_of_week":   now.strftime("%A"),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /format  — deterministic WhatsApp formatting (no LLM tokens)
# ─────────────────────────────────────────────────────────────────────────────

class FormatRequest(BaseModel):
    text: str


_FILLER_PHRASES = re.compile(
    r"^(great question!|certainly!|of course!|i'd be happy to|as an ai|"
    r"according to the search results|based on my knowledge|"
    r"i've already shared|sure!|absolutely!)\s*",
    re.IGNORECASE | re.MULTILINE,
)

_TABLE_ROW_RE  = re.compile(r"\|")
_TABLE_SEP_RE  = re.compile(r"^\s*\|[\s|:-]+\|\s*$")


def _format_for_whatsapp(text: str) -> str:
    """
    Pure-Python deterministic WhatsApp formatting.

    Rules applied (in order):
    1. HTML-entity decode
    2. Strip code fences (```)
    3. Convert **bold** → *bold*,  __italic__ → _italic_
    4. Normalise bullet chars (-, *, +) → •
    5. Remove Markdown headings (#)
    6. Convert Markdown tables → • bullet lists
    7. Strip filler opener phrases
    8. Collapse excess blank lines (max 2 → 1)
    9. Strip excess spaces
    10. Hard cap at 3 800 chars with ellipsis
    """
    if not text:
        return text

    out = html.unescape(text)

    # Remove code fences
    out = re.sub(r"```[^\n]*\n?", "", out)
    out = out.replace("`", "")

    # Bold / italic
    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out)
    out = re.sub(r"__(.+?)__",     r"_\1_", out)

    # Bullet normalisation
    out = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "• ", out)

    # Remove headings
    out = re.sub(r"(?m)^#{1,6}\s+", "", out)

    # Tables → bullets
    lines      = out.splitlines()
    converted  = []
    i          = 0
    while i < len(lines):
        ln = lines[i]
        if _TABLE_SEP_RE.match(ln):
            i += 1
            continue
        if _TABLE_ROW_RE.search(ln):
            cells = [c.strip() for c in ln.strip().strip("|").split("|") if c.strip()]
            if cells:
                converted.append("• " + "  —  ".join(cells))
            i += 1
            continue
        converted.append(ln)
        i += 1
    out = "\n".join(converted)

    # Remove filler openers
    out = _FILLER_PHRASES.sub("", out)

    # Collapse ≥3 blank lines → 1
    out = re.sub(r"\n{3,}", "\n\n", out)

    # Trailing / leading spaces per line
    out = "\n".join(ln.rstrip() for ln in out.splitlines())

    # Hard length cap
    out = out.strip()
    if len(out) > 3800:
        out = out[:3800].rstrip() + "…"

    return out


@app.post("/format")
async def format_text(req: FormatRequest):
    """
    Deterministic WhatsApp-safe formatting — no LLM required.
    Converts Markdown / HTML-entity text to WhatsApp markup.

    POST /format
    {"text": "**Hello** world!\n- item 1\n- item 2"}

    Returns:
    {"text": "*Hello* world!\n• item 1\n• item 2", "changed": true}
    """
    original = req.text
    formatted = _format_for_whatsapp(original)
    return {"text": formatted, "changed": formatted != original}


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=7000, reload=False)
