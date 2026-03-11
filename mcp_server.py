#!/usr/bin/env python3
"""
mcp_server.py — Shimmi MCP Server v1.0.0

Provides three live-data tools over HTTP (JSON) on port 7000:
  GET /news?q=<query>&country=in    World/India news via GNews free tier
  GET /stocks?symbols=RELIANCE,TCS  Indian stocks via Yahoo Finance (yfinance, no key)
  GET /weather?city=Hyderabad       Weather via Open-Meteo (completely free, no key)

All endpoints return JSON. Designed to be called from agent_engine._live_search
or any HTTP client. No API keys required for stocks and weather.
For news, set GNEWS_API_KEY in .env (free tier: 100 requests/day).

Quickstart:
  pip install fastapi uvicorn yfinance httpx
  uvicorn mcp_server:app --host 0.0.0.0 --port 7000
"""

import os
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s mcp:%(message)s")

app = FastAPI(title="Shimmi MCP Server", version="1.0.0")

UTC = timezone.utc
_HTTP: Optional[httpx.AsyncClient] = None

# ─────────────────────────── startup / shutdown ──────────────────────────────

@app.on_event("startup")
async def _startup():
    global _HTTP
    _HTTP = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    logger.info("🚀 MCP server ready on :7000")

@app.on_event("shutdown")
async def _shutdown():
    if _HTTP:
        await _HTTP.aclose()

# ─────────────────────────── /health ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}

# ─────────────────────────── /news ───────────────────────────────────────────

GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")   # free tier at gnews.io — 100 req/day

@app.get("/news")
async def get_news(
    q: str = Query("top headlines", description="Search query"),
    country: str = Query("in", description="2-letter country code (in, us, gb …)"),
    lang: str = Query("en", description="Language"),
    max_results: int = Query(6, ge=1, le=10),
):
    """
    Fetch latest news headlines.
    Uses GNews API (free tier) when GNEWS_API_KEY is set,
    falls back to RSS via rss2json public API otherwise.
    """
    if GNEWS_KEY:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={q}&lang={lang}&country={country}&max={max_results}"
            f"&apikey={GNEWS_KEY}"
        )
        try:
            resp = await _HTTP.get(url)
            data = resp.json()
            articles = data.get("articles", [])
            results = [
                {
                    "title":       a.get("title", ""),
                    "description": a.get("description", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "url":         a.get("url", ""),
                    "published":   a.get("publishedAt", ""),
                }
                for a in articles
            ]
            return {"source": "gnews", "query": q, "count": len(results), "articles": results}
        except Exception as e:
            logger.warning("gnews.error  q=%r  err=%s", q, e)

    # Fallback: Google News RSS → rss2json (public, no key, ~10 req/min free)
    rss_q = q.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={rss_q}&hl=en-IN&gl=IN&ceid=IN:en"
    rss2json_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&count={max_results}"
    try:
        resp = await _HTTP.get(rss2json_url)
        data = resp.json()
        items = data.get("items", [])
        results = [
            {
                "title":       i.get("title", ""),
                "description": i.get("description", "")[:200],
                "source":      i.get("author", ""),
                "url":         i.get("link", ""),
                "published":   i.get("pubDate", ""),
            }
            for i in items
        ]
        return {"source": "rss2json+google_news", "query": q, "count": len(results), "articles": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"news fetch failed: {e}")


# ─────────────────────────── /stocks ─────────────────────────────────────────

# yfinance is a well-known unofficial Yahoo Finance wrapper, completely free
# NSE tickers on Yahoo Finance use the ".NS" suffix, BSE uses ".BO"
_NSE_DEFAULTS = [
    "^NSEI",       # Nifty 50
    "^BSESN",      # Sensex / BSE 30
    "^NSEBANK",    # Nifty Bank
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS",
]

@app.get("/stocks")
async def get_stocks(
    symbols: str = Query(
        ",".join(_NSE_DEFAULTS[:6]),
        description="Comma-separated Yahoo Finance tickers. "
                    "NSE stocks = TICKER.NS, e.g. RELIANCE.NS,TCS.NS. "
                    "Indices: ^NSEI=Nifty50, ^BSESN=Sensex",
    ),
):
    """
    Fetch live Indian stock / index prices via yfinance (Yahoo Finance).
    No API key required. Data is delayed ~15 min for free tier.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(status_code=503, detail="yfinance not installed: pip install yfinance")

    ticker_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    results = []

    def _fetch_sync():
        out = []
        for sym in ticker_list:
            try:
                t = yf.Ticker(sym)
                info = t.fast_info
                price      = getattr(info, "last_price",      None)
                prev_close = getattr(info, "previous_close",  None)
                currency   = getattr(info, "currency",        "INR")
                name       = getattr(info, "display_name", None) or sym

                change     = None
                change_pct = None
                if price and prev_close:
                    change     = round(price - prev_close, 2)
                    change_pct = round((change / prev_close) * 100, 2)

                out.append({
                    "symbol":     sym,
                    "name":       name,
                    "price":      round(price, 2)     if price      else None,
                    "prev_close": round(prev_close, 2) if prev_close else None,
                    "change":     change,
                    "change_pct": change_pct,
                    "currency":   currency,
                    "as_of":      datetime.now(UTC).isoformat(),
                })
            except Exception as e:
                out.append({"symbol": sym, "error": str(e)})
        return out

    results = await asyncio.to_thread(_fetch_sync)
    return {
        "source": "yfinance (Yahoo Finance, ~15min delay)",
        "count": len(results),
        "stocks": results,
    }


# ─────────────────────────── /weather ────────────────────────────────────────

# Open-Meteo: completely free, no API key, high accuracy, ECMWF model
# Geocoding: Open-Meteo's own geocoding API (also free)

@app.get("/weather")
async def get_weather(
    city: str = Query(..., description="City name, e.g. Hyderabad or London"),
    country: str = Query("", description="Optional 2-letter country code, e.g. IN, GB"),
    days: int = Query(3, ge=1, le=7, description="Forecast days (1-7)"),
):
    """
    Fetch current weather + 3-day forecast via Open-Meteo (free, no API key).
    First geocodes city → lat/lon, then fetches weather for those coordinates.
    """
    # Step 1: Geocode city name → lat/lon
    geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country:
        geo_params["country_code"] = country.upper()

    geo_resp = await _HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=geo_params,
    )
    geo_data = geo_resp.json()
    results_geo = geo_data.get("results", [])
    if not results_geo:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    loc         = results_geo[0]
    lat         = loc["latitude"]
    lon         = loc["longitude"]
    tz          = loc.get("timezone", "Asia/Kolkata")
    city_full   = f"{loc.get('name', city)}, {loc.get('country', '')}"

    # Step 2: Fetch weather
    wx_resp = await _HTTP.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":               lat,
            "longitude":              lon,
            "timezone":               tz,
            "forecast_days":          days,
            "current":                ",".join([
                "temperature_2m", "relative_humidity_2m",
                "apparent_temperature", "weather_code",
                "wind_speed_10m", "precipitation",
            ]),
            "daily":                  ",".join([
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "uv_index_max",
            ]),
        },
    )
    wx = wx_resp.json()
    current = wx.get("current", {})
    daily   = wx.get("daily", {})

    # WMO weather interpretation codes → human description
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

    forecast_days = []
    times = daily.get("time", [])
    for i, day in enumerate(times):
        forecast_days.append({
            "date":        day,
            "condition":   _wmo(daily.get("weather_code", [None]*10)[i] or 0),
            "temp_max_c":  daily.get("temperature_2m_max", [None]*10)[i],
            "temp_min_c":  daily.get("temperature_2m_min", [None]*10)[i],
            "rain_mm":     daily.get("precipitation_sum",  [None]*10)[i],
            "uv_index":    daily.get("uv_index_max",       [None]*10)[i],
        })

    return {
        "source":   "Open-Meteo (free, no API key)",
        "city":     city_full,
        "lat":      lat,
        "lon":      lon,
        "timezone": tz,
        "current": {
            "temp_c":       current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kph":     current.get("wind_speed_10m"),
            "rain_mm":      current.get("precipitation"),
            "condition":    _wmo(current.get("weather_code", 0) or 0),
            "as_of":        current.get("time"),
        },
        "forecast": forecast_days,
    }


# ─────────────────────────── entrypoint ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=7000, reload=False)
