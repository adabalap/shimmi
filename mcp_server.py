#!/usr/bin/env python3
"""
mcp_server.py — Shimmi MCP Server v1.1.0

Changes vs v1.0.0:
  NEW  /currency?from=USD&to=INR&amount=100  — live exchange rates (no API key)
  NEW  /timezone?city=Tokyo                  — world clock lookup (no API key)
  FIX  /stocks returns consistent price formatting even when yfinance is slow
  FIX  /health includes uptime + version

Provides live-data tools over HTTP (JSON) on port 7000:
  GET /health
  GET /news?q=<query>&country=in
  GET /stocks?symbols=RELIANCE.NS,^NSEI
  GET /weather?city=Hyderabad&country=IN&days=3
  GET /currency?from=USD&to=INR&amount=100
  GET /timezone?city=London

No API keys required for stocks, weather, currency, or timezone.
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

app     = FastAPI(title="Shimmi MCP Server", version="1.1.0")
UTC     = timezone.utc
_HTTP:   Optional[httpx.AsyncClient] = None
_START   = time.time()


# ─────────────────────────── startup / shutdown ──────────────────────────────

@app.on_event("startup")
async def _startup():
    global _HTTP
    _HTTP = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    logger.info("🚀 MCP server v1.1.0 ready on :7000")


@app.on_event("shutdown")
async def _shutdown():
    if _HTTP:
        await _HTTP.aclose()


# ─────────────────────────── /health ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "version": "1.1.0",
        "uptime_s": round(time.time() - _START),
        "ts":      datetime.now(UTC).isoformat(),
    }


# ─────────────────────────── /news ───────────────────────────────────────────

GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")


@app.get("/news")
async def get_news(
    q:           str = Query("top headlines"),
    country:     str = Query("in"),
    lang:        str = Query("en"),
    max_results: int = Query(6, ge=1, le=10),
):
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
            results  = [
                {
                    "title":       a.get("title",       ""),
                    "description": a.get("description", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "url":         a.get("url",          ""),
                    "published":   a.get("publishedAt",  ""),
                }
                for a in articles
            ]
            return {"source": "gnews", "query": q, "count": len(results), "articles": results}
        except Exception as e:
            logger.warning("gnews.error  q=%r  err=%s", q, e)

    # Fallback: Google News RSS → rss2json
    rss_q   = q.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={rss_q}&hl=en-IN&gl=IN&ceid=IN:en"
    r2j_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&count={max_results}"
    try:
        resp  = await _HTTP.get(r2j_url)
        data  = resp.json()
        items = data.get("items", [])
        results = [
            {
                "title":       i.get("title",       ""),
                "description": i.get("description", "")[:200],
                "source":      i.get("author",       ""),
                "url":         i.get("link",         ""),
                "published":   i.get("pubDate",      ""),
            }
            for i in items
        ]
        return {"source": "rss2json+google_news", "query": q, "count": len(results), "articles": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"news fetch failed: {e}")


# ─────────────────────────── /stocks ─────────────────────────────────────────

_NSE_DEFAULTS = [
    "^NSEI", "^BSESN", "^NSEBANK",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS", "BAJFINANCE.NS",
]


@app.get("/stocks")
async def get_stocks(
    symbols: str = Query(",".join(_NSE_DEFAULTS[:6])),
):
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(status_code=503, detail="yfinance not installed: pip install yfinance")

    ticker_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    def _fetch_sync():
        out = []
        for sym in ticker_list:
            try:
                fi    = yf.Ticker(sym).fast_info
                price = getattr(fi, "last_price",     None)
                prev  = getattr(fi, "previous_close", None)
                cur   = getattr(fi, "currency",       "INR")
                name  = getattr(fi, "display_name",   None) or sym
                chg   = chg_pct = None
                if price and prev:
                    chg     = round(price - prev, 2)
                    chg_pct = round((chg / prev) * 100, 2)
                out.append({
                    "symbol":     sym,
                    "name":       name,
                    "price":      round(price, 2)     if price else None,
                    "prev_close": round(prev, 2)      if prev  else None,
                    "change":     chg,
                    "change_pct": chg_pct,
                    "currency":   cur,
                    "as_of":      datetime.now(UTC).isoformat(),
                })
            except Exception as e:
                out.append({"symbol": sym, "error": str(e)})
        return out

    results = await asyncio.to_thread(_fetch_sync)
    return {
        "source": "yfinance (Yahoo Finance, ~15min delay)",
        "count":  len(results),
        "stocks": results,
    }


# ─────────────────────────── /weather ────────────────────────────────────────

@app.get("/weather")
async def get_weather(
    city:    str = Query(...),
    country: str = Query(""),
    days:    int = Query(3, ge=1, le=7),
):
    geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country:
        geo_params["country_code"] = country.upper()

    geo_resp = await _HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=geo_params,
    )
    geo_data    = geo_resp.json()
    results_geo = geo_data.get("results", [])
    if not results_geo:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    loc       = results_geo[0]
    lat       = loc["latitude"]
    lon       = loc["longitude"]
    tz        = loc.get("timezone", "Asia/Kolkata")
    city_full = f"{loc.get('name', city)}, {loc.get('country', '')}"

    wx_resp = await _HTTP.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "timezone": tz,
            "forecast_days": days,
            "current": ",".join([
                "temperature_2m", "relative_humidity_2m",
                "apparent_temperature", "weather_code",
                "wind_speed_10m", "precipitation",
            ]),
            "daily": ",".join([
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "uv_index_max",
            ]),
        },
    )
    wx      = wx_resp.json()
    current = wx.get("current", {})
    daily   = wx.get("daily",   {})

    def _wmo(code: int) -> str:
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Heavy thunderstorm",
        }
        return codes.get(code, f"Code {code}")

    forecast_days = []
    for i, day in enumerate(daily.get("time", [])):
        forecast_days.append({
            "date":       day,
            "condition":  _wmo(daily.get("weather_code",      [None]*10)[i] or 0),
            "temp_max_c": daily.get("temperature_2m_max",     [None]*10)[i],
            "temp_min_c": daily.get("temperature_2m_min",     [None]*10)[i],
            "rain_mm":    daily.get("precipitation_sum",      [None]*10)[i],
            "uv_index":   daily.get("uv_index_max",           [None]*10)[i],
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


# ─────────────────────────── /currency ───────────────────────────────────────
# Uses exchangerate.host (free, no API key, ~180 currencies)

@app.get("/currency")
async def get_currency(
    from_cur: str = Query(..., alias="from", description="Source currency, e.g. USD"),
    to_cur:   str = Query(..., alias="to",   description="Target currency, e.g. INR"),
    amount:   float = Query(1.0, ge=0.01, description="Amount to convert"),
):
    """
    Live currency conversion via Frankfurter (ECB rates, free, no API key).
    Rates update daily from the European Central Bank.
    """
    from_cur = from_cur.upper().strip()
    to_cur   = to_cur.upper().strip()

    try:
        resp = await _HTTP.get(
            "https://api.frankfurter.app/latest",
            params={"from": from_cur, "to": to_cur},
        )
        data = resp.json()
        rate = data.get("rates", {}).get(to_cur)
        if rate is None:
            raise HTTPException(status_code=404, detail=f"Rate not found for {from_cur}→{to_cur}")
        converted = round(amount * rate, 4)
        return {
            "source":    "Frankfurter (ECB rates, daily)",
            "from":      from_cur,
            "to":        to_cur,
            "rate":      rate,
            "amount":    amount,
            "converted": converted,
            "as_of":     data.get("date", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"currency fetch failed: {e}")


# ─────────────────────────── /timezone ───────────────────────────────────────
# Uses worldtimeapi.org (free, no API key)

@app.get("/timezone")
async def get_timezone(
    city: str = Query(..., description="City name, e.g. Tokyo, London, New York"),
):
    """
    World clock for a city via Open-Meteo geocoding + worldtimeapi.
    Returns current local time, UTC offset, and timezone name.
    """
    # Geocode city to get timezone string
    geo_resp = await _HTTP.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
    )
    geo_data = geo_resp.json().get("results", [])
    if not geo_data:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    loc     = geo_data[0]
    tz_name = loc.get("timezone", "UTC")
    lat     = loc["latitude"]
    lon     = loc["longitude"]
    city_full = f"{loc.get('name', city)}, {loc.get('country', '')}"

    try:
        tz_resp = await _HTTP.get(
            f"https://timeapi.io/api/time/current/coordinate",
            params={"latitude": lat, "longitude": lon},
        )
        tz_data = tz_resp.json()
        local_time = tz_data.get("dateTime", "")
        utc_offset = tz_data.get("utcOffset", "")
        day_of_week = tz_data.get("dayOfWeek", "")
    except Exception:
        # Simple fallback: just return tz name
        local_time = ""
        utc_offset = ""
        day_of_week = ""

    return {
        "source":     "Open-Meteo geocoding + timeapi.io",
        "city":       city_full,
        "timezone":   tz_name,
        "local_time": local_time,
        "utc_offset": utc_offset,
        "day_of_week": day_of_week,
    }


# ─────────────────────────── entrypoint ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=7000, reload=False)
