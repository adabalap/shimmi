"""
live_data.py — Shimmi v2.9.2

Direct free-API implementations for structured live data:
  • Weather   — Open-Meteo (free, no API key, ECMWF model)
  • Stocks    — yfinance / Yahoo Finance (free, no API key, ~15min delay)
  • News      — GNews free tier or Google News RSS fallback

Called by agent_engine._live_search when query intent is detected.
Results are pre-formatted as WhatsApp-ready strings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("app.live_data")

UTC          = timezone.utc
GNEWS_KEY    = os.getenv("GNEWS_API_KEY", "")
_HTTP: Optional[httpx.AsyncClient] = None


def _http_client() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None or _HTTP.is_closed:
        _HTTP = httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    return _HTTP


# ─────────────────────────── Weather ─────────────────────────────────────────

_WMO_CODES = {
    0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Foggy 🌫️", 48: "Icy fog 🌫️",
    51: "Light drizzle 🌦️", 53: "Drizzle 🌦️", 55: "Heavy drizzle 🌧️",
    61: "Light rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
    71: "Light snow 🌨️", 73: "Snow 🌨️", 75: "Heavy snow ❄️",
    80: "Rain showers 🌦️", 81: "Moderate showers 🌧️", 82: "Heavy showers ⛈️",
    95: "Thunderstorm ⛈️", 96: "Thunderstorm + hail ⛈️", 99: "Severe thunderstorm ⛈️",
}


async def get_weather(city: str, country: str = "India") -> Optional[str]:
    """
    Fetch current weather + 3-day forecast for city.
    Returns pre-formatted WhatsApp string or None on failure.
    """
    http = _http_client()
    try:
        # Step 1: Geocode
        geo_resp = await http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        geo = geo_resp.json().get("results", [])
        if not geo:
            logger.warning("live_data.weather  city_not_found=%r", city)
            return None

        loc      = geo[0]
        lat, lon = loc["latitude"], loc["longitude"]
        tz       = loc.get("timezone", "Asia/Kolkata")
        city_name = f"{loc.get('name', city)}, {loc.get('country', country)}"

        # Step 2: Weather
        wx_resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":      lat,
                "longitude":     lon,
                "timezone":      tz,
                "forecast_days": 3,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "weather_code,wind_speed_10m,precipitation",
                "daily":   "weather_code,temperature_2m_max,temperature_2m_min,"
                           "precipitation_sum,uv_index_max",
            },
        )
        wx = wx_resp.json()
        c  = wx.get("current", {})
        d  = wx.get("daily", {})

        def wmo(code):
            return _WMO_CODES.get(int(code or 0), f"Code {code}")

        lines = [
            f"📍 *{city_name}*",
            f"🌡️ *{c.get('temperature_2m', '?')}°C* · Feels {c.get('apparent_temperature', '?')}°C · {wmo(c.get('weather_code', 0))}",
            f"💧 Humidity {c.get('relative_humidity_2m', '?')}%  💨 Wind {c.get('wind_speed_10m', '?')} km/h  "
            f"🌧️ Rain {c.get('precipitation', 0) or 0} mm",
            "",
            "📅 *3-day forecast*",
        ]
        times = d.get("time", [])
        for i, day_str in enumerate(times[:3]):
            try:
                day_label = datetime.fromisoformat(day_str).strftime("%a %d %b")
            except ValueError:
                day_label = day_str
            t_max = d.get("temperature_2m_max", [None]*4)[i]
            t_min = d.get("temperature_2m_min", [None]*4)[i]
            rain  = d.get("precipitation_sum",  [0]*4)[i] or 0
            cond  = wmo(d.get("weather_code", [0]*4)[i] or 0)
            uv    = d.get("uv_index_max", [None]*4)[i]
            uv_str = f"  UV {uv}" if uv is not None else ""
            lines.append(f"• *{day_label}*: {cond} · ↑{t_max}°C / ↓{t_min}°C · Rain {rain}mm{uv_str}")

        lines.append("")
        lines.append("_Source: Open-Meteo (ECMWF)_")
        result = "\n".join(lines)
        logger.info("🌤️  live_data.weather.ok  city=%s  len=%d", city_name, len(result))
        return result

    except Exception as exc:
        logger.warning("live_data.weather.error  city=%r  err=%s", city, exc)
        return None


# ─────────────────────────── Indian Stocks ───────────────────────────────────

# Default watchlist: major indices + blue-chips
_DEFAULT_SYMBOLS = [
    ("^NSEI",       "Nifty 50"),
    ("^BSESN",      "Sensex"),
    ("^NSEBANK",    "Nifty Bank"),
    ("RELIANCE.NS", "Reliance"),
    ("TCS.NS",      "TCS"),
    ("INFY.NS",     "Infosys"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("ICICIBANK.NS","ICICI Bank"),
    ("WIPRO.NS",    "Wipro"),
    ("BAJFINANCE.NS","Bajaj Finance"),
]


async def get_indian_stocks(symbols: Optional[list[str]] = None) -> Optional[str]:
    """
    Fetch Indian stock/index prices via yfinance (Yahoo Finance, free).
    Returns pre-formatted WhatsApp string or None on failure.
    """
    try:
        import yfinance as yf  # pip install yfinance
    except ImportError:
        logger.warning("live_data.stocks — yfinance not installed (pip install yfinance)")
        return None

    watchlist = [(s, s) for s in symbols] if symbols else _DEFAULT_SYMBOLS

    def _fetch_sync() -> list[dict]:
        out = []
        for sym, label in watchlist:
            try:
                t     = yf.Ticker(sym)
                fi    = t.fast_info
                price = getattr(fi, "last_price", None)
                prev  = getattr(fi, "previous_close", None)
                cur   = getattr(fi, "currency", "INR")
                if price is None:
                    continue
                chg     = round(price - prev, 2) if prev else 0
                chg_pct = round((chg / prev) * 100, 2) if prev else 0
                out.append({
                    "label":     label,
                    "symbol":    sym,
                    "price":     price,
                    "chg":       chg,
                    "chg_pct":   chg_pct,
                    "currency":  cur,
                })
            except Exception as e:
                logger.debug("stocks.skip  sym=%s  err=%s", sym, e)
        return out

    stocks = await asyncio.to_thread(_fetch_sync)
    if not stocks:
        return None

    today = datetime.now(UTC).strftime("%a %d %b %Y")
    lines = [f"📈 *Indian Markets — {today}*", "_(~15 min delay · Yahoo Finance)_", ""]
    for s in stocks:
        arrow = "🟢" if s["chg_pct"] >= 0 else "🔴"
        cur   = "₹" if s["currency"] == "INR" else s["currency"] + " "
        lines.append(
            f"{arrow} *{s['label']}* — {cur}{s['price']:,.2f}  "
            f"({s['chg_pct']:+.2f}%)"
        )
    result = "\n".join(lines)
    logger.info("📈 live_data.stocks.ok  count=%d", len(stocks))
    return result


async def get_stock_by_name(query: str) -> Optional[str]:
    """
    Try to resolve a company name from the query and fetch its stock price.
    E.g. "Reliance" → RELIANCE.NS
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    # Extract potential ticker: uppercase words 2-12 chars
    words = re.findall(r"\b([A-Z][A-Za-z]{1,11})\b", query)
    candidates = [w.upper() + ".NS" for w in words if w.upper() not in
                  {"WHAT", "HOW", "THE", "ARE", "NSE", "BSE", "MARKET", "STOCK", "TODAY"}]
    if not candidates:
        return None

    return await get_indian_stocks(symbols=candidates[:5])


# ─────────────────────────── News ────────────────────────────────────────────

async def get_news(query: str = "India top news", country: str = "IN") -> Optional[str]:
    """
    Fetch latest news headlines.
    Uses GNews API (free, 100 req/day) when GNEWS_API_KEY is set,
    falls back to Google News RSS via rss2json public API.
    """
    http = _http_client()

    if GNEWS_KEY:
        try:
            resp = await http.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q":       query,
                    "lang":    "en",
                    "country": country.lower(),
                    "max":     6,
                    "apikey":  GNEWS_KEY,
                },
            )
            data     = resp.json()
            articles = data.get("articles", [])
            if articles:
                lines = [f"📰 *Latest News*", ""]
                for a in articles[:6]:
                    title  = a.get("title", "")
                    source = a.get("source", {}).get("name", "")
                    if title:
                        lines.append(f"• *{title[:90]}*" +
                                     (f"  _({source})_" if source else ""))
                lines.append("")
                lines.append("_Source: GNews_")
                result = "\n".join(lines)
                logger.info("📰 live_data.news.gnews  count=%d", len(articles))
                return result
        except Exception as exc:
            logger.warning("live_data.news.gnews_err  err=%s", exc)

    # Fallback: Google News RSS via rss2json public API (no key, ~10 req/min free)
    try:
        q_enc   = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en"
        r2j_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&count=6"
        resp    = await http.get(r2j_url)
        data    = resp.json()
        items   = data.get("items", [])
        if not items:
            return None
        lines = [f"📰 *Latest News*", ""]
        for it in items[:6]:
            title  = it.get("title", "")
            source = it.get("author", "")
            if title:
                # Strip trailing source appended by Google News: "Title - Source"
                title = re.sub(r"\s*-\s*[^-]{3,40}$", "", title).strip()
                lines.append(f"• *{title[:90]}*" + (f"  _({source})_" if source else ""))
        lines.append("")
        lines.append("_Source: Google News_")
        result = "\n".join(lines)
        logger.info("📰 live_data.news.rss  count=%d", len(items))
        return result
    except Exception as exc:
        logger.warning("live_data.news.rss_err  err=%s", exc)
        return None
