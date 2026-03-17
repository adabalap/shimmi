"""
live_data.py — Shimmi v3.0.3

Changes vs v3.0.1:
  ARCH-1  All data fetching now routes through mcp_client → mcp_server (port 7000)
          as the canonical source. Direct HTTP calls to Open-Meteo / Yahoo /
          GNews remain as fallbacks if the MCP server is unreachable.

  WHY:    The MCP sidecar (shimmi-mcp.service) is already running.  Duplicating
          the same HTTP calls here wastes connections, makes config split across
          two codebases, and means bug fixes have to be applied in two places.
          With MCP-first routing, the bot continues working even if the MCP
          service is temporarily down (fallback kicks in automatically).

  Functions exposed to agent_engine:
    get_weather(city, country)      → WhatsApp-formatted string | None
    get_indian_stocks(symbols)      → WhatsApp-formatted string | None
    get_news(query, country)        → WhatsApp-formatted string | None
    get_stock_by_name(query)        → WhatsApp-formatted string | None
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
UTC       = timezone.utc
GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")

_HTTP: Optional[httpx.AsyncClient] = None


def _http_client() -> httpx.AsyncClient:
    global _HTTP
    if _HTTP is None or _HTTP.is_closed:
        _HTTP = httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    return _HTTP


# ─────────────────────────── Helpers ─────────────────────────────────────────

_WMO_CODES = {
    0: "Clear sky ☀️", 1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Foggy 🌫️", 48: "Icy fog 🌫️",
    51: "Light drizzle 🌦️", 53: "Drizzle 🌦️", 55: "Heavy drizzle 🌧️",
    61: "Light rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
    71: "Light snow 🌨️", 73: "Snow 🌨️", 75: "Heavy snow ❄️",
    80: "Rain showers 🌦️", 81: "Moderate showers 🌧️", 82: "Heavy showers ⛈️",
    95: "Thunderstorm ⛈️", 96: "Thunderstorm + hail ⛈️", 99: "Severe thunderstorm ⛈️",
}


def _wmo(code) -> str:
    return _WMO_CODES.get(int(code or 0), f"Code {code}")


# ─────────────────────────── Weather ─────────────────────────────────────────

def _format_weather_mcp(data: dict) -> str:
    """Format MCP /weather JSON response → WhatsApp string."""
    city    = data.get("city", "Unknown")
    cur     = data.get("current", {})
    fc_days = data.get("forecast", [])

    lines = [
        f"📍 *{city}*",
        f"🌡️ *{cur.get('temp_c', '?')}°C*  Feels {cur.get('feels_like_c', '?')}°C  "
        f"{cur.get('condition', '')}",
        f"💧 Humidity {cur.get('humidity_pct', '?')}%  "
        f"💨 Wind {cur.get('wind_kph', '?')} km/h  "
        f"🌧️ Rain {cur.get('rain_mm', 0) or 0} mm",
        "",
        "📅 *3-day forecast*",
    ]
    for day in fc_days[:3]:
        try:
            label = datetime.fromisoformat(day["date"]).strftime("%a %d %b")
        except Exception:
            label = day.get("date", "")
        cond  = day.get("condition",  "")
        t_max = day.get("temp_max_c", "?")
        t_min = day.get("temp_min_c", "?")
        rain  = day.get("rain_mm",    0) or 0
        uv    = day.get("uv_index")
        uv_s  = f"  UV {uv}" if uv is not None else ""
        lines.append(f"• *{label}*: {cond} · ↑{t_max}°C / ↓{t_min}°C · Rain {rain}mm{uv_s}")

    lines += ["", "_Source: Open-Meteo via MCP (ECMWF)_"]
    return "\n".join(lines)


async def get_weather(city: str, country: str = "India") -> Optional[str]:
    """Fetch weather. Tries MCP server first, then direct Open-Meteo fallback."""

    # ── Try MCP ──────────────────────────────────────────────────────────────
    try:
        from .mcp_client import mcp_weather
        data = await mcp_weather(city=city, country=country, days=3)
        if data and data.get("current"):
            result = _format_weather_mcp(data)
            logger.info("🌤️  live_data.weather.mcp  city=%s  len=%d", city, len(result))
            return result
    except Exception as e:
        logger.debug("live_data.weather.mcp_skip  err=%s", e)

    # ── Fallback: direct Open-Meteo ──────────────────────────────────────────
    http = _http_client()
    try:
        geo_resp = await http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        geo = geo_resp.json().get("results", [])
        if not geo:
            logger.warning("live_data.weather  city_not_found=%r", city)
            return None
        loc       = geo[0]
        lat, lon  = loc["latitude"], loc["longitude"]
        tz        = loc.get("timezone", "Asia/Kolkata")
        city_name = f"{loc.get('name', city)}, {loc.get('country', country)}"
        wx_resp = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "timezone": tz,
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
        lines = [
            f"📍 *{city_name}*",
            f"🌡️ *{c.get('temperature_2m', '?')}°C*  Feels {c.get('apparent_temperature', '?')}°C  "
            f"{_wmo(c.get('weather_code', 0))}",
            f"💧 Humidity {c.get('relative_humidity_2m', '?')}%  "
            f"💨 Wind {c.get('wind_speed_10m', '?')} km/h  "
            f"🌧️ Rain {c.get('precipitation', 0) or 0} mm",
            "", "📅 *3-day forecast*",
        ]
        for i, day_str in enumerate(d.get("time", [])[:3]):
            try:
                day_label = datetime.fromisoformat(day_str).strftime("%a %d %b")
            except ValueError:
                day_label = day_str
            t_max = d.get("temperature_2m_max", [None]*4)[i]
            t_min = d.get("temperature_2m_min", [None]*4)[i]
            rain  = d.get("precipitation_sum",  [0]*4)[i] or 0
            cond  = _wmo(d.get("weather_code", [0]*4)[i] or 0)
            uv    = d.get("uv_index_max", [None]*4)[i]
            uv_s  = f"  UV {uv}" if uv is not None else ""
            lines.append(f"• *{day_label}*: {cond} · ↑{t_max}°C / ↓{t_min}°C · Rain {rain}mm{uv_s}")
        lines += ["", "_Source: Open-Meteo (ECMWF)_"]
        result = "\n".join(lines)
        logger.info("🌤️  live_data.weather.direct  city=%s  len=%d", city_name, len(result))
        return result
    except Exception as exc:
        logger.warning("live_data.weather.error  city=%r  err=%s", city, exc)
        return None


# ─────────────────────────── Indian Stocks ───────────────────────────────────

_DEFAULT_SYMBOLS = [
    ("^NSEI",        "Nifty 50"),
    ("^BSESN",       "Sensex"),
    ("^NSEBANK",     "Nifty Bank"),
    ("RELIANCE.NS",  "Reliance"),
    ("TCS.NS",       "TCS"),
    ("INFY.NS",      "Infosys"),
    ("HDFCBANK.NS",  "HDFC Bank"),
    ("ICICIBANK.NS", "ICICI Bank"),
    ("WIPRO.NS",     "Wipro"),
    ("BAJFINANCE.NS","Bajaj Finance"),
]


def _format_stocks_mcp(data: dict) -> Optional[str]:
    """Format MCP /stocks JSON response → WhatsApp string."""
    stocks = data.get("stocks", [])
    if not stocks:
        return None
    today = datetime.now(UTC).strftime("%a %d %b %Y")
    lines = [f"📈 *Indian Markets — {today}*", "_(~15 min delay · Yahoo Finance via MCP)_", ""]
    skipped = []
    for s in stocks:
        if s.get("error"):
            skipped.append(s.get("symbol", "?"))
            continue
        price   = s.get("price")
        chg_pct = s.get("change_pct", 0) or 0
        cur     = "₹" if s.get("currency", "INR") == "INR" else (s.get("currency", "") + " ")
        arrow   = "🟢" if chg_pct >= 0 else "🔴"
        name    = s.get("name", s.get("symbol", ""))
        if price is None:
            skipped.append(s.get("symbol", "?"))
            continue
        lines.append(f"{arrow} *{name}* — {cur}{price:,.2f}  ({chg_pct:+.2f}%)")
    if len(lines) <= 3:
        # All symbols had errors or null prices — return an informative message
        sym_list = ", ".join(skipped) if skipped else "requested symbols"
        return (
            f"📊 *Stock Data Unavailable*\n"
            f"Could not fetch price for {sym_list}.\n"
            f"_(Yahoo Finance may not recognise this ticker or market is closed. "
            f"Try adding .NS for NSE or .BO for BSE, e.g. PAYTM.NS)_"
        )
    if skipped:
        lines.append(f"\n_⚠️ No data for: {', '.join(skipped)}_")
    return "\n".join(lines)


async def get_indian_stocks(symbols: Optional[list] = None) -> Optional[str]:
    """Fetch Indian stocks. Tries MCP server first, then yfinance fallback."""

    # ── Try MCP ──────────────────────────────────────────────────────────────
    try:
        from .mcp_client import mcp_stocks
        sym_str = ",".join(symbols) if symbols else ",".join(s for s, _ in _DEFAULT_SYMBOLS[:8])
        data = await mcp_stocks(symbols=sym_str)
        if data and data.get("stocks"):
            result = _format_stocks_mcp(data)
            if result:
                logger.info("📈 live_data.stocks.mcp  count=%d  symbols=%r",
                            len(data["stocks"]), sym_str)
                return result
    except Exception as e:
        logger.debug("live_data.stocks.mcp_skip  err=%s", e)

    # ── Fallback: direct yfinance ────────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("live_data.stocks — yfinance not installed")
        return None

    watchlist = [(s, s) for s in symbols] if symbols else _DEFAULT_SYMBOLS

    def _fetch_sync() -> list:
        out = []
        for sym, label in watchlist:
            try:
                fi    = yf.Ticker(sym).fast_info
                price = getattr(fi, "last_price", None)
                prev  = getattr(fi, "previous_close", None)
                cur   = getattr(fi, "currency", "INR")
                if price is None:
                    continue
                chg     = round(price - prev, 2) if prev else 0
                chg_pct = round((chg / prev) * 100, 2) if prev else 0
                out.append({"label": label, "symbol": sym, "price": price,
                             "chg": chg, "chg_pct": chg_pct, "currency": cur})
            except Exception as e:
                logger.debug("stocks.skip  sym=%s  err=%s", sym, e)
        return out

    stocks = await asyncio.to_thread(_fetch_sync)
    if not stocks:
        sym_list = ", ".join(str(s) for s in symbols) if symbols else "requested symbols"
        logger.info("📈 live_data.stocks.no_data  symbols=%r", sym_list)
        return (
            f"📊 *Stock Data Unavailable*\n"
            f"Could not fetch price for {sym_list}.\n"
            "_(Market may be closed, or the ticker may not be recognised by Yahoo Finance. "
            "Try NSE format: PAYTM.NS, RELIANCE.NS, etc.)_"
        )
    today = datetime.now(UTC).strftime("%a %d %b %Y")
    lines = [f"📈 *Indian Markets — {today}*", "_(~15 min delay · Yahoo Finance)_", ""]
    for s in stocks:
        arrow = "🟢" if s["chg_pct"] >= 0 else "🔴"
        cur   = "₹" if s["currency"] == "INR" else s["currency"] + " "
        lines.append(f"{arrow} *{s['label']}* — {cur}{s['price']:,.2f}  ({s['chg_pct']:+.2f}%)")
    result = "\n".join(lines)
    logger.info("📈 live_data.stocks.direct  count=%d", len(stocks))
    return result


async def get_stock_by_name(query: str) -> Optional[str]:
    words      = re.findall(r"\b([A-Z][A-Za-z]{1,11})\b", query)
    candidates = [
        w.upper() + ".NS" for w in words
        if w.upper() not in {"WHAT", "HOW", "THE", "ARE", "NSE", "BSE",
                              "MARKET", "STOCK", "TODAY", "GIVE", "TELL"}
    ]
    if not candidates:
        return None
    return await get_indian_stocks(symbols=candidates[:5])


# ─────────────────────────── News ────────────────────────────────────────────

def _format_news_mcp(data: dict, query: str) -> Optional[str]:
    """Format MCP /news JSON response → WhatsApp string."""
    articles = data.get("articles", [])
    if not articles:
        return None
    lines = ["📰 *Latest News*", ""]
    for a in articles[:6]:
        title  = a.get("title", "").strip()
        source = a.get("source", "")
        if title:
            title = re.sub(r"\s*-\s*[^-]{3,40}$", "", title).strip()
            lines.append(f"• *{title[:90]}*" + (f"  _({source})_" if source else ""))
    lines += ["", f"_Source: {data.get('source', 'MCP News')}_"]
    return "\n".join(lines)


async def get_news(query: str = "India top news", country: str = "IN") -> Optional[str]:
    """Fetch news headlines. Tries MCP first, then GNews / RSS fallback."""

    # ── Try MCP ──────────────────────────────────────────────────────────────
    try:
        from .mcp_client import mcp_news
        data = await mcp_news(query=query, country=country.lower())
        if data and data.get("articles"):
            result = _format_news_mcp(data, query)
            if result:
                logger.info("📰 live_data.news.mcp  count=%d", len(data["articles"]))
                return result
    except Exception as e:
        logger.debug("live_data.news.mcp_skip  err=%s", e)

    # ── Fallback: GNews direct ───────────────────────────────────────────────
    http = _http_client()
    if GNEWS_KEY:
        try:
            resp = await http.get(
                "https://gnews.io/api/v4/search",
                params={"q": query, "lang": "en", "country": country.lower(),
                        "max": 6, "apikey": GNEWS_KEY},
            )
            articles = resp.json().get("articles", [])
            if articles:
                lines = ["📰 *Latest News*", ""]
                for a in articles[:6]:
                    title  = a.get("title", "")
                    source = a.get("source", {}).get("name", "")
                    if title:
                        lines.append(f"• *{title[:90]}*" + (f"  _({source})_" if source else ""))
                lines += ["", "_Source: GNews_"]
                result = "\n".join(lines)
                logger.info("📰 live_data.news.gnews  count=%d", len(articles))
                return result
        except Exception as exc:
            logger.warning("live_data.news.gnews_err  err=%s", exc)

    # ── Fallback: Google News RSS via rss2json ───────────────────────────────
    try:
        q_enc   = query.replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en"
        r2j_url = f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}&count=6"
        resp    = await http.get(r2j_url)
        items   = resp.json().get("items", [])
        if not items:
            return None
        lines = ["📰 *Latest News*", ""]
        for it in items[:6]:
            title  = it.get("title", "")
            source = it.get("author", "")
            if title:
                title = re.sub(r"\s*-\s*[^-]{3,40}$", "", title).strip()
                lines.append(f"• *{title[:90]}*" + (f"  _({source})_" if source else ""))
        lines += ["", "_Source: Google News_"]
        result = "\n".join(lines)
        logger.info("📰 live_data.news.rss  count=%d", len(items))
        return result
    except Exception as exc:
        logger.warning("live_data.news.rss_err  err=%s", exc)
        return None
