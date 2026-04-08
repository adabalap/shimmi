"""
live_data.py — Shimmi v3.1.0

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

Changes vs v3.0.3:
  FIX-3  Added _normalize_news_query() — synchronous module-level function that
         tools.py imports. Was missing entirely, causing ImportError on every
         news tool call (logged as tools.dispatch.error  tool=news).
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


# FIX-3: _normalize_news_query — synchronous module-level function imported by tools.py.
# tools.py calls this BEFORE calling get_news() so the query is clean regardless of
# whether it arrives via tool_call JSON or the agent's live_search path.
# _rewrite_news_query (below) is the async LLM-powered version used inside get_news().
_NORMALIZE_META = re.compile(
    r"\b(morning|evening|night|daily|round.?up|roundup|briefing|top (news|stories|headlines)"
    r"|latest news|news today|current news|headlines today|what.?s (happening|new|on)|news round)\b",
    re.IGNORECASE,
)

def _normalize_news_query(query: str) -> str:
    """
    Synchronous fast-path normaliser for meta-phrase news queries.
    Converts phrases like 'morning news round up' or 'top stories today'
    → 'India top news today' so GNews returns real headlines instead of 0 results.

    This is a zero-cost pre-filter that runs before any LLM or HTTP call.
    The async _rewrite_news_query (below) handles longer-tail rewrites via LLM.
    """
    if not query:
        return "India top news"
    if _NORMALIZE_META.search(query):
        return "India top news today"
    return query


# Meta-phrases that GNews treats as literal search terms and returns 0 results.
# Map them to sensible topic queries that actually match headlines.
async def _rewrite_news_query(raw_query: str) -> str:
    """
    ARCH-3: Replace the lookup table with a lightweight 8B LLM rewriter.

    A lookup table of meta-phrases is a maintenance debt — every new phrasing
    needs a code change. An LLM rewriter handles any variant the user sends
    without touching the codebase.

    The 8B model rewrites "morning news round up" → "India top news today",
    "latest cricket score" → "India cricket live score", etc.
    Costs ~50 tokens (negligible vs compound-beta's 800+).

    Falls back to the raw query if the LLM is unavailable — GNews will try
    and the RSS fallback covers the case where it returns 0 articles.
    """
    # Only rewrite if the query looks like a meta-phrase, not a real topic.
    # Real topics: "Red Sea crisis", "India GDP", "IPL Kolkata Knight Riders"
    # Meta-phrases: "morning news", "latest news", "top stories", "news round up"
    _META_SIGNALS = re.compile(
        r"\b(morning|round up|roundup|briefing|summary|update|top (news|stories|headlines)|"
        r"latest news|news today|current news|daily news|headlines today|"
        r"cricket score|cricket update|ipl score)\b",
        re.IGNORECASE,
    )
    if not _META_SIGNALS.search(raw_query):
        return raw_query  # looks like a real topic — send as-is

    try:
        from .agent_engine import _groq_raw
        result = await _groq_raw(
            [
                {
                    "role": "system",
                    "content": (
                        "Convert the user's news request into a concise GNews search term (2-5 words). "
                        "GNews is a keyword search engine — return only searchable topic words, "
                        "no meta-phrases like 'round up', 'morning', 'briefing', 'update'. "
                        "Examples:\n"
                        "  'morning news round up' → 'India top news today'\n"
                        "  'give me the latest cricket score' → 'India cricket live score'\n"
                        "  'IPL score today' → 'IPL cricket score today'\n"
                        "  'world news today' → 'world top headlines today'\n"
                        "Return ONLY the search term, nothing else."
                    ),
                },
                {"role": "user", "content": raw_query},
            ],
            max_tokens=20,
            chat_id="news_rewrite",
            label="news_query_rewrite",
            role="extract",
            timeout=5.0,
        )
        rewritten = result.strip().strip('"\'`').strip()
        if rewritten and len(rewritten) < 80 and rewritten.lower() != raw_query.lower():
            logger.info("📰 news.query_rewritten  %r → %r", raw_query[:60], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.debug("news.rewrite_skip  err=%s", e)

    return raw_query


async def get_news(query: str = "India top news", country: str = "IN") -> Optional[str]:
    """Fetch news headlines. Tries MCP first, then GNews / RSS fallback."""

    # ARCH-3: Rewrite meta-phrase queries via lightweight 8B LLM call.
    # This handles ANY phrasing variant without a lookup table.
    effective_query = await _rewrite_news_query(query)

    # ── Try MCP ──────────────────────────────────────────────────────────────
    try:
        from .mcp_client import mcp_news
        data = await mcp_news(query=effective_query, country=country.lower())
        if data and data.get("articles"):
            count = len(data["articles"])
            result = _format_news_mcp(data, effective_query)
            if result:
                logger.info("📰 live_data.news.mcp  count=%d", count)
                # FIX-THIN-RESULT: If we got fewer than 3 articles and the user asked
                # for a roundup/top-N, try a broader fallback query to get more.
                # Evidence: "top news stories India" → count=1, result_len=120.
                if count < 3:
                    logger.info("📰 live_data.news.thin_result  count=%d  trying broader query", count)
                    fallback_q = f"{country} news today"
                    data2 = await mcp_news(query=fallback_q, country=country.lower())
                    if data2 and len(data2.get("articles", [])) > count:
                        r2 = _format_news_mcp(data2, fallback_q)
                        if r2:
                            logger.info("📰 live_data.news.mcp_broader  count=%d", len(data2["articles"]))
                            return r2
                return result
        # FIX-EMPTY-RESULT: GNews returned 0 articles (200 OK, empty body).
        # The original query was a meta-phrase. Try the country-level fallback.
        if not data or data.get("count", 0) == 0:
            fallback_q = f"{country} news today"
            logger.info("📰 live_data.news.empty_fallback  %r → %r", query, fallback_q)
            data2 = await mcp_news(query=fallback_q, country=country.lower())
            if data2 and data2.get("articles"):
                result = _format_news_mcp(data2, fallback_q)
                if result:
                    logger.info("📰 live_data.news.mcp_fallback  count=%d", len(data2["articles"]))
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
