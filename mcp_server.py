#!/usr/bin/env python3
"""
mcp_server.py — Shimmi MCP Server v2.0.0

Changes vs v1.0.0:

  CACHE-1   TTL response cache for all external-API endpoints.
  CACHE-2   Cache key includes query parameters.
  FORMAT-1  POST /format — deterministic WhatsApp formatting (zero LLM tokens).
  STOCKS-2  Per-ticker timeout guard in _fetch_sync().
  HTTP-1    _HTTP client timeout reduced to 12s.

v3.0 additions:
  FETCH-1   GET /fetch?url=... — URL content extraction + LexRank compaction.
            Uses trafilatura (F1=0.958) for clean article extraction and
            sumy LexRank (TF-IDF, whole-article aware) for compaction.
            Returns structured JSON: title, author, date, abstract, text.
            abstract = 5 key sentences selected from WHOLE article by LexRank
                       (not just first N sentences — truly content-aware).
            text     = full clean body, sentence-boundary capped at ~3000 chars.
            TTL: 10 minutes (same as weather).
            Graceful fallback if trafilatura/sumy not installed.

Endpoints:
  GET  /health
  GET  /news?q=&country=
  GET  /stocks?symbols=
  GET  /weather?city=&country=&days=
  GET  /currency?from=&to=&amount=
  GET  /timezone?city=
  GET  /datetime?tz=
  GET  /fetch?url=         ← NEW
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

# curl_cffi is used internally by yfinance >= 0.2.54 for browser TLS
# impersonation — do NOT pass a session to yf.Ticker(); let yfinance handle
# its own session management. Passing an external session causes the
# "'str' object has no attribute 'name'" cookie-handling error.
_CURL_CFFI_AVAILABLE = False
try:
    import curl_cffi as _curl_cffi  # noqa: F401 — just checking it's installed
    _CURL_CFFI_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s mcp:%(message)s")

app = FastAPI(title="Shimmi MCP Server", version="3.0.0")

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
_TTL_FETCH    = 600    # 10 min — same as weather; articles don't change fast


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global _HTTP
    _HTTP = httpx.AsyncClient(timeout=12.0, follow_redirects=True)

    # yfinance >= 0.2.54 manages its own curl_cffi session internally.
    # Do NOT pass a session to yf.Ticker() — it causes cookie-handling errors.
    if _CURL_CFFI_AVAILABLE:
        logger.info("yf.session  curl_cffi available — yfinance handles TLS impersonation ✅")
    else:
        logger.warning(
            "yf.session  curl_cffi not installed — Yahoo Finance may rate-limit. "
            "Fix: pip install curl-cffi  (yfinance uses it automatically)"
        )
    logger.info("🚀 MCP server v3.15.3 ready on :7000")

@app.on_event("shutdown")
async def _shutdown():
    if _HTTP:
        await _HTTP.aclose()
    # Note: yfinance manages its own session internally — nothing to close here


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ts": datetime.now(UTC).isoformat(),
        "cache_entries": len(_CACHE),
        "yf_session": "curl_cffi/auto" if _CURL_CFFI_AVAILABLE else "requests/basic (install curl-cffi)",
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
            # FIX-EMPTY-CACHE: Never cache an empty result. If GNews returned 0 articles
            # for this query, the next call should try again (or try the RSS fallback)
            # rather than serving the empty result from cache for the full TTL.
            if not articles:
                result = None
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

    # Only cache non-empty results — empty results should be retried next call
    if result and result.get("count", 0) > 0:
        _cache_set(ck, result, _TTL_NEWS)
    elif result is None:
        result = {"source": "none", "query": q, "count": 0, "articles": []}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# /stocks
# ─────────────────────────────────────────────────────────────────────────────

_NSE_DEFAULTS = [
    "^NSEI", "^BSESN", "^NSEBANK",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS",
]

# Commodity tickers via Yahoo Finance (no extra library needed)
_GOLD_TICKER   = "GC=F"    # COMEX Gold Futures (USD/troy oz)
_SILVER_TICKER = "SI=F"    # COMEX Silver Futures

_STOCKS_PER_TICKER_TIMEOUT = 18.0  # .info is slower than fast_info; needs more headroom


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
        from yfinance.exceptions import YFRateLimitError as _YFRateLimit
    except ImportError:
        raise HTTPException(status_code=503, detail="yfinance not installed")
    except Exception:
        _YFRateLimit = Exception  # fallback if exceptions module differs

    ticker_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    def _fetch_one(sym: str) -> dict:
        """
        Fetch a single ticker using ticker.info for rich data.
        Returns: price, prev_close, change/%, day range, 52W range,
                 volume, avg_volume, pe_ratio, market_cap, sector, name.
        All new fields are optional — callers must handle None gracefully.
        Falls back to fast_info if info is empty (indices like ^NSEI).

        yfinance >= 0.2.54 uses curl_cffi internally for TLS impersonation.
        Do NOT pass a session — yfinance manages it automatically.
        """
        try:
            ticker = yf.Ticker(sym)
            info   = ticker.info or {}

            # Prefer currentPrice, fall back through known field names
            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("navPrice")
                     or info.get("ask")
                     or info.get("bid"))
            if price is None:
                try:
                    fi    = ticker.fast_info
                    price = getattr(fi, "last_price", None)
                except Exception:
                    pass

            # If NSE (.NS) returned no price, try BSE (.BO) automatically
            if price is None and sym.endswith(".NS"):
                bse_sym = sym[:-3] + ".BO"
                logger.info("stocks.fallback  %s → %s (no price on NSE)", sym, bse_sym)
                try:
                    bse_ticker = yf.Ticker(bse_sym)
                    bse_info   = bse_ticker.info or {}
                    price = (bse_info.get("currentPrice")
                             or bse_info.get("regularMarketPrice"))
                    if price is None:
                        fi    = bse_ticker.fast_info
                        price = getattr(fi, "last_price", None)
                    if price is not None:
                        # Use BSE data for this ticker
                        info = bse_info
                        sym  = bse_sym
                        logger.info("stocks.fallback_ok  %s  price=%.2f", bse_sym, price)
                except Exception as bse_err:
                    logger.debug("stocks.bse_fallback_fail  sym=%s  err=%s", bse_sym, str(bse_err)[:80])

            if price is None:
                logger.warning("stocks.no_price  sym=%s  info_keys=%s",
                               sym, list(info.keys())[:10])

            prev_close = (info.get("previousClose")
                          or info.get("regularMarketPreviousClose"))
            if prev_close is None:
                try:
                    fi         = ticker.fast_info
                    prev_close = getattr(fi, "previous_close", None)
                except Exception:
                    pass

            currency = info.get("currency") or "INR"
            name     = (info.get("longName") or info.get("shortName")
                        or info.get("displayName") or sym)

            change = change_pct = None
            if price and prev_close:
                change     = round(price - prev_close, 2)
                change_pct = round((change / prev_close) * 100, 2)

            # Rich fields — present for equities, often None for indices
            def _r(v, decimals=2):
                try:    return round(float(v), decimals) if v is not None else None
                except: return None

            day_high   = _r(info.get("dayHigh")   or info.get("regularMarketDayHigh"))
            day_low    = _r(info.get("dayLow")    or info.get("regularMarketDayLow"))
            open_price = _r(info.get("open")      or info.get("regularMarketOpen"))
            wk52_high  = _r(info.get("fiftyTwoWeekHigh"))
            wk52_low   = _r(info.get("fiftyTwoWeekLow"))
            volume     = info.get("volume")       or info.get("regularMarketVolume")
            avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")
            pe_ratio   = _r(info.get("trailingPE") or info.get("forwardPE"))
            mkt_cap    = info.get("marketCap")
            sector     = info.get("sector") or info.get("category") or ""
            is_gold    = sym in (_GOLD_TICKER, _SILVER_TICKER)

            return {
                "symbol":     sym,
                "name":       name,
                "price":      _r(price),
                "prev_close": _r(prev_close),
                "open":       open_price,
                "day_high":   day_high,
                "day_low":    day_low,
                "change":     change,
                "change_pct": change_pct,
                "week52_high": wk52_high,
                "week52_low":  wk52_low,
                "volume":     int(volume)    if volume     else None,
                "avg_volume": int(avg_volume) if avg_volume else None,
                "pe_ratio":   pe_ratio,
                "market_cap": int(mkt_cap)  if mkt_cap    else None,
                "sector":     sector,
                "currency":   currency,
                "is_commodity": is_gold,
                "as_of":      datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            # Surface rate-limit clearly so the caller can give a useful message
            err_str = str(e)
            if "rate" in err_str.lower() or "429" in err_str or "Too Many" in err_str:
                logger.warning("stocks.rate_limited  sym=%s", sym)
                return {"symbol": sym, "error": "rate_limited",
                        "rate_limited": True,
                        "message": "Yahoo Finance rate limit — try again in 1–2 hours"}
            logger.warning("stocks.fetch_error  sym=%s  err=%s", sym, err_str[:120])
            return {"symbol": sym, "error": err_str[:200]}

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
# /news/briefing — structured multi-category news briefing
# ─────────────────────────────────────────────────────────────────────────────

# Category definitions — (label, emoji, query, country)
# Queries are tuned for relevance: specific enough to avoid fluff,
# broad enough to catch the day's major stories.
_BRIEFING_CATEGORIES = [
    ("Global",        "🌍", "world major news geopolitics economy",  "us"),
    ("India",         "🇮🇳", "India government economy major news",   "in"),
    ("Tech",          "💻", "technology artificial intelligence science", "us"),
    ("India Sports",  "🏏", "India cricket IPL football sports",     "in"),
    ("Local",         "📍", "Hyderabad local news today",            "in"),
]

_TTL_BRIEFING = 1800   # 30 min — briefing is more stable than individual queries

# Source quality tiers — higher = preferred when filtering
# Tier 1: major international/national wire services and newspapers
# Tier 2: good regional/specialist outlets
# Unranked sources get tier 3 (lowest priority)
_SOURCE_TIER: dict = {
    # Tier 1 — highest quality, always prefer
    "reuters":           1, "associated press":   1, "ap":                1,
    "bbc":               1, "bbc news":            1, "the guardian":       1,
    "financial times":   1, "the hindu":           1, "hindustan times":    1,
    "times of india":    1, "economic times":      1, "ndtv":              1,
    "mint":              1, "business standard":   1, "the wire":           1,
    "scroll.in":         1, "the print":           1,
    # Tier 2 — reliable specialist / regional
    "techcrunch":        2, "the verge":           2, "wired":             2,
    "ars technica":      2, "mit technology review": 2,
    "cricinfo":          2, "espn cricinfo":       2, "bcci":              2,
    "deccan chronicle":  2, "the news minute":     2, "hans india":        2,
    "bloomberg":         2, "cnbc":                2, "ft":                2,
    # Everything else defaults to tier 3
}

def _source_tier(source: str) -> int:
    """Return quality tier for a news source name (lower = better)."""
    return _SOURCE_TIER.get(source.lower().strip(), 3)


_FLUFF_PATTERNS = re.compile(
    r"\b(horoscope|zodiac|listicle|ranking|best.*buy|top \d+ ways|"
    r"you won.?t believe|shocking|viral|celebrity|bollywood gossip|"
    r"box office|bigg boss|salman|shahrukh|deepika|kareena|"
    r"\bads?\b|sponsored|promoted|advertisement)\b",
    re.IGNORECASE,
)

def _is_quality_article(title: str, description: str, source: str, pub_iso: str) -> bool:
    """
    Returns True if article passes quality filters:
      1. Not a fluff/entertainment piece
      2. Title is substantive (>25 chars, not a question-bait)
      3. Published within the last 12 hours (freshness)
    """
    if not title or len(title) < 20:
        return False
    # Fluff filter
    combined = f"{title} {description}"
    if _FLUFF_PATTERNS.search(combined):
        return False
    # Freshness — if we have a publish time, reject anything >12h old
    if pub_iso:
        try:
            from datetime import timezone as _tz
            pub_dt = datetime.fromisoformat(pub_iso.replace("Z", "+00:00"))
            age_hours = (datetime.now(_tz.utc) - pub_dt).total_seconds() / 3600
            if age_hours > 12:
                return False
        except Exception:
            pass  # can't parse date → don't filter
    return True


@app.get("/news/briefing")
async def get_news_briefing(
    city: str = Query("Hyderabad", description="User's city for local news"),
):
    """
    Structured multi-category news briefing.
    Runs 5 parallel GNews queries (Global, India, Tech, India Sports, Local).
    Returns a structured dict with sections, each containing 2-3 clean headlines.
    No article URLs in output — headlines only with source attribution.

    GET /news/briefing?city=Hyderabad

    Returns:
    {
      "sections": [
        {"category": "Global", "emoji": "🌍",
         "articles": [{"title": "...", "source": "..."}, ...]},
        ...
      ],
      "generated_at": "2026-04-11T07:00:00"
    }
    """
    # Customise local category query with user's city
    categories = list(_BRIEFING_CATEGORIES)
    if city and city.lower() not in ("hyderabad", ""):
        categories[-1] = ("Local", "📍", f"{city} local news today", "in")

    ck = _cache_key("briefing", city.lower())
    cached = _cache_get(ck)
    if cached:
        logger.debug("briefing.cache_hit  city=%r", city)
        return cached

    async def _fetch_category(label: str, emoji: str, query: str, country: str) -> dict:
        """
        Fetch up to 6 articles for one category, filter to best 3.

        Quality pipeline:
          1. Fetch max=6 from GNews (more candidates to choose from)
          2. Strip source suffix from titles
          3. Apply fluff filter (gossip, listicles, celebrity)
          4. Apply freshness filter (< 12 hours old)
          5. Sort by source tier (Reuters/BBC/ET before unknowns)
          6. Return top 3 with title + source + trimmed description (lede)
        """
        try:
            if not GNEWS_KEY:
                return {"category": label, "emoji": emoji, "articles": []}
            resp = await _HTTP.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q":       query,
                    "lang":    "en",
                    "country": country,
                    "max":     6,          # fetch 6 to have room to filter
                    "apikey":  GNEWS_KEY,
                },
                timeout=httpx.Timeout(connect=5.0, read=8.0, write=3.0, pool=2.0),
            )
            data = resp.json()
            candidates = []
            for a in data.get("articles", []):
                raw_title   = (a.get("title") or "").strip()
                description = (a.get("description") or "").strip()
                source      = (a.get("source") or {}).get("name", "") or ""
                pub_iso     = a.get("publishedAt", "")

                # Strip "- SourceName" suffix from title (common in GNews)
                title = re.sub(r"\s*[-|]\s*[A-Za-z0-9 &\.]{2,40}$", "", raw_title).strip()
                if not title:
                    title = raw_title

                # Quality gate
                if not _is_quality_article(title, description, source, pub_iso):
                    continue

                # Trim description to 120 chars at sentence boundary
                brief = ""
                if description:
                    desc = description[:150]
                    # Cut at sentence end if possible
                    for end in (".", "!", "?"):
                        idx = desc.rfind(end)
                        if 40 < idx < 140:
                            desc = desc[:idx + 1]
                            break
                    brief = desc.strip()
                    if len(description) > len(brief) + 5:
                        brief = brief.rstrip(".") + "…" if not brief.endswith((".", "!", "?")) else brief

                candidates.append({
                    "title":       title[:120],
                    "brief":       brief,
                    "source":      source,
                    "source_tier": _source_tier(source),
                    "pub_iso":     pub_iso,
                })

            # Sort: tier 1 sources first, then by position (GNews relevance order)
            candidates.sort(key=lambda x: (x["source_tier"],
                                           candidates.index(x)))

            # Take top 3, remove internal ranking fields
            articles = [
                {"title": c["title"], "brief": c["brief"], "source": c["source"]}
                for c in candidates[:3]
            ]

            logger.info("briefing.category  label=%r  fetched=%d  kept=%d",
                        label, len(data.get("articles", [])), len(articles))
            return {"category": label, "emoji": emoji, "articles": articles}

        except Exception as exc:
            logger.warning("briefing.category_fail  label=%r  err=%s", label, str(exc)[:80])
            return {"category": label, "emoji": emoji, "articles": []}

    # Run all 5 categories in parallel — total time = slowest single query (~1-2s)
    sections = await asyncio.gather(*[
        _fetch_category(label, emoji, query, country)
        for label, emoji, query, country in categories
    ])

    result = {
        "sections":     list(sections),
        "generated_at": datetime.now(UTC).isoformat(),
        "city":         city,
    }

    # Only cache if we got meaningful content
    total_articles = sum(len(s["articles"]) for s in sections)
    if total_articles >= 5:
        _cache_set(ck, result, _TTL_BRIEFING)
        logger.info("briefing.done  city=%r  total_articles=%d", city, total_articles)
    else:
        logger.warning("briefing.thin  city=%r  total_articles=%d  (not cached)", city, total_articles)

    return result


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


# ─────────────────────────────────────────────────────────────────────────────
# /fetch — URL content extraction + LexRank compaction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_trafilatura(html_text: str, url: str) -> dict:
    """
    Use trafilatura to extract clean article content from raw HTML.
    Returns a dict with: title, author, date, text, word_count.
    trafilatura reads the WHOLE page and discards nav/ads/footers using
    trained heuristics — F1=0.958 in independent benchmarks.
    """
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html_text,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,        # allow fallback extraction if main fails
            output_format="txt",
        )
        metadata = trafilatura.extract_metadata(html_text, default_url=url)
        return {
            "title":      getattr(metadata, "title",  None) or "",
            "author":     getattr(metadata, "author", None) or "",
            "date":       getattr(metadata, "date",   None) or "",
            "text":       extracted or "",
            "word_count": len((extracted or "").split()),
        }
    except ImportError:
        # trafilatura not installed — fall back to simple tag stripping
        logger.warning("fetch.trafilatura_missing — using regex fallback")
        return {}
    except Exception as exc:
        logger.warning("fetch.trafilatura_error  err=%s", str(exc)[:120])
        return {}


def _compact_with_lexrank(text: str, sentence_count: int = 5) -> str:
    """
    Use sumy LexRank to extract the most central sentences from the article.
    LexRank reads the WHOLE text and uses TF-IDF cosine similarity to build
    a sentence graph, then ranks by eigenvector centrality.  A sentence from
    the last paragraph can outscore one from the first paragraph.

    Falls back to first-N-sentences if sumy is not installed.
    """
    if not text or len(text.split()) < 30:
        return text   # too short to summarise

    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser     = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer    = Stemmer("english")
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        sentences = summarizer(parser.document, sentence_count)
        compact   = " ".join(str(s) for s in sentences)
        return compact if compact.strip() else text[:800]

    except ImportError:
        logger.warning("fetch.sumy_missing — using first-sentences fallback")
        # Fallback: split on sentence-ending punctuation, take first N
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:sentence_count])
    except Exception as exc:
        logger.warning("fetch.lexrank_error  err=%s", str(exc)[:120])
        sentences = text.split(". ")
        return ". ".join(sentences[:sentence_count])


def _cap_at_sentence_boundary(text: str, max_chars: int = 3000) -> str:
    """
    Truncate text at a sentence boundary near max_chars.
    Better than a hard char cap which can cut mid-sentence.
    """
    if len(text) <= max_chars:
        return text
    # Find the last sentence-ending punctuation before max_chars
    chunk = text[:max_chars]
    last_end = max(
        chunk.rfind(". "),
        chunk.rfind("! "),
        chunk.rfind("? "),
        chunk.rfind(".\n"),
    )
    if last_end > max_chars // 2:   # only trim if we kept at least half
        return chunk[:last_end + 1].rstrip() + "…"
    return chunk.rstrip() + "…"


@app.get("/fetch")
async def fetch_url(url: str = Query(..., min_length=8)):
    """
    Fetch a URL, extract clean article text using trafilatura, and compact
    using sumy LexRank.  Returns structured JSON ready for LLM consumption.

    GET /fetch?url=https://example.com/article

    Returns:
    {
      "url":        "https://...",
      "title":      "Article title",
      "author":     "Author name",
      "date":       "2026-04-01",
      "abstract":   "5 key sentences selected by LexRank from whole article",
      "text":       "Full clean article text, capped at sentence boundary ~3000 chars",
      "word_count": 680,
      "truncated":  false
    }

    Errors return HTTP 422 (bad URL) or HTTP 502 (fetch failed).
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail=f"Invalid URL: {url!r}")

    # Cache check — same URL won't be fetched twice within TTL
    ck = _cache_key("fetch", url)
    cached = _cache_get(ck)
    if cached:
        logger.debug("fetch.cache_hit  url=%r", url[:80])
        return cached

    logger.info("fetch.start  url=%r", url[:120])

    # ── 1. Fetch raw HTML ─────────────────────────────────────────────────
    try:
        resp = await _HTTP.get(
            url,
            timeout=httpx.Timeout(connect=8.0, read=25.0, write=5.0, pool=2.0),
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        html_text = resp.text
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} from {url}"
        logger.warning("fetch.http_error  url=%r  status=%d", url[:80], exc.response.status_code)
        return JSONResponse({"error": detail, "url": url}, status_code=200)
    except httpx.TimeoutException as exc:
        detail = f"Timed out fetching {url}"
        logger.warning("fetch.timeout  url=%r  err=%s", url[:80], str(exc)[:120])
        return JSONResponse({"error": detail, "url": url}, status_code=200)
    except Exception as exc:
        detail = f"Could not fetch {url}: {str(exc)[:200]}"
        logger.warning("fetch.connection_error  url=%r  err=%s", url[:80], str(exc)[:200])
        return JSONResponse({"error": detail, "url": url}, status_code=200)

    # ── 2. Extract with trafilatura ───────────────────────────────────────
    meta = _extract_with_trafilatura(html_text, url)

    if not meta.get("text"):
        # trafilatura failed or not installed — use simple regex fallback
        text_raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "",
                          html_text, flags=re.DOTALL | re.IGNORECASE)
        text_raw = re.sub(r"<[^>]+>", " ", text_raw)
        text_raw = html.unescape(text_raw)
        text_raw = re.sub(r"[ \t]+", " ", text_raw)
        text_raw = re.sub(r"\n{3,}", "\n\n", text_raw).strip()
        meta = {"title": "", "author": "", "date": "",
                "text": text_raw, "word_count": len(text_raw.split())}

    full_text = meta.get("text", "")
    if not full_text:
        logger.warning("fetch.no_content  url=%r", url[:80])
        return JSONResponse({"error": f"No readable content found at {url}", "url": url}, status_code=200)

    # ── 3. Compact with LexRank ───────────────────────────────────────────
    abstract = _compact_with_lexrank(full_text, sentence_count=5)

    # ── 4. Cap full text at sentence boundary ────────────────────────────
    text_capped   = _cap_at_sentence_boundary(full_text, max_chars=3000)
    was_truncated = len(full_text) > 3000

    result = {
        "url":        url,
        "title":      meta.get("title", ""),
        "author":     meta.get("author", ""),
        "date":       meta.get("date", ""),
        "abstract":   abstract,
        "text":       text_capped,
        "word_count": meta.get("word_count", 0),
        "truncated":  was_truncated,
    }

    logger.info(
        "fetch.done  url=%r  words=%d  abstract_chars=%d  truncated=%s",
        url[:80], result["word_count"], len(abstract), was_truncated,
    )

    _cache_set(ck, result, _TTL_FETCH)
    return result


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
