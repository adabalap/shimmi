"""
live_data.py — Shimmi v3.17.3

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


def _fmt_crore(n: Optional[int]) -> str:
    """Format market cap in readable Indian format (Cr / L Cr)."""
    if not n:
        return ""
    cr = n / 1e7
    if cr >= 1_00_000:
        return f"₹{cr/1_00_000:.1f}L Cr"
    if cr >= 1_000:
        return f"₹{cr:,.0f} Cr"
    return f"₹{cr:.0f} Cr"


def _fmt_52w_position(price, low, high) -> str:
    """Return a text label for where price sits in its 52-week range."""
    if not all([price, low, high]) or high <= low:
        return ""
    pct = (price - low) / (high - low) * 100
    if pct >= 90:   return "🔥 Near 52W high"
    if pct >= 70:   return "📈 Upper range"
    if pct >= 40:   return "➡️  Mid range"
    if pct >= 20:   return "📉 Lower range"
    return "⚠️  Near 52W low"


def _format_stocks_mcp(data: dict, inr_rate: Optional[float] = None) -> Optional[str]:
    """Format MCP /stocks JSON response → WhatsApp string.

    Uses rich fields (52W range, P/E, volume, sector) when present.
    Falls back gracefully to basic price + change when fields are absent
    so the existing fallback yfinance path continues to work unchanged.
    """
    stocks = data.get("stocks", [])
    if not stocks:
        return None

    today    = datetime.now(UTC).strftime("%a %d %b %Y")
    is_gold  = any(s.get("is_commodity") for s in stocks if not s.get("error"))
    header   = "🥇 *Commodities*" if is_gold else f"📈 *Markets — {today}*"
    lines    = [header, "_(~15 min delay · Yahoo Finance)_", ""]
    skipped  = []

    for s in stocks:
        if s.get("error"):
            if s.get("rate_limited"):
                # Propagate rate-limit message immediately — no point continuing
                return (
                    "⏳ *Yahoo Finance is temporarily rate-limiting this server.*\n"
                    "Stock prices are unavailable right now.\n"
                    "_Try again in 1–2 hours — this clears on its own._"
                )
            skipped.append(s.get("symbol", "?"))
            continue

        price   = s.get("price")
        if price is None:
            skipped.append(s.get("symbol", "?"))
            continue

        chg_pct  = s.get("change_pct") or 0
        chg_abs  = s.get("change") or 0
        currency = s.get("currency", "INR")
        cur_sym  = "₹" if currency == "INR" else ("$" if currency == "USD" else currency + " ")
        arrow    = "🟢" if chg_pct >= 0 else "🔴"
        name     = s.get("name") or s.get("symbol", "")
        sym      = s.get("symbol", "")

        # ── Basic price line ─────────────────────────────────────────────
        lines.append(f"{arrow} *{name}*  ({sym})")
        lines.append(f"   💰 {cur_sym}{price:,.2f}  {chg_abs:+,.2f} ({chg_pct:+.2f}%)")

        # ── Day range ───────────────────────────────────────────────────
        dh, dl = s.get("day_high"), s.get("day_low")
        if dh and dl:
            lines.append(f"   📅 Today: {cur_sym}{dl:,.2f} – {cur_sym}{dh:,.2f}")

        # ── 52-week range ───────────────────────────────────────────────
        wh, wl = s.get("week52_high"), s.get("week52_low")
        if wh and wl:
            pos   = _fmt_52w_position(price, wl, wh)
            lines.append(f"   📊 52W: {cur_sym}{wl:,.2f} ↔ {cur_sym}{wh:,.2f}  {pos}")

        # ── Volume ──────────────────────────────────────────────────────
        vol, avg_vol = s.get("volume"), s.get("avg_volume")
        if vol and avg_vol and avg_vol > 0:
            vol_ratio = vol / avg_vol
            vol_note  = "⬆️ High" if vol_ratio > 1.5 else ("⬇️ Low" if vol_ratio < 0.5 else "Normal")
            lines.append(f"   📦 Vol: {vol/1e6:.1f}M  ({vol_note} vs {avg_vol/1e6:.1f}M avg)")

        # ── Fundamentals (equities only) ────────────────────────────────
        detail_parts = []
        pe = s.get("pe_ratio")
        if pe:
            detail_parts.append(f"P/E {pe:.1f}")
        mc = s.get("market_cap")
        if mc:
            detail_parts.append(_fmt_crore(mc))
        sec = s.get("sector")
        if sec:
            detail_parts.append(sec)
        if detail_parts:
            lines.append(f"   🏭 {' · '.join(detail_parts)}")

        # ── Gold/commodity INR auto-conversion ────────────────────────────
        if s.get("is_commodity") and currency == "USD" and price and inr_rate:
            per_10g_inr = (price / 31.1035) * 10 * inr_rate
            lines.append(
                f"   🇮🇳 ≈ ₹{per_10g_inr:,.0f}/10g  "
                f"(at ₹{inr_rate:.2f}/USD)"
            )
        elif s.get("is_commodity") and currency == "USD":
            lines.append(f"   🇮🇳 Priced in USD/troy oz")

        # ── Brief market signal ─────────────────────────────────────────
        signals = []
        if wh and wl and price:
            pct_from_high = (wh - price) / wh * 100
            pct_from_low  = (price - wl) / wl * 100 if wl else 0
            if pct_from_high <= 3:
                signals.append("Trading near 52W high")
            elif pct_from_low <= 3:
                signals.append("Trading near 52W low ⚠️")
        if vol and avg_vol and avg_vol > 0:
            if vol / avg_vol > 2.0:
                signals.append("Unusually high volume 📢")
            elif vol / avg_vol < 0.3:
                signals.append("Very thin volume")
        if chg_pct and abs(chg_pct) >= 4:
            signals.append(f"{'Sharp move up' if chg_pct > 0 else 'Sharp sell-off'} today")
        if signals:
            lines.append(f"   💡 {' · '.join(signals)}")

        lines.append("")   # blank line between stocks

    if not any(True for s in stocks if not s.get("error") and s.get("price")):
        sym_list = ", ".join(skipped) if skipped else "requested symbols"
        return (
            f"📊 *Stock Data Unavailable*\n"
            f"Could not fetch price for: *{sym_list}*\n\n"
            f"Possible reasons:\n"
            f"• Market is closed (NSE/BSE: Mon–Fri 9:15AM–3:30PM IST)\n"
            f"• Ticker not listed on Yahoo Finance\n"
            f"• Try the full name, e.g. _PAYTM_ or _PAYTM.NS_ or _PAYTM.BO_\n"
            f"_(Data source: Yahoo Finance · ~15 min delay)_"
        )
    if skipped:
        lines.append(f"_⚠️ No data for: {', '.join(skipped)}_")
    return "\n".join(lines).rstrip()


async def get_indian_stocks(symbols: Optional[list] = None) -> Optional[str]:
    """Fetch Indian stocks. Tries MCP server first, then yfinance fallback."""

    # ── Try MCP ──────────────────────────────────────────────────────────────
    try:
        from .mcp_client import mcp_stocks, mcp_currency
        sym_str = ",".join(symbols) if symbols else ",".join(s for s, _ in _DEFAULT_SYMBOLS[:8])
        data = await mcp_stocks(symbols=sym_str)
        if data and data.get("stocks"):
            # Auto-fetch USD/INR for gold/commodity display
            inr_rate = None
            _COMMODITY_TICKERS = {"GC=F", "SI=F", "CL=F", "NG=F"}
            has_commodity = symbols and any(s in _COMMODITY_TICKERS for s in symbols)
            if has_commodity:
                try:
                    fx = await mcp_currency("USD", "INR", 1.0)
                    if fx and fx.get("converted"):
                        inr_rate = float(fx["converted"])
                except Exception:
                    pass
            result = _format_stocks_mcp(data, inr_rate=inr_rate)
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



async def get_portfolio_review(holdings_json: str) -> Optional[str]:
    """
    Full portfolio P&L review.

    holdings_json: JSON string from portfolio_holdings fact:
        [{"symbol": "PAYTM.NS", "qty": 100, "avg_price": 1000}, ...]

    Returns a WhatsApp-formatted portfolio card with:
      - Current price, day change for each stock
      - P&L per holding (unrealised gain/loss)
      - Total portfolio value vs cost basis
      - Overall return %
    """
    import json as _json
    from .mcp_client import mcp_stocks

    # ── Parse holdings ──────────────────────────────────────────────────────
    try:
        holdings = _json.loads(holdings_json)
        if not isinstance(holdings, list) or not holdings:
            return None
    except (_json.JSONDecodeError, TypeError):
        return None

    # Normalise: ensure each holding has symbol, qty, avg_price
    valid = []
    for h in holdings:
        sym = str(h.get("symbol", "") or "").strip().upper()
        if not sym:
            continue
        # Add .NS if bare symbol
        if "." not in sym and not sym.startswith("^") and sym not in {"GC=F","SI=F"}:
            sym += ".NS"
        try:
            qty       = float(h.get("qty", 0) or 0)
            avg_price = float(h.get("avg_price", 0) or 0)
        except (TypeError, ValueError):
            qty = avg_price = 0
        if qty > 0 and avg_price > 0:
            valid.append({"symbol": sym, "qty": qty, "avg_price": avg_price})

    if not valid:
        return None

    # ── Fetch current prices from MCP ───────────────────────────────────────
    symbols_str = ",".join(h["symbol"] for h in valid)
    data = await mcp_stocks(symbols=symbols_str)
    if not data or not data.get("stocks"):
        return None

    # Build a lookup dict from MCP response
    price_map = {}
    for s in data["stocks"]:
        if not s.get("error") and s.get("price") is not None:
            price_map[s["symbol"]] = s

    # ── Calculate P&L ───────────────────────────────────────────────────────
    today     = datetime.now(UTC).strftime("%a %d %b %Y")
    lines     = [f"📊 *Portfolio Review — {today}*", ""]

    total_cost    = 0.0
    total_current = 0.0
    any_data      = False

    for h in valid:
        sym       = h["symbol"]
        qty       = h["qty"]
        avg_price = h["avg_price"]
        cost      = qty * avg_price
        total_cost += cost

        mcp = price_map.get(sym)
        if not mcp:
            lines.append(f"⚪ *{sym}*  —  No data available")
            lines.append("")
            continue

        any_data      = True
        cur_price     = mcp["price"]
        cur_value     = qty * cur_price
        total_current += cur_value

        pnl        = cur_value - cost
        pnl_pct    = (pnl / cost * 100) if cost else 0
        day_chg    = mcp.get("change_pct") or 0
        name       = mcp.get("name") or sym

        arrow      = "🟢" if pnl >= 0 else "🔴"
        day_arrow  = "▲" if day_chg >= 0 else "▼"
        cur_sym    = "₹" if mcp.get("currency","INR") == "INR" else "$"

        lines.append(f"{arrow} *{name}*  ({sym})")
        lines.append(
            f"   💰 {cur_sym}{cur_price:,.2f}  "
            f"({day_arrow}{abs(day_chg):.2f}% today)"
        )
        lines.append(
            f"   📦 {qty:.0f} shares  ×  avg {cur_sym}{avg_price:,.2f}"
        )

        # Cost vs current
        lines.append(
            f"   💼 Cost: {cur_sym}{cost:,.0f}  →  "
            f"Now: {cur_sym}{cur_value:,.0f}"
        )

        # P&L line
        pnl_sign = "+" if pnl >= 0 else ""
        lines.append(
            f"   {'📈' if pnl >= 0 else '📉'} P&L: "
            f"{pnl_sign}{cur_sym}{pnl:,.0f}  ({pnl_sign}{pnl_pct:.1f}%)"
        )

        # 52-week context if available
        wh, wl = mcp.get("week52_high"), mcp.get("week52_low")
        if wh and wl:
            pos = _fmt_52w_position(cur_price, wl, wh)
            lines.append(f"   📊 52W: {cur_sym}{wl:,.0f} ↔ {cur_sym}{wh:,.0f}  {pos}")

        lines.append("")

    if not any_data:
        return None

    # ── Portfolio summary ───────────────────────────────────────────────────
    if total_cost > 0 and total_current > 0:
        total_pnl     = total_current - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100
        total_sign    = "+" if total_pnl >= 0 else ""
        summary_icon  = "🟢" if total_pnl >= 0 else "🔴"

        lines.append("─" * 30)
        lines.append(f"*Portfolio Summary*")
        lines.append(f"   💰 Invested:  ₹{total_cost:,.0f}")
        lines.append(f"   💎 Current:   ₹{total_current:,.0f}")
        lines.append(
            f"   {summary_icon} Total P&L:  "
            f"{total_sign}₹{total_pnl:,.0f}  ({total_sign}{total_pnl_pct:.1f}%)"
        )

    logger.info(
        "📊 portfolio_review  holdings=%d  cost=%.0f  current=%.0f",
        len(valid), total_cost, total_current,
    )
    return "\n".join(lines).rstrip()


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
            json_mode=False,   # returns plain text search term, not JSON
        )
        rewritten = result.strip().strip('"\'`').strip()
        if rewritten and len(rewritten) < 80 and rewritten.lower() != raw_query.lower():
            logger.info("📰 news.query_rewritten  %r → %r", raw_query[:60], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.debug("news.rewrite_skip  err=%s", e)

    return raw_query



async def _generate_story_insight(title: str, source: str,
                                    article_abstract: str = "") -> dict:
    """
    Use Gemini to generate editorial insight for the 'One Story' section.

    article_abstract: real article body (3 LexRank sentences) fetched from source.
    When present, Gemini grounds its insight in actual journalism rather than
    guessing from the headline alone. Quality difference is significant.

    Returns dict with why, now, next, takeaway.
    Falls back to empty strings if Gemini is unavailable/quota-limited.
    """
    try:
        from .agent_engine import _groq_raw
        if article_abstract and len(article_abstract) > 80:
            # Grounded: real article content available
            prompt = (
                f"News headline: \"{title}\" (Source: {source})\n\n"
                f"Article content:\n{article_abstract[:600]}\n\n"
                "Based on the actual article above, write editorial insight with EXACTLY these 4 fields:\n"
                "WHY: One sentence — why this story matters right now.\n"
                "NOW: One sentence — the key fact or development from the article.\n"
                "NEXT: One sentence — what to watch for next.\n"
                "TAKEAWAY: One short sentence — net implication for a professional.\n\n"
                "Rules: Each answer is ONE sentence. No bullets. No markdown. "
                "Be specific — use facts from the article, not generic commentary. "
                "Output only the 4 labelled lines."
            )
        else:
            # Fallback: headline only
            prompt = (
                f"News headline: \"{title}\" (Source: {source})\n\n"
                "Write a concise editorial insight with EXACTLY these 4 fields:\n"
                "WHY: One sentence — why this story matters right now.\n"
                "NOW: One sentence — the immediate context or trigger.\n"
                "NEXT: One sentence — what to watch for next.\n"
                "TAKEAWAY: One short sentence — net implication for a professional.\n\n"
                "Rules: One sentence each. No bullets. No markdown. Be direct. "
                "Output only the 4 labelled lines."
            )
        raw = await _groq_raw(
            [
                {
                    "role": "system",
                    "content": "You are a senior news analyst writing a morning briefing for busy professionals.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            chat_id="briefing_insight",
            label="story_insight",
            role="orchestrate",   # use Gemini (primary orchestrator) for quality
            json_mode=False,
            timeout=12.0,
        )
        # Parse the 4 labelled lines
        result = {"why": "", "now": "", "next": "", "takeaway": ""}
        for line in (raw or "").splitlines():
            line = line.strip()
            for key in ("WHY", "NOW", "NEXT", "TAKEAWAY"):
                if line.upper().startswith(key + ":"):
                    result[key.lower()] = line[len(key)+1:].strip()
                    break
        if any(result.values()):
            logger.info("📰 story_insight.done  title=%r", title[:60])
            return result
    except Exception as exc:
        logger.debug("story_insight.skip  err=%s", str(exc)[:80])
    return {"why": "", "now": "", "next": "", "takeaway": ""}


async def get_news_briefing(city: str = "Hyderabad") -> Optional[str]:
    """
    World-class WhatsApp morning briefing — jobs-based structure.

    8 sections, each with a distinct purpose:
      🧭 Executive Snapshot   — what's the context/mood today (10-sec read)
      🔄 What Changed Overnight — new facts from last 8 hours
      🎯 One Story to Understand — deep editorial on the day's key story
      🇮🇳 India Lens           — global stories filtered for India relevance
      💹 Markets Quick         — Nifty/Sensex/Rupee signals
      🤖 Tech & AI Radar       — what's shifting in tech
      ⏰ What to Watch Today   — forward-looking, actionable
      📍 Local                 — city-specific headlines
    """
    from .mcp_client import mcp_news_briefing, mcp_stocks, mcp_weather

    data = await mcp_news_briefing(city=city)
    if not data:
        return None

    snapshot     = data.get("snapshot", [])
    changed      = data.get("changed", [])
    story_raw    = data.get("story")
    ai_story_raw = data.get("ai_story")
    ai_headlines = data.get("ai_headlines", [])
    india_lens   = data.get("india_lens", [])
    tech_radar   = data.get("tech_radar", [])
    watch_today  = data.get("watch_today", [])
    local        = data.get("local", [])
    sports       = data.get("sports", [])

    total = sum(len(x) for x in [snapshot, changed, india_lens]
                if isinstance(x, list))
    if total < 3:
        logger.info("📰 news_briefing.thin  total=%d  falling back", total)
        return None

    # ── Parallel: story insight + AI insight + markets + weather ───────────
    story_insight = {}
    ai_insight    = {}
    markets_text  = ""
    weather_text  = ""

    async def _get_insight():
        nonlocal story_insight
        if story_raw and story_raw.get("title"):
            story_insight = await _generate_story_insight(
                story_raw["title"],
                story_raw.get("source", ""),
                article_abstract=story_raw.get("abstract", ""),  # real article body
            )

    async def _get_ai_insight():
        nonlocal ai_insight
        if ai_story_raw and ai_story_raw.get("title"):
            ai_insight = await _generate_story_insight(
                ai_story_raw["title"],
                ai_story_raw.get("source", ""),
                article_abstract=ai_story_raw.get("abstract", ""),
            )

    async def _get_markets():
        nonlocal markets_text
        try:
            mdata = await mcp_stocks(symbols="^NSEI,^BSESN,USDINR=X")
            if mdata and mdata.get("stocks"):
                parts = []
                labels = {"^NSEI": "Nifty", "^BSESN": "Sensex", "USDINR=X": "₹/USD"}
                for s in mdata["stocks"]:
                    sym   = s.get("symbol", "")
                    price = s.get("price")
                    chg   = s.get("change_pct")
                    if price and sym in labels:
                        arrow = "▲" if (chg or 0) >= 0 else "▼"
                        parts.append(
                            f"{labels[sym]} {arrow}{abs(chg or 0):.1f}%"
                        )
                markets_text = "  •  ".join(parts)
        except Exception as exc:
            logger.debug("briefing.markets_skip  err=%s", exc)

    async def _get_weather():
        nonlocal weather_text
        try:
            wdata = await mcp_weather(city=city, country="IN", days=1)
            if wdata and wdata.get("current"):
                cur  = wdata["current"]
                temp = cur.get("temp_c")
                desc = cur.get("condition", "")
                if temp is not None:
                    weather_text = f"{temp:.0f}°C, {desc}" if desc else f"{temp:.0f}°C"
        except Exception as exc:
            logger.debug("briefing.weather_skip  err=%s", exc)

    await asyncio.gather(_get_insight(), _get_ai_insight(), _get_markets(), _get_weather())

    # ── Format ─────────────────────────────────────────────────────────────
    now_ist = datetime.now(UTC)
    # Determine greeting by IST hour (UTC+5:30)
    ist_hour = (now_ist.hour + 5) % 24 + (1 if now_ist.minute >= 30 else 0)
    if ist_hour < 12:
        greeting = "Good morning ☀️"
    elif ist_hour < 17:
        greeting = "Good afternoon 🌤️"
    else:
        greeting = "Good evening 🌆"

    date_str = now_ist.strftime("%a, %d %b")
    lines = [
        f"*{greeting} Here's your briefing for {date_str}*",
        "",
    ]

    # ── 🧭 Executive Snapshot ──────────────────────────────────────────────
    if snapshot:
        lines.append("🧭 *Executive Snapshot*")
        for a in snapshot:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 🔄 What Changed Overnight ─────────────────────────────────────────
    if changed:
        lines.append("🔄 *What Changed Overnight*")
        for a in changed:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 🎯 One Story to Understand Today ──────────────────────────────────
    if story_raw and story_raw.get("title"):
        lines.append("🎯 *One Story to Understand Today*")
        src = f"  _({story_raw['source']})_" if story_raw.get("source") else ""
        lines.append(f"*{story_raw['title'][:110]}*{src}")
        if story_insight.get("why"):
            lines.append(f"*Why it matters:* {story_insight['why']}")
        if story_insight.get("now"):
            lines.append(f"*Right now:* {story_insight['now']}")
        if story_insight.get("next"):
            lines.append(f"*What's next:* {story_insight['next']}")
        if story_insight.get("takeaway"):
            lines.append(f"👉 {story_insight['takeaway']}")
        lines.append("")

    # ── 🇮🇳 India Lens ─────────────────────────────────────────────────────
    if india_lens:
        lines.append("🇮🇳 *India Lens*")
        for a in india_lens:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 💹 Markets Quick ───────────────────────────────────────────────────
    lines.append("💹 *Markets Quick*")
    if markets_text:
        lines.append(f"• {markets_text}")
    else:
        lines.append("• Markets data unavailable")
    if weather_text:
        lines.append(f"• {city}: {weather_text}")
    lines.append("")

    # ── 🤖 AI & the World ─────────────────────────────────────────────────
    # Dedicated AI section — model releases, regulation, industry shifts
    # Top AI story has a full editorial (like One Story), rest are headlines
    has_ai = ai_story_raw or ai_headlines
    if has_ai:
        lines.append("🤖 *AI & the World*")
        if ai_story_raw and ai_story_raw.get("title"):
            src = f"  _({ai_story_raw['source']})_" if ai_story_raw.get("source") else ""
            lines.append(f"*{ai_story_raw['title'][:110]}*{src}")
            if ai_insight.get("why"):
                lines.append(f"  _{ai_insight['why']}_")
            if ai_insight.get("takeaway"):
                lines.append(f"  👉 _{ai_insight['takeaway']}_")
        for a in ai_headlines:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 💻 Tech Radar (non-AI) ─────────────────────────────────────────────
    if tech_radar:
        lines.append("💻 *Tech*")
        for a in tech_radar:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── ⏰ What to Watch Today ─────────────────────────────────────────────
    if watch_today:
        lines.append("⏰ *What to Watch Today*")
        for a in watch_today:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 🏏 Sports ──────────────────────────────────────────────────────────
    if sports:
        lines.append("🏏 *Sports*")
        for a in sports:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # ── 📍 Local ───────────────────────────────────────────────────────────
    if local:
        lines.append(f"📍 *{city}*")
        for a in local:
            src = f"  _({a['source']})_" if a.get("source") else ""
            lines.append(f"• {a['title'][:100]}{src}")
        lines.append("")

    # Trim trailing blank
    while lines and lines[-1] == "":
        lines.pop()

    result = "\n".join(lines)
    logger.info("📰 live_data.news_briefing.done  sections=8  city=%r  chars=%d",
                city, len(result))
    return result



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
