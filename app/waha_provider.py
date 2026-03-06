from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple

import httpx

from .config import settings
from .retry import async_retry
from .utils import canonical_text, sanitize_for_whatsapp, sha1_hex

logger = logging.getLogger("app.waha")

HTTPX_WAHA: Optional[httpx.AsyncClient] = None
_init_lock = asyncio.Lock()

# These will be managed in main.py but are declared here for clarity
OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = 300.0

_NEWLINE = chr(10)

def outbound_hash(chat_id: str, msg: str) -> str:
    return sha1_hex(chat_id + _NEWLINE + canonical_text(msg))

def _emoji_prefix(text: str) -> str:
    """Applies the bot emoji prefix if it's enabled in the settings."""
    if not str(os.getenv("BOT_EMOJI_PREFIX_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on"):
        return text
    if not (emoji := (os.getenv("BOT_EMOJI") or "").strip()):
        return text
    if not (s := (text or "").lstrip()):
        return s
    if s.startswith(emoji):
        return s
    return f"{emoji} {s}"

async def init_waha() -> None:
    global HTTPX_WAHA
    if HTTPX_WAHA: return
    if not settings.waha_api_url:
        logger.error("❌ waha.misconfig WAHA_API_URL is empty")
        return
    async with _init_lock:
        if HTTPX_WAHA: return
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        headers = {"X-Api-Key": settings.waha_api_key} if settings.waha_api_key else None
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
        HTTPX_WAHA = httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers)
        logger.info("✅ waha.init url=%s session=%s", settings.waha_api_url, settings.waha_session)

async def close_waha() -> None:
    global HTTPX_WAHA
    if HTTPX_WAHA:
        try: await HTTPX_WAHA.aclose()
        finally: HTTPX_WAHA = None

async def start_typing(chat_id: str) -> None:
    if not HTTPX_WAHA: return
    try:
        await HTTPX_WAHA.post(f"{settings.waha_api_url}/startTyping", json={"session": settings.waha_session, "chatId": chat_id})
    except Exception: pass

async def stop_typing(chat_id: str) -> None:
    if not HTTPX_WAHA: return
    try:
        await HTTPX_WAHA.post(f"{settings.waha_api_url}/stopTyping", json={"session": settings.waha_session, "chatId": chat_id})
    except Exception: pass

async def typing_keepalive(chat_id: str, stop_event: asyncio.Event) -> None:
    try:
        await start_typing(chat_id)
        refresh = 4.0
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=refresh)
                break
            except asyncio.TimeoutError:
                await start_typing(chat_id)
                refresh = min(10.0, refresh * 1.2)
    finally:
        await stop_typing(chat_id)

async def _post(path: str, payload: dict) -> dict:
    if not HTTPX_WAHA: raise RuntimeError("waha not initialized")
    async def _do() -> dict:
        resp = await HTTPX_WAHA.post(f"{settings.waha_api_url}/{path.lstrip('/')}", json=payload)
        resp.raise_for_status()
        return resp.json() or {}
    return await async_retry(_do, max_attempts=3, base_delay=0.4, max_delay=4.0)

async def send_text(chat_id: str, text: str) -> Tuple[str, dict]:
    """Sends text and returns the final message sent AND the API response."""
    if not HTTPX_WAHA: await init_waha()
    if not HTTPX_WAHA: return "", {}

    final_msg = _emoji_prefix(sanitize_for_whatsapp(text or ""))
    if not final_msg: return "", {}

    payload = {"session": settings.waha_session, "chatId": chat_id, "text": final_msg}
    try:
        response_data = await _post("sendText", payload)
        return final_msg, response_data
    except Exception as e:
        logger.error("waha.send_text.failed error=%s", str(e))
        return final_msg, {}


