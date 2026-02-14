from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import httpx

from .config import settings
from .retry import async_retry
from .utils import canonical_text, sanitize_for_whatsapp, sha1_hex

logger = logging.getLogger("app.waha")

HTTPX_WAHA: Optional[httpx.AsyncClient] = None
_init_lock = asyncio.Lock()

OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = 300.0

_NEWLINE = chr(10)


def outbound_hash(chat_id: str, msg: str) -> str:
    return sha1_hex(chat_id + _NEWLINE + canonical_text(msg))


def _purge_outbound() -> None:
    now = time.time()
    cutoff = now - OUTBOUND_TTL_SEC
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_TXT.pop(k, None)


async def init_waha() -> None:
    global HTTPX_WAHA
    if HTTPX_WAHA:
        return
    if not settings.waha_api_url:
        logger.error("❌ waha.misconfig WAHA_API_URL is empty")
        return

    async with _init_lock:
        if HTTPX_WAHA:
            return
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        headers = {"X-Api-Key": settings.waha_api_key} if settings.waha_api_key else None
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
        HTTPX_WAHA = httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers)
        logger.info("✅ waha.init url=%s session=%s", settings.waha_api_url, settings.waha_session)


async def close_waha() -> None:
    global HTTPX_WAHA
    if HTTPX_WAHA:
        try:
            await HTTPX_WAHA.aclose()
        finally:
            HTTPX_WAHA = None


async def start_typing(chat_id: str) -> None:
    if not HTTPX_WAHA:
        return
    try:
        await HTTPX_WAHA.post(f"{settings.waha_api_url}/startTyping", json={"session": settings.waha_session, "chatId": chat_id})
    except Exception:
        pass


async def stop_typing(chat_id: str) -> None:
    if not HTTPX_WAHA:
        return
    try:
        await HTTPX_WAHA.post(f"{settings.waha_api_url}/stopTyping", json={"session": settings.waha_session, "chatId": chat_id})
    except Exception:
        pass


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
    if not HTTPX_WAHA:
        raise RuntimeError("waha not initialized")

    async def _do() -> dict:
        resp = await HTTPX_WAHA.post(f"{settings.waha_api_url}/{path.lstrip('/')}", json=payload)
        resp.raise_for_status()
        try:
            return resp.json() or {}
        except Exception:
            return {}

    return await async_retry(_do, max_attempts=3, base_delay=0.4, max_delay=4.0)


async def send_text(chat_id: str, text: str) -> dict:
    if not HTTPX_WAHA:
        await init_waha()
    if not HTTPX_WAHA:
        return {}

    msg = sanitize_for_whatsapp(text or "")
    if not msg:
        return {}

    _purge_outbound()

    h = outbound_hash(chat_id, msg)
    payload = {"session": settings.waha_session, "chatId": chat_id, "text": msg}
    data = await _post("sendText", payload)

    msg_id = data.get("id") or (data.get("message") or {}).get("id")
    if msg_id:
        OUTBOUND_CACHE_IDS[str(msg_id)] = time.time()

    OUTBOUND_CACHE_TXT[h] = time.time()
    return data
