from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from typing import Dict, Optional

import httpx

from . import config
from .utils import canonical_text, sanitize_for_whatsapp

logger = logging.getLogger('app.waha')

OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = float(os.getenv('OUTBOUND_TTL_SEC', '300'))

HTTPX_WAHA: Optional[httpx.AsyncClient] = None
_init_lock = asyncio.Lock()

LOG_OUTBOUND_PREVIEW = os.getenv('LOG_OUTBOUND_PREVIEW', '1').lower() in ('1','true','yes','on')
LOG_OUTBOUND_TEXT = os.getenv('LOG_OUTBOUND_TEXT', '0').lower() in ('1','true','yes','on')


def _purge_outbound_caches():
    now = time.time()
    cutoff = now - OUTBOUND_TTL_SEC
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_TXT.pop(k, None)


async def init_client():
    global HTTPX_WAHA
    if HTTPX_WAHA:
        return
    if not (config.WAHA_API_URL or '').strip():
        logger.error('❌ waha.misconfig WAHA_API_URL is empty')
        return

    async with _init_lock:
        if HTTPX_WAHA:
            return
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        headers = {'X-Api-Key': config.WAHA_API_KEY} if (config.WAHA_API_KEY or '').strip() else None
        timeout = httpx.Timeout(connect=5.0, read=float(os.getenv('WAHA_TIMEOUT','30')), write=30.0, pool=5.0)
        HTTPX_WAHA = httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers)
        logger.info('✅ waha.init url=%s session=%s', config.WAHA_API_URL, config.WAHA_SESSION)


async def close_client():
    global HTTPX_WAHA
    if HTTPX_WAHA:
        try:
            await HTTPX_WAHA.aclose()
        finally:
            HTTPX_WAHA = None


async def typing_keepalive(chat_id: str, stop_event: asyncio.Event):
    if not HTTPX_WAHA:
        await stop_event.wait()
        return

    async def _start():
        try:
            await HTTPX_WAHA.post(f'{config.WAHA_API_URL}/startTyping', json={'session': config.WAHA_SESSION, 'chatId': chat_id})
        except Exception:
            pass

    async def _stop():
        try:
            await HTTPX_WAHA.post(f'{config.WAHA_API_URL}/stopTyping', json={'session': config.WAHA_SESSION, 'chatId': chat_id})
        except Exception:
            pass

    try:
        await _start()
        refresh = 4.0
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=refresh)
                break
            except asyncio.TimeoutError:
                await _start()
                refresh = min(10.0, refresh * 1.2)
    finally:
        await _stop()


async def send_message(chat_id: str, content: str) -> tuple[bool, Optional[str]]:
    if not HTTPX_WAHA:
        logger.error('❌ waha.not_initialized')
        return False, None

    text = sanitize_for_whatsapp(content or '')
    if not text:
        return False, None

    _purge_outbound_caches()

    key = hashlib.sha1(f'{chat_id}\n{canonical_text(text)}'.encode('utf-8')).hexdigest()
    if len(text) > 80:
        ts = OUTBOUND_CACHE_TXT.get(key)
        if ts and (time.time() - ts) < OUTBOUND_TTL_SEC:
            logger.info('♻️ waha.dedup_skip chat_id=%s', chat_id)
            return True, None

    if LOG_OUTBOUND_PREVIEW:
        if LOG_OUTBOUND_TEXT:
            logger.info('📤 wa.out chat_id=%s len=%s text=%s', chat_id, len(text), text)
        else:
            logger.info('📤 wa.out chat_id=%s len=%s preview=%s', chat_id, len(text), text[:220])

    payload = {'session': config.WAHA_SESSION, 'chatId': chat_id, 'text': text}
    url = f'{config.WAHA_API_URL}/sendText'

    start = time.perf_counter()
    tries = 0
    success = False
    last_status = None
    msg_id = None

    while tries < 2 and not success:
        tries += 1
        try:
            resp = await HTTPX_WAHA.post(url, json=payload)
            last_status = resp.status_code
            if 200 <= resp.status_code < 300:
                success = True
                data = {}
                try:
                    data = resp.json() or {}
                except Exception:
                    data = {}
                msg_id = data.get('id') or (data.get('message') or {}).get('id')
                if msg_id:
                    OUTBOUND_CACHE_IDS[str(msg_id)] = time.time()
                OUTBOUND_CACHE_TXT[key] = time.time()
                break
            body = (resp.text or '')[:400]
            logger.error('❌ waha.http_error status=%s body=%s', resp.status_code, body)
            await asyncio.sleep(0.2 + random.random() * 0.4)
        except Exception as e:
            logger.error('❌ waha.error err=%s', str(e)[:160])
            break

    ms = int((time.perf_counter() - start) * 1000)
    logger.info('✅ waha.send_done chat_id=%s success=%s status=%s ms=%s', chat_id, success, last_status, ms)
    return success, msg_id
