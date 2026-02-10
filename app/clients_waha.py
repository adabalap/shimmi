# app/clients_waha.py
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import time
from typing import Dict, Optional

import httpx

from . import config
from .utils import canonical_text

logger = logging.getLogger("app")

# Dedup caches for echo & idempotency
OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = float(os.getenv("OUTBOUND_TTL_SEC", "300"))

HTTPX_WAHA: Optional[httpx.AsyncClient] = None
_init_lock = asyncio.Lock()

# Logging controls (avoid leaking content in logs)
LOG_OUTBOUND_PREVIEW = os.getenv("LOG_OUTBOUND_PREVIEW", "0").strip().lower() in ("1", "true", "yes", "on")

# WhatsApp formatting controls
WA_STRIP_MARKDOWN = os.getenv("WA_STRIP_MARKDOWN", "1") == "1"
WA_NORMALIZE_BULLETS = os.getenv("WA_NORMALIZE_BULLETS", "0") == "1"  # default OFF
WA_WRAP_COL = int(os.getenv("WA_WRAP_COL", "0"))  # 0 = no wrap

# Emoji/branding
BOT_EMOJI = os.getenv("BOT_EMOJI", "").strip()
BOT_EMOJI_PREFIX_ENABLED = os.getenv("BOT_EMOJI_PREFIX_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
EMOJI_POLICY = os.getenv("EMOJI_POLICY", "normal").strip().lower()  # none|minimal|normal
EMOJI_MAX_PER_MSG = int(os.getenv("EMOJI_MAX_PER_MSG", "6"))

STAR_SINGLE_WORD = re.compile(r"(?<!\*)\*(\w[\w\-']{2,48})\*(?!\*)")
BOLD_DOUBLE = re.compile(r"\*\*(.+?)\*\*")
CODE_FENCE = re.compile(r"```[^`]*```", re.DOTALL)
BACKTICKS = re.compile(r"`([^`]*)`")
DUP_SPACES = re.compile(r"[ \t]{2,}")

# Basic emoji coverage (not exhaustive)
_EMOJI_RE = re.compile(
    "["                     
    "\U0001F300-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]"
)


def _purge_outbound_caches(nowt: float | None = None) -> None:
    nowt = nowt or time.time()
    cutoff = nowt - OUTBOUND_TTL_SEC
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if ts < cutoff:
            OUTBOUND_CACHE_TXT.pop(k, None)


def _sanitize_whatsapp(text: str) -> str:
    if not text:
        return text
    t = text

    if WA_STRIP_MARKDOWN:
        # Keep fenced block content but remove backticks
        t = CODE_FENCE.sub(lambda m: m.group(0).replace("`", ""), t)
        t = BACKTICKS.sub(lambda m: m.group(1), t)
        t = BOLD_DOUBLE.sub(lambda m: m.group(1), t)
        t = STAR_SINGLE_WORD.sub(lambda m: m.group(1), t)

    if WA_NORMALIZE_BULLETS:
        t = re.sub(r"(?m)^\s*-\s+", "• ", t)
        t = re.sub(r"(?m)^\s*\*\s+", "• ", t)

    t = DUP_SPACES.sub(" ", t)

    if WA_WRAP_COL and WA_WRAP_COL >= 40:
        out_lines = []
        for line in t.splitlines():
            line = line.strip()
            if len(line) <= WA_WRAP_COL:
                out_lines.append(line)
                continue
            buf = line
            while len(buf) > WA_WRAP_COL:
                cut = buf.rfind(" ", 0, WA_WRAP_COL)
                cut = cut if cut != -1 else WA_WRAP_COL
                out_lines.append(buf[:cut].strip())
                buf = buf[cut:].lstrip()
            if buf:
                out_lines.append(buf)
        t = "\n".join(out_lines)

    return t.strip()


def _apply_emoji_policy(text: str) -> str:
    if EMOJI_POLICY == "none":
        return _EMOJI_RE.sub("", text)

    if EMOJI_POLICY == "minimal":
        out, count = [], 0
        for ch in text:
            if _EMOJI_RE.match(ch):
                if count >= EMOJI_MAX_PER_MSG:
                    continue
                count += 1
            out.append(ch)
        return "".join(out)

    return text  # normal


async def init_client():
    global HTTPX_WAHA
    if HTTPX_WAHA:
        return

    async with _init_lock:
        if HTTPX_WAHA:
            return

        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        headers = {"X-Api-Key": config.WAHA_API_KEY} if getattr(config, "WAHA_API_KEY", "") else None
        timeout = httpx.Timeout(
            connect=5.0,
            read=float(os.getenv("WAHA_TIMEOUT", "30")),
            write=30.0,
            pool=5.0,
        )
        HTTPX_WAHA = httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers)
        logger.info("waha_client_initialized url=%s session=%s", config.WAHA_API_URL, config.WAHA_SESSION)


async def close_client():
    global HTTPX_WAHA
    if HTTPX_WAHA:
        try:
            await HTTPX_WAHA.aclose()
        finally:
            HTTPX_WAHA = None
    logger.info("waha_client_closed")


async def start_typing(chat_id: str):
    if not HTTPX_WAHA:
        return
    try:
        await HTTPX_WAHA.post(
            f"{config.WAHA_API_URL}/startTyping",
            json={"session": config.WAHA_SESSION, "chatId": chat_id},
        )
    except Exception:
        pass


async def stop_typing(chat_id: str):
    if not HTTPX_WAHA:
        return
    try:
        await HTTPX_WAHA.post(
            f"{config.WAHA_API_URL}/stopTyping",
            json={"session": config.WAHA_SESSION, "chatId": chat_id},
        )
    except Exception:
        pass


async def typing_keepalive(chat_id: str, stop_event: asyncio.Event):
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


async def send_message(chat_id: str, content: str) -> bool:
    global HTTPX_WAHA
    if not HTTPX_WAHA:
        logger.error("waha_client_not_initialized")
        return False

    text = (content or "").strip()
    if not text:
        return False

    _purge_outbound_caches()

    sanitized = _sanitize_whatsapp(text)
    sanitized = _apply_emoji_policy(sanitized)

    # Optional bot emoji prefix
    if BOT_EMOJI_PREFIX_ENABLED and BOT_EMOJI and not sanitized.startswith(BOT_EMOJI):
        sanitized = f"{BOT_EMOJI} {sanitized}"

    # Idempotency: skip if same recent message already sent to this chat
    key = hashlib.sha1(f"{chat_id}\n{canonical_text(sanitized)}".encode("utf-8")).hexdigest()
    ts = OUTBOUND_CACHE_TXT.get(key)
    if ts and (time.time() - ts) < OUTBOUND_TTL_SEC:
        logger.info("waha_send_dedup_skip chat_id=%s", chat_id)
        return True

    if LOG_OUTBOUND_PREVIEW:
        logger.info("📨 outbound.preview chat_id=%s len=%s text=%s", chat_id, len(sanitized), sanitized[:420])
    else:
        logger.info("📨 outbound.preview chat_id=%s len=%s", chat_id, len(sanitized))

    payload = {"session": config.WAHA_SESSION, "chatId": chat_id, "text": sanitized}
    url = f"{config.WAHA_API_URL}/sendText"

    start = time.perf_counter()
    tries = 0
    success = False
    last_status = None

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
                    pass
                msg_id = data.get("id") or (data.get("message") or {}).get("id")
                if msg_id:
                    OUTBOUND_CACHE_IDS[str(msg_id)] = time.time()
                OUTBOUND_CACHE_TXT[key] = time.time()
                break

            body = (resp.text or "")[:500]
            # WAHA known soft-fail
            if resp.status_code == 500 and "markedUnread" in body:
                success = True
                OUTBOUND_CACHE_TXT[key] = time.time()
                logger.warning("waha_send_soft_fail_markedUnread")
                break

            logger.error("waha_send_http_error status=%s body=%s", resp.status_code, body)
            await asyncio.sleep(0.2 + random.random() * 0.4)

        except httpx.RequestError as e:
            logger.error("waha_send_request_error err=%s", e)
            await asyncio.sleep(0.3 + random.random() * 0.4)
        except Exception as e:
            logger.error("waha_send_unexpected_error err=%s", e)
            break

    dur_ms = int((time.perf_counter() - start) * 1000)
    logger.info("waha_send_done chat_id=%s success=%s status=%s duration_ms=%s", chat_id, success, last_status, dur_ms)
    return success
