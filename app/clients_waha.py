
# app/clients_waha.py
from __future__ import annotations
import asyncio, time, hashlib, logging, os, re
from typing import Optional, Dict
import httpx
from . import config

logger = logging.getLogger("app")

OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = float(os.getenv("OUTBOUND_TTL_SEC", "300"))

HTTPX_WAHA: Optional[httpx.AsyncClient] = None

# ---------- WhatsApp formatting controls ----------
WA_STRIP_MARKDOWN = os.getenv("WA_STRIP_MARKDOWN", "1") == "1"
WA_NORMALIZE_BULLETS = os.getenv("WA_NORMALIZE_BULLETS", "0") == "1"   # default OFF
WA_WRAP_COL = int(os.getenv("WA_WRAP_COL", "0"))  # 0 = no wrap

# Emoji/branding
BOT_EMOJI = os.getenv("BOT_EMOJI", "").strip()
BOT_EMOJI_PREFIX_ENABLED = os.getenv("BOT_EMOJI_PREFIX_ENABLED", "0").strip().lower() in ("1","true","yes","on")
EMOJI_POLICY = os.getenv("EMOJI_POLICY", "normal").strip().lower()  # none|minimal|normal
EMOJI_MAX_PER_MSG = int(os.getenv("EMOJI_MAX_PER_MSG", "6"))

STAR_SINGLE_WORD = re.compile(r"(?<!\*)\*(\w[\w\-']{2,48})\*(?!\*)")  # *Word* → Word
BOLD_DOUBLE = re.compile(r"\*\*(.+?)\*\*")  # **phrase** → phrase
CODE_FENCE = re.compile(r"```[^`]*```", re.DOTALL)
BACKTICKS = re.compile(r"`([^`]*)`")
DUP_SPACES = re.compile(r"[ \t]{2,}")

def _sanitize_whatsapp(text: str) -> str:
    if not text:
        return text
    t = text

    if WA_STRIP_MARKDOWN:
        t = CODE_FENCE.sub(lambda m: m.group(0).replace("`", ""), t)
        t = BACKTICKS.sub(lambda m: m.group(1), t)
        t = BOLD_DOUBLE.sub(lambda m: m.group(1), t)
        t = STAR_SINGLE_WORD.sub(lambda m: m.group(1), t)

    if WA_NORMALIZE_BULLETS:
        t = re.sub(r"(?m)^\s*-\s+", "• ", t)
        t = re.sub(r"(?m)^\s*\*\s+(?!\S)", "• ", t)

    t = DUP_SPACES.sub(" ", t)
    if WA_WRAP_COL and WA_WRAP_COL >= 40:
        out = []
        for line in t.splitlines():
            if len(line) <= WA_WRAP_COL:
                out.append(line.strip())
            else:
                buf = line.strip()
                while len(buf) > WA_WRAP_COL:
                    cut = buf.rfind(" ", 0, WA_WRAP_COL)
                    cut = cut if cut != -1 else WA_WRAP_COL
                    out.append(buf[:cut].strip())
                    buf = buf[cut:].lstrip()
                if buf:
                    out.append(buf)
        t = "\n".join(out)

    return t.strip()

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
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    headers = {"X-Api-Key": config.WAHA_API_KEY} if getattr(config, "WAHA_API_KEY", "") else None
    timeout = int(os.getenv("WAHA_TIMEOUT", "30"))
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

def _canonical_text(text: str) -> str:
    return (" ".join((text or "").strip().split()))[:4000]

async def start_typing(chat_id: str):
    if not HTTPX_WAHA: return
    try:
        await HTTPX_WAHA.post(f"{config.WAHA_API_URL}/startTyping",
                              json={"session": config.WAHA_SESSION, "chatId": chat_id})
    except Exception:
        pass

async def stop_typing(chat_id: str):
    if not HTTPX_WAHA: return
    try:
        await HTTPX_WAHA.post(f"{config.WAHA_API_URL}/stopTyping",
                              json={"session": config.WAHA_SESSION, "chatId": chat_id})
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

    sanitized = _sanitize_whatsapp(text)
    sanitized = _apply_emoji_policy(sanitized)

    # Optional bot emoji prefix
    if BOT_EMOJI_PREFIX_ENABLED and BOT_EMOJI and not sanitized.startswith(BOT_EMOJI):
        sanitized = f"{BOT_EMOJI} {sanitized}"

    # Log exact message preview we will send
    logger.info("📨 outbound.preview chat_id=%s len=%s text=%s",
                chat_id, len(sanitized), sanitized[:420])

    payload = {"session": config.WAHA_SESSION, "chatId": chat_id, "text": sanitized}
    url = f"{config.WAHA_API_URL}/sendText"

    start = time.perf_counter()
    tries = 0
    success = False
    last_status = None
    last_body = ""

    while tries < 2 and not success:
        tries += 1
        try:
            resp = await HTTPX_WAHA.post(url, json=payload)
            last_status = resp.status_code
            if 200 <= resp.status_code < 300:
                success = True
                try:
                    data = resp.json() or {}
                except Exception:
                    data = {}
                msg_id = data.get("id") or (data.get("message") or {}).get("id")
                if msg_id:
                    OUTBOUND_CACHE_IDS[str(msg_id)] = time.time()
                OUTBOUND_CACHE_TXT[hashlib.sha1(f"{chat_id}\n{_canonical_text(sanitized)}".encode("utf-8")).hexdigest()] = time.time()
                break

            try:
                last_body = (resp.text or "")[:500]
            except Exception:
                last_body = ""
            if resp.status_code == 500 and "markedUnread" in (last_body or ""):
                success = True
                OUTBOUND_CACHE_TXT[hashlib.sha1(f"{chat_id}\n{_canonical_text(sanitized)}".encode("utf-8")).hexdigest()] = time.time()
                logger.warning("waha_send_soft_fail_markedUnread status=%s", resp.status_code)
                break

            logger.error("waha_send_http_error status=%s body=%s", resp.status_code, last_body)
            await asyncio.sleep(0.3)

        except httpx.RequestError as e:
            logger.error("waha_send_request_error err=%s", e)
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error("waha_send_unexpected_error err=%s", e)
            break

    dur_ms = int((time.perf_counter() - start) * 1000)
    logger.info("waha_send_done chat_id=%s success=%s status=%s duration_ms=%s",
                chat_id, success, last_status, dur_ms)
    return success

