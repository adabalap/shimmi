"""
Main WhatsApp Bot Application
- Production Grade v7.7 (Definitive)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .logging_setup import setup_logging
from .utils import normalize_whatsapp_id, canonical_text, sha1_hex

import app.database as database
from .waha_provider import init_waha, close_waha, send_text, typing_keepalive
from .multi_provider_llm import init_llm, close_llm, smart_complete
from .agent_engine import run_agent

from .ambient_memory import AmbientConfig, AmbientObserver
from .fact_mining import fact_mining_loop
from .reminder_manager import ReminderManager
from .reminder_scheduler import start_reminder_scheduler
from .reminder_commands import handle_reminder_command, detect_command as detect_reminder_command
from .structured_actions import init_actions_store

# --- Core Setup ---
setup_logging()
logger = logging.getLogger("app")
UTC = timezone.utc
app = FastAPI(title="Shimmi Bot", version="7.7.0")

# --- In-Memory State ---
CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}
CHAT_CONTEXT: Dict[str, Dict[str, Any]] = {}
OUTBOUND_CACHE_IDS: Dict[str, float] = {}
OUTBOUND_CACHE_TXT: Dict[str, float] = {}
OUTBOUND_TTL_SEC: float = 300.0

# --- Global Components ---
ambient_observer: Optional[AmbientObserver] = None
reminder_manager: Optional[ReminderManager] = None
reminder_scheduler = None

# --- Helper Functions ---

def compile_prefix_re() -> re.Pattern:
    tokens = [t.strip() for t in (settings.bot_command_prefix or "").split(",") if t.strip()]
    if not tokens: tokens = ["@shimmi", "shimmi"]
    tokens_sorted = sorted(tokens, key=len, reverse=True)
    alts = "|".join(re.escape(t) for t in tokens_sorted)
    return re.compile(r"(^|\s|[,.:;!?])(" + alts + r")(\s|[,.:;!?]|$)", flags=re.IGNORECASE)

PREFIX_RE = compile_prefix_re()

def has_prefix(text: str) -> bool:
    return bool(PREFIX_RE.search(text))

def strip_invocation(text: str) -> str:
    if not text: return ""
    match = PREFIX_RE.search(text)
    if not match: return text.strip()
    out = text[:match.start(2)] + text[match.end(2):]
    return re.sub(r'\s+', ' ', out).strip(" ,:;")

def get_chat_context(chat_id: str) -> Dict[str, Any]:
    CHAT_CONTEXT.setdefault(chat_id, {})
    return CHAT_CONTEXT[chat_id]

def normalize_event(body: dict) -> Tuple[str, str, str, bool, str]:
    root = body.get("payload", body)
    msg = root.get("message", {})
    text = root.get("body", msg.get("text", msg.get("conversation", "")))
    from_me = bool(root.get("fromMe", root.get("from_me", False)))
    chat_id = root.get("chatId", root.get("chat_id", root.get("from", "")))
    sender = root.get("sender", {}).get("id", root.get("participant", chat_id))
    event_id = str(root.get("id", ""))
    return normalize_whatsapp_id(sender), chat_id, text or "", from_me, event_id

def _purge_outbound_caches() -> None:
    now = time.time()
    cutoff = now - OUTBOUND_TTL_SEC
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if ts < cutoff: OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if ts < cutoff: OUTBOUND_CACHE_TXT.pop(k, None)

def outbound_hash(chat_id: str, msg: str) -> str:
    return sha1_hex(chat_id + "\n" + canonical_text(msg))

def is_echo(chat_id: str, text: str) -> bool:
    """Checks if an incoming message is an echo of an outgoing one."""
    # The text hash check is now the primary defense.
    if chat_id and text:
        if outbound_hash(chat_id, text) in OUTBOUND_CACHE_TXT:
            return True
    return False

def _apply_bot_prefix(text: str) -> str:
    """Adds the bot emoji prefix to the text."""
    if not str(os.getenv("BOT_EMOJI_PREFIX_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on"):
        return text
    if not (emoji := (os.getenv("BOT_EMOJI") or "").strip()):
        return text
    if not (s := (text or "").lstrip()):
        return s
    if s.startswith(emoji):
        return s
    return f"{emoji} {s}"

# --- Application Lifecycle & Main Logic ---

@app.on_event("startup")
async def startup() -> None:
    # ... (startup logic remains the same)
    pass

@app.on_event("shutdown")
async def shutdown() -> None:
    # ... (shutdown logic remains the same)
    pass

async def process_message(chat_id: str, sender_id: str, text: str, event_id: str, from_me: bool) -> None:
    bot_reply_raw = ""
    user_text_for_log = ""
    try:
        chat_ctx = get_chat_context(chat_id)
        is_follow_up = chat_ctx.get('awaiting_answer_since', 0) > time.time() - 300
        user_text = text.strip() if is_follow_up else strip_invocation(text)
        user_text_for_log = user_text
        if not user_text: return

        logger.info("➡️  processing: user_text='%s'", user_text)

        if reminder_manager and (cmd := detect_reminder_command(user_text)):
            if response := await handle_reminder_command(user_text, sender_id, chat_id, reminder_manager):
                bot_reply_raw = response
                # ... fall through to send logic ...
        else:
            facts = await database.sqlite_store.get_all_facts(sender_id) if database.sqlite_store else {}
            context = []
            if database.chroma_store:
                snippets = await database.chroma_store.search(chat_id=chat_id, query=user_text, k=5, whatsapp_id=sender_id)
                context = [s.text for s in snippets]

            result = await run_agent(
                user_text=user_text, facts=facts, context=context,
                llm_complete_fn=smart_complete,
                is_follow_up=is_follow_up, last_question=chat_ctx.get('last_question')
            )
            
            chat_ctx.pop('awaiting_answer_since', None)
            chat_ctx.pop('last_question', None)
            if result.question_asked:
                chat_ctx['awaiting_answer_since'] = time.time()
                chat_ctx['last_question'] = result.question_asked
            
            bot_reply_raw = result.reply.text
            
            if database.sqlite_store and result.memory_updates:
                for mu in result.memory_updates:
                    await database.sqlite_store.upsert_fact(sender_id, mu.key, mu.value)

        # --- Send Logic (centralized) ---
        final_bot_reply = _apply_bot_prefix(bot_reply_raw)
        
        # Add to echo cache BEFORE sending
        OUTBOUND_CACHE_TXT[outbound_hash(chat_id, final_bot_reply)] = time.time()
        
        await send_text(chat_id, final_bot_reply)

    finally:
        final_log_reply = _apply_bot_prefix(bot_reply_raw)
        logger.info("⬅️  replying to '%s': bot_reply='%s'", user_text_for_log, final_log_reply)

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        sender_id, chat_id, text, from_me, event_id = normalize_event(body)

        if not (settings.allowed_chat_jids and chat_id in settings.allowed_chat_jids):
            return JSONResponse({"status": "ok", "message": "chat not in allowlist"})
        if not text: return JSONResponse({"status": "ok", "message": "empty"})

        _purge_outbound_caches()
        # The text-based echo check is now the primary defense.
        if is_echo(chat_id, text):
            return JSONResponse({"status": "ok", "message": "echo ignored"})

        if ambient_observer:
            await ambient_observer.observe(chat_id=chat_id, sender_id=sender_id, text=text, is_group="@g.us" in chat_id, event_id=event_id)
        
        # We can't trust from_me, so it's only a weak secondary check.
        if from_me and not settings.allow_fromme:
            return JSONResponse({"status": "ok", "message": "fromMe ignored"})

        chat_ctx = get_chat_context(chat_id)
        is_awaiting_reply = chat_ctx.get('awaiting_answer_since', 0) > time.time() - 300
        if "@g.us" in chat_id and not is_awaiting_reply and not has_prefix(text):
            return JSONResponse({"status": "ok", "message": "group chat requires prefix"})
            
        if (time.perf_counter() - CHAT_LAST_MSG_TS.get(chat_id, 0.0)) * 1000.0 < settings.message_debounce_ms:
            return JSONResponse({"status": "ok", "message": "debounced"})
        CHAT_LAST_MSG_TS[chat_id] = time.perf_counter()

        q = CHAT_QUEUES.get(chat_id)
        if not q:
            q = asyncio.Queue(maxsize=settings.llm_max_queue_per_chat)
            CHAT_QUEUES[chat_id] = q
            async def _worker():
                while True:
                    try:
                        item = await q.get()
                        await process_message(**item)
                    except asyncio.CancelledError: break
                    finally: q.task_done()
            CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

        await asyncio.wait_for(q.put({"chat_id": chat_id, "sender_id": sender_id, "text": text, "event_id": event_id, "from_me": from_me}), timeout=5.0)
        return JSONResponse({"status": "ok", "message": "enqueued"})
    except asyncio.TimeoutError:
        return JSONResponse({"status": "ok", "message": "queue timeout"})
    except Exception:
        logger.exception("webhook.error")
        return JSONResponse({"status": "error", "message": "internal server error"}, status_code=500)

