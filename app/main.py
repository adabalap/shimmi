from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_setup import setup_logging, log_startup_env
from .utils import verify_signature, canonical_text
from . import config

from .db import init_db, close_db, save_message, fetchall, execute
from .clients_waha import (
    init_client as init_waha,
    close_client as close_waha,
    send_message,
    typing_keepalive,
    OUTBOUND_CACHE_IDS,
    OUTBOUND_CACHE_TXT,
    OUTBOUND_TTL_SEC,
)
from .clients_llm import init_clients as init_llm, close_clients as close_llm, groq_chat, groq_live_search
from .observe import handle_observe_command, observe_ingest

setup_logging()
logger = logging.getLogger("app")

# optional chroma
CHROMA_AVAILABLE = True
try:
    from .chroma import add_text as chroma_add_text, query as chroma_query, warmup as chroma_warmup
except Exception:
    CHROMA_AVAILABLE = False


def chroma_enabled() -> bool:
    return CHROMA_AVAILABLE and bool(getattr(config, "CHROMA_ENABLED", True))


app = FastAPI()

CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}


def log_kv(event: str, **kw):
    def _s(v):
        if isinstance(v, str) and len(v) > 420:
            return v[:420] + "…"
        return v
    logger.info("%s %s", event, " ".join(f"{k}={_s(v)}" for k, v in kw.items()))


def looks_group(j: Optional[str]) -> bool:
    return bool(j and j.endswith("@g.us"))


def looks_broadcast(j: Optional[str]) -> bool:
    return bool(j and j.endswith("@broadcast"))


def looks_channel(j: Optional[str]) -> bool:
    return bool(j and j.endswith("@newsletter"))


def is_groupish(j: Optional[str]) -> bool:
    return looks_group(j) or looks_broadcast(j) or looks_channel(j)


def group_allowed(chat_id: Optional[str]) -> bool:
    if not chat_id:
        return False
    if looks_channel(chat_id):
        return True
    if not looks_group(chat_id):
        return True
    allow = getattr(config, "ALLOWED_GROUP_JIDS", None)
    if not allow:
        return True
    return chat_id in allow


def normalize_jid(jid: Optional[str]) -> Optional[str]:
    if not jid:
        return None
    if jid.endswith("@s.whatsapp.net"):
        return jid.replace("@s.whatsapp.net", "@c.us")
    return jid


# Prefix matching anywhere in sentence
_PREFIX_RE: Optional[re.Pattern] = None

def _prefixes() -> list[str]:
    raw = getattr(config, "BOT_COMMAND_PREFIX", "") or ""
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(p).strip() for p in raw if str(p).strip()]
    s = str(raw).strip()
    return [s] if s else []


def _compile_prefix_re():
    global _PREFIX_RE
    alts = [re.escape(p.lstrip("@")) for p in _prefixes()]
    if not alts:
        _PREFIX_RE = re.compile(r"a^")
        return
    _PREFIX_RE = re.compile(r"(?i)(?:^|\s)@?(%s)\b" % "|".join(alts))


def has_prefix(text: Optional[str]) -> bool:
    if not text:
        return False
    if _PREFIX_RE is None:
        _compile_prefix_re()
    return bool(_PREFIX_RE.search(text))


def strip_prefix(text: str) -> str:
    if _PREFIX_RE is None:
        _compile_prefix_re()
    s = text or ""
    m = _PREFIX_RE.search(s)
    if not m:
        return s.strip()
    start, end = m.span()
    before = s[:start].strip()
    after = s[end:].lstrip(" ,:;-\t")
    if not before:
        return after.strip()
    joined = (before + " " + after).strip()
    return re.sub(r"\s+", " ", joined)


def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1","true","yes","y","on"}:
        return True
    if s in {"0","false","no","n","off"}:
        return False
    return default


def needs_live_search(user_text: str) -> bool:
    t = (user_text or "").lower()
    keys = ["latest", "today", "current", "news", "updates", "score", "price", "weather", "trending"]
    return any(k in t for k in keys)


def infer_time_window(text: str) -> Tuple[Optional[str], Optional[str]]:
    now = datetime.now(timezone.utc)
    tl = (text or "").lower()
    if "since yesterday" in tl or "yesterday" in tl:
        start = now - timedelta(days=1)
        start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        return start.isoformat(), now.isoformat()
    m = re.search(r"since\s+(\d+)\s+days", tl)
    if m:
        days = max(1, int(m.group(1)))
        return (now - timedelta(days=days)).isoformat(), now.isoformat()
    return None, None


async def summarize_from_db(chat_id: str, limit: int = 60, *, since_iso: Optional[str] = None, until_iso: Optional[str] = None) -> str:
    rows = await fetchall("SELECT timestamp, role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?", (chat_id, limit))
    if not rows:
        return ""
    rows = list(reversed(rows))
    if since_iso and until_iso:
        rows = [r for r in rows if r[0] and (str(r[0]) >= since_iso) and (str(r[0]) <= until_iso)]
    convo = "\n".join([f"{r[1].upper()}: {r[2]}" for r in rows if r[2]])
    sys = "You summarize ONLY the provided chat lines. Plain text. No markdown. Do not invent details."
    reply, ok, _ = await groq_chat(chat_id, sys, convo, temperature=0.0, max_tokens=320)
    return reply.strip() if ok and reply else ""


async def assemble_context(chat_id: str, user_text: str) -> str:
    if not chroma_enabled():
        return ""
    try:
        rag = await chroma_query(chat_id=chat_id, text=user_text, k=3)
        return rag or ""
    except Exception:
        return ""


async def process_message(chat_id: str, text: str, sender_id: str):
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))
    try:
        user_text = strip_prefix((text or "").strip())
        log_kv("💬 process.begin", chat_id=chat_id, sender_id=sender_id, text=user_text[:160])

        if any(x in user_text.lower() for x in ("summarise", "summarize", "recap", "tl;dr")):
            since, until = infer_time_window(user_text)
            s = await summarize_from_db(chat_id, since_iso=since, until_iso=until)
            if not s:
                s = "I don’t have enough chat history yet to summarize."
            sent = await send_message(chat_id, s)
            log_kv("🧾 summarize", chat_id=chat_id, sent=sent, len=len(s))
            return

        if getattr(config, "LIVE_SEARCH_ENABLED", False) and needs_live_search(user_text):
            reply, ok, _ = await groq_live_search(chat_id, user_text)
            if ok and reply:
                sent = await send_message(chat_id, reply)
                log_kv("🌐 live.reply", chat_id=chat_id, sent=sent, len=len(reply))
                return

        context = await assemble_context(chat_id, user_text)
        persona = getattr(config, "BOT_PERSONA_NAME", "Stateful AI BOT")
        sys = "\n".join([
            f"You are {persona}, a privacy-safe stateful WhatsApp assistant.",
            "Write in 1–4 short sentences.",
            "Avoid markdown.",
            "If user asks for live info and LIVE_SEARCH is disabled, say it can be enabled.",
        ])
        prompt = f"{user_text}\n\nSNIPPETS:\n{context}" if context else user_text
        reply, ok, _ = await groq_chat(chat_id, sys, prompt, temperature=0.3, max_tokens=700)
        if not ok or not reply:
            reply = "Sorry, try again later."
        sent = await send_message(chat_id, reply)
        log_kv("📤 reply", chat_id=chat_id, sent=sent, len=len(reply))

    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass
        log_kv("💬 process.end", chat_id=chat_id)


def normalize_lid_or_chat(body: dict) -> Tuple[Optional[str], Optional[str], str, bool, str, Optional[str]]:
    root = body.get("payload") or body.get("data") or {}
    data_obj = root.get("_data") or {}
    key = data_obj.get("key") or {}

    text = (
        root.get("body")
        or (root.get("message") or {}).get("text")
        or (root.get("message") or {}).get("conversation")
        or data_obj.get("body")
        or ""
    )

    from_me = bool(root.get("fromMe") or root.get("from_me") or False)

    # WAHA often provides the session jid in body.me.id
    me_obj = body.get("me") or root.get("me") or {}
    me_id = me_obj.get("id") if isinstance(me_obj, dict) else None

    key_remote = key.get("remoteJid")
    remote_jid = root.get("remoteJid") or root.get("chatId") or root.get("chat_id")
    from_field = root.get("from")
    to_field = root.get("to")
    participant = root.get("participant") or data_obj.get("author")
    sender_obj = root.get("sender") or {}
    sender_id_raw = sender_obj.get("id") or participant or from_field or remote_jid

    chat_id_raw = key_remote or remote_jid or from_field or to_field
    chat_id = normalize_jid(chat_id_raw)
    sender_id = normalize_jid(sender_id_raw)

    event_id = (root.get("id") or body.get("id") or "")
    if not event_id:
        event_id = (data_obj.get("id") or "")

    return sender_id, chat_id, text, from_me, str(event_id or ""), normalize_jid(me_id)


@app.on_event("startup")
async def startup():
    _compile_prefix_re()

    await init_db()
    await init_waha()
    await init_llm()

    env_keys = [
        "BOT_PERSONA_NAME", "BOT_COMMAND_PREFIX", "CHROMA_ENABLED",
        "OBSERVE_GROUPS_DEFAULT", "OBSERVE_MIN_TEXT_LEN", "OBSERVE_MAX_EMBED_PER_MIN",
        "OBSERVE_RETENTION_DEFAULT_DAYS", "OBSERVE_REDACTION_DEFAULT",
        "LOG_LEVEL", "LOG_FORMAT", "ACCESS_LOG_LEVEL",
        "WAHA_API_URL", "WAHA_SESSION",
        "LIVE_SEARCH_ENABLED", "LIVE_SEARCH_MODEL",
    ]
    log_startup_env(logger, keys=env_keys)

    logger.info("🚀 startup.complete waha=%s session=%s", getattr(config, "WAHA_API_URL", ""), getattr(config, "WAHA_SESSION", ""))
    logger.info("📚 chroma.status enabled=%s available=%s", chroma_enabled(), CHROMA_AVAILABLE)

    if chroma_enabled():
        try:
            dim = await chroma_warmup()
            logger.info("📚 chroma.warmup dim=%s", dim)
        except Exception as e:
            logger.warning("📚 chroma.warmup.fail err=%s", str(e)[:200])


@app.on_event("shutdown")
async def shutdown():
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()
    await close_db()
    logger.info("🧹 shutdown.complete")


@app.post("/webhook")
async def webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("X-WAHA-HMAC") or request.headers.get("X-Webhook-Signature") or request.headers.get("X-Signature")
    if not verify_signature(raw, sig):
        logger.warning("🔒 webhook.sig_fail")
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        logger.warning("🔒 webhook.json_fail")
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    evt = (body.get("event") or "").lower()
    if evt == "message":
        return JSONResponse({"status": "ok", "message": "ignored"})

    sender_id, chat_id, text, from_me, event_id, me_id = normalize_lid_or_chat(body)

    # Apply allowlist early and avoid logging message text for disallowed group chats
    if looks_group(chat_id) and not group_allowed(chat_id):
        logger.info("↪️ webhook.ignore reason=group_not_allowed chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "group not allowed"})

    logger.info("📩 webhook.recv evt=%s chat_id=%s sender_id=%s text=%s", evt, chat_id, sender_id, (text or "")[:120])

    if not chat_id:
        return JSONResponse({"status": "ok", "message": "no chat id"})

    # --- Echo suppression (FIXED) ---
    # Some WAHA setups incorrectly set fromMe=true for inbound events.
    # Therefore, we only treat fromMe as echo if it is consistent with our own session id
    # OR matches outbound caches.
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)

    is_echo = False
    if event_id and event_id in OUTBOUND_CACHE_IDS:
        is_echo = True
    if chat_id and text:
        h = hashlib.sha1(f"{chat_id}\n{canonical_text(text)}".encode("utf-8")).hexdigest()
        if h in OUTBOUND_CACHE_TXT:
            is_echo = True

    # only trust from_me if we can also link it to our session id
    if from_me and me_id and sender_id and sender_id == me_id:
        is_echo = True

    if is_echo:
        logger.info("♻️ webhook.echo_ignored chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    # observe commands in groups
    if looks_group(chat_id) and (text or "").strip().lower().startswith("/observe"):
        msg = await handle_observe_command(chat_id, sender_id or chat_id, text)
        if msg:
            await send_message(chat_id, msg)
        return JSONResponse({"status": "ok", "message": "observe handled"})

    # ambient observe: in groups, when no prefix anywhere
    if looks_group(chat_id) and (not has_prefix(text)) and chroma_enabled():
        await observe_ingest(chat_id, sender_id or chat_id, text or "", chroma_add_text=chroma_add_text, db_exec=execute)
        logger.info("👁️ webhook.observed chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "observed"})

    # strict prefix gating for groups/broadcasts: prefix may appear anywhere
    if (looks_group(chat_id) or looks_broadcast(chat_id)) and not has_prefix(text):
        logger.info("↪️ webhook.ignore reason=no_group_prefix chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "no group prefix"})

    # DM gating
    allow_nlp = _as_bool(getattr(config, "ALLOW_NLP_WITHOUT_PREFIX", True), True)
    if (not is_groupish(chat_id)) and (not allow_nlp) and not has_prefix(text):
        logger.info("↪️ webhook.ignore reason=no_dm_prefix chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "no dm prefix"})

    # save inbound
    await save_message(chat_id, sender_id or chat_id, "user", strip_prefix(text or ""), event_id=event_id)

    # debounce
    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < getattr(config, "MESSAGE_DEBOUNCE_MS", 800):
        logger.info("↪️ webhook.debounced chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "debounced"})

    # per-chat queue
    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=getattr(config, "LLM_MAX_QUEUE_PER_CHAT", 3))
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            logger.info("🧵 worker.spawned chat_id=%s", chat_id)
            try:
                while True:
                    item = await q.get()
                    try:
                        try:
                            await process_message(chat_id, item["text"], item["sender_id"])
                        except Exception:
                            logger.exception("🟥 worker.process_error chat_id=%s", chat_id)
                    finally:
                        q.task_done()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("🟥 worker.crashed chat_id=%s", chat_id)

        CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

    try:
        await asyncio.wait_for(q.put({"text": text or "", "sender_id": sender_id or chat_id}), timeout=getattr(config, "LLM_QUEUE_WAIT_SEC", 20))
    except asyncio.TimeoutError:
        await send_message(chat_id, "I'm busy; try again in a few seconds.")
        logger.warning("⏳ queue.timeout chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    logger.info("✅ webhook.enqueued chat_id=%s", chat_id)
    return JSONResponse({"status": "ok", "message": "enqueued"})


@app.get("/healthz")
async def health():
    return {"status": "ok"}
