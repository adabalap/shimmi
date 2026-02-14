from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_setup import setup_logging, log_startup_env
from app.config import settings
from app.utils import (
    verify_signature,
    canonical_text,
    normalize_jid,
    looks_group,
    looks_broadcast,
    is_groupish,
    group_allowed,
    has_prefix,
    strip_prefix,
    compile_prefix_re,
)
import app.database as db
from app.waha_provider import (
    init_waha,
    close_waha,
    send_text,
    send_buttons,
    send_list,
    typing_keepalive,
    OUTBOUND_CACHE_IDS,
    OUTBOUND_CACHE_TXT,
    OUTBOUND_TTL_SEC,
)
from app.agent_engine import init_llm, close_llm, run_agent

setup_logging()
logger = logging.getLogger("app")
UTC = timezone.utc

app = FastAPI()

CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}


def trace(step: str, **kw):
    parts = []
    for k, v in kw.items():
        if isinstance(v, str) and len(v) > 260:
            v = v[:260] + "…"
        parts.append(f"{k}={v}")
    logger.info("🧭 %s | %s", step, " ".join(parts))


def normalize_event(body: dict) -> Tuple[Optional[str], Optional[str], str, bool, str]:
    root = body.get("payload") or body.get("data") or body
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

    key_remote = key.get("remoteJid")
    remote_jid = root.get("remoteJid") or root.get("chatId") or root.get("chat_id")

    from_field = root.get("from")
    to_field = root.get("to")
    participant = root.get("participant") or data_obj.get("author")
    sender_obj = root.get("sender") or {}

    sender_raw = sender_obj.get("id") or participant or from_field or remote_jid
    chat_raw = key_remote or remote_jid or from_field or to_field

    sender_id = normalize_jid(sender_raw)
    chat_id = normalize_jid(chat_raw)
    event_id = str(root.get("id") or body.get("id") or data_obj.get("id") or "")

    return sender_id, chat_id, text, from_me, event_id


async def process_message(chat_id: str, sender_id: str, text: str, event_id: str):
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))

    try:
        user_text = strip_prefix((text or "").strip())
        trace("BRAIN start", chat_id=chat_id, sender_id=sender_id, user_text=user_text)

        facts = await db.sqlite_store.get_all_facts(sender_id) if db.sqlite_store else {}
        trace("BRAIN retrieved facts", sender_id=sender_id, fact_keys=str(list(facts.keys())))

        context_items = []
        if db.chroma_store:
            rel = await db.chroma_store.search(chat_id=chat_id, query=user_text, k=settings.chroma_top_k)
            rec = await db.chroma_store.recent_window(chat_id=chat_id, k=settings.chroma_recent_k)
            merged = {c.id: c for c in (rel + rec)}
            context_items = [
                {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance}
                for c in list(merged.values())[:10]
            ]
        trace("BRAIN searched context", chat_id=chat_id, snippets=str(len(context_items)), chroma=str(bool(db.chroma_store)))

        trace("BRAIN calling LLM", context_snippets=str(len(context_items)))
        result = await run_agent(chat_id=chat_id, user_text=user_text, facts=facts, context=context_items)
        trace("BRAIN LLM returned", reply_type=result.reply.type, has_memory_update=str(bool(result.memory_update)))

        reply = result.reply
        trace("ACTION sending", chat_id=chat_id, reply_type=reply.type, preview=reply.text[:140])

        send_res = {}
        try:
            if reply.type == "buttons" and getattr(reply, "buttons", None):
                send_res = await send_buttons(chat_id, reply.text, reply.buttons)  # type: ignore
                if not send_res:
                    send_res = await send_text(chat_id, reply.text)
            elif reply.type == "list" and getattr(reply, "list", None):
                send_res = await send_list(chat_id, reply.text, reply.list)  # type: ignore
                if not send_res:
                    send_res = await send_text(chat_id, reply.text)
            else:
                send_res = await send_text(chat_id, reply.text)
        except Exception as e:
            trace("ACTION send fallback", err=str(e)[:160])
            send_res = await send_text(chat_id, reply.text)

        allow_store = (not looks_group(chat_id)) or group_allowed(chat_id)
        ts_out = datetime.now(UTC).isoformat()
        out_id = str(send_res.get("id") or (send_res.get("message") or {}).get("id") or f"out-{event_id}" or f"out-{int(time.time()*1000)}")
        trace("AMBIENT store outbound", allowed=str(allow_store), out_id=out_id)

        if allow_store and db.sqlite_store:
            await db.sqlite_store.log_message(chat_id=chat_id, whatsapp_id=sender_id, direction="out", text=reply.text, ts=ts_out, event_id=out_id)
        if allow_store and db.chroma_store:
            await db.chroma_store.add_message(chat_id=chat_id, whatsapp_id=sender_id, direction="out", text=reply.text, ts=ts_out, message_id=out_id)

        if result.memory_update and db.sqlite_store:
            status = await db.sqlite_store.upsert_fact(sender_id, result.memory_update.get("key", ""), result.memory_update.get("value", ""))
            trace("MEMORY upsert", status=status, sender_id=sender_id, key=result.memory_update.get("key", ""))

        trace("BRAIN end", chat_id=chat_id)

    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    compile_prefix_re()
    db.init_stores()
    await init_waha()
    await init_llm()

    log_startup_env(
        logger,
        keys=[
            "DATA_DIR",
            "WAHA_API_URL",
            "WAHA_SESSION",
            "CHROMA_ENABLED",
            "ALLOWED_GROUP_JIDS",
            "ALLOW_FROM_ME_MESSAGES",
            "ALLOW_NLP_WITHOUT_PREFIX",
            "LIVE_SEARCH_ENABLED",
        ],
    )
    trace("STARTUP complete", chroma_enabled=str(settings.chroma_enabled), allow_from_me=str(settings.allow_from_me_messages))


@app.on_event("shutdown")
async def shutdown():
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()
    trace("SHUTDOWN complete")


@app.post("/webhook")
async def webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("X-WAHA-HMAC") or request.headers.get("X-Webhook-Signature") or request.headers.get("X-Signature")

    if not verify_signature(raw, sig):
        trace("WEBHOOK rejected", reason="bad_signature")
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        trace("WEBHOOK rejected", reason="bad_json")
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    evt = str(body.get("event") or "").lower()
    if evt == "message":
        trace("WEBHOOK ignored", reason="duplicate_event_message")
        return JSONResponse({"status": "ok", "message": "ignored"})

    sender_id, chat_id, text, from_me, event_id = normalize_event(body)

    trace(
        "WEBHOOK normalized",
        evt=evt,
        from_me=str(from_me),
        chat_id=str(chat_id),
        sender_id=str(sender_id),
        event_id=str(event_id),
        text=(text or "")[:140],
    )

    if not chat_id or not sender_id:
        trace("WEBHOOK ignored", reason="missing_ids")
        return JSONResponse({"status": "ok", "message": "missing ids"})

    # ✅ privacy boundary
    if looks_group(chat_id) and not group_allowed(chat_id):
        trace("WEBHOOK ignored", reason="group_not_allowed", chat_id=chat_id)
        return JSONResponse({"status": "ok", "message": "group not allowed"})

    # --- strong echo protection ---
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)

    if event_id and event_id in OUTBOUND_CACHE_IDS:
        trace("WEBHOOK ignored", reason="echo_cache_event_id", chat_id=chat_id, event_id=event_id)
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    if chat_id and text:
        h = hashlib.sha1(f"{chat_id}\n{canonical_text(text)}".encode("utf-8")).hexdigest()
        if h in OUTBOUND_CACHE_TXT:
            trace("WEBHOOK ignored", reason="echo_cache_text_hash", chat_id=chat_id)
            return JSONResponse({"status": "ok", "message": "echo ignored"})

    # fromMe handling (self-test mode)
    if from_me and not settings.allow_from_me_messages:
        trace("WEBHOOK ignored", reason="from_me_true and ALLOW_FROM_ME_MESSAGES=0", chat_id=chat_id)
        return JSONResponse({"status": "ok", "message": "fromMe ignored"})
    if from_me and settings.allow_from_me_messages:
        trace("WEBHOOK allowed", reason="from_me_true self-test mode (not echo)", chat_id=chat_id)

    # Ambient store inbound (DMs + allowed groups)
    cleaned_in = strip_prefix((text or "").strip())
    ts_in = datetime.now(UTC).isoformat()
    allow_store = (not looks_group(chat_id)) or group_allowed(chat_id)
    trace("AMBIENT store inbound", allowed=str(allow_store), chat_id=chat_id, length=str(len(cleaned_in)))

    if allow_store and db.sqlite_store and cleaned_in:
        await db.sqlite_store.log_message(chat_id=chat_id, whatsapp_id=sender_id, direction="in", text=cleaned_in, ts=ts_in, event_id=event_id or None)
    if allow_store and db.chroma_store and cleaned_in:
        await db.chroma_store.add_message(chat_id=chat_id, whatsapp_id=sender_id, direction="in", text=cleaned_in, ts=ts_in, message_id=event_id or f"in-{int(time.time()*1000)}")

    # Group prefix gating
    if (looks_group(chat_id) or looks_broadcast(chat_id)) and not has_prefix(text):
        trace("WEBHOOK gated", gate="group_prefix", chat_id=chat_id, text=(text or "")[:120])
        return JSONResponse({"status": "ok", "message": "no group prefix"})

    # DM gating
    if (not is_groupish(chat_id)) and (not settings.allow_nlp_without_prefix) and not has_prefix(text):
        trace("WEBHOOK gated", gate="dm_prefix", chat_id=chat_id, text=(text or "")[:120])
        return JSONResponse({"status": "ok", "message": "no dm prefix"})

    # debounce
    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < settings.message_debounce_ms:
        trace("WEBHOOK debounced", chat_id=chat_id, debounce_ms=str(settings.message_debounce_ms))
        return JSONResponse({"status": "ok", "message": "debounced"})

    # per-chat queue
    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=settings.llm_max_queue_per_chat)
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            trace("WORKER spawned", chat_id=chat_id)
            try:
                while True:
                    item = await q.get()
                    try:
                        await process_message(chat_id=chat_id, sender_id=item["sender_id"], text=item["text"], event_id=item["event_id"])
                    except Exception:
                        logger.exception("worker.process_error chat_id=%s", chat_id)
                    finally:
                        q.task_done()
            except asyncio.CancelledError:
                pass

        CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

    try:
        await asyncio.wait_for(q.put({"text": text or "", "sender_id": sender_id, "event_id": event_id}), timeout=settings.llm_queue_wait_sec)
    except asyncio.TimeoutError:
        await send_text(chat_id, "I'm busy; try again in a few seconds.")
        trace("QUEUE timeout", chat_id=chat_id)
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    trace("QUEUE enqueued", chat_id=chat_id, queue_size=str(q.qsize()))
    return JSONResponse({"status": "ok", "message": "enqueued"})


@app.get("/healthz")
async def health():
    return {"status": "ok"}
