"""Main application with all integrations"""
from __future__ import annotations

import asyncio
import logging
import time
import groq
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_setup import setup_logging
from .config import settings
from .utils import (
    verify_signature, canonical_text, has_prefix, strip_invocation,
    compile_prefix_re, chat_is_allowed, canonical_user_key
, normalize_whatsapp_id)
import app.database as database
from .waha_provider import (
    init_waha, close_waha, send_text, typing_keepalive,
    OUTBOUND_CACHE_IDS, OUTBOUND_CACHE_TXT, OUTBOUND_TTL_SEC, outbound_hash
)
from .multi_provider_llm import init_llm, close_llm, smart_complete
from .agent_engine import run_agent

setup_logging()
logger = logging.getLogger("app")
UTC = timezone.utc

app = FastAPI()

CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}


def log_allowed(chat_id: Optional[str], msg: str, **kw: Any) -> None:
    if not chat_is_allowed(chat_id):
        return
    def _s(v: Any) -> Any:
        if isinstance(v, str) and len(v) > 240:
            return v[:240] + "…"
        return v
    logger.info("%s %s", msg, " ".join(f"{k}={_s(v)}" for k, v in kw.items()))


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

    def _norm(j: Optional[str]) -> Optional[str]:
        if not j:
            return None
        if j.endswith("@s.whatsapp.net"):
            return j.replace("@s.whatsapp.net", "@c.us")
        return j

    sender_id = _norm(sender_raw)
    chat_id = _norm(chat_raw)
    event_id = str(root.get("id") or body.get("id") or data_obj.get("id") or "")
    return sender_id, chat_id, text, from_me, event_id


async def _ambient_store(*, chat_id: str, sender_key: str, text: str, event_id: str) -> None:
    if not chat_is_allowed(chat_id):
        return

    cleaned = strip_invocation((text or "").strip())
    if not cleaned:
        return

    ts_in = datetime.now(UTC).isoformat()
    if database.sqlite_store:
        await database.sqlite_store.log_message(
            chat_id=chat_id, whatsapp_id=sender_key, direction="in",
            text=cleaned, ts=ts_in, event_id=event_id or None
        )
    if database.chroma_store:
        await database.chroma_store.add_message(
            chat_id=chat_id, whatsapp_id=sender_key, direction="in",
            text=cleaned, ts=ts_in,
            message_id=event_id or ("in-" + str(int(time.time()*1000)))
        )


def _purge_outbound_caches() -> None:
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)


def _is_echo(chat_id: str, text: str, event_id: str) -> bool:
    if event_id and event_id in OUTBOUND_CACHE_IDS:
        return True
    if chat_id and text:
        h = outbound_hash(chat_id, text)
        if h in OUTBOUND_CACHE_TXT:
            return True
    return False


async def process_message(chat_id: str, sender_id: str, text: str, event_id: str, from_me: bool) -> None:
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))

    sender_key = normalize_whatsapp_id(sender_id) or (sender_id or "")

    try:
        user_text = strip_invocation((text or "").strip())
        log_allowed(chat_id, "🧠 flow.begin", chat=str(chat_id), sender=str(sender_id),
                   sender_key=str(sender_key), fromMe=from_me, text=user_text)

        # Get facts
        facts = await database.sqlite_store.get_all_facts(sender_key) if database.sqlite_store else {}
        log_allowed(chat_id, "🧠 facts.loaded", count=len(facts), keys=",".join(list(facts.keys())[:10]))

        # Get context with user filtering for anti-hallucination
        context_items: List[Dict[str, Any]] = []
        if database.chroma_store:
            rel = await database.chroma_store.search(
                chat_id=chat_id,
                query=user_text,
                k=settings.chroma_top_k,
                whatsapp_id=sender_key  # ANTI-HALLUCINATION: Only this user's messages
            )
            rec = await database.chroma_store.recent_window(
                chat_id=chat_id,
                k=settings.chroma_recent_k,
                whatsapp_id=sender_key  # ANTI-HALLUCINATION: Only this user's messages
            )
            
            merged = {c.id: c for c in (rel + rec)}
            context_items.extend([
                {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance}
                for c in list(merged.values())[:10]
            ])

        # Run agent
        result = await run_agent(
            chat_id=chat_id,
            user_text=user_text,
            facts=facts,
            context=context_items,
            llm_complete_fn=smart_complete,
            whatsapp_id=sender_key  # Pass for tracking
        )

        send_res = await send_text(chat_id, result.reply.text)
        log_allowed(chat_id, "🧠 flow.action", sent=bool(send_res), id=str(send_res.get("id") or ""))

        # Log outbound
        ts_out = datetime.now(UTC).isoformat()
        out_id = str(
            send_res.get("id")
            or (send_res.get("message") or {}).get("id")
            or ("out-" + event_id)
            or ("out-" + str(int(time.time()*1000)))
        )
        
        if database.sqlite_store:
            await database.sqlite_store.log_message(
                chat_id=chat_id, whatsapp_id=sender_key, direction="out",
                text=result.reply.text, ts=ts_out, event_id=out_id
            )
        if database.chroma_store:
            await database.chroma_store.add_message(
                chat_id=chat_id, whatsapp_id=sender_key, direction="out",
                text=result.reply.text, ts=ts_out, message_id=out_id
            )

        # Save memory updates
        if database.sqlite_store and result.memory_updates:
            for mu in result.memory_updates:
                status = await database.sqlite_store.upsert_fact(sender_key, mu.key, mu.value)
                log_allowed(chat_id, "🧠 memory", status=status, key=mu.key)

        log_allowed(chat_id, "🧠 flow.end")

    except groq.RateLimitError:
        logger.error("rate_limit.error chat=%s", chat_id)
        try:
            await send_text(chat_id, "I'm at capacity. Please try again in a few minutes. 🙏")
        except:
            pass
    except Exception as e:
        logger.exception("process_message.error chat=%s", chat_id)
        try:
            await send_text(chat_id, "I encountered an error. Please try again. 🤖")
        except:
            pass
    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass


@app.on_event("startup")
async def startup() -> None:
    compile_prefix_re()
    database.init_stores()  # This function now exists!
    await init_waha()
    await init_llm()
    logger.info("startup.ready allowlist_count=%s", len(settings.allowed_chat_jids or []))


@app.on_event("shutdown")
async def shutdown() -> None:
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()


@app.post("/webhook")
async def webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("X-WAHA-HMAC") or request.headers.get("X-Webhook-Signature") or request.headers.get("X-Signature")
    if not verify_signature(raw, sig):
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    sender_id, chat_id, text, from_me, event_id = normalize_event(body)

    if not chat_is_allowed(chat_id):
        return JSONResponse({"status": "ok", "message": "chat not allowed"})

    log_allowed(chat_id, "📩 webhook.recv", chat=str(chat_id), sender=str(sender_id),
               fromMe=from_me, text=(text or "")[:120])

    if not (text or "").strip():
        log_allowed(chat_id, "↪️ webhook.ignore", reason="empty_text")
        return JSONResponse({"status": "ok", "message": "empty"})

    _purge_outbound_caches()

    if chat_id and _is_echo(chat_id, canonical_text(text or ""), event_id):
        log_allowed(chat_id, "↪️ webhook.ignore", reason="echo")
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    sender_key = normalize_whatsapp_id(sender_id) or (sender_id or "")
    
    await _ambient_store(chat_id=chat_id, sender_key=sender_key, text=text or "", event_id=event_id)

    if from_me and not settings.allow_fromme:
        log_allowed(chat_id, "↪️ webhook.ignore", reason="fromMe_disabled")
        return JSONResponse({"status": "ok", "message": "fromMe ignored"})

    if (not settings.allow_nlp_without_prefix) and not has_prefix(text):
        log_allowed(chat_id, "↪️ webhook.ignore", reason="no_prefix")
        return JSONResponse({"status": "ok", "message": "no prefix"})

    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < settings.message_debounce_ms:
        log_allowed(chat_id, "↪️ webhook.ignore", reason="debounced")
        return JSONResponse({"status": "ok", "message": "debounced"})

    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=settings.llm_max_queue_per_chat)
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            log_allowed(chat_id, "🧵 worker.spawned")
            try:
                while True:
                    item = await q.get()
                    try:
                        await process_message(
                            chat_id=chat_id,
                            sender_id=item["sender_id"],
                            text=item["text"],
                            event_id=item["event_id"],
                            from_me=item["from_me"]
                        )
                    except Exception:
                        logger.exception("worker.error chat=%s", chat_id)
                    finally:
                        q.task_done()
            except asyncio.CancelledError:
                pass

        CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

    try:
        await asyncio.wait_for(
            q.put({"text": text or "", "sender_id": sender_id, "event_id": event_id, "from_me": from_me}),
            timeout=settings.llm_queue_wait_sec
        )
    except asyncio.TimeoutError:
        await send_text(chat_id, "I'm busy; try again in a few seconds.")
        log_allowed(chat_id, "⏳ queue.timeout")
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    log_allowed(chat_id, "✅ webhook.enqueued")
    return JSONResponse({"status": "ok", "message": "enqueued"})


@app.get("/healthz")
async def health():
    return {"status": "ok", "version": "6.0.1"}
