from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_setup import setup_logging
from .config import settings
from .utils import (
    verify_signature,
    canonical_text,
    has_prefix,
    strip_invocation,
    compile_prefix_re,
    chat_is_allowed,
    canonical_user_key,
)
import app.database as database
from .waha_provider import (
    init_waha,
    close_waha,
    send_text,
    typing_keepalive,
    OUTBOUND_CACHE_IDS,
    OUTBOUND_CACHE_TXT,
    OUTBOUND_TTL_SEC,
    outbound_hash,
)
from .agent_engine import init_llm, close_llm, run_agent

setup_logging()
logger = logging.getLogger("app")
UTC = timezone.utc

app = FastAPI()

CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}   # chat_id -> perf_counter of last message

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log_allowed(chat_id: Optional[str], msg: str, **kw: Any) -> None:
    if not chat_is_allowed(chat_id):
        return

    def _s(v: Any) -> Any:
        if isinstance(v, str) and len(v) > 240:
            return v[:240] + "…"
        return v

    logger.info("%s %s", msg, " ".join(f"{k}={_s(v)}" for k, v in kw.items()))


# ---------------------------------------------------------------------------
# Webhook event normalisation
# ---------------------------------------------------------------------------

def normalize_event(
    body: dict,
) -> Tuple[Optional[str], Optional[str], str, bool, str]:
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
    event_id = str(
        root.get("id") or body.get("id") or data_obj.get("id") or ""
    )
    return sender_id, chat_id, text, from_me, event_id


# ---------------------------------------------------------------------------
# Ambient storage helper
# FIX #1: only called for messages we actually intend to process, so bot
# echoes and non-prefixed messages are never recorded as inbound.
# ---------------------------------------------------------------------------

async def _ambient_store(
    *, chat_id: str, sender_key: str, text: str, event_id: str
) -> None:
    if not chat_is_allowed(chat_id):
        return

    cleaned = strip_invocation((text or "").strip())
    if not cleaned:
        return

    ts_in = datetime.now(UTC).isoformat()
    if database.sqlite_store:
        await database.sqlite_store.log_message(
            chat_id=chat_id,
            whatsapp_id=sender_key,
            direction="in",
            text=cleaned,
            ts=ts_in,
            event_id=event_id or None,
        )
    if database.chroma_store:
        await database.chroma_store.add_message(
            chat_id=chat_id,
            whatsapp_id=sender_key,
            direction="in",
            text=cleaned,
            ts=ts_in,
            message_id=event_id or ("in-" + str(int(time.time() * 1000))),
        )


# ---------------------------------------------------------------------------
# Outbound echo cache helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core message processor
# FIX #6: outbound logging and memory updates only happen AFTER a successful send.
# FIX #9: conversation history injected from SQLite.
# ---------------------------------------------------------------------------

async def process_message(
    chat_id: str,
    sender_id: str,
    text: str,
    event_id: str,
    from_me: bool,
) -> None:
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))
    sender_key = canonical_user_key(sender_id) or (sender_id or "")

    try:
        user_text = strip_invocation((text or "").strip())
        log_allowed(
            chat_id,
            "flow.begin",
            chat=str(chat_id),
            sender=str(sender_id),
            fromMe=from_me,
            text=user_text,
        )

        facts = (
            await database.sqlite_store.get_all_facts(sender_key)
            if database.sqlite_store
            else {}
        )
        log_allowed(
            chat_id, "facts.loaded", count=len(facts), keys=",".join(list(facts.keys())[:10])
        )

        # --- Build semantic context (vector search) -------------------------
        context_items: List[Dict[str, Any]] = []
        if database.chroma_store:
            rel = await database.chroma_store.search(
                chat_id=chat_id, query=user_text, k=settings.chroma_top_k
            )
            rec = await database.chroma_store.recent_window(
                chat_id=chat_id, k=settings.chroma_recent_k
            )
            merged = {c.id: c for c in (rel + rec)}
            context_items = [
                {
                    "id": c.id,
                    "text": c.text,
                    "metadata": c.metadata,
                    "distance": c.distance,
                }
                for c in list(merged.values())[:10]
            ]

        # FIX #9: ordered turn-by-turn history from SQLite
        history: List[Dict[str, str]] = []
        if database.sqlite_store and settings.history_turns > 0:
            history = await database.sqlite_store.get_recent_messages(
                chat_id, limit=settings.history_turns
            )

        result = await run_agent(
            chat_id=chat_id,
            user_text=user_text,
            facts=facts,
            context=context_items,
            history=history,
        )

        # FIX #6: only log/update memory when send succeeds ─────────────────
        send_res = await send_text(chat_id, result.reply.text)
        if not send_res:
            log_allowed(chat_id, "flow.send_failed — skipping memory update")
            return

        log_allowed(
            chat_id, "flow.action", sent=True, id=str(send_res.get("id") or "")
        )

        ts_out = datetime.now(UTC).isoformat()
        out_id = str(
            send_res.get("id")
            or (send_res.get("message") or {}).get("id")
            or ("out-" + event_id)
            or ("out-" + str(int(time.time() * 1000)))
        )
        if database.sqlite_store:
            await database.sqlite_store.log_message(
                chat_id=chat_id,
                whatsapp_id=sender_key,
                direction="out",
                text=result.reply.text,
                ts=ts_out,
                event_id=out_id,
            )
        if database.chroma_store:
            await database.chroma_store.add_message(
                chat_id=chat_id,
                whatsapp_id=sender_key,
                direction="out",
                text=result.reply.text,
                ts=ts_out,
                message_id=out_id,
            )

        if database.sqlite_store:
            if result.memory_updates:
                for mu in result.memory_updates:
                    status = await database.sqlite_store.upsert_fact(
                        sender_key, mu.key, mu.value
                    )
                    log_allowed(chat_id, "memory", status=status, key=mu.key)
            else:
                log_allowed(chat_id, "memory", status="none")

        log_allowed(chat_id, "flow.end")

    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FIX #4 / #11: worker factory with idle TTL — workers exit after silence
# ---------------------------------------------------------------------------

def _spawn_worker(chat_id: str) -> asyncio.Task:
    """Create a per-chat worker task that auto-exits after idle TTL."""
    q = CHAT_QUEUES[chat_id]
    idle_ttl = float(settings.worker_idle_ttl_sec)

    async def _worker() -> None:
        log_allowed(chat_id, "worker.spawned")
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=idle_ttl)
                except asyncio.TimeoutError:
                    # No message for idle_ttl seconds — self-destruct cleanly
                    log_allowed(chat_id, "worker.idle_exit")
                    CHAT_QUEUES.pop(chat_id, None)
                    CHAT_WORKERS.pop(chat_id, None)
                    CHAT_LAST_MSG_TS.pop(chat_id, None)
                    return

                try:
                    await process_message(
                        chat_id=chat_id,
                        sender_id=item["sender_id"],
                        text=item["text"],
                        event_id=item["event_id"],
                        from_me=item["from_me"],
                    )
                except Exception:
                    logger.exception("worker.error chat=%s", chat_id)
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            pass

    return asyncio.create_task(_worker())


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    compile_prefix_re()
    database.init_stores()
    await init_waha()
    await init_llm()
    if not settings.allowed_chat_jids:
        logger.warning(
            "startup.warning ALLOWED_GROUP_JIDS is empty — "
            "ALL incoming messages will be rejected. "
            "Set ALLOWED_GROUP_JIDS in .env to enable the bot."
        )
    logger.info(
        "startup.ready allowlist_count=%s", len(settings.allowed_chat_jids or [])
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def webhook(request: Request):
    raw = await request.body()
    sig = (
        request.headers.get("X-WAHA-HMAC")
        or request.headers.get("X-Webhook-Signature")
        or request.headers.get("X-Signature")
    )
    if not verify_signature(raw, sig):
        return JSONResponse(
            {"status": "error", "message": "Invalid signature"}, status_code=401
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON"}, status_code=400
        )

    sender_id, chat_id, text, from_me, event_id = normalize_event(body)

    if not chat_is_allowed(chat_id):
        return JSONResponse({"status": "ok", "message": "chat not allowed"})

    log_allowed(
        chat_id,
        "webhook.recv",
        chat=str(chat_id),
        sender=str(sender_id),
        fromMe=from_me,
        text=(text or "")[:120],
    )

    if not (text or "").strip():
        log_allowed(chat_id, "webhook.ignore", reason="empty_text")
        return JSONResponse({"status": "ok", "message": "empty"})

    _purge_outbound_caches()

    if chat_id and _is_echo(chat_id, canonical_text(text or ""), event_id):
        log_allowed(chat_id, "webhook.ignore", reason="echo")
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    # FIX #1: apply fromMe and prefix guards BEFORE _ambient_store
    if from_me and not settings.allow_fromme:
        log_allowed(chat_id, "webhook.ignore", reason="fromMe_disabled")
        return JSONResponse({"status": "ok", "message": "fromMe ignored"})

    if (not settings.allow_nlp_without_prefix) and not has_prefix(text):
        log_allowed(chat_id, "webhook.ignore", reason="no_prefix")
        return JSONResponse({"status": "ok", "message": "no prefix"})

    # Only store messages we've decided to actually process
    sender_key = canonical_user_key(sender_id) or (sender_id or "")
    await _ambient_store(
        chat_id=chat_id, sender_key=sender_key, text=text or "", event_id=event_id
    )

    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < settings.message_debounce_ms:
        log_allowed(chat_id, "webhook.ignore", reason="debounced")
        return JSONResponse({"status": "ok", "message": "debounced"})

    # FIX #4/#11: create queue + worker only if one doesn't already exist
    if chat_id not in CHAT_QUEUES:
        CHAT_QUEUES[chat_id] = asyncio.Queue(maxsize=settings.llm_max_queue_per_chat)
        CHAT_WORKERS[chat_id] = _spawn_worker(chat_id)

    q = CHAT_QUEUES[chat_id]
    try:
        await asyncio.wait_for(
            q.put(
                {
                    "text": text or "",
                    "sender_id": sender_id,
                    "event_id": event_id,
                    "from_me": from_me,
                }
            ),
            timeout=settings.llm_queue_wait_sec,
        )
    except asyncio.TimeoutError:
        await send_text(chat_id, "I'm a bit overloaded right now — please try again in a few seconds.")
        log_allowed(chat_id, "queue.timeout")
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    log_allowed(chat_id, "webhook.enqueued")
    return JSONResponse({"status": "ok", "message": "enqueued"})


# ---------------------------------------------------------------------------
# FIX #12: observability endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    """Operational dashboard — active workers, queue depths, circuit breakers."""
    from .agent_engine import MODEL_CIRCUIT, STICKY_MODEL
    import time as _time

    now = _time.monotonic()
    workers_info = {}
    for cid, task in CHAT_WORKERS.items():
        q = CHAT_QUEUES.get(cid)
        workers_info[cid] = {
            "queue_depth": q.qsize() if q else 0,
            "task_done": task.done(),
        }

    circuit_info = {
        m: max(0.0, round(open_until - now, 1))
        for m, open_until in MODEL_CIRCUIT.items()
        if open_until > now
    }

    return {
        "status": "ok",
        "active_workers": len(CHAT_WORKERS),
        "workers": workers_info,
        "open_circuits": circuit_info,
        "sqlite_enabled": database.sqlite_store is not None,
        "chroma_enabled": database.chroma_store is not None,
        "allowlist_count": len(settings.allowed_chat_jids or []),
    }
