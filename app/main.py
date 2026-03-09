"""
main.py — Shimmi v2.9.2

Changes vs v2.7.0:
  - Background reminder scheduler started in lifespan
  - _reminder / _cancel_reminder special memory keys parsed and routed to reminders table
  - Startup log shows PORT for visibility
  - User reminders loaded alongside facts and passed to run_agent
  - worker.msg_error no longer re-raises — worker survives any per-message failure
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_setup import setup_logging
from .config import settings
from .utils import (
    verify_signature, canonical_text, has_prefix, strip_invocation,
    compile_prefix_re, chat_is_allowed, canonical_user_key,
)
import app.database as database
from .waha_provider import (
    init_waha, close_waha, send_text, typing_keepalive,
    OUTBOUND_CACHE_IDS, OUTBOUND_CACHE_TXT, OUTBOUND_TTL_SEC, outbound_hash,
)
from .agent_engine import (
    init_llm, close_llm, run_agent, extract_reply_memory, VALID_GROQ_PREFIXES,
    _is_reminder_duplicate,
)
from .trace import Trace
from .signature import append_signature
from .scheduler import run_reminder_loop

setup_logging()
logger = logging.getLogger("app")
UTC    = timezone.utc

CHAT_QUEUES:      Dict[str, asyncio.Queue] = {}
CHAT_WORKERS:     Dict[str, asyncio.Task]  = {}
CHAT_LAST_MSG_TS: Dict[str, float]          = {}
_WORKER_IDLE_TIMEOUT_SEC: float = 3_600.0
BOT_IDENTITY = "shimmi-bot"
_REMINDER_TASK: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Startup integrity check
# ---------------------------------------------------------------------------

def _integrity_check() -> None:
    issues: List[str] = []
    if not settings.groq_api_key:
        issues.append("GROQ_API_KEY is not set — LLM will be disabled")
    if not settings.waha_api_url:
        issues.append("WAHA_API_URL is not set — cannot send messages")
    if not settings.allowed_chat_jids and not settings.allow_all_chats:
        issues.append(
            "ALLOWED_GROUP_JIDS is empty and ALLOW_ALL_CHATS=false — ALL messages ignored"
        )
    if settings.live_search_enabled and not settings.live_search_model:
        issues.append("LIVE_SEARCH_ENABLED=1 but LIVE_SEARCH_MODEL is empty")
    pool = settings.groq_model_pool or []
    for m in pool:
        if not any(m.startswith(p) for p in VALID_GROQ_PREFIXES):
            issues.append(
                f"GROQ_MODEL_POOL contains invalid model {m!r}. "
                "Valid names start with: llama-, compound-beta, mixtral-, gemma-. "
                "Fix GROQ_MODEL_POOL in .env."
            )
    if pool and pool[0] == "llama-3.1-8b-instant":
        issues.append(
            "GROQ_MODEL_POOL starts with llama-3.1-8b-instant (8B model). "
            "Recommend putting llama-3.3-70b-versatile first for faster, higher-quality responses. "
            "Current: GROQ_MODEL_POOL=llama-3.1-8b-instant,llama-3.3-70b-versatile — "
            "Recommended: GROQ_MODEL_POOL=llama-3.3-70b-versatile,llama-3.1-8b-instant"
        )
    for issue in issues:
        logger.warning("⚠️  CONFIG: %s", issue)
    if not issues:
        logger.info("✅ integrity_check — all config OK")


# ---------------------------------------------------------------------------
# Special memory key handler + reminder saver
# ---------------------------------------------------------------------------

async def _save_reminders(
    sender_key:  str,
    chat_id:     str,
    reminders:   list,         # List[ReminderEntry] from AgentResult
    existing:    list,         # List[Reminder] already in DB (pending)
) -> int:
    """
    Save new reminders to DB.  Skips duplicates (BUG-4 fix).
    Returns count of reminders actually saved.
    """
    saved = 0
    for r in reminders:
        text        = (r.text or "").strip()
        trigger_iso = (r.trigger_iso or "").strip()
        if not text or not trigger_iso:
            logger.warning("reminder.skip — blank  text=%r  trigger=%r", text[:50], trigger_iso[:30])
            continue
        # Dedup: skip if matching pending reminder already exists
        if _is_reminder_duplicate(text, trigger_iso, existing):
            logger.info("🔔 reminder.dedup  trigger=%s  text=%r", trigger_iso, text[:60])
            continue
        try:
            rid = await database.sqlite_store.add_reminder(
                whatsapp_id=sender_key,
                chat_id=chat_id,
                text=text,
                trigger_iso=trigger_iso,
            )
            logger.info("🔔 reminder.scheduled  id=%d  chat=%s  trigger=%s  text=%r",
                        rid, chat_id, trigger_iso, text[:60])
            saved += 1
        except Exception as exc:
            logger.error("reminder.save_failed  err=%s  text=%r", exc, text[:60])
    return saved


async def _reply_extract_and_save_bg(
    chat_id:     str,
    sender_key:  str,
    user_text:   str,
    bot_reply:   str,
    known_facts: dict,
) -> None:
    """
    BUG-1 fix: fire-and-forget task that runs AFTER the reply is sent.
    Extracts structured data (lists, names, etc.) that the bot confirmed
    in its reply text, then saves anything new to the facts DB.
    """
    try:
        updates = await extract_reply_memory(
            chat_id=chat_id,
            user_text=user_text,
            bot_reply=bot_reply,
            existing_facts=known_facts,
        )
        if not updates or not database.sqlite_store:
            return
        for mu in updates:
            try:
                status = await database.sqlite_store.upsert_fact(sender_key, mu.key, mu.value)
                if status in ("created", "updated"):
                    logger.info(
                        "🧠 reply_memory.%s  sender=%s  key=%s  value=%r",
                        status, sender_key, mu.key, mu.value,
                    )
            except Exception as exc:
                logger.error("reply_memory.save_failed  key=%s  err=%s", mu.key, exc)
    except Exception as exc:
        logger.error("reply_memory.bg_error  err=%s", exc)


# ---------------------------------------------------------------------------
# Webhook event normalisation
# ---------------------------------------------------------------------------

def normalize_event(body: dict) -> Tuple[Optional[str], Optional[str], str, bool, str]:
    root       = body.get("payload") or body.get("data") or body
    data_obj   = root.get("_data") or {}
    key        = data_obj.get("key") or {}
    text       = (
        root.get("body")
        or (root.get("message") or {}).get("text")
        or (root.get("message") or {}).get("conversation")
        or data_obj.get("body")
        or ""
    )
    from_me    = bool(root.get("fromMe") or root.get("from_me") or False)
    key_remote = key.get("remoteJid")
    remote_jid = root.get("remoteJid") or root.get("chatId") or root.get("chat_id")
    from_field = root.get("from")
    to_field   = root.get("to")
    participant = root.get("participant") or data_obj.get("author")
    sender_obj  = root.get("sender") or {}
    sender_raw  = sender_obj.get("id") or participant or from_field or remote_jid
    chat_raw    = key_remote or remote_jid or from_field or to_field

    def _norm(j):
        return j.replace("@s.whatsapp.net", "@c.us") if j else None

    return (
        _norm(sender_raw),
        _norm(chat_raw),
        text,
        from_me,
        str(root.get("id") or body.get("id") or data_obj.get("id") or ""),
    )


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------

async def _store_out_bg(*, chat_id, text, ts, out_id) -> None:
    try:
        if database.sqlite_store:
            await database.sqlite_store.log_message(
                chat_id=chat_id, whatsapp_id=BOT_IDENTITY,
                direction="out", text=text, ts=ts, event_id=out_id,
            )
        if database.chroma_store:
            await database.chroma_store.add_message(
                chat_id=chat_id, whatsapp_id=BOT_IDENTITY,
                direction="out", text=text, ts=ts, message_id=out_id,
            )
    except Exception:
        logger.exception("store_out_bg.error  chat=%s", chat_id)


async def _ambient_store(*, chat_id, sender_key, text, event_id) -> None:
    if not chat_is_allowed(chat_id):
        return
    cleaned = strip_invocation((text or "").strip())
    if not cleaned:
        return
    ts_in = datetime.now(UTC).isoformat()
    if database.sqlite_store:
        await database.sqlite_store.log_message(
            chat_id=chat_id, whatsapp_id=sender_key,
            direction="in", text=cleaned, ts=ts_in, event_id=event_id or None,
        )
    if database.chroma_store:
        await database.chroma_store.add_message(
            chat_id=chat_id, whatsapp_id=sender_key,
            direction="in", text=cleaned, ts=ts_in,
            message_id=event_id or ("in-" + str(int(time.time() * 1000))),
        )


def _purge_outbound_caches() -> None:
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)


def _is_echo(chat_id, text, event_id) -> bool:
    if event_id and event_id in OUTBOUND_CACHE_IDS:
        return True
    if chat_id and text and outbound_hash(chat_id, text) in OUTBOUND_CACHE_TXT:
        return True
    return False


# ---------------------------------------------------------------------------
# Core message processor
# ---------------------------------------------------------------------------

async def process_message(
    chat_id: str, sender_id: str, text: str, event_id: str, from_me: bool,
) -> None:
    async with Trace(event_id=event_id, chat_id=chat_id, sender_id=sender_id) as trace:

        stop_evt       = asyncio.Event()
        keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))

        sender_key = canonical_user_key(sender_id) or sender_id or ""
        user_text  = strip_invocation((text or "").strip())

        trace.tag(
            from_me=from_me, sender_key=sender_key,
            text_len=len(user_text), text_preview=user_text[:100],
        )
        logger.info(
            "📨 msg.in  event=%s  chat=%s  sender=%s  text=%r",
            event_id, chat_id, sender_key, user_text[:120],
        )

        try:
            # ── 1. load facts ───────────────────────────────────────────────
            with trace.step("facts_load"):
                facts = (
                    await database.sqlite_store.get_all_facts(sender_key)
                    if database.sqlite_store else {}
                )
                trace.tag(
                    facts_count=len(facts),
                    facts=", ".join(f"{k}={v!r}" for k, v in list(facts.items())[:15]) or "∅",
                )
                if facts:
                    logger.info(
                        "📋 facts.loaded  sender=%s  count=%d  %s",
                        sender_key, len(facts),
                        "  ".join(f"{k}={v!r}" for k, v in facts.items()),
                    )
                else:
                    logger.info("📋 facts.new_user  sender=%s", sender_key)

            # ── 2. load reminders ───────────────────────────────────────────
            with trace.step("reminders_load"):
                reminders = (
                    await database.sqlite_store.get_user_reminders(sender_key)
                    if database.sqlite_store else []
                )
                pending = [r for r in reminders if r.status == "pending"]
                trace.tag(reminders_pending=len(pending), reminders_total=len(reminders))
                if pending:
                    logger.info(
                        "🔔 reminders.loaded  sender=%s  pending=%d",
                        sender_key, len(pending),
                    )

            # ── 3. build context ────────────────────────────────────────────
            context_items: List[Dict[str, Any]] = []
            with trace.step("context_build"):
                if database.chroma_store:
                    rel = await database.chroma_store.search(
                        chat_id=chat_id, query=user_text, k=settings.chroma_top_k,
                    )
                    rec = await database.chroma_store.recent_window(
                        chat_id=chat_id, k=settings.chroma_recent_k,
                    )
                    merged_ctx = {c.id: c for c in (rel + rec)}
                    context_items = [
                        {"id": c.id, "text": c.text, "metadata": c.metadata, "distance": c.distance}
                        for c in list(merged_ctx.values())[:20]
                    ]
                trace.tag(context_total=len(context_items))
                logger.info(
                    "📚 context.built  total=%d", len(context_items),
                )

            # ── 4. run agentic loop ─────────────────────────────────────────
            with trace.step("agent_run"):
                result = await run_agent(
                    chat_id=chat_id,
                    user_text=user_text,
                    facts=facts,
                    context=context_items,
                    reminders=reminders,
                    trace=trace,
                )
                trace.tag(
                    agent_iterations=result.iterations,
                    reply_len=len(result.reply.text),
                    reply_preview=result.reply.text[:100],
                    memory_updates=len(result.memory_updates),
                    reminders_scheduled=len(result.reminders),
                )
                logger.info(
                    "🤖 agent.done  iter=%d  reply_len=%d  memory_updates=%d  reminders=%d  preview=%r",
                    result.iterations, len(result.reply.text),
                    len(result.memory_updates), len(result.reminders), result.reply.text[:100],
                )

            # ── 5. signature ────────────────────────────────────────────────
            with trace.step("signature"):
                reply_with_sig = append_signature(result.reply.text, chat_id)
                trace.tag(final_len=len(reply_with_sig))

            # ── 6. persist memory + reminders (BEFORE send so WAHA errors can't lose facts)
            with trace.step("memory_save"):
                saved = created = updated = 0
                save_errors: List[str] = []

                if database.sqlite_store:
                    # 6a. Save regular fact updates from orchestrator
                    for mu in result.memory_updates:
                        try:
                            status = await database.sqlite_store.upsert_fact(
                                sender_key, mu.key, mu.value,
                            )
                            if status == "created":
                                created += 1
                                logger.info(
                                    "🧠 memory.new      sender=%s  key=%s  value=%r",
                                    sender_key, mu.key, mu.value,
                                )
                            elif status == "updated":
                                updated += 1
                                logger.info(
                                    "🧠 memory.updated  sender=%s  key=%s  value=%r",
                                    sender_key, mu.key, mu.value,
                                )
                            saved += 1
                        except Exception as exc:
                            err_msg = f"{type(exc).__name__}: {exc}"
                            save_errors.append(f"{mu.key}: {err_msg}")
                            logger.error(
                                "🧠 memory.save_failed  sender=%s  key=%s  err=%s",
                                sender_key, mu.key, exc,
                            )

                    # 6b. Save reminders from AgentResult (dedup via pending list)
                    if result.reminders:
                        rems_saved = await _save_reminders(
                            sender_key=sender_key,
                            chat_id=chat_id,
                            reminders=result.reminders,
                            existing=pending,
                        )
                        trace.tag(reminders_saved=rems_saved)

                trace.tag(
                    facts_saved=saved, facts_created=created, facts_updated=updated,
                    facts_attempted=len(result.memory_updates),
                    **({"save_errors": "; ".join(save_errors)} if save_errors else {}),
                )
                if result.memory_updates:
                    logger.info(
                        "🧠 memory.summary  sender=%s  attempted=%d  "
                        "created=%d  updated=%d  errors=%d",
                        sender_key, len(result.memory_updates), created, updated, len(save_errors),
                    )
                else:
                    logger.info("🧠 memory.none  sender=%s", sender_key)

            # ── 7. send ─────────────────────────────────────────────────────
            msg_id = ""
            with trace.step("send"):
                send_res = await send_text(chat_id, reply_with_sig)
                from .waha_provider import _extract_msg_id
                msg_id   = _extract_msg_id(send_res)
                trace.tag(sent=bool(send_res), msg_id=msg_id[:40] if msg_id else "")
                logger.info(
                    "📤 msg.sent  chat=%s  msg_id=%s  len=%d",
                    chat_id,
                    msg_id if msg_id else "(see WAHA DEBUG log for raw response)",
                    len(reply_with_sig),
                )

            # ── 8. store outgoing + post-reply memory (fire-and-forget) ────
            ts_out = datetime.now(UTC).isoformat()
            out_id = msg_id or f"out-{event_id}" or f"out-{int(time.time()*1000)}"
            asyncio.create_task(
                _store_out_bg(chat_id=chat_id, text=result.reply.text, ts=ts_out, out_id=out_id)
            )
            # Skip reply_extract when bot just asked a clarifying question —
            # there's no personal info in a question reply.
            # NOTE: AgentResult schemas may not include reply.action (pydantic model changes).
            # Using getattr keeps the worker resilient across schema versions.
            reply_action = getattr(result.reply, 'action', None)
            if reply_action is None:
                # Backward/forward compatibility: some schemas use intent/type instead of action.
                reply_action = getattr(result.reply, 'intent', None) or getattr(result.reply, 'type', None)
            reply_text = (getattr(result.reply, 'text', None) or '')
            _reply_is_question = (
                (isinstance(reply_action, str) and reply_action == 'ask_user')
                or (len(reply_text) < 120 and reply_text.rstrip().endswith('?'))
            )
            if not _reply_is_question:
                asyncio.create_task(
                    _reply_extract_and_save_bg(
                        chat_id=chat_id,
                        sender_key=sender_key,
                        user_text=user_text,
                        bot_reply=result.reply.text,
                        known_facts=facts,
                    )
                )

        finally:
            stop_evt.set()
            try:
                await keepalive_task
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-chat worker
# ---------------------------------------------------------------------------

async def _chat_worker(chat_id: str, q: asyncio.Queue) -> None:
    logger.info("🧵 worker.start  chat=%s", chat_id)
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=_WORKER_IDLE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.info("🧵 worker.idle_exit  chat=%s  (1h idle)", chat_id)
                break
            try:
                await process_message(
                    chat_id=chat_id,
                    sender_id=item["sender_id"],
                    text=item["text"],
                    event_id=item["event_id"],
                    from_me=item["from_me"],
                )
            except Exception:
                logger.exception("worker.msg_error  chat=%s", chat_id)
                # Swallow — worker must stay alive for next message
            finally:
                q.task_done()
    except asyncio.CancelledError:
        pass
    finally:
        CHAT_QUEUES.pop(chat_id, None)
        CHAT_WORKERS.pop(chat_id, None)
        logger.info("🧵 worker.exit  chat=%s", chat_id)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _REMINDER_TASK
    compile_prefix_re()
    _integrity_check()
    database.init_stores()
    await init_waha()
    await init_llm()

    # Start background reminder scheduler
    _REMINDER_TASK = asyncio.create_task(
        run_reminder_loop(check_interval_sec=60),
        name="reminder_scheduler",
    )
    logger.info("🕐 scheduler.task_created")

    port = int(os.getenv("PORT", "6000"))
    logger.info(
        "🚀 startup.ready  http://0.0.0.0:%d  "
        "allowlist=%d  allow_all=%s  live_search=%s  chroma=%s  model_pool=%s",
        port,
        len(settings.allowed_chat_jids or []),
        settings.allow_all_chats,
        settings.live_search_enabled,
        settings.chroma_enabled,
        settings.groq_model_pool,
    )

    yield

    logger.info("🛑 shutdown.begin")
    if _REMINDER_TASK:
        _REMINDER_TASK.cancel()
        try:
            await _REMINDER_TASK
        except asyncio.CancelledError:
            pass
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()
    logger.info("🛑 shutdown.complete")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Webhook
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
        logger.warning("webhook.auth_fail  ip=%s", request.client.host if request.client else "?")
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    sender_id, chat_id, text, from_me, event_id = normalize_event(body)

    if not chat_is_allowed(chat_id):
        logger.debug("webhook.skip  reason=allowlist  chat=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "chat not allowed"})

    if not (text or "").strip():
        logger.debug("webhook.skip  reason=empty  event=%s", event_id)
        return JSONResponse({"status": "ok", "message": "empty"})

    _purge_outbound_caches()

    if chat_id and _is_echo(chat_id, canonical_text(text or ""), event_id):
        logger.debug("webhook.skip  reason=echo  event=%s", event_id)
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    sender_key = canonical_user_key(sender_id) or sender_id or ""
    await _ambient_store(chat_id=chat_id, sender_key=sender_key, text=text or "", event_id=event_id)

    if from_me and not settings.allow_fromme:
        logger.debug("webhook.skip  reason=fromMe  event=%s", event_id)
        return JSONResponse({"status": "ok", "message": "fromMe ignored"})

    if (not settings.allow_nlp_without_prefix) and not has_prefix(text):
        logger.debug("webhook.skip  reason=no_prefix  event=%s", event_id)
        return JSONResponse({"status": "ok", "message": "no prefix"})

    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < settings.message_debounce_ms:
        logger.debug("webhook.skip  reason=debounced  event=%s", event_id)
        return JSONResponse({"status": "ok", "message": "debounced"})

    q      = CHAT_QUEUES.get(chat_id)
    worker = CHAT_WORKERS.get(chat_id)
    if not q or (worker is not None and worker.done()):
        q = asyncio.Queue(maxsize=settings.llm_max_queue_per_chat)
        CHAT_QUEUES[chat_id]  = q
        CHAT_WORKERS[chat_id] = asyncio.create_task(_chat_worker(chat_id, q))

    try:
        await asyncio.wait_for(
            q.put({"text": text or "", "sender_id": sender_id, "event_id": event_id, "from_me": from_me}),
            timeout=settings.llm_queue_wait_sec,
        )
    except asyncio.TimeoutError:
        await send_text(chat_id, "I'm busy right now — try again in a moment.")
        logger.warning("⏳ queue.timeout  chat=%s  event=%s", chat_id, event_id)
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    logger.info(
        "✅ webhook.enqueued  event=%s  chat=%s  depth=%d",
        event_id, chat_id, q.qsize(),
    )
    return JSONResponse({"status": "ok", "message": "enqueued"})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def health():
    return {
        "status":           "ok",
        "workers":          len(CHAT_WORKERS),
        "queues":           {cid: q.qsize() for cid, q in CHAT_QUEUES.items()},
        "live_search":      settings.live_search_enabled,
        "chroma":           settings.chroma_enabled,
        "model_pool":       settings.groq_model_pool,
        "reminder_task":    _REMINDER_TASK is not None and not (_REMINDER_TASK.done() if _REMINDER_TASK else True),
    }
