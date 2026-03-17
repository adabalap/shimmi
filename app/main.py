"""
main.py — Shimmi v3.2.0

Changes vs v3.1.0:
  - Version bump to match agent_engine.py and mcp_server.py
  - Health endpoint version updated
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
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
from .database import DeleteOutcome, normalize_key
from .waha_provider import (
    init_waha, close_waha, send_text, typing_keepalive,
    OUTBOUND_CACHE_IDS, OUTBOUND_CACHE_TXT, OUTBOUND_TTL_SEC, outbound_hash,
)
from .agent_engine import (
    init_llm, close_llm, run_agent, extract_reply_memory, consolidate_user_facts,
    VALID_GROQ_PREFIXES, VALID_GEMINI_PREFIXES,
    _is_reminder_duplicate, MODEL_CIRCUIT, PROVIDER_CIRCUIT, _TOKEN_BUDGET,
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

# FIX-P0-3: inbound event dedup — prevents double-processing when WAHA retries
# the same webhook event (common in WhatsApp delivery retries).
_INBOUND_SEEN:     Dict[str, float] = {}   # event_id → monotonic timestamp
_INBOUND_SEEN_TTL: float            = 30.0  # seconds to remember event_id


def _inbound_seen_check(event_id: str) -> bool:
    """Returns True (= duplicate, skip) if event_id was seen within TTL."""
    now = time.monotonic()
    # Prune old entries to prevent unbounded growth
    stale = [k for k, ts in _INBOUND_SEEN.items() if now - ts > 60.0]
    for k in stale:
        _INBOUND_SEEN.pop(k, None)
    if event_id in _INBOUND_SEEN:
        return True
    _INBOUND_SEEN[event_id] = now
    return False


# ---------------------------------------------------------------------------
# Startup integrity check
# ---------------------------------------------------------------------------

def _integrity_check() -> None:
    issues: List[str] = []
    warnings: List[str] = []

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

    # Gemini check — warning only (Groq fallback still works)
    if not settings.gemini_api_key:
        warnings.append(
            "GEMINI_API_KEY is not set. Gemini is the primary orchestrator with a "
            "much higher free-tier quota (1M tokens/min vs Groq 100K tokens/day). "
            "Get a free key at https://aistudio.google.com/apikey and set GEMINI_API_KEY."
        )

    pool = settings.groq_model_pool or []
    for m in pool:
        if not any(m.startswith(p) for p in VALID_GROQ_PREFIXES):
            issues.append(
                f"GROQ_MODEL_POOL contains invalid model {m!r}. "
                "Valid names start with: llama-, compound-beta, mixtral-, gemma-. "
                "Fix GROQ_MODEL_POOL in .env."
            )
    if any(m.startswith("compound-beta") for m in pool):
        warnings.append(
            "GROQ_MODEL_POOL contains compound-beta-mini. This model shares the "
            "llama-3.3-70b daily token bucket (100K/day). Remove it from "
            "GROQ_MODEL_POOL — it is used automatically for live search."
        )

    gem_pool = settings.gemini_model_pool or []
    for m in gem_pool:
        if not any(m.startswith(p) for p in VALID_GEMINI_PREFIXES):
            warnings.append(
                f"GEMINI_MODEL_POOL contains non-Gemini model {m!r}. "
                "Expected names start with: gemini-"
            )

    for issue in issues:
        logger.warning("⚠️  CONFIG_ERROR: %s", issue)
    for warn in warnings:
        logger.warning("⚠️  CONFIG_WARN: %s", warn)
    if not issues and not warnings:
        logger.info("✅ integrity_check — all config OK")
    elif not issues:
        logger.info("✅ integrity_check — config OK (with warnings)")


# ---------------------------------------------------------------------------
# FIX-7: User-facing error messages
# ---------------------------------------------------------------------------

def _rate_limit_reply(exc: Exception) -> str:
    """Compose a user-friendly reply when all LLM providers are rate-limited."""
    msg = str(exc)
    # Groq: "Please try again in 1h4m54.368s"
    m = re.search(
        r"try again in\s+(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:([\d.]+)s)?",
        msg, re.IGNORECASE,
    )
    wait_str = ""
    if m:
        h    = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        if h:
            wait_str = f" (about {h}h {mins}m)"
        elif mins:
            wait_str = f" (about {mins} min)"

    # Gemini: "retry after N seconds"
    if not wait_str:
        m2 = re.search(r"retry[^\d]*(\d+)\s*second", msg, re.IGNORECASE)
        if m2:
            secs = int(m2.group(1))
            wait_str = f" (about {secs//60} min)" if secs >= 60 else f" ({secs}s)"

    is_gemini = "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
    provider  = "Gemini" if is_gemini else "Groq"

    return (
        f"⚡ I've hit my {provider} AI quota{wait_str}. "
        "Please try again later — I'll be back to full speed soon! 🙏"
    )


# ---------------------------------------------------------------------------
# Special memory key handler + reminder saver
# ---------------------------------------------------------------------------

async def _save_reminders(
    sender_key:  str,
    chat_id:     str,
    reminders:   list,
    existing:    list,
) -> int:
    saved = 0
    for r in reminders:
        text        = (r.text or "").strip()
        trigger_iso = (r.trigger_iso or "").strip()
        if not text or not trigger_iso:
            logger.warning("reminder.skip — blank  text=%r  trigger=%r", text[:50], trigger_iso[:30])
            continue
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
            logger.info(
                "🔔 reminder.scheduled  id=%d  chat=%s  trigger=%s  text=%r",
                rid, chat_id, trigger_iso, text[:60],
            )
            saved += 1
        except Exception as exc:
            logger.error("reminder.save_failed  err=%s  text=%r", exc, text[:60])
    return saved


async def _reply_extract_and_save_bg(
    chat_id:     str,
    sender_key:  str,
    user_text:   str,
    bot_reply:   str,
    known_facts: dict,  # noqa: ARG001 — kept for API compat
) -> None:
    """
    Fire-and-forget: extract facts from the bot's own reply and persist them.
    Uses extract_reply_memory() which handles DB writes internally.
    All exceptions are suppressed — this is best-effort enrichment.
    """
    try:
        await extract_reply_memory(
            reply_text=bot_reply,
            chat_id=chat_id,
            sender_key=sender_key,
        )
    except Exception as exc:
        logger.debug("reply_memory.bg_suppressed  err=%s", str(exc)[:80])


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
    from_me     = bool(root.get("fromMe") or root.get("from_me") or False)
    key_remote  = key.get("remoteJid")
    remote_jid  = root.get("remoteJid") or root.get("chatId") or root.get("chat_id")
    from_field  = root.get("from")
    to_field    = root.get("to")
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


async def _ambient_extract_bg(*, chat_id: str, sender_key: str, text: str) -> None:
    """
    Background LLM memory extraction for messages that bypassed the main
    process_message() pipeline (no-prefix group messages).

    Runs _extract_memory + verify + upsert entirely in the background.
    All exceptions are suppressed — purely best-effort enrichment.
    This is what makes non-prefix group chat messages contribute to
    long-term memory, not just context.
    """
    try:
        from .agent_engine import _extract_memory, _verify_updates
        if not database.sqlite_store:
            return
        # Load existing facts so extraction knows what's new
        existing = await database.sqlite_store.get_all_facts(sender_key)
        updates = await _extract_memory(text, chat_id, existing_facts=existing)
        if not updates:
            return
        approved = await _verify_updates(
            updates, chat_id, existing_facts=existing, user_text=text,
        )
        if not approved:
            return
        created = updated = 0
        for u in approved:
            if getattr(u, "delete", False):
                continue   # never auto-delete from ambient observation
            status = await database.sqlite_store.upsert_fact(
                sender_key, normalize_key(u.key), u.value
            )
            if status == "created":
                created += 1
            elif status == "updated":
                updated += 1
        if created or updated:
            logger.info(
                "🧠 ambient_memory.saved  sender=%s  created=%d  updated=%d",
                sender_key, created, updated,
            )
    except Exception as exc:
        logger.debug("ambient_extract_bg.suppressed  sender=%s  err=%s",
                     sender_key, str(exc)[:80])


async def _ambient_store(*, chat_id, sender_key, text, event_id) -> None:
    if not chat_is_allowed(chat_id):
        return
    cleaned = strip_invocation((text or "").strip())
    if not cleaned:
        return
    ts_in = datetime.now(UTC).isoformat()
    stored_sqlite = stored_chroma = False
    if database.sqlite_store:
        await database.sqlite_store.log_message(
            chat_id=chat_id, whatsapp_id=sender_key,
            direction="in", text=cleaned, ts=ts_in, event_id=event_id or None,
        )
        stored_sqlite = True
    if database.chroma_store:
        await database.chroma_store.add_message(
            chat_id=chat_id, whatsapp_id=sender_key,
            direction="in", text=cleaned, ts=ts_in,
            message_id=event_id or ("in-" + str(int(time.time() * 1000))),
        )
        stored_chroma = True

    logger.info(
        "📥 ambient.stored  sender=%s  len=%d  sqlite=%s  chroma=%s  preview=%r",
        sender_key, len(cleaned), stored_sqlite, stored_chroma, cleaned[:60],
    )

    # ── Long-term memory extraction (fire-and-forget) ────────────────────────
    # Even when the message has no bot prefix and won't trigger a full LLM
    # response, we still extract personal facts into long-term memory.
    # Trigger for: first-person signals OR substantive messages (>15 chars)
    # that may contain facts like "Going to Goa next week", "My dog is sick" etc.
    _low = cleaned.lower()
    _personal = ("i ", "i'm", "i am", "my ", "me ", "mine", "myself", "i've", "i'd", "i'll",
                 "we ", "we're", "we are", "our ", "going to", "planning to", "will be")
    _has_signal = any(h in _low for h in _personal) or len(cleaned) > 15
    if sender_key and _has_signal:
        asyncio.create_task(
            _ambient_extract_bg(chat_id=chat_id, sender_key=sender_key, text=cleaned),
            name=f"ambient_extract:{sender_key}",
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
                    # LLM-driven dedup: fire-and-forget — merges semantic
                    # duplicates (favourite_colour / favorite_color etc.)
                    # without a hand-written alias map.
                    asyncio.create_task(
                        consolidate_user_facts(sender_key),
                        name=f"consolidate:{sender_key}",
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
                logger.info("📚 context.built  total=%d", len(context_items))

            # ── 4. run agentic loop ─────────────────────────────────────────
            with trace.step("agent_run"):
                try:
                    result = await run_agent(
                        chat_id=chat_id,
                        sender_key=sender_key,   # P1-GUARD: authenticated, never from LLM
                        user_text=user_text,
                        facts=facts,
                        context=context_items,
                        reminders=reminders,
                        trace=trace,
                    )
                except Exception as agent_exc:
                    # FIX-7: catch rate-limit / LLM failure and send user-facing message
                    exc_str = str(agent_exc)
                    if "429" in exc_str or "rate_limit" in exc_str.lower():
                        friendly = _rate_limit_reply(agent_exc)
                        logger.warning(
                            "⚡ rate_limit.user_reply  chat=%s  err=%s",
                            chat_id, exc_str[:200],
                        )
                        try:
                            await send_text(chat_id, friendly)
                        except Exception:
                            pass
                    raise   # re-raise so trace captures fatal_error

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

            # ── 5b. reply-extract (ambient memory) ──────────────────────────
            # Runs after a genuine LLM response to capture data the bot confirmed.
            # Skipped for shortcut responses — the bot echoed an existing DB value;
            # there is nothing new to extract. Running it there wastes ~140ms and
            # risks re-saving a stale value over a freshly updated one.
            _re_text     = getattr(result.reply, "text", None) or ""
            _is_shortcut = getattr(result, "provider_used", "") == "shortcut"
            _is_question = (len(_re_text) < 120 and _re_text.rstrip().endswith("?"))
            if _re_text and not _is_shortcut and not _is_question:
                with trace.step("reply_extract"):
                    await _reply_extract_and_save_bg(
                        chat_id=chat_id,
                        sender_key=sender_key,
                        user_text=user_text,
                        bot_reply=_re_text,
                        known_facts=facts,
                    )

            # ── 6. persist memory + reminders ──────────────────────────────
            with trace.step("memory_save"):
                saved = created = updated = unchanged = 0
                save_errors: List[str] = []

                if database.sqlite_store:
                    # P1-GUARD: sender_key is derived from the WAHA webhook payload,
                    # never from LLM output. Explicitly verify it is not empty.
                    assert sender_key, "sender_key must be set before writing to DB"

                    for mu in result.memory_updates:
                        try:
                            if getattr(mu, "delete", False):
                                # P1-FEAT-2 + P1-GUARD: guarded deletion
                                confirmed = getattr(mu, "confirm", False)
                                outcome = await database.sqlite_store.delete_fact(
                                    sender_key, mu.key, confirmed=confirmed,
                                )
                                # ── Branch on structured DeleteOutcome enum ──────
                                if outcome == DeleteOutcome.DELETED:
                                    logger.info(
                                        "🗑️  memory.deleted  sender=%s  key=%s  confirmed=%s",
                                        sender_key, mu.key, confirmed,
                                    )
                                    saved += 1

                                elif outcome == DeleteOutcome.NEEDS_CONFIRM:
                                    # High-stakes list key (shopping_list / grocery_list /
                                    # todo_list) — queue confirmation and rewrite reply.
                                    current_val = facts.get(mu.key, "")
                                    from .agent_engine import _register_pending_delete
                                    _register_pending_delete(sender_key, mu.key, current_val)
                                    key_label = mu.key.replace("_", " ")
                                    confirm_text = (
                                        f"⚠️ Are you sure you want to clear your "
                                        f"*{key_label}*? Reply *yes* to confirm or "
                                        f"*no* to keep it."
                                    )
                                    # ReplyPayload is frozen Pydantic — use object.__setattr__
                                    object.__setattr__(result.reply, "text", confirm_text)
                                    logger.info(
                                        "⏳ memory.delete_needs_confirm  sender=%s  key=%s",
                                        sender_key, mu.key,
                                    )

                                elif outcome == DeleteOutcome.NOT_FOUND:
                                    # Key not in DB — no-op, not an error.
                                    logger.debug(
                                        "🗑️  memory.delete_noop  sender=%s  key=%s  (not in DB)",
                                        sender_key, mu.key,
                                    )

                                elif outcome in (DeleteOutcome.BLOCKED, DeleteOutcome.EMPTY_KEY):
                                    logger.warning(
                                        "🚫 memory.delete_blocked  sender=%s  key=%s  outcome=%s",
                                        sender_key, mu.key, outcome,
                                    )
                                    save_errors.append(f"{mu.key}: {outcome}")
                            else:
                                status = await database.sqlite_store.upsert_fact(
                                    sender_key, mu.key, mu.value,
                                )
                                if status == "created":
                                    created += 1
                                    saved += 1
                                    logger.info(
                                        "🧠 memory.new      sender=%s  key=%s  value=%r",
                                        sender_key, mu.key, mu.value,
                                    )
                                elif status == "updated":
                                    updated += 1
                                    saved += 1
                                    logger.info(
                                        "🧠 memory.updated  sender=%s  key=%s  value=%r",
                                        sender_key, mu.key, mu.value,
                                    )
                                else:  # "unchanged"
                                    unchanged += 1
                        except Exception as exc:
                            err_msg = f"{type(exc).__name__}: {exc}"
                            save_errors.append(f"{mu.key}: {err_msg}")
                            logger.error(
                                "🧠 memory.save_failed  sender=%s  key=%s  err=%s",
                                sender_key, mu.key, exc,
                            )

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
                    facts_unchanged=unchanged, facts_attempted=len(result.memory_updates),
                    **( {"save_errors": "; ".join(save_errors)} if save_errors else {} ),
                )
                if result.memory_updates:
                    logger.info(
                        "🧠 memory.summary  sender=%s  attempted=%d  "
                        "created=%d  updated=%d  unchanged=%d  errors=%d",
                        sender_key, len(result.memory_updates), created, updated, unchanged, len(save_errors),
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

            # ── 8. store outgoing message (fire-and-forget) ────────────────
            ts_out = datetime.now(UTC).isoformat()
            out_id = msg_id or f"out-{event_id}" or f"out-{int(time.time()*1000)}"
            asyncio.create_task(
                _store_out_bg(chat_id=chat_id, text=result.reply.text, ts=ts_out, out_id=out_id)
            )
            # Note: reply_extract now runs in step 5b (awaited before memory_save).

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

    _REMINDER_TASK = asyncio.create_task(
        run_reminder_loop(check_interval_sec=60),
        name="reminder_scheduler",
    )
    logger.info("🕐 scheduler.task_created")

    port = int(os.getenv("PORT", "6000"))
    primary_orch = (
        f"Gemini({settings.gemini_orchestrator_model})"
        if settings.gemini_enabled
        else f"Groq({settings.orchestrator_model})"
    )
    logger.info(
        "🚀 startup.ready  http://0.0.0.0:%d  "
        "allowlist=%d  allow_all=%s  live_search=%s  chroma=%s  "
        "primary_orchestrator=%s  extraction=%s  gemini=%s",
        port,
        len(settings.allowed_chat_jids or []),
        settings.allow_all_chats,
        settings.live_search_enabled,
        settings.chroma_enabled,
        primary_orch,
        settings.extraction_model,
        "✅" if settings.gemini_enabled else "❌ (set GEMINI_API_KEY for 15× more quota)",
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

    # FIX-P0-3: reject duplicate event_ids (WAHA retries same event on delivery failure)
    if event_id and _inbound_seen_check(event_id):
        logger.debug("webhook.dedup  event=%s  chat=%s  (already seen)", event_id, chat_id)
        return JSONResponse({"status": "ok", "message": "duplicate"})

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
    import time as _time
    now = _time.monotonic()
    model_circuits = {
        m: "open" if now >= ts else f"tripped ({ts - now:.0f}s remaining)"
        for m, ts in MODEL_CIRCUIT.items()
    }
    provider_circuits = {
        p: "open" if now >= ts else f"tripped ({ts - now:.0f}s remaining)"
        for p, ts in PROVIDER_CIRCUIT.items()
    }
    # Token budget fractions
    from .agent_engine import _budget_fraction
    budget = {
        "groq_70b": f"{_budget_fraction('groq_70b', settings.groq_70b_daily_limit)*100:.1f}% of {settings.groq_70b_daily_limit:,}/day",
        "groq_8b":  f"{_budget_fraction('groq_8b', 500_000)*100:.1f}% of 500K/day",
    }
    return {
        "status":            "ok",
        "version":           "3.2.0",
        "workers":           len(CHAT_WORKERS),
        "queues":            {cid: q.qsize() for cid, q in CHAT_QUEUES.items()},
        "providers": {
            "gemini": {
                "enabled": settings.gemini_enabled,
                "orchestrator": settings.gemini_orchestrator_model,
                "extraction":   settings.gemini_extraction_model,
            },
            "groq": {
                "orchestrator": settings.orchestrator_model,
                "extraction":   settings.extraction_model,
                "pool":         settings.groq_model_pool,
            },
        },
        "live_search":       settings.live_search_enabled,
        "chroma":            settings.chroma_enabled,
        "model_circuits":    model_circuits,
        "provider_circuits": provider_circuits,
        "token_budget":      budget,
        "reminder_task":     _REMINDER_TASK is not None and not (_REMINDER_TASK.done() if _REMINDER_TASK else True),
    }
