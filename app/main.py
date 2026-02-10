from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .utils import verify_signature, canonical_text
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
from .clients_llm import init_clients as init_llm, close_clients as close_llm, groq_chat
from .clients_llm_extractor import llm_extract_facts_open, llm_judge_facts
from .memory_admission import admit_fact, normalize_fact, admit_holding, normalize_holding
from .memory import (
    get_user_facts,
    upsert_user_fact,
    insert_user_record,
    classify_and_persist,
    build_profile_snapshot_text,
)
from .observe import handle_observe_command, observe_ingest

# Diagnostics router (protected via DIAG_ENABLED/DIAG_API_KEY in diag.py)
from .diag import router as diag_router

# Optional Chroma (local FAISS RAG)
CHROMA_AVAILABLE = True
try:
    from .chroma import add_text as chroma_add_text, query as chroma_query, warmup as chroma_warmup
except Exception:
    CHROMA_AVAILABLE = False

def chroma_enabled() -> bool:
    return CHROMA_AVAILABLE and bool(getattr(config, "CHROMA_ENABLED", True))

# ---------------- Logging ----------------
logger = logging.getLogger("app")
if not logger.handlers:
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s:%(message)s")
    level_name = str(getattr(config, "LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=fmt)

logging.getLogger("httpx").setLevel(logging.WARNING)

def log_kv(event: str, **kw):
    def _s(v):
        if isinstance(v, str) and len(v) > 420:
            return v[:420] + "…"
        return v
    logger.info("%s %s", event, " ".join(f"{k}={_s(v)}" for k, v in kw.items()))

# ---------------- App + Routers ----------------
app = FastAPI()
app.include_router(diag_router)

# ---------------- Per-chat queues ----------------
CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}

# ---------------- JID helpers ----------------
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
    if not getattr(config, "ALLOWED_GROUP_JIDS", None):
        return True
    return chat_id in config.ALLOWED_GROUP_JIDS

def normalize_jid(jid: Optional[str]) -> Optional[str]:
    if not jid:
        return None
    if jid.endswith("@s.whatsapp.net"):
        return jid.replace("@s.whatsapp.net", "@c.us")
    return jid

def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default

# ---------------- Prefix handling (STRICT + anchored) ----------------
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
    """Strict: prefix must be at start (optional @), allow ':' or ',' after."""
    global _PREFIX_RE
    alts = [re.escape(p.lstrip("@")) for p in _prefixes()]
    if not alts:
        _PREFIX_RE = re.compile(r"^$")
        return
    _PREFIX_RE = re.compile(r"^\s*@?(%s)\b[,:]?\s+" % "|".join(alts), re.IGNORECASE)

def has_prefix(text: Optional[str]) -> bool:
    if not text:
        return False
    if _PREFIX_RE is None:
        _compile_prefix_re()
    return bool(_PREFIX_RE.match(text))

def strip_prefix(text: str) -> str:
    if _PREFIX_RE is None:
        _compile_prefix_re()
    m = _PREFIX_RE.match(text or "")
    return (text[m.end():] if m else (text or "")).strip()

async def _strip_vocative_prefix(reply: str) -> str:
    """Remove accidental leading prefix used as a name (e.g., 'Spock, ...')."""
    if not reply:
        return reply
    alts = [re.escape(p.lstrip("@")) for p in _prefixes()]
    if not alts:
        return reply.strip()
    return re.sub(r"^\s*@?(%s)\b[,:]?\s*" % "|".join(alts), "", reply, flags=re.IGNORECASE).strip()

# ---------------- Intent detection ----------------
_GREET_RE = re.compile(r"\b(hello|hey|hi|good (?:morning|afternoon|evening))\b", re.I)
_WEATHER_RE = re.compile(r"\b(weather|forecast|temperature|rain|humidity|wind|climate)\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summarise|summarize|recap|conversation so far|tl;dr)\b", re.I)
_PROFILE_Q_RE = re.compile(
    r"\b(what(?:'s|\s+is)?\s+my|where\s+do\s+i|who\s+is\s+my|tell\s+me\s+everything\s+you|based\s+on\s+everything\s+you\s+know\s+about\s+me)\b",
    re.I,
)
_HOLDING_HINT_RE = re.compile(r"\b(i have stocks|i hold|shares of|purchase value|avg price|average price)\b", re.I)

def detect_intent(text: str) -> str:
    tl = (text or "").strip().lower()
    if _GREET_RE.search(tl):
        return "greeting"
    if _WEATHER_RE.search(tl):
        return "weather"
    if _SUMMARY_RE.search(tl):
        return "summarize"
    if _PROFILE_Q_RE.search(tl):
        return "profile_query"
    if _HOLDING_HINT_RE.search(tl):
        return "holdings_intake"
    return "general_qa"

# ---------------- Summarize grounded in DB ----------------
async def summarize_from_db(chat_id: str, limit: int = 18) -> str:
    rows = await fetchall(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?",
        (chat_id, limit),
    )
    if not rows:
        return ""
    rows = list(reversed(rows))
    # ✅ This is the exact line that must NOT break:
    convo = "\n".join(f"{r[0].upper()}: {r[1]}" for r in rows if r[1])

    sys = (
        "You summarize ONLY the provided chat lines. Plain text. No markdown. "
        "Do not invent details not present in the lines."
    )
    reply, ok, _ = await groq_chat(chat_id, sys, convo, temperature=0.0, max_tokens=220)
    return reply.strip() if ok and reply else ""

# ---------------- Context assembly ----------------
async def assemble_context(intent: str, sender_id: str, chat_id: str, user_text: str) -> str:
    lines: List[str] = []

    if intent == "profile_query":
        snap = await build_profile_snapshot_text(sender_id)
        if snap:
            lines.append("PROFILE_FACTS\n" + snap)

        facts = await get_user_facts(sender_id, namespace="default")
        if facts:
            kv = "\n".join(f"{k}: {v}" for k, v in facts[:80])
            lines.append("FACTS\n" + kv)

    # For general QA, use RAG snippets if enabled
    if intent == "general_qa" and chroma_enabled():
        try:
            rag = await chroma_query(chat_id=chat_id, text=user_text, k=3)
            if rag:
                lines.append("SNIPPETS\n" + rag)
        except Exception:
            pass

    # For summarize intent, we do NOT use RAG; we summarize from DB lines.
    return "\n\n".join(lines)

# ---------------- Core message processor ----------------
async def process_message(chat_id: str, text: str, sender_id: str):
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))

    try:
        raw_text = (text or "").strip()
        user_text = strip_prefix(raw_text)

        log_kv("process.begin", chat_id=chat_id, sender_id=sender_id, text=user_text[:160])

        # Handle commands (lists/help/forget)
        from .handlers import handle_commands
        handled, cmd_reply = await handle_commands(sender_id, chat_id, user_text)
        if handled:
            out = cmd_reply or "Done."
            sent = await send_message(chat_id, out)
            if sent:
                await save_message(chat_id, sender_id, "assistant", out, event_id=f"local:{time.time()}")
            return

        intent = detect_intent(user_text)

        # ✅ Hard no-memory guard for profile queries (prevents hallucinations)
        if intent == "profile_query":
            snap = await build_profile_snapshot_text(sender_id)
            facts = await get_user_facts(sender_id, namespace="default")
            if not snap and not facts:
                msg = (
                    "I don’t have any saved details about you yet. "
                    "If you’d like, tell me a few basics like: “my name is …”, “I live in …”, “my job is …”."
                )
                sent = await send_message(chat_id, msg)
                if sent:
                    await save_message(chat_id, sender_id, "assistant", msg, event_id=f"local:{time.time()}")
                return

        # Deterministic tiny extractor (city/country)
        await classify_and_persist(chat_id, sender_id, user_text)

        # Add addressed messages to RAG (for later recall)
        if chroma_enabled() and len(user_text) >= 60:
            try:
                await chroma_add_text(chat_id=chat_id, sender_id=sender_id, text=user_text)
            except Exception:
                pass

        # LLM extraction gating (only if message looks like fact-statement)
        extraction_mode = str(getattr(config, "FACTS_EXTRACTION_MODE", "hybrid")).lower()
        min_conf = float(getattr(config, "FACTS_MIN_CONF", 0.80))
        verify_enabled = _as_bool(getattr(config, "FACTS_VERIFICATION", True), True)

        t_low = user_text.lower()
        strong_fact_signal = any(p in t_low for p in [
            "my name is", "i'm ", "i am ", "i live in", "my birthday", "i work", "my coffee",
            "i adopted", "i'm allergic", "my favorite color", "my hobbies",
            "i hold", "shares of", "purchase price", "avg price",
        ])

        if extraction_mode in {"llm", "hybrid"} and strong_fact_signal:
            existing = dict(await get_user_facts(sender_id, namespace="default"))
            raw = await llm_extract_facts_open(chat_id, user_text, known_facts=existing)

            facts = raw.get("facts", []) or []
            recs = raw.get("records", []) or []

            fact_candidates = []
            for f in facts:
                f = normalize_fact(f)
                if admit_fact(f, user_text=user_text):
                    fact_candidates.append(f)

            record_candidates = []
            for r in recs:
                r = normalize_holding(r)
                if r and admit_holding(r):
                    record_candidates.append(r)

            approved, judge_conf, why = True, 1.0, ""
            if (fact_candidates or record_candidates) and verify_enabled:
                payload = {"facts": fact_candidates, "records": record_candidates}
                approved, judge_conf, why = await llm_judge_facts(chat_id, user_text, payload)
                log_kv("llm.judge", approved=approved, conf=judge_conf, why=why)

            facts_persisted = 0
            if (fact_candidates or record_candidates) and approved and judge_conf >= min_conf:
                for f in fact_candidates:
                    ok = await upsert_user_fact(
                        sender_id,
                        key=f["key"],
                        value=f["value"],
                        namespace=f.get("namespace", "default"),
                        value_type=f.get("type", "text"),
                        confidence=judge_conf,
                    )
                    if ok:
                        facts_persisted += 1

                for r in record_candidates:
                    await insert_user_record(sender_id, record_type="holding", data=r, confidence=judge_conf)

                # If user is stating facts (not asking), send short ACK
                if facts_persisted and "?" not in user_text:
                    ack = "✅ Got it — I’ve saved that."
                    sent = await send_message(chat_id, ack)
                    if sent:
                        await save_message(chat_id, sender_id, "assistant", ack, event_id=f"local:{time.time()}")
                    return

        # Summarize: grounded in DB
        if intent == "summarize":
            s = await summarize_from_db(chat_id, limit=18)
            if not s:
                s = "I don’t have enough chat history yet to summarize."
            s = await _strip_vocative_prefix(s)
            sent = await send_message(chat_id, s)
            if sent:
                await save_message(chat_id, sender_id, "assistant", s, event_id=f"local:{time.time()}")
            return

        # Build context for normal response
        context = await assemble_context(intent, sender_id, chat_id, user_text)

        sys = "\n".join([
            f"You are {config.BOT_PERSONA_NAME}, a helpful WhatsApp assistant.",
            "Write naturally in short paragraphs (1–3 sentences), not lists.",
            "Use lists ONLY if the user asks for steps/bullets/summary.",
            "Never invent personal facts. If information is not in CONTEXT, say you don’t know.",
            "Never address the user by a command prefix.",
            "Avoid markdown formatting.",
        ])

        prompt = f"{user_text}\n\nCONTEXT:\n{context}" if context else user_text
        reply, ok, _ = await groq_chat(chat_id, sys, prompt, temperature=0.5, max_tokens=650)
        if not ok or not reply:
            reply = "Sorry, try again later."

        reply = await _strip_vocative_prefix(reply)
        sent = await send_message(chat_id, reply)
        if sent:
            await save_message(chat_id, sender_id, "assistant", reply, event_id=f"local:{time.time()}")

    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass
        log_kv("process.end", chat_id=chat_id)

# ---------------- Webhook parsing ----------------
def normalize_lid_or_chat(body: dict) -> Tuple[Optional[str], Optional[str], str, dict]:
    root = body.get("payload") or body.get("data") or {}
    data_obj = root.get("_data") or {}
    key = data_obj.get("key") or {}
    me = body.get("me") or {}

    me_id = normalize_jid(me.get("id") or None)
    me_lid_lit = me.get("lid") or None

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
    sender_id_raw = sender_obj.get("id") or participant or from_field or remote_jid

    chat_id_raw = None
    if key_remote:
        chat_id_raw = key_remote
    elif remote_jid:
        chat_id_raw = remote_jid
    elif from_me:
        chat_id_raw = to_field or from_field
    else:
        chat_id_raw = from_field or to_field

    if key_remote and me_lid_lit and key_remote == me_lid_lit and me_id:
        chat_id = me_id
    else:
        chat_id = normalize_jid(chat_id_raw)

    sender_id = normalize_jid(sender_id_raw)
    dbg = {"text": (text or "")[:160], "from_me": from_me}
    return sender_id, chat_id, text, dbg

# ---------------- Startup/Shutdown ----------------
@app.on_event("startup")
async def startup():
    _compile_prefix_re()
    await init_db()
    await init_waha()
    await init_llm()

    log_kv("startup.complete", waha=getattr(config, "WAHA_API_URL", ""), session=getattr(config, "WAHA_SESSION", ""))
    log_kv("chroma.status", enabled=chroma_enabled(), available=CHROMA_AVAILABLE)

    if chroma_enabled():
        try:
            dim = await chroma_warmup()
            log_kv("chroma.warmup", dim=dim)
        except Exception as e:
            log_kv("chroma.warmup.fail", err=str(e)[:200])

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
    log_kv("shutdown.complete")

# ---------------- Webhook endpoint ----------------
@app.post("/webhook")
async def webhook(request: Request):
    raw = await request.body()
    if not verify_signature(
        raw,
        request.headers.get("X-WAHA-HMAC")
        or request.headers.get("X-Webhook-Signature")
        or request.headers.get("X-Signature"),
    ):
        log_kv("webhook.sig_fail")
        return JSONResponse({"status": "error", "message": "Invalid signature"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        log_kv("webhook.json_fail")
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    evt = (body.get("event") or "").lower()
    if evt not in {"message", "message.any", "message.created", "message.edited"}:
        return JSONResponse({"status": "ok", "message": f"ignored {evt}"})

    sender_id, chat_id, text, dbg = normalize_lid_or_chat(body)
    if not chat_id:
        return JSONResponse({"status": "ok", "message": "no chat id"})

    if looks_channel(chat_id):
        return JSONResponse({"status": "ok", "message": "channel ignored"})

    if not group_allowed(chat_id):
        return JSONResponse({"status": "ok", "message": "group not allowed"})

    # Purge echo caches
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)

    event_id = (body.get("payload") or {}).get("id") or body.get("id") or ""

    # Echo filter
    is_echo = False
    if event_id and str(event_id) in OUTBOUND_CACHE_IDS:
        is_echo = True
    if chat_id and text and hashlib.sha1(f"{chat_id}\n{canonical_text(text)}".encode("utf-8")).hexdigest() in OUTBOUND_CACHE_TXT:
        is_echo = True
    if is_echo:
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    # /observe commands (admin-controlled can be added later; this is pilot-friendly)
    if looks_group(chat_id) and (text or "").strip().lower().startswith("/observe"):
        msg = await handle_observe_command(chat_id, sender_id or chat_id, text)
        if msg:
            await send_message(chat_id, msg)
        return JSONResponse({"status": "ok", "message": "observe handled"})

    # Ambient observation: group + no prefix => ingest only, never reply
    if looks_group(chat_id) and not has_prefix(text) and chroma_enabled():
        await observe_ingest(chat_id, sender_id or chat_id, text or "", chroma_add_text=chroma_add_text, db_exec=execute)
        return JSONResponse({"status": "ok", "message": "observed"})

    # Strict prefix gating for group/broadcast
    if (looks_group(chat_id) or looks_broadcast(chat_id)) and not has_prefix(text):
        return JSONResponse({"status": "ok", "message": "no group prefix"})

    # Optional DM prefix gating
    allow_nlp = _as_bool(getattr(config, "ALLOW_NLP_WITHOUT_PREFIX", True), True)
    if (not is_groupish(chat_id)) and (not allow_nlp) and not has_prefix(text):
        return JSONResponse({"status": "ok", "message": "no dm prefix"})

    # Save inbound message (dedup by UNIQUE(chat_id,event_id))
    await save_message(chat_id, sender_id or chat_id, "user", strip_prefix(text or ""), event_id=event_id)

    # Debounce
    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < getattr(config, "MESSAGE_DEBOUNCE_MS", 800):
        return JSONResponse({"status": "ok", "message": "debounced"})

    # Per-chat queue
    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=getattr(config, "LLM_MAX_QUEUE_PER_CHAT", 3))
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            try:
                while True:
                    item = await q.get()
                    try:
                        await process_message(chat_id, item["text"], item["sender_id"])
                    finally:
                        q.task_done()
            except asyncio.CancelledError:
                pass

        CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

    item = {"text": text or "", "sender_id": sender_id or chat_id}
    try:
        await asyncio.wait_for(q.put(item), timeout=getattr(config, "LLM_QUEUE_WAIT_SEC", 20))
    except asyncio.TimeoutError:
        await send_message(chat_id, "I'm busy; try again in a few seconds.")
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    return JSONResponse({"status": "ok", "message": "enqueued"})

@app.get("/healthz")
async def health():
    return {"status": "ok"}
