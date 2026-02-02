
# app/main.py
from __future__ import annotations
import asyncio, time, hashlib, logging, re, os, json
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .utils import verify_signature, canonical_text
from .db import init_db, save_message
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

CHROMA_AVAILABLE = True
try:
    from .chroma import add_text as chroma_add_text, query as chroma_query, add_profile_snapshot, warmup as chroma_warmup
except Exception:
    CHROMA_AVAILABLE = False

def chroma_enabled() -> bool:
    return CHROMA_AVAILABLE and bool(getattr(config, "CHROMA_ENABLED", True))

logger = logging.getLogger("app")
if not logger.handlers:
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s:%(message)s")
    level_name = str(getattr(config, "LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=fmt)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(
    getattr(logging, os.getenv("ACCESS_LOG_LEVEL", "INFO").upper(), logging.INFO)
)

def log_kv(event: str, **kw):
    def _s(v):
        if isinstance(v, str) and len(v) > 420:
            return v[:420] + "…"
        return v
    logger.info("%s %s", event, " ".join(f"{k}={_s(v)}" for k, v in kw.items()))

CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}

def looks_group(j: Optional[str]) -> bool: return bool(j and j.endswith("@g.us"))
def looks_broadcast(j: Optional[str]) -> bool: return bool(j and j.endswith("@broadcast"))
def looks_channel(j: Optional[str]) -> bool: return bool(j and j.endswith("@newsletter"))
def is_groupish(j: Optional[str]) -> bool: return looks_group(j) or looks_broadcast(j) or looks_channel(j)

def group_allowed(chat_id: Optional[str]) -> bool:
    if not chat_id: return False
    if looks_channel(chat_id): return True
    if not looks_group(chat_id): return True
    if not getattr(config, "ALLOWED_GROUP_JIDS", None): return True
    return chat_id in config.ALLOWED_GROUP_JIDS

def normalize_jid(jid: Optional[str]) -> Optional[str]:
    if not jid: return None
    if jid.endswith("@s.whatsapp.net"): return jid.replace("@s.whatsapp.net", "@c.us")
    return jid

def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    s = str(v).strip().lower()
    if s in {"1","true","yes","y","on"}: return True
    if s in {"0","false","no","n","off"}: return False
    return default

def _prefixes() -> list[str]:
    raw = getattr(config, "BOT_COMMAND_PREFIX", "") or ""
    if isinstance(raw, str):
        return [p.strip().lower() for p in raw.split(",") if str(p).strip()]
    if isinstance(raw, (list, tuple, set)):
        out = []
        for p in raw:
            s = str(p).strip().lower()
            if s: out.append(s)
        return out
    return [p.strip().lower() for p in str(raw).split(",") if str(p).strip()]

def has_prefix(text: Optional[str]) -> bool:
    if not text: return False
    t = (" " + (text or "").strip().lower() + " ")
    for p in _prefixes():
        if not p: continue
        if p in t or t.startswith(" " + p + " ") or t.endswith(" " + p + " ") or (" " + p + " ") in t:
            return True
    return False

# ---------- Intent detection (regex, robust to typos & "do I" forms) ----------
_GREET_RE = re.compile(r"\b(hello|hey|hi|good (?:morning|afternoon|evening))\b", re.I)
_WEATHER_RE = re.compile(r"\b(weather|forecast|temperature|rain|humidity|wind|climate)\b", re.I)
_FINANCE_RE = re.compile(r"\b(stock market|market update|nifty|sensex|indices|stocks today|market news|finance news)\b", re.I)
_HOLDING_HINT_RE = re.compile(r"\b(i have stocks|i hold|shares of|purchase value|avg price|average price)\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summarise|summarize|recap|conversation so far|tl;dr)\b", re.I)

# Base profile-question patterns
_PROFILE_Q_RE = re.compile(
    r"(?:"                                   # any of:
    r"\bwhat(?:'s|\s+is)?\s+my\b|"           # what's my / what is my
    r"\bwhere\s+do\s+i\b|"                   # where do i ...
    r"\bwho\s+is\s+my\b|"                    # who is my ...
    r"\bwhen\s+is\s+my\b|"                   # when is my ...
    r"\bhow\s+many\b.*\bmy\b|"               # how many ... my
    r"\bwhat\b.*\bdo\s+i\b|"                 # what ... do i ...
    r"\bwhich\b.*\bdo\s+i\b|"                # which ... do i ...
    r"\bwhich\b.*\bdid\s+i\b|"               # which ... did i ...
    r"\bbased on everything you know about me\b|" # holistic recall
    r"\btell me everything you (?:remember|remeber|rememebr|rember) about me\b" # tolerate common typos
    r")",
    re.I
)

# Heuristic keywords that imply "about me"
_PROFILE_KEYWORDS = re.compile(
    r"\b(name|age|birthday|birthdate|city|live|location|address|zip|postal|country|"
    r"university|degree|education|graduation|job|work|role|company|employer|"
    r"coffee|order|favorite|favourite|color|colour|hobby|hobbies|trail|"
    r"pet|pets|allergy|allergic|podcast|band|music|genre|car|commute|neighborhood|neighbourhood|"
    r"run|running|mileage|miles|trip|travel|destination|goal|language|best friend)\b",
    re.I
)

def _looks_profile_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t: return False
    if _PROFILE_Q_RE.search(t): return True
    # additional heuristic: first-person question + profile keywords
    if re.search(r"\b(i|my|me)\b", t) and _PROFILE_KEYWORDS.search(t):
        return True
    # questions starting with what/which + profile keyword and first person
    if re.search(r"\b(what|which|where|who|when|how)\b", t) and re.search(r"\b(i|my|me)\b", t):
        return True
    return False

def detect_intent(text: str) -> str:
    t = (text or "").strip()
    tl = t.lower()
    if _GREET_RE.search(tl): return "greeting"
    if _WEATHER_RE.search(tl): return "weather"
    if _looks_profile_like_question(tl): return "profile_query"
    if _FINANCE_RE.search(tl): return "finance_news"
    if _HOLDING_HINT_RE.search(tl): return "holdings_intake"
    if _SUMMARY_RE.search(tl): return "summarize"
    return "general_qa"

def should_use_chroma(intent: str) -> bool:
    return intent in {"general_qa", "summarize", "finance_news"}

_TIME_PATS = [
    (re.compile(r"\blast\s+week\b", re.I), 7),
    (re.compile(r"\blast\s+(\d+)\s+days?\b", re.I), None),
    (re.compile(r"\byesterday\b", re.I), 1),
    (re.compile(r"\btoday\b", re.I), 0),
    (re.compile(r"\blast\s+month\b", re.I), 30),
]

def _infer_time_window(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text: return None, None
    t = text.strip().lower()
    now = datetime.now(timezone.utc)
    m = re.search(r"last\s+(\d+)\s+days?", t)
    if m:
        days = max(1, int(m.group(1)))
        return (now - timedelta(days=days)).isoformat(), now.isoformat()
    for pat, days in _TIME_PATS:
        if pat.search(t):
            if days == 0:
                start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
                return start.isoformat(), now.isoformat()
            elif days == 1:
                start = now - timedelta(days=1)
                start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
                end = start + timedelta(days=1)
                return start.isoformat(), end.isoformat()
            elif days:
                return (now - timedelta(days=days)).isoformat(), now.isoformat()
    return None, None

async def assemble_context(intent: str, sender_id: str, chat_id: str, user_text: str) -> str:
    lines: List[str] = []

    if intent == "weather":
        facts = await get_user_facts(sender_id, namespace="default")
        loc_keys = {"city", "state", "country", "postal_code"}
        loc = [(k, v) for k, v in facts if k.lower() in loc_keys]
        if loc:
            s = "\n".join(f"{k}: {v}" for k, v in loc)
            lines.append("LOCATION\n" + s)

    if should_use_chroma(intent) and chroma_enabled():
        since, until = _infer_time_window(user_text)
        try:
            rag = await chroma_query(chat_id=chat_id, text=user_text, k=3, since_iso=since, until_iso=until)
            if rag:
                lines.append("SNIPPETS\n" + rag)
                log_kv("context.chroma.general", k=3)
        except Exception:
            pass

    if intent in {"self_profile", "profile_query"}:
        snap = await build_profile_snapshot_text(sender_id)
        if snap:
            lines.append("PROFILE_FACTS\n" + snap)
        facts = await get_user_facts(sender_id, namespace="default")
        if facts:
            kv = "\n".join(f"{k}: {v}" for k, v in facts[:120])  # higher cap for recall
            lines.append("FACTS\n" + kv)

    # Heuristic safety: if intent fell through to general_qa but the question looks profile-ish, inject anyway
    if intent == "general_qa" and _looks_profile_like_question(user_text):
        snap = await build_profile_snapshot_text(sender_id)
        if snap:
            lines.append("PROFILE_FACTS\n" + snap)
        facts = await get_user_facts(sender_id, namespace="default")
        if facts:
            kv = "\n".join(f"{k}: {v}" for k, v in facts[:120])
            lines.append("FACTS\n" + kv)

    return "\n\n".join(lines)

# Diagnostics router (kept)
from .diag import router as diag_router
app = FastAPI()
app.include_router(diag_router)

@app.on_event("startup")
async def startup():
    await init_db()
    await init_waha()
    await init_llm()
    log_kv("startup.complete", waha=config.WAHA_API_URL, session=config.WAHA_SESSION)
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
        try: t.cancel()
        except Exception: pass
    await close_waha()
    await close_llm()
    log_kv("shutdown.complete")

async def _sanitize_vocative(reply: str, sender_id: str) -> str:
    tl = reply.lower()
    if "not spock" in tl:
        return reply
    facts = dict(await get_user_facts(sender_id, "default"))
    name = facts.get("name", "").strip()
    if "spock" in tl:
        if name:
            reply = re.sub(r"\b[Ss]pock\b", name, reply)
        else:
            reply = re.sub(r"\b[Ss]pock[,! ]*", "", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()
    return reply

def _trim_context_for_retry(intent: str, user_text: str, context: str) -> str:
    """
    Build a trimmed, high-signal prompt for retry:
      - profile-ish → FACTS only
      - otherwise → drop SNIPPETS
    """
    if not context:
        return user_text
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    if _looks_profile_like_question(user_text) or intent in {"profile_query", "self_profile"}:
        facts = [b for b in blocks if b.startswith("FACTS")]
        if not facts:
            # fallback: keep PROFILE_FACTS if FACTS missing
            facts = [b for b in blocks if b.startswith("PROFILE_FACTS")]
        skinny = "\n\n".join(facts[:1])
        return f"{user_text}\n\nCONTEXT:\n{skinny}" if skinny else user_text
    # general -> keep non-SNIPPETS
    non_snippets = [b for b in blocks if not b.startswith("SNIPPETS")]
    skinny = "\n\n".join(non_snippets)
    return f"{user_text}\n\nCONTEXT:\n{skinny}" if skinny else user_text

async def process_message(chat_id: str, text: str, sender_id: str):
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))
    try:
        log_kv("💬 process.begin", chat_id=chat_id, sender_id=sender_id, text=(text or "")[:160])

        from .handlers import handle_commands
        if await handle_commands(sender_id, chat_id, text):
            ok = await send_message(chat_id, "Done.")
            log_kv("process.command", sent=ok)
            return

        extraction_mode = str(getattr(config, "FACTS_EXTRACTION_MODE", "hybrid")).lower()
        min_conf       = float(getattr(config, "FACTS_MIN_CONF", 0.80))
        verify_enabled = _as_bool(getattr(config, "FACTS_VERIFICATION", True), True)

        t_low = (text or "").lower()
        likely_facts_or_records = any(w in t_low for w in [
            "live in", "my ", "i am ", "i'm ", "city", "country", "zip", "postal", "pincode",
            "color", "preference", "i have stocks", "i hold", "shares of", "purchase value", "avg price", "average price",
            "birthday", "allergic", "name is ", "i work as", "my coffee", "hobby", "hobbies", "pet", "pets",
            "podcast", "band", "music", "car", "commute", "neighborhood", "running mileage"
        ])

        facts_persisted = 0
        records_persisted = 0

        if extraction_mode in {"llm", "hybrid"} and likely_facts_or_records:
            existing = dict(await get_user_facts(sender_id, namespace="default"))
            log_kv("🧠 llm.extract.begin")
            raw = await llm_extract_facts_open(chat_id, text, known_facts=existing)
            facts = raw.get("facts", []) or []
            recs  = raw.get("records", []) or []

            fact_candidates = []
            for f in facts:
                f = normalize_fact(f)
                if admit_fact(f):
                    fact_candidates.append(f)

            record_candidates = []
            for r in recs:
                r = normalize_holding(r)
                if r and admit_holding(r):
                    record_candidates.append(r)

            logger.info("🧠 llm.extract.end count=%s facts=%s records=%s",
                        len(fact_candidates) + len(record_candidates),
                        len(fact_candidates),
                        len(record_candidates))

            approved, judge_conf = True, 1.0
            if (fact_candidates or record_candidates) and verify_enabled:
                payload = {"facts": fact_candidates, "records": record_candidates}
                log_kv("🧠 llm.judge.begin")
                approved, judge_conf, why = await llm_judge_facts(chat_id, text, payload)
                log_kv("🧠 llm.judge.end", approved=approved, conf=judge_conf, why=why)

            if (fact_candidates or record_candidates) and approved and judge_conf >= min_conf:
                persisted_keys = []
                for f in fact_candidates:
                    if await upsert_user_fact(
                        sender_id,
                        key=f["key"],
                        value=f["value"],
                        value_type=f.get("type", "text"),
                        confidence=judge_conf
                    ):
                        facts_persisted += 1
                        persisted_keys.append(f["key"])

                for r in record_candidates:
                    if await insert_user_record(sender_id, record_type="holding", data=r, confidence=judge_conf):
                        records_persisted += 1

                if facts_persisted:
                    log_kv("💾 facts.persisted_open", count=facts_persisted, keys=",".join(sorted(set(persisted_keys))))
                if records_persisted:
                    log_kv("💾 records.persisted_open", count=records_persisted, type="holding")

        summary = await classify_and_persist(chat_id, sender_id, text)
        log_kv("💾 process.persisted", facts=summary.get("facts"), records=summary.get("records"))

        if chroma_enabled() and (facts_persisted or records_persisted or summary.get("facts")):
            try:
                profile_text = await build_profile_snapshot_text(sender_id)
                if profile_text:
                    await add_profile_snapshot(chat_id=chat_id, sender_id=sender_id, text=profile_text)
            except Exception:
                pass

        if chroma_enabled():
            try:
                if (text and len(text.strip()) > 60) or (facts_persisted or records_persisted):
                    await chroma_add_text(chat_id=chat_id, sender_id=sender_id, text=text)
                    log_kv("📚 chroma_add", id="auto", sender_id=sender_id, chat_id=chat_id)
            except Exception:
                pass

        intent  = detect_intent(text or "")
        context = await assemble_context(intent, sender_id, chat_id, text or "")

        if intent == "weather" and not context:
            ask = "To share a forecast, I’ll need your city or ZIP/postal code."
            sent = await send_message(chat_id, ask)
            log_kv("📤 process.reply", sent=sent, reply_len=len(ask), preview=ask)
            return

        if intent == "self_profile" and not context:
            msg = "I don’t have your location saved yet. You can tell me, e.g., “my city is Hyderabad, India 500083”."
            sent = await send_message(chat_id, msg)
            log_kv("📤 process.reply", sent=sent, reply_len=len(msg), preview=msg)
            return

        # System style
        minimal_emoji = os.getenv("EMOJI_POLICY","normal").lower() in ("none","minimal")
        style_lines = [
            f"You are {config.BOT_PERSONA_NAME}, a helpful WhatsApp assistant.",
            "Write naturally in short paragraphs (1–3 sentences), not lists.",
            "Use lists ONLY if the user asks for a list, steps, bullets, or a summary.",
            "When referencing saved user details, ALWAYS speak in second person (e.g., “your name is …”); never claim them as your own.",
            "NEVER address the user by a command prefix (e.g., 'Spock'); if the user's name is known, use that; otherwise, no salutation.",
            "Avoid markdown bold/italics. Keep punctuation and spacing clean.",
            "Do not fabricate live metrics; if no live tool is available, be general and invite a follow-up.",
            "Answer ONLY what the user asked; no unrelated info."
        ]
        if not minimal_emoji:
            style_lines.insert(2, "A few tasteful emoji are OK, but only when they add clarity.")
        sys = "\n".join(style_lines) + "\n"

        def _full_prompt():
            return f"{text}\n\nCONTEXT:\n{context}" if context else text

        # ---- LLM reply + resilient retry ----
        logger.info("🧠 llm.reply.begin intent=%s model=%s", intent, getattr(config, "LLM_MODEL", "unknown"))
        reply, ok, meta = await groq_chat(chat_id, sys, _full_prompt())

        if not ok or not reply:
            # Retry once with trimmed, high-signal prompt
            skinny_prompt = _trim_context_for_retry(intent, text, context)
            await asyncio.sleep(0.25)
            reply2, ok2, meta2 = await groq_chat(chat_id, sys, skinny_prompt)
            if ok2 and reply2:
                reply, ok, meta = reply2, ok2, meta2

        used_model = (meta or {}).get("model") if isinstance(meta, dict) else None
        tools_used = (meta or {}).get("tools", {})
        search_used = bool(tools_used.get("search_used")) if isinstance(tools_used, dict) else False
        usage = (meta or {}).get("usage", {})
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        log_kv("🧠 llm.reply.end", ok=ok, model=(used_model or getattr(config, "LLM_MODEL", "unknown")), search_used=search_used, tokens=total_tokens)

        if not ok or not reply:
            reply = "Sorry, try again later."

        reply = await _sanitize_vocative(reply, sender_id)

        sent = await send_message(chat_id, reply)
        log_kv("📤 process.reply", sent=sent, reply_len=len(reply or ""), preview=(reply or "")[:300])

    finally:
        stop_evt.set()
        try: await keepalive_task
        except Exception: pass
        log_kv("💬 process.end", chat_id=chat_id)

# ---- Webhook (unchanged) ----
def normalize_lid_or_chat(body: dict) -> Tuple[Optional[str], Optional[str], str, dict]:
    root = body.get("payload") or body.get("data") or {}
    data_obj = root.get("_data") or {}
    key = data_obj.get("key") or {}
    me  = body.get("me") or {}

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
    to_field   = root.get("to")
    participant = root.get("participant") or data_obj.get("author")
    sender_obj  = root.get("sender") or {}
    sender_id_raw = sender_obj.get("id") or participant or from_field or remote_jid

    chat_id_raw = None
    if key_remote: chat_id_raw = key_remote
    elif remote_jid: chat_id_raw = remote_jid
    elif from_me: chat_id_raw = to_field or from_field
    else: chat_id_raw = from_field or to_field

    if key_remote and me_lid_lit and key_remote == me_lid_lit and me_id:
        chat_id = me_id
    else:
        chat_id = normalize_jid(chat_id_raw)

    sender_id = normalize_jid(sender_id_raw)

    dbg = {
        "text": text[:160],
        "from_me": from_me,
        "key_remote": key_remote,
        "remote_jid": remote_jid,
        "from_field": from_field,
        "to_field": to_field,
        "participant": participant,
        "sender_id_raw": sender_id_raw,
        "chat_id_raw": chat_id_raw,
        "me_id": me_id,
        "me_lid": me_lid_lit,
    }
    return sender_id, chat_id, text, dbg

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
        logger.debug("webhook.ignored evt=%s", evt)
        return JSONResponse({"status": "ok", "message": f"ignored {evt}"})

    sender_id, chat_id, text, dbg = normalize_lid_or_chat(body)
    if not chat_id:
        log_kv("webhook.no_chat", reason="cannot resolve chat_id")
        return JSONResponse({"status": "ok", "message": "no chat id"})

    if looks_channel(chat_id):
        logger.debug("webhook.channel_ignored chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "channel ignored"})

    if not group_allowed(chat_id):
        logger.debug("webhook.blocked_group chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "group not allowed"})

    event_id = (body.get("payload") or {}).get("id") or body.get("id") or ""
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)

    is_echo = False
    if event_id and str(event_id) in OUTBOUND_CACHE_IDS: is_echo = True
    if chat_id and text and (
        hashlib.sha1(f"{chat_id}\n{canonical_text(text)}".encode("utf-8")).hexdigest()
        in OUTBOUND_CACHE_TXT
    ):
        is_echo = True

    if chat_id and text is not None and not is_echo:
        await save_message(chat_id, sender_id or chat_id, "user", text or "", event_id=event_id)

    if is_echo:
        logger.debug("webhook.echo_ignored")
        return JSONResponse({"status": "ok", "message": "echo ignored"})

    if (looks_group(chat_id) or looks_broadcast(chat_id)) and not has_prefix(text):
        logger.debug("webhook.no_group_prefix chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "no group prefix"})

    allow_nlp = _as_bool(getattr(config, "ALLOW_NLP_WITHOUT_PREFIX", True), True)
    if (not is_groupish(chat_id)) and (not allow_nlp) and (not has_prefix(text)):
        logger.debug("webhook.no_dm_prefix chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "no dm prefix"})

    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < getattr(config, "MESSAGE_DEBOUNCE_MS", 800):
        logger.debug("webhook.debounced chat_id=%s", chat_id)
        return JSONResponse({"status": "ok", "message": "debounced"})

    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=getattr(config, "LLM_MAX_QUEUE_PER_CHAT", 3))
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            logger.debug("worker.spawned chat_id=%s", chat_id)
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
        _ = await send_message(chat_id, "I'm busy; try again in a few seconds.")
        log_kv("webhook.queue_timeout", chat_id=chat_id)
        return JSONResponse({"status": "ok", "message": "queue timeout"})

    logger.debug("webhook.enqueued chat_id=%s", chat_id)
    return JSONResponse({"status": "ok", "message": "enqueued"})

@app.get("/healthz")
async def health():
    return {"status": "ok"}

