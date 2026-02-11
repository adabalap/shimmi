from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logging_setup import setup_logging, log_startup_env, log_event
from .utils import verify_signature, canonical_text
from . import config

from .db import init_db, close_db, save_message, execute, get_chat_prefs, set_chat_prefs
from .memory import get_recent_messages, get_user_facts, upsert_user_fact, clear_user_memory, get_messages_in_window, day_window_iso
from .memory_extractor import extract_facts
from .prompting import format_history, format_facts, build_system, build_user_prompt
from .response_guard import sanitize_reply
from .clients_waha import init_client as init_waha, close_client as close_waha, send_message, typing_keepalive, OUTBOUND_CACHE_IDS, OUTBOUND_CACHE_TXT, OUTBOUND_TTL_SEC
from .clients_llm import init_clients as init_llm, close_clients as close_llm, groq_chat, groq_live_search
from .observe import observe_ingest_group, observe_ingest_dm, handle_observe_command

setup_logging()
logger = logging.getLogger('app')

CHROMA_AVAILABLE = True
try:
    from .chroma import add_text as chroma_add_text, query as chroma_query, warmup as chroma_warmup
except Exception:
    CHROMA_AVAILABLE = False


def chroma_enabled() -> bool:
    return CHROMA_AVAILABLE and bool(config.CHROMA_ENABLED)


def looks_group(j: Optional[str]) -> bool:
    return bool(j and j.endswith('@g.us'))


def looks_broadcast(j: Optional[str]) -> bool:
    return bool(j and j.endswith('@broadcast'))


def looks_channel(j: Optional[str]) -> bool:
    return bool(j and j.endswith('@newsletter'))


def is_groupish(j: Optional[str]) -> bool:
    return looks_group(j) or looks_broadcast(j) or looks_channel(j)


def group_allowed(chat_id: Optional[str]) -> bool:
    if not chat_id:
        return False
    if looks_channel(chat_id):
        return False
    if not looks_group(chat_id):
        return True
    if not config.ALLOWED_GROUP_JIDS:
        return True
    return chat_id in config.ALLOWED_GROUP_JIDS


def normalize_jid(jid: Optional[str]) -> Optional[str]:
    if not jid:
        return None
    if jid.endswith('@s.whatsapp.net'):
        return jid.replace('@s.whatsapp.net', '@c.us')
    return jid


_PREFIX_RE: Optional[re.Pattern] = None

def _compile_prefix_re():
    global _PREFIX_RE
    alts = [re.escape(p.strip().lstrip('@')) for p in (config.BOT_COMMAND_PREFIX or '').split(',') if p.strip()]
    _PREFIX_RE = re.compile(r'(?i)(?:^|\s)@?(%s)\b' % '|'.join(alts)) if alts else re.compile(r'a^')


def has_prefix(text: Optional[str]) -> bool:
    if not text:
        return False
    if _PREFIX_RE is None:
        _compile_prefix_re()
    return bool(_PREFIX_RE.search(text))


def strip_prefix(text: str) -> str:
    if _PREFIX_RE is None:
        _compile_prefix_re()
    s = text or ''
    m = _PREFIX_RE.search(s)
    if not m:
        return s.strip()
    start, end = m.span()
    before = s[:start].strip()
    after = s[end:].lstrip(' ,:;-\t')
    if not before:
        return after.strip()
    return re.sub(r'\s+', ' ', (before + ' ' + after).strip())


def needs_live_search(user_text: str) -> bool:
    t = (user_text or '').lower()
    return any(k in t for k in ('latest','current','news','updates','score','price','weather','trending'))


def is_today_recap(user_text: str) -> bool:
    t = (user_text or '').lower().strip()
    return any(p in t for p in ('what did i do today','recap my day','today recap','summarize my day','what happened today'))


def normalize_payload(body: dict) -> Tuple[Optional[str], Optional[str], str, bool, str, Optional[str]]:
    root = body.get('payload') or body.get('data') or {}
    data_obj = root.get('_data') or {}
    key = data_obj.get('key') or {}

    text = (
        root.get('body')
        or (root.get('message') or {}).get('text')
        or (root.get('message') or {}).get('conversation')
        or data_obj.get('body')
        or ''
    )

    from_me = bool(root.get('fromMe') or root.get('from_me') or False)

    me_obj = body.get('me') or root.get('me') or {}
    me_id = me_obj.get('id') if isinstance(me_obj, dict) else None

    remote_jid = (key.get('remoteJid') or root.get('remoteJid') or root.get('chatId') or root.get('chat_id') or root.get('from') or root.get('to'))
    participant = root.get('participant') or data_obj.get('author')
    sender_obj = root.get('sender') or {}
    sender_id_raw = sender_obj.get('id') or participant or root.get('from') or remote_jid

    event_id = (root.get('id') or body.get('id') or data_obj.get('id') or '')

    return normalize_jid(sender_id_raw), normalize_jid(remote_jid), text, from_me, str(event_id or ''), normalize_jid(me_id)


def _purge_echo_cache():
    nowt = time.time()
    for k, ts in list(OUTBOUND_CACHE_IDS.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_IDS.pop(k, None)
    for k, ts in list(OUTBOUND_CACHE_TXT.items()):
        if nowt - ts > OUTBOUND_TTL_SEC:
            OUTBOUND_CACHE_TXT.pop(k, None)


def is_echo(chat_id: str, text: str, event_id: str, *, from_me: bool, sender_id: Optional[str], me_id: Optional[str]) -> bool:
    _purge_echo_cache()
    if event_id and event_id in OUTBOUND_CACHE_IDS:
        return True
    if chat_id and text:
        h = hashlib.sha1(f'{chat_id}\n{canonical_text(text)}'.encode('utf-8')).hexdigest()
        if h in OUTBOUND_CACHE_TXT:
            return True
    if from_me and sender_id and me_id and sender_id == me_id:
        return True
    return False


async def build_today_recap(chat_id: str) -> str:
    start_iso, end_iso = day_window_iso(tz_name=config.APP_TIMEZONE)
    msgs = await get_messages_in_window(chat_id, start_iso=start_iso, end_iso=end_iso, role='user', limit=200)
    if not msgs:
        return "I don't have any saved messages from today yet."
    lines = [m[2] for m in msgs if m[2]]
    if len(lines) <= 8:
        return 'Today, you mentioned:\n' + '\n'.join([f'• {l}' for l in lines])
    convo = '\n'.join([f'USER: {l}' for l in lines])
    sys = 'Summarize ONLY the provided USER lines into 3-6 bullets. Do not add anything else.'
    reply, ok, meta, model = await groq_chat(chat_id, system=sys, user=convo, temperature=0.0, max_tokens=250)
    return reply.strip() if ok and reply else ('Today, you mentioned:\n' + '\n'.join([f'• {l}' for l in lines[:10]]))


CHAT_QUEUES: Dict[str, asyncio.Queue] = {}
CHAT_WORKERS: Dict[str, asyncio.Task] = {}
CHAT_LAST_MSG_TS: Dict[str, float] = {}

app = FastAPI()


async def process_message(chat_id: str, sender_id: str, raw_text: str, inbound_id: Optional[int]):
    stop_evt = asyncio.Event()
    keepalive_task = asyncio.create_task(typing_keepalive(chat_id, stop_evt))
    try:
        user_text = strip_prefix((raw_text or '').strip())
        log_event(logger, '💬 process.begin', chat_id=chat_id, sender_id=sender_id, text=user_text[:160])

        tl = user_text.lower().strip()
        if tl in ('/forget me','forget me'):
            await clear_user_memory(sender_id)
            out = '✅ Done. I have cleared your saved facts.'
            ok, out_id = await send_message(chat_id, out)
            log_event(logger, '📤 send.forget', chat_id=chat_id, ok=ok, msg_id=out_id or '')
            await save_message(chat_id, sender_id, 'assistant', out, event_id=out_id or '')
            return

        if is_today_recap(user_text):
            out = await build_today_recap(chat_id)
            ok, out_id = await send_message(chat_id, out)
            log_event(logger, '📤 send.today', chat_id=chat_id, ok=ok, msg_id=out_id or '')
            await save_message(chat_id, sender_id, 'assistant', out, event_id=out_id or '')
            return

        history = await get_recent_messages(chat_id, limit=12)
        facts = await get_user_facts(sender_id, namespace='default')
        hist_block = format_history(history, max_turns=10)
        facts_block = format_facts([(k, v, c) for k, v, c in facts], max_items=35)

        snippets = ''
        if chroma_enabled():
            snippets = await chroma_query(chat_id=chat_id, text=user_text, k=3)

        # Live web search when enabled
        if config.LIVE_SEARCH_ENABLED and needs_live_search(user_text):
            log_event(logger, '🧠 llm.live.start', chat_id=chat_id, model=config.LIVE_SEARCH_MODEL)
            reply, ok, meta, model = await groq_live_search(chat_id, user=user_text)
            log_event(logger, '🧠 llm.live.end', chat_id=chat_id, ok=ok, model=model, ms=meta.get('ms',''))
            if ok and reply:
                reply = sanitize_reply(reply, [(k, v, c) for k, v, c in facts])
                ok2, out_id = await send_message(chat_id, reply)
                log_event(logger, '📤 send.reply', chat_id=chat_id, ok=ok2, msg_id=out_id or '')
                await save_message(chat_id, sender_id, 'assistant', reply, event_id=out_id or '')
                return

        system = build_system(config.BOT_PERSONA_NAME)
        prompt = build_user_prompt(user_text, facts_block=facts_block, snippets=snippets, history_block=hist_block)
        log_event(logger, '🧠 llm.chat.start', chat_id=chat_id)
        reply, ok, meta, model = await groq_chat(chat_id, system=system, user=prompt)
        log_event(logger, '🧠 llm.chat.end', chat_id=chat_id, ok=ok, model=model, ms=meta.get('ms',''))
        if not ok or not reply:
            reply = 'Sorry, I had trouble responding. Please try again.'

        reply = sanitize_reply(reply, [(k, v, c) for k, v, c in facts])
        ok2, out_id = await send_message(chat_id, reply)
        log_event(logger, '📤 send.reply', chat_id=chat_id, ok=ok2, msg_id=out_id or '')
        await save_message(chat_id, sender_id, 'assistant', reply, event_id=out_id or '')

        # Fact extraction
        if (config.FACTS_EXTRACTION_MODE or 'off').lower() != 'off':
            extracted = await extract_facts(chat_id, user_text)
            persisted = []
            for k, v, c in extracted:
                if c < float(config.FACTS_MIN_CONF):
                    continue
                changed = await upsert_user_fact(sender_id, k, v, confidence=c, source_msg_id=inbound_id)
                if changed:
                    persisted.append(k)
            log_event(logger, '🧠 memory.persist', chat_id=chat_id, n=len(persisted), keys=','.join(persisted)[:160])

    finally:
        stop_evt.set()
        try:
            await keepalive_task
        except Exception:
            pass
        log_event(logger, '💬 process.end', chat_id=chat_id)


@app.on_event('startup')
async def startup():
    _compile_prefix_re()
    await init_db()
    await init_waha()
    await init_llm()

    env_keys = [
        'BOT_PERSONA_NAME','BOT_COMMAND_PREFIX','CHROMA_ENABLED','OBSERVE_DMS_DEFAULT',
        'LOG_LEVEL','LOG_FORMAT','ACCESS_LOG_LEVEL',
        'WAHA_API_URL','WAHA_SESSION',
        'LIVE_SEARCH_ENABLED','LIVE_SEARCH_MODEL',
        'FACTS_EXTRACTION_MODE','FACTS_MIN_CONF',
    ]
    log_startup_env(logger, keys=env_keys)

    if chroma_enabled():
        try:
            dim = await chroma_warmup()
            logger.info('📚 chroma.warmup dim=%s', dim)
        except Exception:
            pass


@app.on_event('shutdown')
async def shutdown():
    for _, t in list(CHAT_WORKERS.items()):
        try:
            t.cancel()
        except Exception:
            pass
    await close_waha()
    await close_llm()
    await close_db()
    logger.info('🧹 shutdown.complete')


@app.post('/webhook')
async def webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get('X-WAHA-HMAC') or request.headers.get('X-Webhook-Signature') or request.headers.get('X-Signature')
    if not verify_signature(raw, sig):
        return JSONResponse({'status':'error','message':'Invalid signature'}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'status':'error','message':'Invalid JSON'}, status_code=400)

    evt = (body.get('event') or '').lower()
    if evt == 'message':
        return JSONResponse({'status':'ok','message':'ignored'})

    sender_id, chat_id, text, from_me, event_id, me_id = normalize_payload(body)

    # SILENT ignores (channels + non-allowed groups)
    if looks_channel(chat_id):
        return JSONResponse({'status':'ok','message':'ignored'})
    if looks_group(chat_id) and not group_allowed(chat_id):
        return JSONResponse({'status':'ok','message':'ignored'})

    log_event(logger, '📩 webhook.recv', evt=evt, chat_id=chat_id, sender_id=sender_id, text=(text or '')[:120])

    if not chat_id:
        return JSONResponse({'status':'ok','message':'no chat id'})

    if is_echo(chat_id, text or '', event_id or '', from_me=from_me, sender_id=sender_id, me_id=me_id):
        log_event(logger, '♻️ webhook.echo_ignored', chat_id=chat_id)
        return JSONResponse({'status':'ok','message':'echo ignored'})

    # Observe command only in groups
    if looks_group(chat_id) and (text or '').strip().lower().startswith('/observe'):
        msg = await handle_observe_command(chat_id, text, get_prefs=get_chat_prefs, set_prefs=set_chat_prefs)
        if msg:
            ok, out_id = await send_message(chat_id, msg)
            log_event(logger, '📤 send.observe', chat_id=chat_id, ok=ok, msg_id=out_id or '')
        return JSONResponse({'status':'ok','message':'observe handled'})

    # Always store inbound message (DM ingestion B depends on this)
    inbound_id = await save_message(chat_id, sender_id or chat_id, 'user', strip_prefix(text or ''), event_id=event_id or '')

    # DM ingestion B: embed passively
    if (not looks_group(chat_id)) and chroma_enabled():
        await observe_ingest_dm(chat_id, sender_id or chat_id, text or '', chroma_add_text=chroma_add_text, db_exec=execute)

    # Group ambient observe if no prefix
    if looks_group(chat_id) and (not has_prefix(text)) and chroma_enabled():
        await observe_ingest_group(chat_id, sender_id or chat_id, text or '', chroma_add_text=chroma_add_text, db_exec=execute, get_prefs=get_chat_prefs)
        log_event(logger, '👁️ webhook.observed', chat_id=chat_id)
        return JSONResponse({'status':'ok','message':'observed'})

    # Group/broadcast reply gating
    if (looks_group(chat_id) or looks_broadcast(chat_id)) and not has_prefix(text):
        return JSONResponse({'status':'ok','message':'no group prefix'})

    # DM reply gating: if strict prefix required, do not reply; still stored+embedded.
    if (not is_groupish(chat_id)) and (not bool(config.ALLOW_NLP_WITHOUT_PREFIX)) and not has_prefix(text):
        return JSONResponse({'status':'ok','message':'stored'})

    # Debounce
    last = CHAT_LAST_MSG_TS.get(chat_id, 0.0)
    nowp = time.perf_counter()
    CHAT_LAST_MSG_TS[chat_id] = nowp
    if (nowp - last) * 1000.0 < int(config.MESSAGE_DEBOUNCE_MS):
        return JSONResponse({'status':'ok','message':'debounced'})

    q = CHAT_QUEUES.get(chat_id)
    if not q:
        q = asyncio.Queue(maxsize=int(config.LLM_MAX_QUEUE_PER_CHAT))
        CHAT_QUEUES[chat_id] = q

        async def _worker():
            log_event(logger, '🧵 worker.spawned', chat_id=chat_id)
            try:
                while True:
                    item = await q.get()
                    try:
                        await process_message(chat_id, item['sender_id'], item['text'], inbound_id=item.get('inbound_id'))
                    finally:
                        q.task_done()
            except asyncio.CancelledError:
                pass

        CHAT_WORKERS[chat_id] = asyncio.create_task(_worker())

    try:
        await asyncio.wait_for(q.put({'text': text or '', 'sender_id': sender_id or chat_id, 'inbound_id': inbound_id}), timeout=int(config.LLM_QUEUE_WAIT_SEC))
    except asyncio.TimeoutError:
        ok, out_id = await send_message(chat_id, "I'm busy; try again in a few seconds.")
        log_event(logger, '⏳ queue.timeout', chat_id=chat_id, ok=ok, msg_id=out_id or '')
        return JSONResponse({'status':'ok','message':'queue timeout'})

    log_event(logger, '✅ webhook.enqueued', chat_id=chat_id)
    return JSONResponse({'status':'ok','message':'enqueued'})


@app.get('/healthz')
async def healthz():
    return {'status':'ok'}
