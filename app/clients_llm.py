from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional, Dict, Tuple, Any

from groq import Groq

from . import config

logger = logging.getLogger('app.llm')

GROQ_CLIENT: Optional[Groq] = None
MODEL_CIRCUIT: Dict[str, float] = {}
STICKY_MODEL: Dict[str, Tuple[str, float]] = {}
STICKY_TTL_SEC = 600
_circuit_lock = asyncio.Lock()
GROQ_INFLIGHT = asyncio.Semaphore(int(config.GROQ_MAX_INFLIGHT or 5))


def _model_open(model: str) -> bool:
    return time.monotonic() >= MODEL_CIRCUIT.get(model, 0.0)


async def _open_circuit(model: str, cooldown: float):
    async with _circuit_lock:
        MODEL_CIRCUIT[model] = time.monotonic() + cooldown


def _sticky_or_healthy(chat_id: str, pool: list[str]) -> str:
    key = f'groq:{chat_id}'
    sticky = STICKY_MODEL.get(key)
    if sticky and (time.monotonic() - sticky[1]) < STICKY_TTL_SEC and _model_open(sticky[0]):
        return sticky[0]
    for m in pool:
        if _model_open(m):
            STICKY_MODEL[key] = (m, time.monotonic())
            return m
    STICKY_MODEL[key] = (pool[0], time.monotonic())
    return pool[0]


async def init_clients():
    global GROQ_CLIENT
    if GROQ_CLIENT:
        return
    if not (config.GROQ_API_KEY or '').strip():
        logger.warning('🧠 llm.init groq_enabled=0 (missing GROQ_API_KEY)')
        return
    GROQ_CLIENT = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT)
    logger.info('🧠 llm.init groq_enabled=1 models=%s', ','.join(config.GROQ_MODEL_POOL))


async def close_clients():
    pass


def _usage_dict(resp) -> dict:
    try:
        u = getattr(resp, 'usage', None)
        if not u:
            return {}
        if hasattr(u, 'model_dump'):
            return u.model_dump()
        if isinstance(u, dict):
            return u
    except Exception:
        pass
    return {}


async def groq_chat(chat_id: str, *, system: str, user: str, temperature: float = 0.3, max_tokens: int = 700) -> tuple[str, bool, dict, str]:
    if not GROQ_CLIENT:
        return 'Sorry, AI unavailable.', False, {}, ''

    model = _sticky_or_healthy(chat_id, list(config.GROQ_MODEL_POOL))
    payload = {
        'model': model,
        'messages': [{'role':'system','content':system},{'role':'user','content':user}],
        'temperature': float(max(0.0, min(temperature, 1.5))),
        'max_tokens': int(max_tokens),
    }

    async with GROQ_INFLIGHT:
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            txt = (resp.choices[0].message.content or '').strip()
            meta = {'ms': int((time.perf_counter()-t0)*1000), 'usage': _usage_dict(resp)}
            return txt, True, meta, model
        except Exception as e:
            await _open_circuit(model, 12.0 + random.random()*3.0)
            meta = {'ms': int((time.perf_counter()-t0)*1000), 'err': str(e)[:180]}
            return '', False, meta, model


async def groq_live_search(chat_id: str, *, user: str, max_tokens: int = 900) -> tuple[str, bool, dict, str]:
    if not GROQ_CLIENT:
        return 'Sorry, AI unavailable.', False, {}, ''

    model = (config.LIVE_SEARCH_MODEL or 'groq/compound-mini').strip()
    payload: Dict[str, Any] = {
        'model': model,
        'messages': [
            {'role':'system','content':'Use web search to answer with up-to-date information. Include source links when relevant.'},
            {'role':'user','content':user},
        ],
        'temperature': 0.2,
        'max_tokens': int(max_tokens),
        'compound_custom': {'tools': {'enabled_tools': ['web_search']}},
    }

    async with GROQ_INFLIGHT:
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(lambda: GROQ_CLIENT.chat.completions.create(**payload))
            txt = (resp.choices[0].message.content or '').strip()
            meta = {'ms': int((time.perf_counter()-t0)*1000), 'usage': _usage_dict(resp)}
            try:
                et = getattr(resp.choices[0].message, 'executed_tools', None)
                meta['executed_tools_n'] = len(et) if et else 0
            except Exception:
                meta['executed_tools_n'] = 0
            return txt, True, meta, model
        except Exception as e:
            meta = {'ms': int((time.perf_counter()-t0)*1000), 'err': str(e)[:180]}
            return '', False, meta, model
