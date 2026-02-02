# app/clients_llm.py
from __future__ import annotations
import httpx, time, asyncio
from typing import Optional, List
from . import config
from .db import inc_usage

HTTPX_GROQ: Optional[httpx.AsyncClient] = None
HTTPX_GEMINI: Optional[httpx.AsyncClient] = None
HTTPX_GEMINI_SUMMARY: Optional[httpx.AsyncClient] = None

MODEL_CIRCUIT: dict[str, float] = {}
STICKY_MODEL: dict[str, tuple[str, float]] = {}
STICKY_TTL_SEC = 600

GEMINI_LAST_CALL = 0.0
GEMINI_MIN_INTERVAL = 3.0

async def init_clients():
    global HTTPX_GROQ, HTTPX_GEMINI, HTTPX_GEMINI_SUMMARY
    HTTPX_GROQ = httpx.AsyncClient(timeout=config.GROQ_TIMEOUT, headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"} if config.GROQ_API_KEY else None)
    HTTPX_GEMINI = httpx.AsyncClient(timeout=config.GEMINI_TIMEOUT)
    HTTPX_GEMINI_SUMMARY = httpx.AsyncClient(timeout=config.GEMINI_TIMEOUT)

async def close_clients():
    for c in (HTTPX_GROQ, HTTPX_GEMINI, HTTPX_GEMINI_SUMMARY):
        try:
            if c: await c.aclose()
        except Exception:
            pass

def _model_open(model: str) -> bool:
    return time.perf_counter() >= MODEL_CIRCUIT.get(model, 0.0)

def _open_circuit(model: str, cooldown: float):
    MODEL_CIRCUIT[model] = time.perf_counter() + cooldown

def _sticky_or_healthy(chat_id: str, pool: list[str]) -> str:
    key = f"groq:{chat_id}"
    sticky = STICKY_MODEL.get(key)
    if sticky and (time.perf_counter() - sticky[1]) < STICKY_TTL_SEC and _model_open(sticky[0]):
        return sticky[0]
    for m in pool:
        if _model_open(m):
            STICKY_MODEL[key] = (m, time.perf_counter())
            return m
    STICKY_MODEL[key] = (pool[0], time.perf_counter())
    return pool[0]

async def groq_chat(chat_id: str, system_text: str, user_text: str) -> tuple[str, bool, dict]:
    if not HTTPX_GROQ or not config.GROQ_API_KEY:
        return "Sorry, AI unavailable.", False, {}
    model = _sticky_or_healthy(chat_id, config.GROQ_MODEL_POOL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.6,
        "max_tokens": 800
    }
    try:
        resp = await HTTPX_GROQ.post("https://api.groq.com/openai/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = (data.get('choices') or [{}])[0].get('message') or {}
        txt = (msg.get('content') or '').strip()
        usage = data.get('usage') or {}
        await inc_usage(model, usage.get('total_tokens') or 0)
        return txt, True, data
    except httpx.HTTPStatusError as e:
        code = e.response.status_code if e.response else None
        if code in (429, 500, 502, 503):
            _open_circuit(model, 120.0)
        return "", False, {}
    except Exception:
        return "", False, {}

# Gemini is mainly used by the summary worker; it’s optional to the core Q&A.
async def gemini_chat(chat_id: str, system_text: str, prompt_text: str, use_summary_key: bool = False) -> tuple[str, bool, dict]:
    client = HTTPX_GEMINI_SUMMARY if use_summary_key and config.GEMINI_SUMMARY_API_KEY else HTTPX_GEMINI
    api_key = config.GEMINI_SUMMARY_API_KEY if use_summary_key and config.GEMINI_SUMMARY_API_KEY else config.GEMINI_API_KEY
    if not client or not api_key:
        return "", False, {}

    # simple throttle
    global GEMINI_LAST_CALL
    delay = max(0.0, GEMINI_MIN_INTERVAL - (time.perf_counter() - GEMINI_LAST_CALL))
    if delay > 0:
        await asyncio.sleep(delay)
    GEMINI_LAST_CALL = time.perf_counter()

    model = config.GEMINI_MODEL_POOL[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "systemInstruction": {"parts": [{"text": system_text}]},
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.6, "topP": 0.9}
    }
    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        cand = (data.get('candidates') or [{}])[0]
        part = (cand.get('content') or {}).get('parts', [{}])[0]
        txt = (part.get('text') or '').strip()
        return txt, True, data
    except Exception:
        return "", False, {}

