# app/config.py
from __future__ import annotations
import os
from dotenv import load_dotenv

# Load .env if present
for _p in ('.env', '.env.local'):
    if os.path.exists(_p):
        load_dotenv(_p)

get = os.getenv

# ---- Core settings ----
WAHA_API_URL: str = (get('WAHA_API_URL','http://localhost:3000/api').rstrip('/'))
WAHA_API_KEY: str = get('WAHA_API_KEY','')
WAHA_SESSION: str = get('WAHA_SESSION','default')
WHATSAPP_JID: str = (get('WHATSAPP_JID','') or '').strip()

BOT_PERSONA_NAME: str = get('BOT_PERSONA_NAME','Shimmi')
BOT_COMMAND_PREFIX: list[str] = [p.strip() for p in (get('BOT_COMMAND_PREFIX','@shimmi,shimmi').split(',')) if p.strip()]
ALLOW_NLP_WITHOUT_PREFIX: bool = get('ALLOW_NLP_WITHOUT_PREFIX','false').lower() == 'true'
BOT_EMOJI: str = get('BOT_EMOJI','🤖')
BOT_EMOJI_PREFIX_ENABLED: bool = get('BOT_EMOJI_PREFIX_ENABLED','1') == '1'

ALLOWED_GROUP_JIDS: set[str] = {j.strip() for j in (get('ALLOWED_GROUP_JIDS','').split(',')) if j.strip()}

# LLMs
GROQ_API_KEY: str = (get('GROQ_API_KEY','') or '').strip()
GROQ_MODEL_POOL: list[str] = [m.strip() for m in get('GROQ_MODEL_POOL', get('GROQ_MODEL','llama-3.3-70b-versatile')).split(',') if m.strip()]
GROQ_TIMEOUT: int = int(get('GROQ_TIMEOUT','60'))

GEMINI_API_KEY: str = (get('GEMINI_API_KEY','') or '').strip()
GEMINI_SUMMARY_API_KEY: str = (get('GEMINI_SUMMARY_API_KEY','') or '').strip()
GEMINI_MODEL_POOL: list[str] = [m.strip() for m in get('GEMINI_MODEL_POOL', get('GEMINI_MODEL','gemini-2.0-flash')).split(',') if m.strip()]
GEMINI_TIMEOUT: int = int(get('GEMINI_TIMEOUT','60'))

# DB & Memory
DB_FILE: str = get('DB_FILE','bot_memory.db')
CHROMA_ENABLED: bool = get('CHROMA_ENABLED','1') == '1'
CHROMA_PATH: str = get('CHROMA_PATH','./chroma_data')
EMBEDDING_MODEL: str = get('EMBEDDING_MODEL','all-MiniLM-L6-v2')

# Context / RAG
MAX_RECENT_HISTORY: int = int(get('MAX_RECENT_HISTORY','6'))
CHROMA_N_RESULTS: int = int(get('CHROMA_N_RESULTS','3'))
MAX_DOC_CHARS: int = int(get('MAX_DOC_CHARS','300'))
CONTEXT_CHAR_BUDGET: int = int(get('CONTEXT_CHAR_BUDGET','3000'))

# Rates / Queues
LLM_CALLS_PER_MINUTE: int = int(get('LLM_CALLS_PER_MINUTE','10'))
LLM_BURST_COOLDOWN_SEC: int = int(get('LLM_BURST_COOLDOWN_SEC','60'))
LLM_MAX_QUEUE_PER_CHAT: int = int(get('LLM_MAX_QUEUE_PER_CHAT','3'))
LLM_QUEUE_WAIT_SEC: int = int(get('LLM_QUEUE_WAIT_SEC','20'))
PER_CHAT_COOLDOWN_SEC: float = float(get('PER_CHAT_COOLDOWN_SEC','60'))
MESSAGE_DEBOUNCE_MS: int = int(get('MESSAGE_DEBOUNCE_MS','1000'))

# Timezone & log
APP_TIMEZONE: str = get('APP_TIMEZONE','Asia/Kolkata')
LOG_LEVEL: str = get('LOG_LEVEL','INFO').upper()

# Workers
SUMMARY_ENABLED: bool = get('SUMMARY_ENABLED','1') == '1'
SUMMARY_INTERVAL_SEC: int = int(get('SUMMARY_INTERVAL_SEC','3600'))
SUMMARY_WINDOW_MSGS: int = int(get('SUMMARY_WINDOW_MSGS','10'))
SUMMARY_MAX_TOKENS: int = int(get('SUMMARY_MAX_TOKENS','120'))
SUMMARY_IDLE_ONLY: bool = get('SUMMARY_IDLE_ONLY','1') == '1'
SUMMARY_IDLE_THRESHOLD_SEC: float = float(get('SUMMARY_IDLE_THRESHOLD_SEC','900'))

# Salience (kept off by default in this MVP)
SALIENCE_ENABLED: bool = get('SALIENCE_ENABLED','0') == '1'
SALIENCE_INTERVAL_SEC: int = int(get('SALIENCE_INTERVAL_SEC','1800'))
SALIENCE_DECAY_PER_DAY: float = float(get('SALIENCE_DECAY_PER_DAY','0.02'))
SALIENCE_BOOST_ON_REPEAT: float = float(get('SALIENCE_BOOST_ON_REPEAT','0.05'))
SALIENCE_CONF_MIN: float = float(get('SALIENCE_CONF_MIN','0.30'))
SALIENCE_CONF_MAX: float = float(get('SALIENCE_CONF_MAX','0.98'))

# Security
WEBHOOK_SECRET: bytes = (get('WEBHOOK_SECRET','').encode('utf-8') if get('WEBHOOK_SECRET') else b'')

