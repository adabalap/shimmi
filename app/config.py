from __future__ import annotations

from .env_loader import load_env
load_env(override=False)

import os
from typing import List


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ('1','true','yes','on')


def _int(v: str | None, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _float(v: str | None, default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _list(v: str | None, default: List[str]) -> List[str]:
    if not v:
        return list(default)
    return [p.strip() for p in str(v).split(',') if p.strip()]


LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s %(levelname)s %(name)s:%(message)s')
ACCESS_LOG_LEVEL = os.getenv('ACCESS_LOG_LEVEL', 'WARNING')

APP_TIMEZONE = os.getenv('APP_TIMEZONE', 'UTC')
DB_FILE = os.getenv('DB_FILE', os.getenv('SQLITE_PATH', 'bot_memory.db'))

BOT_PERSONA_NAME = os.getenv('BOT_PERSONA_NAME', 'Stateful AI BOT')
BOT_COMMAND_PREFIX = os.getenv('BOT_COMMAND_PREFIX', 'spock')

WAHA_API_URL = os.getenv('WAHA_API_URL', '')
WAHA_SESSION = os.getenv('WAHA_SESSION', 'default')
WAHA_API_KEY = os.getenv('WAHA_API_KEY', '')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_TIMEOUT = _float(os.getenv('GROQ_TIMEOUT', '30'), 30.0)
GROQ_MODEL_POOL = _list(os.getenv('GROQ_MODEL_POOL'), ['llama-3.3-70b-versatile','llama-3.1-8b-instant'])
GROQ_MAX_INFLIGHT = _int(os.getenv('GROQ_MAX_INFLIGHT', '5'), 5)

LIVE_SEARCH_ENABLED = _bool(os.getenv('LIVE_SEARCH_ENABLED', '0'), False)
LIVE_SEARCH_MODEL = os.getenv('LIVE_SEARCH_MODEL', 'groq/compound-mini')

CHROMA_ENABLED = _bool(os.getenv('CHROMA_ENABLED', '1'), True)
ALLOW_NLP_WITHOUT_PREFIX = _bool(os.getenv('ALLOW_NLP_WITHOUT_PREFIX', '1'), True)

MESSAGE_DEBOUNCE_MS = _int(os.getenv('MESSAGE_DEBOUNCE_MS', '800'), 800)
LLM_MAX_QUEUE_PER_CHAT = _int(os.getenv('LLM_MAX_QUEUE_PER_CHAT', '3'), 3)
LLM_QUEUE_WAIT_SEC = _int(os.getenv('LLM_QUEUE_WAIT_SEC', '20'), 20)

FACTS_EXTRACTION_MODE = os.getenv('FACTS_EXTRACTION_MODE', 'hybrid')
FACTS_MIN_CONF = _float(os.getenv('FACTS_MIN_CONF', '0.85'), 0.85)

OBSERVE_GROUPS_DEFAULT = os.getenv('OBSERVE_GROUPS_DEFAULT', 'off')
OBSERVE_DMS_DEFAULT = os.getenv('OBSERVE_DMS_DEFAULT', 'on')  # off|on|topics
OBSERVE_RETENTION_DEFAULT_DAYS = _int(os.getenv('OBSERVE_RETENTION_DEFAULT_DAYS', '30'), 30)
OBSERVE_REDACTION_DEFAULT = os.getenv('OBSERVE_REDACTION_DEFAULT', 'on')
OBSERVE_MIN_TEXT_LEN = _int(os.getenv('OBSERVE_MIN_TEXT_LEN', '60'), 60)
OBSERVE_MAX_EMBED_PER_MIN = _int(os.getenv('OBSERVE_MAX_EMBED_PER_MIN', '10'), 10)

ALLOWED_GROUP_JIDS = _list(os.getenv('ALLOWED_GROUP_JIDS'), [])
