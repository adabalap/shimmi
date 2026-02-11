from __future__ import annotations

"""Runtime configuration.

This module loads environment variables from a .env file *before* reading them.
This is required when deployments store configuration only in /opt/shimmi/.env.

Load order / precedence:
1) Already-set process environment (systemd, docker, etc.)
2) Values loaded from .env file
3) Defaults

"""

from .env_loader import load_env
load_env(override=False)

import importlib
import os
from typing import List


def _is_set(v) -> bool:
    return v is not None and not (isinstance(v, str) and v.strip() == "")


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


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
    parts = [p.strip() for p in str(v).split(",")]
    return [p for p in parts if p]


_root = None
try:
    _root = importlib.import_module("config")
except Exception:
    _root = None


def _get(name: str, default=None):
    if _root is not None and hasattr(_root, name):
        v = getattr(_root, name)
        if _is_set(v):
            return v
    return default


# --- Logging ---
LOG_LEVEL = _get("LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
LOG_FORMAT = _get("LOG_FORMAT", os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s:%(message)s"))
ACCESS_LOG_LEVEL = _get("ACCESS_LOG_LEVEL", os.getenv("ACCESS_LOG_LEVEL", "WARNING"))

# --- App basics ---
APP_TIMEZONE = _get("APP_TIMEZONE", os.getenv("APP_TIMEZONE", "UTC"))
DB_FILE = _get("DB_FILE", os.getenv("DB_FILE", os.getenv("SQLITE_PATH", "bot_memory.db")))

BOT_PERSONA_NAME = _get("BOT_PERSONA_NAME", os.getenv("BOT_PERSONA_NAME", "Stateful AI BOT"))
BOT_COMMAND_PREFIX = _get("BOT_COMMAND_PREFIX", os.getenv("BOT_COMMAND_PREFIX", "spock"))

# --- WAHA ---
WAHA_API_URL = _get("WAHA_API_URL", os.getenv("WAHA_API_URL", ""))
WAHA_SESSION = _get("WAHA_SESSION", os.getenv("WAHA_SESSION", "default"))
WAHA_API_KEY = _get("WAHA_API_KEY", os.getenv("WAHA_API_KEY", ""))
WEBHOOK_SECRET = _get("WEBHOOK_SECRET", os.getenv("WEBHOOK_SECRET", ""))

# --- LLM (Groq) ---
GROQ_API_KEY = _get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
GROQ_TIMEOUT = _get("GROQ_TIMEOUT", _float(os.getenv("GROQ_TIMEOUT", "30"), 30.0))
GROQ_MODEL_POOL = _get(
    "GROQ_MODEL_POOL",
    _list(os.getenv("GROQ_MODEL_POOL"), ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
)
GROQ_MAX_INFLIGHT = _get("GROQ_MAX_INFLIGHT", _int(os.getenv("GROQ_MAX_INFLIGHT", "5"), 5))

LIVE_SEARCH_ENABLED = _get("LIVE_SEARCH_ENABLED", _bool(os.getenv("LIVE_SEARCH_ENABLED", "0"), False))
LIVE_SEARCH_MODEL = _get("LIVE_SEARCH_MODEL", os.getenv("LIVE_SEARCH_MODEL", "groq/compound-mini"))

# --- Behavior toggles ---
CHROMA_ENABLED = _get("CHROMA_ENABLED", _bool(os.getenv("CHROMA_ENABLED", "1"), True))
ALLOW_NLP_WITHOUT_PREFIX = _get("ALLOW_NLP_WITHOUT_PREFIX", _bool(os.getenv("ALLOW_NLP_WITHOUT_PREFIX", "1"), True))

MESSAGE_DEBOUNCE_MS = _get("MESSAGE_DEBOUNCE_MS", _int(os.getenv("MESSAGE_DEBOUNCE_MS", "800"), 800))
LLM_MAX_QUEUE_PER_CHAT = _get("LLM_MAX_QUEUE_PER_CHAT", _int(os.getenv("LLM_MAX_QUEUE_PER_CHAT", "3"), 3))
LLM_QUEUE_WAIT_SEC = _get("LLM_QUEUE_WAIT_SEC", _int(os.getenv("LLM_QUEUE_WAIT_SEC", "20"), 20))

# --- Ambient observe defaults ---
OBSERVE_GROUPS_DEFAULT = _get("OBSERVE_GROUPS_DEFAULT", os.getenv("OBSERVE_GROUPS_DEFAULT", "off"))
OBSERVE_RETENTION_DEFAULT_DAYS = _get("OBSERVE_RETENTION_DEFAULT_DAYS", _int(os.getenv("OBSERVE_RETENTION_DEFAULT_DAYS", "30"), 30))
OBSERVE_REDACTION_DEFAULT = _get("OBSERVE_REDACTION_DEFAULT", os.getenv("OBSERVE_REDACTION_DEFAULT", "on"))
OBSERVE_MIN_TEXT_LEN = _get("OBSERVE_MIN_TEXT_LEN", _int(os.getenv("OBSERVE_MIN_TEXT_LEN", "60"), 60))
OBSERVE_MAX_EMBED_PER_MIN = _get("OBSERVE_MAX_EMBED_PER_MIN", _int(os.getenv("OBSERVE_MAX_EMBED_PER_MIN", "10"), 10))

# --- Group allowlist ---
ALLOWED_GROUP_JIDS = _get("ALLOWED_GROUP_JIDS", None)
if isinstance(ALLOWED_GROUP_JIDS, str):
    ALLOWED_GROUP_JIDS = [s.strip() for s in ALLOWED_GROUP_JIDS.split(",") if s.strip()]
