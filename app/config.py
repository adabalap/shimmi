from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _int(v: str | None, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

@dataclass(frozen=True)
class Settings:
    # Core
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    app_timezone: str = os.getenv("APP_TIMEZONE", "UTC")

    bot_persona_name: str = os.getenv("BOT_PERSONA_NAME", "Shimmi")
    bot_command_prefix: str = os.getenv("BOT_COMMAND_PREFIX", "@shimmi,shimmi")
    allow_nlp_without_prefix: bool = _bool(os.getenv("ALLOW_NLP_WITHOUT_PREFIX", "1"), True)

    # allow processing fromMe messages (self-test), still loop-safe via echo cache
    allow_from_me_messages: bool = _bool(os.getenv("ALLOW_FROM_ME_MESSAGES", "0"), False)

    # WAHA
    waha_api_url: str = os.getenv("WAHA_API_URL", "").rstrip("/")
    waha_api_key: str = os.getenv("WAHA_API_KEY", "")
    waha_session: str = os.getenv("WAHA_SESSION", "default")
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")

    allowed_group_jids: list[str] = None

    # LLM (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_pool: list[str] = None
    groq_timeout: float = float(os.getenv("GROQ_TIMEOUT", "60"))
    groq_max_inflight: int = _int(os.getenv("GROQ_MAX_INFLIGHT", "5"), 5)

    # Live search (Groq compound web_search)
    live_search_enabled: bool = _bool(os.getenv("LIVE_SEARCH_ENABLED", "0"), False)
    live_search_model: str = os.getenv("LIVE_SEARCH_MODEL", "groq/compound-mini")

    # Chroma
    chroma_enabled: bool = _bool(os.getenv("CHROMA_ENABLED", "1"), True)
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "shimmi_conversations")
    chroma_embed_model: str = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_top_k: int = _int(os.getenv("CHROMA_TOP_K", "10"), 10)
    chroma_recent_k: int = _int(os.getenv("CHROMA_RECENT_K", "10"), 10)

    # Robustness
    message_debounce_ms: int = _int(os.getenv("MESSAGE_DEBOUNCE_MS", "800"), 800)
    llm_max_queue_per_chat: int = _int(os.getenv("LLM_MAX_QUEUE_PER_CHAT", "3"), 3)
    llm_queue_wait_sec: int = _int(os.getenv("LLM_QUEUE_WAIT_SEC", "20"), 20)

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s:%(message)s")
    access_log_level: str = os.getenv("ACCESS_LOG_LEVEL", "WARNING")

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        allow = [s.strip() for s in os.getenv("ALLOWED_GROUP_JIDS", "").split(",") if s.strip()]
        object.__setattr__(self, "allowed_group_jids", allow or None)

        pool = [s.strip() for s in os.getenv("GROQ_MODEL_POOL", "").split(",") if s.strip()]
        object.__setattr__(self, "groq_model_pool", pool or ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "shimmi.sqlite"

    @property
    def chroma_dir(self) -> Path:
        d = self.data_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d

settings = Settings()
