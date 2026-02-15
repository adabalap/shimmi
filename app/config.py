from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _int(v: Optional[str], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _float(v: Optional[str], default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # Core identity
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    app_timezone: str = os.getenv("APP_TIMEZONE", "UTC")
    bot_persona_name: str = os.getenv("BOT_PERSONA_NAME", "Shimmi")
    bot_command_prefix: str = os.getenv("BOT_COMMAND_PREFIX", "@shimmi,shimmi")
    
    # Bot behavior
    allow_nlp_without_prefix: bool = _bool(os.getenv("ALLOW_NLP_WITHOUT_PREFIX", "1"), True)
    allow_fromme: bool = _bool(os.getenv("ALLOW_FROMME", "0"), False)
    
    # WhatsApp / WAHA
    waha_api_url: str = os.getenv("WAHA_API_URL", "").rstrip("/")
    waha_api_key: str = os.getenv("WAHA_API_KEY", "")
    waha_session: str = os.getenv("WAHA_SESSION", "default")
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")
    allowed_chat_jids: Optional[List[str]] = None
    
    # LLM Provider Configuration
    # Format: PROVIDER_NAME_ENABLED, PROVIDER_NAME_API_KEY, PROVIDER_NAME_MODELS
    
    # Groq
    groq_enabled: bool = _bool(os.getenv("GROQ_ENABLED", "1"), True)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_models: str = os.getenv("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant")
    groq_timeout: float = _float(os.getenv("GROQ_TIMEOUT", "60"), 60.0)
    groq_daily_limit: int = _int(os.getenv("GROQ_DAILY_LIMIT", "100000"), 100000)
    
    # Gemini
    gemini_enabled: bool = _bool(os.getenv("GEMINI_ENABLED", "1"), True)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_models: str = os.getenv("GEMINI_MODELS", "gemini-2.0-flash-exp,gemini-1.5-flash")
    gemini_timeout: float = _float(os.getenv("GEMINI_TIMEOUT", "60"), 60.0)
    gemini_daily_limit: int = _int(os.getenv("GEMINI_DAILY_LIMIT", "1500000"), 1500000)
    
    # Claude (Anthropic)
    claude_enabled: bool = _bool(os.getenv("CLAUDE_ENABLED", "0"), False)
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    claude_models: str = os.getenv("CLAUDE_MODELS", "claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022")
    claude_timeout: float = _float(os.getenv("CLAUDE_TIMEOUT", "60"), 60.0)
    claude_daily_limit: int = _int(os.getenv("CLAUDE_DAILY_LIMIT", "1000000"), 1000000)
    
    # OpenAI
    openai_enabled: bool = _bool(os.getenv("OPENAI_ENABLED", "0"), False)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_models: str = os.getenv("OPENAI_MODELS", "gpt-4o-mini,gpt-3.5-turbo")
    openai_timeout: float = _float(os.getenv("OPENAI_TIMEOUT", "60"), 60.0)
    openai_daily_limit: int = _int(os.getenv("OPENAI_DAILY_LIMIT", "1000000"), 1000000)
    
    # Provider priority (comma-separated, checked in order)
    llm_provider_priority: str = os.getenv("LLM_PROVIDER_PRIORITY", "groq,gemini,claude,openai")
    llm_max_inflight: int = _int(os.getenv("LLM_MAX_INFLIGHT", "5"), 5)
    
    # Live search
    live_search_enabled: bool = _bool(os.getenv("LIVE_SEARCH_ENABLED", "1"), True)
    live_search_provider: str = os.getenv("LIVE_SEARCH_PROVIDER", "groq")
    live_search_model: str = os.getenv("LIVE_SEARCH_MODEL", "llama-3.1-8b-instant")
    
    # ChromaDB
    chroma_enabled: bool = _bool(os.getenv("CHROMA_ENABLED", "1"), True)
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "shimmi_conversations")
    chroma_embed_model: str = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_top_k: int = _int(os.getenv("CHROMA_TOP_K", "10"), 10)
    chroma_recent_k: int = _int(os.getenv("CHROMA_RECENT_K", "10"), 10)
    
    # Message processing
    message_debounce_ms: int = _int(os.getenv("MESSAGE_DEBOUNCE_MS", "800"), 800)
    llm_max_queue_per_chat: int = _int(os.getenv("LLM_MAX_QUEUE_PER_CHAT", "3"), 3)
    llm_queue_wait_sec: int = _int(os.getenv("LLM_QUEUE_WAIT_SEC", "20"), 20)
    
    # Facts
    facts_verification: bool = _bool(os.getenv("FACTS_VERIFICATION", "1"), True)
    facts_min_conf: float = _float(os.getenv("FACTS_MIN_CONF", "0.85"), 0.85)
    allow_freeform_memory_keys: bool = _bool(os.getenv("ALLOW_FREEFORM_MEMORY_KEYS", "1"), True)
    
    # Ambient memory
    observe_groups_default: str = os.getenv("OBSERVE_GROUPS_DEFAULT", "topics")
    observe_dms_default: str = os.getenv("OBSERVE_DMS_DEFAULT", "on")
    observe_min_text_len: int = _int(os.getenv("OBSERVE_MIN_TEXT_LEN", "60"), 60)
    observe_max_embed_per_min: int = _int(os.getenv("OBSERVE_MAX_EMBED_PER_MIN", "10"), 10)
    observe_redaction_default: str = os.getenv("OBSERVE_REDACTION_DEFAULT", "on")
    observe_retention_default_days: int = _int(os.getenv("OBSERVE_RETENTION_DEFAULT_DAYS", "30"), 30)
    observe_admin_sender_ids: str = os.getenv("OBSERVE_ADMIN_SENDER_IDS", "")
    
    # Memory management
    memory_cleanup_enabled: bool = _bool(os.getenv("MEMORY_CLEANUP_ENABLED", "1"), True)
    memory_cleanup_interval_hours: int = _int(os.getenv("MEMORY_CLEANUP_INTERVAL_HOURS", "24"), 24)
    memory_importance_decay: float = _float(os.getenv("MEMORY_IMPORTANCE_DECAY", "0.02"), 0.02)
    
    # Fact mining
    fact_mining_enabled: bool = _bool(os.getenv("FACT_MINING_ENABLED", "1"), True)
    fact_mining_interval_hours: int = _int(os.getenv("FACT_MINING_INTERVAL_HOURS", "24"), 24)
    fact_mining_lookback_days: int = _int(os.getenv("FACT_MINING_LOOKBACK_DAYS", "7"), 7)
    
    # Structured actions (lists, reminders, todos)
    actions_enabled: bool = _bool(os.getenv("ACTIONS_ENABLED", "1"), True)
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_agent: bool = _bool(os.getenv("DEBUG_AGENT", "0"), False)
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        allow = [s.strip() for s in os.getenv("ALLOWED_GROUP_JIDS", "").split(",") if s.strip()]
        object.__setattr__(self, "allowed_chat_jids", allow or None)
    
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "shimmi.sqlite"
    
    @property
    def chroma_dir(self) -> Path:
        d = self.data_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific provider"""
        return {
            'enabled': getattr(self, f'{provider}_enabled', False),
            'api_key': getattr(self, f'{provider}_api_key', ''),
            'models': [m.strip() for m in getattr(self, f'{provider}_models', '').split(',') if m.strip()],
            'timeout': getattr(self, f'{provider}_timeout', 60.0),
            'daily_limit': getattr(self, f'{provider}_daily_limit', 1000000),
        }
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers in priority order"""
        priority = [p.strip() for p in self.llm_provider_priority.split(',') if p.strip()]
        enabled = []
        for provider in priority:
            config = self.get_provider_config(provider)
            if config['enabled'] and config['api_key']:
                enabled.append(provider)
        return enabled


settings = Settings()
