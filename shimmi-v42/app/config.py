"""
config.py — Shimmi v3.0.3

Changes vs v3.0.2:
  - Added GEMINI_API_KEY + Gemini model routing (primary orchestrator)
  - Gemini free tier: ~1,500 req/day, 1M tokens/min — 15x more than Groq 70B
  - compound-beta-mini removed from default Groq pool (shares 70B daily bucket)
  - Live search model kept as compound-beta-mini (dedicated, tracked separately)
  - Added token_budget_warn_pct / token_budget_block_pct for budget awareness
  - Added mcp_server_url for MCP sidecar
"""
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
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    app_timezone: str = os.getenv("APP_TIMEZONE", "UTC")

    port: int = _int(os.getenv("PORT", "6000"), 6000)
    host: str = os.getenv("HOST", "0.0.0.0")

    bot_persona_name: str    = os.getenv("BOT_PERSONA_NAME",    "Shimmi")
    bot_command_prefix: str  = os.getenv("BOT_COMMAND_PREFIX",  "@shimmi,shimmi")

    allow_nlp_without_prefix: bool = _bool(os.getenv("ALLOW_NLP_WITHOUT_PREFIX", "1"), True)
    allow_fromme: bool             = _bool(os.getenv("ALLOW_FROMME", "0"), False)

    waha_api_url: str    = os.getenv("WAHA_API_URL", "").rstrip("/")
    waha_api_key: str    = os.getenv("WAHA_API_KEY", "")
    waha_session: str    = os.getenv("WAHA_SESSION", "default")
    webhook_secret: str  = os.getenv("WEBHOOK_SECRET", "")

    allow_all_chats: bool              = _bool(os.getenv("ALLOW_ALL_CHATS", "0"), False)
    allowed_chat_jids: Optional[List[str]] = None

    # ── Groq ───────────────────────────────────────────────────────────────
    groq_api_key: str           = os.getenv("GROQ_API_KEY", "")
    groq_model_pool: Optional[List[str]] = None
    groq_timeout: float         = _float(os.getenv("GROQ_TIMEOUT", "45"), 45.0)
    groq_max_inflight: int      = _int(os.getenv("GROQ_MAX_INFLIGHT", "5"), 5)
    orchestrator_model: str     = os.getenv("ORCHESTRATOR_MODEL", "llama-3.3-70b-versatile")
    extraction_model: str       = os.getenv("EXTRACTION_MODEL",   "llama-3.1-8b-instant")
    live_search_model: str      = os.getenv("LIVE_SEARCH_MODEL",  "compound-beta-mini")

    # ── Google Gemini (primary orchestrator) ──────────────────────────────
    # Free tier: ~1500 req/day, 1M tokens/min — far more generous than Groq free.
    # Get your key at https://aistudio.google.com/apikey
    gemini_api_key: str               = os.getenv("GEMINI_API_KEY", "")
    gemini_model_pool: Optional[List[str]] = None
    gemini_orchestrator_model: str    = os.getenv("GEMINI_ORCHESTRATOR_MODEL", "gemini-2.0-flash")
    gemini_extraction_model: str      = os.getenv("GEMINI_EXTRACTION_MODEL",   "gemini-2.0-flash-lite")
    gemini_timeout: float             = _float(os.getenv("GEMINI_TIMEOUT", "30"), 30.0)

    # ── Token budget awareness ─────────────────────────────────────────────
    token_budget_warn_pct: float  = _float(os.getenv("TOKEN_BUDGET_WARN_PCT",  "0.75"), 0.75)
    token_budget_block_pct: float = _float(os.getenv("TOKEN_BUDGET_BLOCK_PCT", "0.92"), 0.92)
    # Groq 70B daily limit (on_demand free tier = 100K tokens/day)
    groq_70b_daily_limit: int = _int(os.getenv("GROQ_70B_DAILY_LIMIT", "100000"), 100_000)

    # ── Live data / MCP ───────────────────────────────────────────────────
    live_search_enabled: bool = _bool(os.getenv("LIVE_SEARCH_ENABLED", "1"), True)
    mcp_server_url: str       = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:7000")

    # ── Agent ─────────────────────────────────────────────────────────────
    agent_max_turns: int = _int(os.getenv("AGENT_MAX_TURNS", "4"), 4)

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_enabled: bool    = _bool(os.getenv("CHROMA_ENABLED", "1"), True)
    chroma_collection: str  = os.getenv("CHROMA_COLLECTION", "shimmi_conversations")
    chroma_embed_model: str = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_top_k: int       = _int(os.getenv("CHROMA_TOP_K",     "10"), 10)
    chroma_recent_k: int    = _int(os.getenv("CHROMA_RECENT_K",  "10"), 10)

    # ── Messaging ────────────────────────────────────────────────────────
    message_debounce_ms: int     = _int(os.getenv("MESSAGE_DEBOUNCE_MS",    "800"), 800)
    llm_max_queue_per_chat: int  = _int(os.getenv("LLM_MAX_QUEUE_PER_CHAT", "3"),   3)
    llm_queue_wait_sec: int      = _int(os.getenv("LLM_QUEUE_WAIT_SEC",     "20"),  20)

    # ── Memory ───────────────────────────────────────────────────────────
    facts_verification: bool         = _bool(os.getenv("FACTS_VERIFICATION",         "1"),    True)
    facts_min_conf: float            = _float(os.getenv("FACTS_MIN_CONF",            "0.85"), 0.85)
    allow_freeform_memory_keys: bool = _bool(os.getenv("ALLOW_FREEFORM_MEMORY_KEYS", "1"),    True)

    # ── Scheduler ─────────────────────────────────────────────────────────
    reminder_check_interval_sec: int = _int(os.getenv("REMINDER_CHECK_INTERVAL_SEC", "60"), 60)

    # ── Observability ─────────────────────────────────────────────────────
    debug_agent: bool   = _bool(os.getenv("DEBUG_AGENT",   "0"), False)
    trace_enabled: bool = _bool(os.getenv("TRACE_ENABLED", "1"), True)

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        allow = [s.strip() for s in os.getenv("ALLOWED_GROUP_JIDS", "").split(",") if s.strip()]
        object.__setattr__(self, "allowed_chat_jids", allow or None)

        # Groq pool — default excludes compound-beta-mini (shares 70B daily bucket)
        pool_str = os.getenv("GROQ_MODEL_POOL", "")
        pool = [s.strip() for s in pool_str.split(",") if s.strip()] if pool_str else []
        if not pool:
            pool = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        object.__setattr__(self, "groq_model_pool", pool)

        # Gemini pool
        gem_str = os.getenv("GEMINI_MODEL_POOL", "")
        gem_pool = [s.strip() for s in gem_str.split(",") if s.strip()] if gem_str else []
        if not gem_pool:
            gem_pool = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
        object.__setattr__(self, "gemini_model_pool", gem_pool)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "shimmi.sqlite"

    @property
    def chroma_dir(self) -> Path:
        d = self.data_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
