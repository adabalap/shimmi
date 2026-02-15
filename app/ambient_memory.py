"""
Ambient Memory System
Silently observes conversations without explicit invocation and builds context

Refactors:
- Fix Chroma where filters using {"$and":[...]} form.
- Use ChromaAmbient.embed_fn (if available) to compute embeddings safely in async context.
- Fix PII regex patterns (email regex was broken).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger("app.ambient")
UTC = timezone.utc


@dataclass
class AmbientConfig:
    """Configuration for ambient memory observation"""

    # Observation modes:
    # - groups_default: "topics" | "on" | "off"
    # - dms_default:    "on" | "topics" | "off"
    groups_default: str = "topics"
    dms_default: str = "on"

    # Filtering
    min_text_len: int = 60
    max_embed_per_min: int = 10

    # Privacy
    redaction_enabled: bool = True
    retention_days: int = 30

    # Admin control
    admin_sender_ids: Set[str] = None

    @classmethod
    def from_env(cls, settings):
        return cls(
            groups_default=getattr(settings, "observe_groups_default", "topics"),
            dms_default=getattr(settings, "observe_dms_default", "on"),
            min_text_len=getattr(settings, "observe_min_text_len", 60),
            max_embed_per_min=getattr(settings, "observe_max_embed_per_min", 10),
            redaction_enabled=str(getattr(settings, "observe_redaction_default", "on")).lower() in ("on", "true", "1"),
            retention_days=getattr(settings, "observe_retention_default_days", 30),
            admin_sender_ids=set(getattr(settings, "observe_admin_sender_ids", "").split(","))
            if getattr(settings, "observe_admin_sender_ids", "")
            else set(),
        )


class PIIRedactor:
    """Detect and redact personally identifiable information"""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "url": r"https?://[^\s]+",
    }

    @classmethod
    def redact(cls, text: str, preserve_context: bool = True) -> str:
        redacted = text
        for pii_type, pattern in cls.PATTERNS.items():
            replacement = f"[{pii_type.upper()}]" if preserve_context else "[REDACTED]"
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        return redacted

    @classmethod
    def has_pii(cls, text: str) -> bool:
        for pattern in cls.PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class AmbientObserver:
    """
    Observes conversations and stores selected messages as ambient context.
    """

    def __init__(self, config: AmbientConfig, chroma_store, sqlite_store):
        self.config = config
        self.chroma = chroma_store  # expects .collection, and optionally .embed_fn
        self.sqlite = sqlite_store

        self.embed_timestamps: List[float] = []
        self.observed_ids: Set[str] = set()

    @staticmethod
    def _and_where(*clauses: Dict[str, Any]) -> Dict[str, Any]:
        clauses = [c for c in clauses if c]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def should_observe(self, *, chat_id: str, sender_id: str, text: str, is_group: bool) -> bool:
        mode = self.config.groups_default if is_group else self.config.dms_default
        if mode == "off":
            return False

        if len(text.strip()) < self.config.min_text_len:
            return False

        msg_hash = hashlib.md5(f"{chat_id}:{sender_id}:{text}".encode()).hexdigest()
        if msg_hash in self.observed_ids:
            return False

        now = datetime.now(UTC).timestamp()
        self.embed_timestamps = [ts for ts in self.embed_timestamps if now - ts < 60]
        if len(self.embed_timestamps) >= self.config.max_embed_per_min:
            logger.warning("ambient.rate_limited chat=%s", chat_id)
            return False

        if mode == "topics":
            topic_markers = ["?", "what", "how", "why", "when", "where", "who", "think", "opinion", "believe", "feel", "recommend"]
            if not any(marker in text.lower() for marker in topic_markers):
                return False

        return True

    async def observe(
        self,
        *,
        chat_id: str,
        sender_id: str,
        text: str,
        is_group: bool,
        event_id: str,
    ) -> bool:
        if not self.should_observe(chat_id=chat_id, sender_id=sender_id, text=text, is_group=is_group):
            return False

        clean_text = PIIRedactor.redact(text, preserve_context=True) if self.config.redaction_enabled else text

        ts = datetime.now(UTC).isoformat()
        msg_id = f"ambient:{chat_id}:{event_id}"
        metadata = {
            "chat_id": chat_id,
            "whatsapp_id": sender_id,
            "direction": "in",
            "ts": ts,
            "is_ambient": True,
            "is_group": is_group,
            "original_length": len(text),
            "redacted": self.config.redaction_enabled and PIIRedactor.has_pii(text),
        }

        try:
            embeddings = None
            if hasattr(self.chroma, "embed_fn") and self.chroma.embed_fn:
                embeddings = await self.chroma.embed_fn.aembed([clean_text])

            if embeddings is not None:
                await asyncio.to_thread(
                    lambda: self.chroma.collection.upsert(
                        ids=[msg_id],
                        documents=[clean_text],
                        metadatas=[metadata],
                        embeddings=embeddings,
                    )
                )
            else:
                # fallback: let Chroma compute embeddings internally
                await asyncio.to_thread(
                    lambda: self.chroma.collection.upsert(
                        ids=[msg_id],
                        documents=[clean_text],
                        metadatas=[metadata],
                    )
                )

            self.observed_ids.add(hashlib.md5(f"{chat_id}:{sender_id}:{text}".encode()).hexdigest())
            self.embed_timestamps.append(datetime.now(UTC).timestamp())

            logger.info("ambient.observed chat=%s sender=%s len=%d redacted=%s", chat_id, sender_id, len(text), metadata["redacted"])
            return True
        except Exception as e:
            logger.error("ambient.failed chat=%s error=%s", chat_id, str(e)[:200])
            return False

    async def get_ambient_context(self, chat_id: str, query: str, k: int = 5) -> List[Dict]:
        try:
            where = self._and_where({"chat_id": chat_id}, {"is_ambient": True})
            result = await asyncio.to_thread(
                lambda: self.chroma.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=where,
                )
            )
            context = []
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metas):
                context.append(
                    {
                        "text": doc,
                        "timestamp": (meta or {}).get("ts", ""),
                        "is_group": (meta or {}).get("is_group", False),
                        "redacted": (meta or {}).get("redacted", False),
                    }
                )
            return context
        except Exception as e:
            logger.error("ambient.context_failed error=%s", str(e)[:200])
            return []

    async def cleanup_old_ambient(self, days: int = None) -> int:
        days = days or self.config.retention_days
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        try:
            result = await asyncio.to_thread(
                lambda: self.chroma.collection.get(where={"is_ambient": True}, include=["metadatas"])
            )
            old_ids = []
            for msg_id, meta in zip(result.get("ids", []), result.get("metadatas", [])):
                if (meta or {}).get("ts", "") < cutoff:
                    old_ids.append(msg_id)

            if old_ids:
                await asyncio.to_thread(lambda: self.chroma.collection.delete(ids=old_ids))
                logger.info("ambient.cleanup deleted=%d days=%d", len(old_ids), days)
                return len(old_ids)

            return 0
        except Exception as e:
            logger.error("ambient.cleanup_failed error=%s", str(e)[:200])
            return 0


class AmbientInsightsExtractor:
    def __init__(self, observer: AmbientObserver, llm_complete_fn):
        self.observer = observer
        self.llm_complete = llm_complete_fn

    @staticmethod
    def _and_where(*clauses: Dict[str, Any]) -> Dict[str, Any]:
        clauses = [c for c in clauses if c]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    async def extract_topics(self, chat_id: str, days: int = 7) -> List[Dict[str, Any]]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        where = self._and_where({"chat_id": chat_id}, {"is_ambient": True})

        result = await asyncio.to_thread(
            lambda: self.observer.chroma.collection.get(where=where, include=["documents", "metadatas"])
        )

        messages = []
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
            if (meta or {}).get("ts", "") >= cutoff:
                messages.append(doc)

        if len(messages) < 5:
            return []

        combined = "\n".join(messages[:50])
        prompt = f"""
Analyze these conversation snippets and identify the main topics discussed.
CONVERSATIONS:
{combined[:3000]}
Extract 3-5 main topics. For each topic provide:
- topic: Short name (2-4 words)
- mentions: How many times it came up
- sentiment: positive/neutral/negative
- key_phrases: 1-2 key phrases
Return JSON only: {{"topics": [{{"topic":"...","mentions":N,"sentiment":"...","key_phrases":[...]}}]}}
"""

        try:
            response = await self.llm_complete(
                system="You are a conversation analyst. Output JSON only.",
                user=prompt,
                temperature=0.2,
                max_tokens=500,
            )
            data = json.loads(response)
            return data.get("topics", [])
        except Exception as e:
            logger.error("topic_extraction.failed error=%s", str(e)[:200])
            return []
