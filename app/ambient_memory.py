"""
Ambient Memory System
Silently observes conversations without explicit invocation and builds context
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger("app.ambient")
UTC = timezone.utc


# ============================================================================
# AMBIENT MEMORY CONFIGURATION
# ============================================================================

@dataclass
class AmbientConfig:
    """Configuration for ambient memory observation"""
    
    # Observation modes
    groups_default: str = "topics"  # topics | on | off
    dms_default: str = "on"  # on | topics | off
    
    # Filtering
    min_text_len: int = 60  # Minimum message length to observe
    max_embed_per_min: int = 10  # Rate limit on embeddings
    
    # Privacy
    redaction_enabled: bool = True
    retention_days: int = 30
    
    # Admin control
    admin_sender_ids: Set[str] = None
    
    @classmethod
    def from_env(cls, settings):
        """Load config from environment settings"""
        return cls(
            groups_default=getattr(settings, 'observe_groups_default', 'topics'),
            dms_default=getattr(settings, 'observe_dms_default', 'on'),
            min_text_len=getattr(settings, 'observe_min_text_len', 60),
            max_embed_per_min=getattr(settings, 'observe_max_embed_per_min', 10),
            redaction_enabled=getattr(settings, 'observe_redaction_default', True) in ('on', 'true', '1', True),
            retention_days=getattr(settings, 'observe_retention_default_days', 30),
            admin_sender_ids=set(getattr(settings, 'observe_admin_sender_ids', '').split(',')) if getattr(settings, 'observe_admin_sender_ids', '') else set()
        )


# ============================================================================
# PII REDACTION
# ============================================================================

class PIIRedactor:
    """Detect and redact personally identifiable information"""
    
    # Patterns for PII detection
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'url': r'https?://[^\s]+',
    }
    
    @classmethod
    def redact(cls, text: str, preserve_context: bool = True) -> str:
        """
        Redact PII from text.
        
        If preserve_context=True, replace with placeholders like [EMAIL] instead of removing.
        """
        redacted = text
        
        for pii_type, pattern in cls.PATTERNS.items():
            if preserve_context:
                replacement = f'[{pii_type.upper()}]'
            else:
                replacement = '[REDACTED]'
            
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        
        return redacted
    
    @classmethod
    def has_pii(cls, text: str) -> bool:
        """Check if text contains PII"""
        for pattern in cls.PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# ============================================================================
# AMBIENT OBSERVER
# ============================================================================

class AmbientObserver:
    """
    Silently observes conversations and extracts:
    - Topics being discussed
    - User interests and patterns
    - Conversation context
    - Sentiment and engagement
    
    WITHOUT requiring explicit bot invocation.
    """
    
    def __init__(self, config: AmbientConfig, chroma_store, sqlite_store):
        self.config = config
        self.chroma = chroma_store
        self.sqlite = sqlite_store
        
        # Rate limiting for embeddings
        self.embed_timestamps = []
        
        # Track what we've already observed
        self.observed_ids: Set[str] = set()
    
    def should_observe(self, *, chat_id: str, sender_id: str, text: str, is_group: bool) -> bool:
        """
        Decide whether to observe this message.
        
        Rules:
        1. Check mode (on/off/topics)
        2. Check minimum length
        3. Check if already observed
        4. Check rate limits
        5. Check admin permissions
        """
        
        # Check mode
        mode = self.config.groups_default if is_group else self.config.dms_default
        
        if mode == "off":
            return False
        
        # Check minimum length
        if len(text.strip()) < self.config.min_text_len:
            return False
        
        # Generate message ID
        msg_id = hashlib.md5(f"{chat_id}:{sender_id}:{text}".encode()).hexdigest()
        if msg_id in self.observed_ids:
            return False
        
        # Check embedding rate limit
        now = datetime.now(UTC).timestamp()
        self.embed_timestamps = [ts for ts in self.embed_timestamps if now - ts < 60]
        
        if len(self.embed_timestamps) >= self.config.max_embed_per_min:
            logger.warning("ambient.rate_limited chat=%s", chat_id)
            return False
        
        # Mode "topics" - only observe if message contains discussion/question markers
        if mode == "topics":
            topic_markers = [
                '?', 'what', 'how', 'why', 'when', 'where', 'who',
                'think', 'opinion', 'believe', 'feel', 'recommend'
            ]
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
        event_id: str
    ) -> bool:
        """
        Observe a message and store it in ambient memory.
        Returns True if observed, False if skipped.
        """
        
        if not self.should_observe(
            chat_id=chat_id,
            sender_id=sender_id,
            text=text,
            is_group=is_group
        ):
            return False
        
        # Redact PII if enabled
        if self.config.redaction_enabled:
            clean_text = PIIRedactor.redact(text, preserve_context=True)
        else:
            clean_text = text
        
        # Store in ChromaDB with ambient metadata
        ts = datetime.now(UTC).isoformat()
        msg_id = f"ambient:{chat_id}:{event_id}"
        
        metadata = {
            'chat_id': chat_id,
            'whatsapp_id': sender_id,
            'direction': 'in',
            'ts': ts,
            'is_ambient': True,
            'is_group': is_group,
            'original_length': len(text),
            'redacted': self.config.redaction_enabled and PIIRedactor.has_pii(text)
        }
        
        try:
            await asyncio.to_thread(
                lambda: self.chroma.collection.upsert(
                    ids=[msg_id],
                    documents=[clean_text],
                    metadatas=[metadata]
                )
            )
            
            # Track that we observed this
            self.observed_ids.add(hashlib.md5(f"{chat_id}:{sender_id}:{text}".encode()).hexdigest())
            self.embed_timestamps.append(datetime.now(UTC).timestamp())
            
            logger.info(
                "ambient.observed chat=%s sender=%s len=%d redacted=%s",
                chat_id, sender_id, len(text), metadata['redacted']
            )
            
            return True
            
        except Exception as e:
            logger.error("ambient.failed chat=%s error=%s", chat_id, str(e)[:200])
            return False
    
    async def get_ambient_context(
        self,
        chat_id: str,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve ambient context relevant to a query.
        This provides background context from unrelated conversations.
        """
        
        try:
            result = await asyncio.to_thread(
                lambda: self.chroma.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where={"chat_id": chat_id, "is_ambient": True}
                )
            )
            
            context = []
            for doc, meta in zip(
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0]
            ):
                context.append({
                    "text": doc,
                    "timestamp": meta.get("ts", ""),
                    "is_group": meta.get("is_group", False),
                    "redacted": meta.get("redacted", False)
                })
            
            return context
            
        except Exception as e:
            logger.error("ambient.context_failed error=%s", str(e)[:200])
            return []
    
    async def cleanup_old_ambient(self, days: int = None) -> int:
        """
        Delete ambient observations older than retention period.
        Returns count of deleted items.
        """
        
        days = days or self.config.retention_days
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        
        try:
            # Get all ambient messages
            result = await asyncio.to_thread(
                lambda: self.chroma.collection.get(
                    where={"is_ambient": True},
                    include=["metadatas"]
                )
            )
            
            # Find old ones
            old_ids = []
            for msg_id, meta in zip(result.get("ids", []), result.get("metadatas", [])):
                if meta.get("ts", "") < cutoff:
                    old_ids.append(msg_id)
            
            if old_ids:
                # Delete them
                await asyncio.to_thread(
                    lambda: self.chroma.collection.delete(ids=old_ids)
                )
                
                logger.info("ambient.cleanup deleted=%d days=%d", len(old_ids), days)
                return len(old_ids)
            
            return 0
            
        except Exception as e:
            logger.error("ambient.cleanup_failed error=%s", str(e)[:200])
            return 0


# ============================================================================
# AMBIENT INSIGHTS EXTRACTOR
# ============================================================================

class AmbientInsightsExtractor:
    """
    Analyzes ambient observations to extract:
    - Common topics
    - User interests
    - Conversation patterns
    - Sentiment trends
    """
    
    def __init__(self, observer: AmbientObserver, llm_complete_fn):
        self.observer = observer
        self.llm_complete = llm_complete_fn
    
    async def extract_topics(
        self,
        chat_id: str,
        days: int = 7
    ) -> List[Dict[str, any]]:
        """
        Extract main topics discussed in ambient observations.
        """
        
        # Get recent ambient messages
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        
        result = await asyncio.to_thread(
            lambda: self.observer.chroma.collection.get(
                where={"chat_id": chat_id, "is_ambient": True},
                include=["documents", "metadatas"]
            )
        )
        
        # Filter by date
        messages = []
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
            if meta.get("ts", "") >= cutoff:
                messages.append(doc)
        
        if len(messages) < 5:
            return []
        
        # Use LLM to extract topics
        combined = "\n".join(messages[:50])  # Limit to avoid token overflow
        
        prompt = f"""
Analyze these conversation snippets and identify the main topics discussed.

CONVERSATIONS:
{combined[:3000]}

Extract 3-5 main topics. For each topic provide:
- topic: Short name (2-4 words)
- mentions: How many times it came up
- sentiment: positive/neutral/negative
- key_phrases: 1-2 key phrases

Return JSON only: {{"topics": [{{"topic": "...", "mentions": N, "sentiment": "...", "key_phrases": [...]}}]}}
"""
        
        try:
            response = await self.llm_complete(
                system="You are a conversation analyst. Output JSON only.",
                user=prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            data = json.loads(response)
            return data.get("topics", [])
            
        except Exception as e:
            logger.error("topic_extraction.failed error=%s", str(e)[:200])
            return []
    
    async def get_user_interests(
        self,
        chat_id: str,
        whatsapp_id: str,
        days: int = 30
    ) -> Dict[str, int]:
        """
        Infer user interests from ambient observations.
        Returns: {interest: frequency}
        """
        
        # This would analyze message content to identify recurring themes
        # For now, return placeholder
        
        topics = await self.extract_topics(chat_id, days)
        
        interests = {}
        for topic in topics:
            interests[topic.get("topic", "")] = topic.get("mentions", 0)
        
        return interests


# ============================================================================
# INTEGRATION
# ============================================================================

async def integrate_ambient_memory(app_main):
    """
    Integration guide for main.py:
    
    1. Initialize in startup:
    
    @app.on_event("startup")
    async def startup():
        from .ambient_memory import AmbientObserver, AmbientConfig
        
        config = AmbientConfig.from_env(settings)
        global ambient_observer
        ambient_observer = AmbientObserver(
            config=config,
            chroma_store=database.chroma_store,
            sqlite_store=database.sqlite_store
        )
    
    2. In webhook handler, BEFORE prefix check:
    
    # Store in ambient memory if not invoking bot
    if not has_prefix(text):
        await ambient_observer.observe(
            chat_id=chat_id,
            sender_id=sender_id,
            text=text,
            is_group=chat_id.endswith("@g.us"),
            event_id=event_id
        )
    
    3. Use ambient context in agent:
    
    ambient_context = await ambient_observer.get_ambient_context(
        chat_id=chat_id,
        query=user_text,
        k=3
    )
    
    # Include in agent prompt
    context_items.extend(ambient_context)
    """
    pass


# ============================================================================
# BACKGROUND CLEANUP TASK
# ============================================================================

async def ambient_cleanup_loop(observer: AmbientObserver):
    """Background task to cleanup old ambient data"""
    
    while True:
        try:
            # Run every 24 hours
            await asyncio.sleep(86400)
            
            deleted = await observer.cleanup_old_ambient()
            logger.info("ambient.cleanup_complete deleted=%d", deleted)
            
        except Exception:
            logger.exception("ambient.cleanup_error")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_ambient_usage():
    """Example of ambient memory in action"""
    
    from .config import settings
    from . import database
    
    # Initialize
    config = AmbientConfig.from_env(settings)
    observer = AmbientObserver(
        config=config,
        chroma_store=database.chroma_store,
        sqlite_store=database.sqlite_store
    )
    
    # Observe a message (user didn't invoke bot)
    observed = await observer.observe(
        chat_id="group123@g.us",
        sender_id="user456@c.us",
        text="I've been thinking about switching to a vegetarian diet. Any recommendations?",
        is_group=True,
        event_id="msg_001"
    )
    
    print(f"Observed: {observed}")
    
    # Later, when user asks bot a related question
    context = await observer.get_ambient_context(
        chat_id="group123@g.us",
        query="vegetarian recipes",
        k=5
    )
    
    print(f"Ambient context: {len(context)} relevant messages")
    
    # Extract insights
    from .improved_agent_engine import smart_complete
    insights = AmbientInsightsExtractor(observer, smart_complete)
    
    topics = await insights.extract_topics("group123@g.us", days=7)
    print(f"Topics discussed: {topics}")
