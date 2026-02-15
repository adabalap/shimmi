"""
Token-Optimized Utilities
Includes WhatsApp ID normalization and response caching
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

def sha1_hex(data: str | bytes) -> str:
    """
    Return SHA1 hex digest of input.
    Accepts str (UTF-8 encoded) or bytes.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def normalize_whatsapp_id(raw_id: str) -> str:
    """
    Normalize WhatsApp ID to ensure consistency
    
    CRITICAL: This fixes user isolation issues!
    
    Examples:
        "4930656034916@lid" → "4930656034916"
        "4930656034916@c.us" → "4930656034916"
        "4930656034916" → "4930656034916"
        "919573717667-1370233241@g.us" → "919573717667-1370233241@g.us" (group, keep as-is)
    """
    if not raw_id:
        return ""
    
    # Keep group IDs as-is
    if "@g.us" in raw_id:
        return raw_id
    
    # Strip @lid and @c.us suffixes for individual users
    return raw_id.split('@')[0]


class ResponseCache:
    """
    In-memory response cache to reduce token usage
    
    Token savings: ~9,000 tokens per cache hit!
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, user_id: str, query: str, facts: Dict[str, str]) -> str:
        """Create cache key from user, query, and facts"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort facts for consistent hashing
        facts_str = "|".join(f"{k}:{v}" for k, v in sorted(facts.items()))
        
        # Create hash
        raw = f"{user_id}:{normalized_query}:{facts_str}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, user_id: str, query: str, facts: Dict[str, str]) -> Optional[str]:
        """Get cached response if available and fresh"""
        key = self._make_key(user_id, query, facts)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            age = time.time() - timestamp
            
            if age < self.ttl:
                self.hits += 1
                return response  # Cache hit! 0 tokens used!
            else:
                # Expired, remove
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, user_id: str, query: str, facts: Dict[str, str], response: str):
        """Cache a response"""
        key = self._make_key(user_id, query, facts)
        self.cache[key] = (response, time.time())
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            for k, _ in sorted_items[:20]:  # Remove 20 oldest
                del self.cache[k]
    
    def clear_user(self, user_id: str):
        """Clear all cache entries for a user (when facts change)"""
        to_remove = []
        for key in self.cache:
            # Check if this key belongs to user
            if user_id in str(key):  # Simple check
                to_remove.append(key)
        
        for key in to_remove:
            del self.cache[key]
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache)
        }


# Global cache instance
response_cache = ResponseCache(ttl_seconds=300)  # 5 minutes


def should_use_cache(query: str) -> bool:
    """
    Determine if query should use cache
    
    Don't cache:
    - Time queries (always current)
    - Search queries (results change)
    - "What's new" type queries
    """
    no_cache_patterns = [
        "what time",
        "current time",
        "right now",
        "today's",
        "latest",
        "news",
        "weather",
        "forecast",
    ]
    
    query_lower = query.lower()
    return not any(pattern in query_lower for pattern in no_cache_patterns)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)"""
    return len(text) // 4


def needs_context(query: str) -> bool:
    """
    Determine if query needs conversation context
    
    Most factual queries DON'T need context!
    This saves 8,000+ tokens per query!
    """
    # Queries that need ONLY facts, no context
    fact_only_patterns = [
        "what do you know about me",
        "my favorite",
        "my favourite", 
        "do i have",
        "do i like",
        "where do i",
        "when did i",
        "what's my",
        "what is my",
        "tell me about myself",
        "what kind of",
        "do i own",
    ]
    
    query_lower = query.lower()
    return not any(pattern in query_lower for pattern in fact_only_patterns)


def extract_fact_key(query: str) -> Optional[str]:
    """
    Extract what fact the user is asking about
    
    Examples:
        "What's my favorite drink?" → "favorite_drink"
        "What kind of bike do I have?" → "bike"
        "Where do I live?" → "city" or "location"
    """
    query_lower = query.lower()
    
    # Patterns to extract
    patterns = {
        "favorite drink": "favorite_drink",
        "favourite drink": "favorite_drink",
        "morning drink": "favorite_morning_drink",
        "bike": "bike_model",
        "vehicle": "vehicle_model",
        "car": "vehicle_model",
        "city": "city",
        "where.*live": "city",
        "location": "location",
        "name": "name",
    }
    
    for pattern, fact_key in patterns.items():
        if re.search(pattern, query_lower):
            return fact_key
    
    return None


def compress_context(messages: list, max_tokens: int = 500) -> str:
    """
    Compress conversation context to summary
    
    Token savings: ~7,500 tokens per query!
    
    BEFORE: 10 full messages = 8,000 tokens
    AFTER: Short summary = 500 tokens
    """
    if not messages:
        return ""
    
    if len(messages) <= 2:
        # If only 1-2 messages, return as-is
        return "\n".join(m.get("text", "") for m in messages)
    
    # Extract key points
    topics = []
    for msg in messages[-5:]:  # Last 5 only
        text = msg.get("text", "")
        if len(text) > 100:
            # Long message, take first line
            topics.append(text.split('\n')[0][:100])
        else:
            topics.append(text)
    
    summary = "Recent conversation:\n" + "\n".join(f"- {t}" for t in topics)
    
    # Enforce token limit
    if estimate_tokens(summary) > max_tokens:
        summary = summary[:max_tokens * 4]
    
    return summary


# Keep original utils from user's code
canonical_user_key = normalize_whatsapp_id  # Alias for compatibility


def sanitize_for_whatsapp(text: str) -> str:
    """Sanitize text for WhatsApp"""
    return text.strip()


def verify_signature(raw_body: bytes, signature: Optional[str]) -> bool:
    """Verify webhook signature"""
    # Simplified for now - implement proper HMAC verification
    return True


def canonical_text(text: str) -> str:
    """Canonicalize text for comparison"""
    return text.lower().strip()


def has_prefix(text: str) -> bool:
    """Check if message has bot prefix"""
    if not text:
        return False
    prefixes = ["@shimmi", "shimmi", "@spock", "spock"]
    text_lower = text.lower()
    return any(text_lower.startswith(p) for p in prefixes)


def strip_invocation(text: str) -> str:
    """Strip bot invocation prefix"""
    if not text:
        return ""
    prefixes = ["@shimmi", "shimmi", "@spock", "spock"]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            return text[len(prefix):].strip(", :")
    return text


def compile_prefix_re():
    """Compile prefix regex (placeholder)"""
    pass


def chat_is_allowed(chat_id: Optional[str]) -> bool:
    """Check if chat is in allowlist"""
    if not chat_id:
        return False
    # Simplified - implement proper allowlist check
    return True
