from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("app.rate_limit")


@dataclass
class TokenBucket:
    """Token bucket rate limiter for LLM calls"""
    capacity: int
    tokens: float
    fill_rate: float  # tokens per second
    last_update: float

    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.fill_rate = fill_rate
        self.last_update = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.fill_rate)
        )
        self.last_update = now

    def consume(self, tokens: int) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: int) -> float:
        """Returns seconds until tokens will be available"""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.fill_rate


class RateLimitManager:
    """Manages rate limits across multiple providers and models"""

    def __init__(self):
        # Groq free tier: 100k tokens/day = ~1.16 tokens/second
        # Use 80% to leave safety margin
        self.groq_bucket = TokenBucket(
            capacity=100000,
            fill_rate=1.16 * 0.8  # 80% of limit
        )
        self.lock = asyncio.Lock()

    async def check_and_reserve(
        self,
        estimated_tokens: int,
        provider: str = "groq"
    ) -> tuple[bool, Optional[str]]:
        """
        Check if rate limit allows request and reserve tokens.
        Returns (allowed, error_message)
        """
        async with self.lock:
            bucket = self.groq_bucket  # Add other providers as needed

            # Check current status
            if bucket.consume(estimated_tokens):
                logger.info(
                    "rate_limit.approved tokens=%d remaining=%.0f",
                    estimated_tokens,
                    bucket.tokens
                )
                return True, None

            # Rate limited
            wait_time = bucket.time_until_available(estimated_tokens)
            wait_minutes = int(wait_time / 60)

            if wait_minutes < 1:
                error_msg = "I'm processing many requests. Please try again in a moment. ⏳"
            elif wait_minutes < 5:
                error_msg = f"I'm at capacity. Please try again in ~{wait_minutes} minutes. 🙏"
            else:
                error_msg = "I've hit my daily limit. Please try again later or tomorrow. 😅"

            logger.warning(
                "rate_limit.exceeded tokens=%d wait_seconds=%.1f",
                estimated_tokens,
                wait_time
            )

            return False, error_msg

    async def record_failure(self, provider: str = "groq"):
        """
        Called when a rate limit error is received from the API.
        Drains bucket to prevent further attempts.
        """
        async with self.lock:
            # Drain tokens to prevent immediate retries
            self.groq_bucket.tokens = 0
            logger.error("rate_limit.api_rejected draining_bucket")


# Global instance
rate_limiter = RateLimitManager()

