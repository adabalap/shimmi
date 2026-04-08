"""
retry.py — Shimmi v3.0.3

Changes vs v3.0.2:
  - _is_rate_limit() now also detects Google Gemini 429 / RESOURCE_EXHAUSTED
    responses so the multi-provider circuit breaker works for both Groq and Gemini.
"""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def _sleep_time(attempt: int, base_delay: float, max_delay: float) -> float:
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return delay * (0.75 + random.random() * 0.5)


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception is a 429 / rate-limit from Groq or Gemini."""
    s = str(exc)
    cls_name = type(exc).__name__
    return (
        "429" in s
        or "rate_limit_exceeded" in s.lower()
        or "RESOURCE_EXHAUSTED" in s          # Google Gemini quota error
        or "quota" in s.lower()
        or "RateLimitError" in cls_name
        or "QuotaError" in cls_name
    )


async def async_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
) -> T:
    """
    Retry with exponential back-off.

    Never retries rate-limit errors: retrying the same provider on a 429
    wastes the remaining quota.  The caller's circuit-breaker trips and
    routes the next call to a different model/provider.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if _is_rate_limit(exc):
                raise   # never retry rate limits
            attempt += 1
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(_sleep_time(attempt, base_delay, max_delay))
