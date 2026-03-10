"""
retry.py — Shimmi v3.0.2

Changes vs v3.0.1:
  - Never retry on RateLimitError (HTTP 429): retrying a rate-limited model
    wastes daily tokens on calls guaranteed to fail. Raise immediately so
    the circuit breaker trips and the NEXT call picks a different model.
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
    """Return True if the exception is a Groq 429 / rate-limit error."""
    s = str(exc)
    return (
        "429" in s
        or "rate_limit_exceeded" in s.lower()
        or "RateLimitError" in type(exc).__name__
    )


async def async_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            # Never retry rate-limit errors — retrying the same model wastes
            # daily tokens on calls guaranteed to fail. The caller's circuit
            # breaker will trip and route the next message to a different model.
            if _is_rate_limit(exc):
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(_sleep_time(attempt, base_delay, max_delay))
