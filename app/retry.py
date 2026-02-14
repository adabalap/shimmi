from __future__ import annotations

import asyncio
import random
from typing import Callable, TypeVar, Awaitable

T = TypeVar("T")


def _sleep(attempt: int, base: float, maxd: float) -> float:
    d = min(maxd, base * (2 ** (attempt - 1)))
    return d * (0.75 + random.random() * 0.5)


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
        except Exception:
            attempt += 1
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(_sleep(attempt, base_delay, max_delay))
