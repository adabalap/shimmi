"""
trace.py — Structured per-message trace.

Every message creates a Trace. Each processing phase calls trace.step().
At the end we emit two log lines to app.trace:

  1. TRACE_JSON {...}  — full structured JSON, one line, machine-readable
     grep + jq friendly: grep TRACE_JSON shimmi-bot.log | tail -1 | jq .

  2. TRACE OK/ERROR ... — one-line summary for tail -f monitoring

The old box waterfall (╔══ MSG TRACE ══╗) was removed — it was identical
to the JSON output but took 800+ characters and made logs unreadable in
tail mode.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("app.trace")


@dataclass
class TraceStep:
    name: str
    started_at: float
    ended_at: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.ended_at:
            return round((self.ended_at - self.started_at) * 1000, 1)
        return 0.0


class Trace:
    """Context manager for a single message lifecycle."""

    def __init__(self, event_id: str, chat_id: str, sender_id: str = ""):
        self.event_id  = event_id  or "?"
        self.chat_id   = chat_id   or "?"
        self.sender_id = sender_id or "?"
        self._started: float = time.perf_counter()
        self._steps: List[TraceStep] = []
        self._pending: Optional[TraceStep] = None
        self._global_tags: Dict[str, Any] = {}

    async def __aenter__(self) -> "Trace":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.tag(fatal_error=f"{exc_type.__name__}: {exc_val}")
        self._finalise()

    @contextmanager
    def step(self, name: str):
        s = TraceStep(name=name, started_at=time.perf_counter())
        self._steps.append(s)
        self._pending = s
        try:
            yield s
            s.ended_at = time.perf_counter()
        except Exception as exc:
            s.ended_at = time.perf_counter()
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._pending = None

    def tag(self, **kwargs: Any) -> None:
        """Attach tags to the currently open step, or globally if none open."""
        if self._pending is not None:
            self._pending.tags.update(kwargs)
        else:
            self._global_tags.update(kwargs)

    # ─────────────────────────────────────────────────────────────────────
    def _finalise(self) -> None:
        total_ms = round((time.perf_counter() - self._started) * 1000, 1)
        failed   = any(s.error for s in self._steps) or "fatal_error" in self._global_tags

        # Single structured JSON line — machine-readable, grep/jq friendly.
        # The box waterfall was removed: it duplicated this data verbosely.
        # To read: grep TRACE_JSON shimmi-bot.log | tail -1 | jq .
        record = {
            "trace": {
                "event_id":  self.event_id,
                "chat_id":   self.chat_id,
                "sender_id": self.sender_id,
                "total_ms":  total_ms,
                "ok":        not failed,
                **self._global_tags,
            },
            "steps": [
                {
                    "name": s.name,
                    "ms":   s.duration_ms,
                    **({"error": s.error} if s.error else {}),
                    **s.tags,
                }
                for s in self._steps
            ],
        }
        logger.info("TRACE_JSON %s", json.dumps(record, ensure_ascii=False, default=str))

        # One-line summary — visible in tail -f without scrolling
        outcome = "ERROR" if failed else "OK"
        total_steps = len(self._steps)
        error_steps = [s.name for s in self._steps if s.error]
        summary_extras = ""
        if error_steps:
            summary_extras = f"  errors=[{','.join(error_steps)}]"
        # Pull key globals for the summary line
        for key in ("reply_preview", "agent_iterations", "memory_updates", "total_ms"):
            pass  # already in record above
        logger.info(
            "TRACE  %s  %s  %.0fms  iter=%s  mem=%s  reply=%s chars%s",
            outcome,
            self.event_id[:16],
            total_ms,
            self._global_tags.get("agent_iterations", 0),
            self._global_tags.get("memory_updates", 0),
            self._global_tags.get("reply_len", 0),
            summary_extras,
        )
