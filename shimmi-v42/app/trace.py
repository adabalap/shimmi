"""
trace.py — Structured per-message trace with rich step logging.

Every message creates a Trace.  Each processing phase appends a TraceStep.
At the end we emit:
  1. A structured JSON log line  (machine-readable, grep/jq friendly)
  2. A human-readable waterfall  (instantly scannable in journalctl / tail)

The waterfall makes it easy to see at a glance:
  - Which phases ran and in what order
  - How long each took
  - What data flowed through (facts, memory, context counts, reply preview)
  - Whether anything failed
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

        # ── 1. structured JSON ────────────────────────────────────────────
        record = {
            "trace": {
                "event_id":  self.event_id,
                "chat_id":   self.chat_id,
                "sender_id": self.sender_id,
                "total_ms":  total_ms,
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

        # ── 2. human-readable waterfall ───────────────────────────────────
        failed = any(s.error for s in self._steps) or "fatal_error" in self._global_tags
        outcome_icon = "✗ ERROR" if failed else "✓ OK"

        lines = [
            "",
            f"╔══ MSG TRACE ══════════════════════════════════════════════",
            f"║  event    {self.event_id}",
            f"║  chat     {self.chat_id}",
            f"║  sender   {self.sender_id}",
            f"║  outcome  {outcome_icon}  ({total_ms} ms total)",
            f"╠══ STEPS ══════════════════════════════════════════════════",
        ]

        for s in self._steps:
            icon = "✗" if s.error else "✓"
            tag_parts = []
            for k, v in s.tags.items():
                v_str = str(v)
                # Truncate long values in the waterfall only
                if len(v_str) > 120:
                    v_str = v_str[:117] + "…"
                tag_parts.append(f"{k}={v_str}")
            tag_str = "  ".join(tag_parts)
            err_str = f"\n║    ↳ ERROR: {s.error}" if s.error else ""
            lines.append(
                f"║  {icon} {s.name:<26} {s.duration_ms:>8.1f}ms"
                + (f"   {tag_str}" if tag_str else "")
                + err_str
            )

        if self._global_tags:
            lines.append(f"╠══ GLOBALS ═════════════════════════════════════════════")
            for k, v in self._global_tags.items():
                lines.append(f"║  {k}={v}")

        lines.append(f"╚═══════════════════════════════════════════════════════════")
        logger.info("\n".join(lines))
