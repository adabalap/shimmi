"""
txlog.py — End-to-end transaction tracer for Shimmi v2.4.0.

Every inbound message creates one TxTrace.  Every step in the message
lifecycle calls tx.step(...).  When the transaction finishes, tx.finish()
writes one JSON line to data/traces/YYYY-MM-DD.jsonl — trivially greppable,
tail-able, or loadable into any analytics tool.

Typical trace file entry:
  {"tx_id":"abc123","chat_id":"...","ts_start":"...","total_ms":1240.5,
   "outcome":"sent","steps":[
     {"step":"webhook.recv","elapsed_ms":0.1,"text_chars":18},
     {"step":"llm.agent_t0","elapsed_ms":420.3,"model":"llama-3.3-70b","latency_ms":415.1},
     ...
   ]}
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("app.tx")
UTC = timezone.utc

_TRACES_DIR: Optional[Path] = None


def _get_traces_dir() -> Path:
    global _TRACES_DIR
    if _TRACES_DIR is None:
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        _TRACES_DIR = data_dir / "traces"
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    return _TRACES_DIR


def _write_trace(record: dict) -> None:
    try:
        day  = datetime.now(UTC).strftime("%Y-%m-%d")
        path = _get_traces_dir() / f"{day}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("txlog.write_failed err=%s", exc)


class TxTrace:
    """
    Lightweight per-request tracer.  One instance per inbound message.
    Async-safe: each request owns its own instance.
    """

    __slots__ = ("tx_id", "chat_id", "event_id", "ts_start", "_t0", "steps", "outcome")

    def __init__(self, chat_id: Optional[str], event_id: Optional[str]) -> None:
        self.tx_id:    str            = uuid.uuid4().hex[:12]
        self.chat_id:  Optional[str]  = chat_id
        self.event_id: Optional[str]  = event_id
        self.ts_start: str            = datetime.now(UTC).isoformat()
        self._t0:      float          = time.monotonic()
        self.steps:    List[Dict[str, Any]] = []
        self.outcome:  str            = "unknown"

    @property
    def elapsed_ms(self) -> float:
        return round((time.monotonic() - self._t0) * 1000, 1)

    def step(self, name: str, **data: Any) -> "TxTrace":
        """Record a named step with optional metadata."""
        elapsed = self.elapsed_ms
        entry: Dict[str, Any] = {"step": name, "elapsed_ms": elapsed}
        entry.update(data)
        self.steps.append(entry)

        # Debug log: truncate long strings so log lines stay readable
        preview = {
            k: (str(v)[:100] + "…" if isinstance(v, str) and len(str(v)) > 100 else v)
            for k, v in data.items()
            if k not in ("raw", "context_items", "facts")
        }
        logger.debug(
            "TX[%s] %-30s +%7.1f ms  %s",
            self.tx_id, name, elapsed,
            "  ".join(f"{k}={v}" for k, v in preview.items()),
        )
        return self

    def llm_step(
        self,
        stage: str,
        model: str,
        prompt_chars: int,
        response_chars: int,
        latency_ms: float,
        **extra: Any,
    ) -> "TxTrace":
        """Convenience method for LLM call steps."""
        return self.step(
            f"llm.{stage}",
            model=model,
            prompt_chars=prompt_chars,
            response_chars=response_chars,
            latency_ms=round(latency_ms, 1),
            **extra,
        )

    def finish(self, outcome: str, **extra: Any) -> None:
        """
        Finalise the trace: write JSONL record and emit INFO log.
        Always call this — even on error paths.
        """
        self.outcome  = outcome
        total_ms      = self.elapsed_ms

        record: Dict[str, Any] = {
            "tx_id":    self.tx_id,
            "chat_id":  self.chat_id,
            "event_id": self.event_id,
            "ts_start": self.ts_start,
            "total_ms": total_ms,
            "outcome":  outcome,
            "steps":    self.steps,
        }
        record.update(extra)
        _write_trace(record)

        logger.info(
            "TX[%s] outcome=%-16s total_ms=%8.1f  steps=%d  chat=%s",
            self.tx_id, outcome, total_ms, len(self.steps), self.chat_id,
        )
