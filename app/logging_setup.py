"""
logging_setup.py — Logging configuration for Shimmi v2.5.0.

Goals:
  - Complete silence on uvicorn HTTP access lines (POST /webhook 200 OK noise)
  - Silence chromadb telemetry errors (posthog version mismatch, not our bug)
  - Silence httpx request/response lines (internal WAHA + Groq API calls)
  - Silence sentence_transformers model-load chatter
  - Keep app.* loggers at the configured level
  - Keep app.trace always at INFO so waterfall logs always appear
"""
from __future__ import annotations

import logging
import logging.config
import os


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)-8s %(name)-18s %(message)s",
    )

    logging.basicConfig(level=level, format=fmt, force=True)

    # ── completely silence noisy third-party loggers ──────────────────────
    _silence = [
        # uvicorn HTTP access (POST /webhook 200 OK — pure noise)
        "uvicorn.access",
        # chromadb posthog telemetry (broken capture() signature — not our bug)
        "chromadb",
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.posthog",
        # httpx/httpcore — internal WAHA + Groq API request lines
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        # sentence_transformers model loading chatter
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        # Groq SDK internals
        "groq",
        "groq._base_client",
        # openai SDK (used under groq)
        "openai",
    ]
    for name in _silence:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # ── keep uvicorn server lifecycle messages (startup, shutdown) ─────────
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # ── trace logger always visible ────────────────────────────────────────
    logging.getLogger("app.trace").setLevel(logging.INFO)
    logging.getLogger("app.tx").setLevel(logging.INFO)
