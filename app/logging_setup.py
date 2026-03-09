"""
logging_setup.py — Shimmi v2.8.0

Silenced (CRITICAL — never shown):
  uvicorn.access               — POST /webhook 200 OK noise
  uvicorn.protocols.*          — "Invalid HTTP request received" (port scanners)
  chromadb.*                   — posthog telemetry API mismatch
  httpx / httpcore             — internal WAHA + Groq request lines
  sentence_transformers.*      — model-load chatter
  groq.* / openai.*            — SDK internals

Kept at INFO (shows startup/shutdown + application ready line):
  uvicorn / uvicorn.error      — shows "Uvicorn running on http://..." ✓

Always at INFO:
  app.trace                    — per-message waterfall
  app.tx                       — TX summary lines
"""
from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level      = getattr(logging, level_name, logging.INFO)
    fmt        = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)-8s %(name)-18s %(message)s",
    )
    logging.basicConfig(level=level, format=fmt, force=True)

    # ── completely silence noisy third-party loggers ──────────────────────
    _silence = [
        "uvicorn.access",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.websockets_impl",
        "chromadb", "chromadb.telemetry",
        "chromadb.telemetry.product", "chromadb.telemetry.product.posthog",
        "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
        "sentence_transformers", "sentence_transformers.SentenceTransformer",
        "groq", "groq._base_client",
        "openai", "openai._base_client",
    ]
    for name in _silence:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # ── show uvicorn startup message ("Uvicorn running on http://…") ──────
    # Setting to INFO here means you'll see the "Listening on" line at startup
    # plus any warnings/errors. Access log (200 OK) is silenced above.
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    # ── trace + tx always visible ──────────────────────────────────────────
    logging.getLogger("app.trace").setLevel(logging.INFO)
    logging.getLogger("app.tx").setLevel(logging.INFO)
