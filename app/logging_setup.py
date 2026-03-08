"""
logging_setup.py — Shimmi v2.7.0

Silenced loggers (CRITICAL — never appear in logs):
  uvicorn.access                   — POST /webhook 200 OK noise
  uvicorn.protocols.*              — "Invalid HTTP request received" (port scanners)
  chromadb.*                       — posthog telemetry API mismatch (not our bug)
  httpx / httpcore                 — internal WAHA + Groq request lines
  sentence_transformers.*          — model-load chatter
  groq.* / openai.*                — SDK internals

Kept at WARNING:
  uvicorn / uvicorn.error          — server lifecycle (startup/shutdown)

Always at INFO (regardless of LOG_LEVEL):
  app.trace                        — per-message waterfall
  app.tx                           — TX summary lines
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
        # uvicorn HTTP access log (200 OK lines)
        "uvicorn.access",
        # uvicorn protocol layer — "Invalid HTTP request received"
        # These come from port scanners / TLS probes on a plain HTTP port.
        # They are harmless and entirely actionless — pure noise.
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.websockets_impl",
        # chromadb posthog telemetry (capture() positional arg mismatch)
        "chromadb",
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.posthog",
        # httpx / httpcore — internal WAHA + Groq API call lines
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        # sentence_transformers model-loading verbosity
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        # Groq and openai SDK internals
        "groq",
        "groq._base_client",
        "openai",
        "openai._base_client",
    ]
    for name in _silence:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # ── keep uvicorn server lifecycle messages (startup / shutdown) ────────
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # ── trace + tx always visible ──────────────────────────────────────────
    logging.getLogger("app.trace").setLevel(logging.INFO)
    logging.getLogger("app.tx").setLevel(logging.INFO)
