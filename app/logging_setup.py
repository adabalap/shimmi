"""
logging_setup.py — Shimmi v2.8.0

Changes vs v2.7.0:
  - uvicorn.error set to INFO (not WARNING) so the startup "Uvicorn running on…"
    message appears — this is the startup listener log the user needs.
  - uvicorn.protocols.* stays at CRITICAL (port-scanner noise unchanged)
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
        "chromadb",
        "chromadb.telemetry",
        "chromadb.telemetry.product",
        "chromadb.telemetry.product.posthog",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "groq",
        "groq._base_client",
        "openai",
        "openai._base_client",
    ]
    for name in _silence:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # ── uvicorn server lifecycle — INFO so startup message appears ────────
    # This lets through: "Uvicorn running on http://0.0.0.0:6000 (Press CTRL+C to quit)"
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    # ── trace + tx always visible ──────────────────────────────────────────
    logging.getLogger("app.trace").setLevel(logging.INFO)
    logging.getLogger("app.tx").setLevel(logging.INFO)
