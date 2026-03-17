"""
logging_setup.py — Shimmi v3.2.0

Changes vs v2.8.0:
  FIX   Added _InvalidHttpFilter to suppress "Invalid HTTP request received"
        WARNING spam from port-scanners and health-check probes that hit the
        plain-HTTP port with TLS or non-HTTP traffic. Completely harmless but
        previously flooded the log on every port scan.
  KEEP  uvicorn.error stays at INFO so the startup "Uvicorn running on…"
        message still appears.
  KEEP  uvicorn.protocols.* stays at CRITICAL (port-scanner noise).
"""
from __future__ import annotations

import logging
import os


class _InvalidHttpFilter(logging.Filter):
    """Suppress uvicorn's harmless 'Invalid HTTP request received' warnings."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid HTTP request received" not in record.getMessage()


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
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    # ── suppress port-scanner "Invalid HTTP request received" noise ───────
    _filter = _InvalidHttpFilter()
    logging.getLogger("uvicorn.error").addFilter(_filter)
    logging.getLogger("uvicorn").addFilter(_filter)

    # ── trace + tx always visible ─────────────────────────────────────────
    logging.getLogger("app.trace").setLevel(logging.INFO)
    logging.getLogger("app.tx").setLevel(logging.INFO)
