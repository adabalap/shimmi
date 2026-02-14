from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s:%(message)s")
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=fmt, force=True)

    access_level = os.getenv("ACCESS_LOG_LEVEL", "WARNING").upper()
    logging.getLogger("uvicorn.access").setLevel(getattr(logging, access_level, logging.WARNING))
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # silence chroma telemetry
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
