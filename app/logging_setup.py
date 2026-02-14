from __future__ import annotations

import logging
import os
import re
from typing import Dict

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s:%(message)s"
_SECRET_KEYS = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|HMAC)", re.IGNORECASE)


def setup_logging() -> None:
    fmt = os.getenv("LOG_FORMAT", DEFAULT_FORMAT)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=fmt, force=True)

    access_level = os.getenv("ACCESS_LOG_LEVEL", "WARNING").upper()
    logging.getLogger("uvicorn.access").setLevel(getattr(logging, access_level, logging.WARNING))
    logging.getLogger("httpx").setLevel(logging.WARNING)


def mask_env(env: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in env.items():
        out[k] = "***" if (_SECRET_KEYS.search(k) and v) else (v or "")
    return out


def log_startup_env(logger: logging.Logger, *, keys: list[str]) -> None:
    env = {k: os.getenv(k, "") for k in keys}
    env = mask_env(env)
    msg = " ".join([f"{k}={env.get(k,'')}" for k in keys])
    logger.info("🧩 env.loaded %s", msg)
