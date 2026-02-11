from __future__ import annotations

from .env_loader import load_env
load_env(override=False)

import logging
import os
import re
from typing import Dict

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s:%(message)s"
_SECRET = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|HMAC)", re.I)


def setup_logging() -> None:
    fmt = os.getenv('LOG_FORMAT', DEFAULT_FORMAT)
    lvl = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO), format=fmt, force=True)

    access_level = os.getenv('ACCESS_LOG_LEVEL', 'WARNING').upper()
    logging.getLogger('uvicorn.access').setLevel(getattr(logging, access_level, logging.WARNING))
    logging.getLogger('httpx').setLevel(logging.WARNING)


def mask_env(env: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in env.items():
        out[k] = '***' if (_SECRET.search(k) and v) else v
    return out


def log_startup_env(logger: logging.Logger, *, keys: list[str]) -> None:
    env = mask_env({k: os.getenv(k, '') for k in keys})
    logger.info('🧩 env.loaded %s', ' '.join([f"{k}={env.get(k,'')}" for k in keys]))


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    def _s(v):
        if v is None:
            return ''
        s = str(v)
        return s if len(s) <= 400 else s[:400] + '…'
    logger.info('%s %s', event, ' '.join([f"{k}={_s(v)}" for k, v in fields.items()]))
