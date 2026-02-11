from __future__ import annotations

"""Lightweight .env loader (no external deps).

Loads KEY=VALUE pairs into os.environ so os.getenv works.

Order:
1) ENV_FILE if set
2) /opt/shimmi/.env
3) ./.env

Does not override existing env unless override=True.
"""

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _parse(line: str) -> Optional[tuple[str, str]]:
    s = (line or '').strip()
    if not s or s.startswith('#'):
        return None
    if s.lower().startswith('export '):
        s = s[7:].lstrip()
    if '=' not in s:
        return None
    k, v = s.split('=', 1)
    k = k.strip(); v = v.strip()
    if not k:
        return None
    if v and v[0] not in ('"', "'"):
        if ' #' in v:
            v = v.split(' #', 1)[0].rstrip()
    return k, _strip_quotes(v)


def load_env(*, override: bool = False) -> Optional[str]:
    global _LOADED
    if _LOADED:
        return None

    candidates = []
    env_file = os.getenv('ENV_FILE', '').strip()
    if env_file:
        candidates.append(Path(env_file))
    candidates.extend([Path('/opt/shimmi/.env'), Path.cwd() / '.env'])

    chosen: Optional[Path] = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                chosen = p
                break
        except Exception:
            continue

    if not chosen:
        _LOADED = True
        return None

    try:
        for line in chosen.read_text(encoding='utf-8', errors='ignore').splitlines():
            kv = _parse(line)
            if not kv:
                continue
            k, v = kv
            if not override and k in os.environ and os.environ[k] != '':
                continue
            os.environ[k] = v
    finally:
        _LOADED = True

    return str(chosen)
