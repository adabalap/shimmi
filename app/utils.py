"""
Core Utilities
- WhatsApp ID normalization
- Hashing
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

def sha1_hex(data: str | bytes) -> str:
    """
    Return SHA1 hex digest of input.
    Accepts str (UTF-8 encoded) or bytes.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()

def normalize_whatsapp_id(raw_id: str) -> str:
    """
    Normalize WhatsApp ID to ensure consistency for USER IDs (not group/chat IDs).
    - Keeps groups as-is (@g.us)
    - Strips suffixes for user ids: @lid, @c.us, @s.whatsapp.net
    """
    if not raw_id:
        return ""
    if "@g.us" in raw_id:
        return raw_id
    return raw_id.split("@")[0]

def canonical_text(text: str) -> str:
    """Return a canonical version of text for hashing and comparison."""
    return (text or "").lower().strip()

def sanitize_for_whatsapp(text: str) -> str:
    """Sanitizes text for WhatsApp's formatting."""
    return (text or "").strip()

