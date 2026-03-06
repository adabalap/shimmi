"""
Core Utilities
- WhatsApp ID normalization
- Hashing
"""
from __future__ import annotations

import hashlib
import re

def sha1_hex(data: str | bytes) -> str:
    """Returns SHA1 hex digest of input."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()

def normalize_whatsapp_id(raw_id: str) -> str:
    """
    Normalizes WhatsApp ID to a consistent format for user identification.
    - Keeps group IDs as-is (@g.us).
    - Strips suffixes for user IDs (@lid, @c.us, etc.).
    """
    if not raw_id:
        return ""
    if "@g.us" in raw_id:
        return raw_id
    return raw_id.split("@")[0]

def canonical_text(text: str) -> str:
    """Returns a canonical version of text for hashing and comparison."""
    return (text or "").lower().strip()

def sanitize_for_whatsapp(text: str) -> str:
    """Sanitizes text for WhatsApp's formatting."""
    return (text or "").strip()


