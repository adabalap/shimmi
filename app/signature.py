"""
signature.py — Bot message signature.

Appends a discreet emoji + dynamic AI tagline to every outgoing message.
The tagline is deterministically varied based on the message content hash
so it feels organic without being random on retries.
"""
from __future__ import annotations

import hashlib
from typing import List

from .config import settings

# ---------------------------------------------------------------------------
# Tagline pool — AI-themed, conversational, never the same twice in a row
# ---------------------------------------------------------------------------
_TAGLINES: List[str] = [
    "Thinking in tokens, speaking in meaning",
    "Intelligence without edges",
    "Every answer is a new beginning",
    "Learning from the world, replying to you",
    "Where context becomes clarity",
    "Patterns, possibilities, answers",
    "AI that listens before it speaks",
    "Not guessing — reasoning",
    "The future of conversation, today",
    "Your memory, amplified",
    "Curiosity, accelerated",
    "Knowledge at the speed of chat",
    "Trained on the world, focused on you",
    "Understanding is the new search",
    "Here before the question fully formed",
    "Minds made of math, hearts made of intent",
    "One question, infinite paths, best answer",
    "Context-aware, always",
    "Built to get you — not just get back to you",
    "The conversation that never forgets",
]


def _pick_tagline(seed_text: str) -> str:
    """Pick a tagline deterministically from message content — stable on retry."""
    idx = int(hashlib.md5(seed_text.encode("utf-8", errors="replace")).hexdigest(), 16)
    return _TAGLINES[idx % len(_TAGLINES)]


def append_signature(text: str, seed: str = "") -> str:
    """
    Append the bot emoji and a dynamic tagline as a discreet footer.

    The signature is visually separated from the message body and uses
    WhatsApp italic so it's clearly metadata, not content.

        [message text]

        🤖 _Thinking in tokens, speaking in meaning_
    """
    if not text:
        return text

    name = settings.bot_persona_name or "Shimmi"
    tagline = _pick_tagline(seed or text)
    footer = f"\n\n🤖 _{tagline} · {name}_"
    return text.rstrip() + footer
