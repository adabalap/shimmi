from __future__ import annotations

import re
from typing import Tuple

from .db import add_collection_items, mark_collection_item, clear_done, fetchall, upsert_fact, execute

LIST_DEFAULT = "shopping"


async def handle_commands(sender_id: str, chat_id: str, text: str) -> Tuple[bool, str]:
    raw = (text or "").strip()
    t = raw.lower()

    if t in {"help", "help me", "capabilities", "what can you do"}:
        return True, """I can help with:
• Q&A
• Remember facts you explicitly share (e.g., “my name is …”, “I live in …”)
• Lists: add / mark / clear / show
• Ambient observe (groups): /observe on|off|status|retention 30d|redaction on|off
"""

    if t in {"/forget me", "forget me"}:
        await execute("DELETE FROM user_facts WHERE sender_id=?", (sender_id,))
        await execute("DELETE FROM user_records WHERE sender_id=?", (sender_id,))
        return True, "✅ Done. I’ve forgotten your saved facts and records."

    return False, ""
