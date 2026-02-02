# app/handlers.py
from __future__ import annotations
import re
from .db import add_collection_items, mark_collection_item, clear_done, fetchall, upsert_fact

LIST_DEFAULT = "shopping"

async def handle_commands(sender_id: str, chat_id: str, text: str) -> bool:
    t = (text or '').strip().lower()

    if t in {"help","help me","capabilities","what can you do"}:
        # The caller (main) replies with a short "Done." to keep the MVP simple,
        # but you can return a string and send a full menu here.
        return True

    # Show lists
    if re.search(r"\bmy\s+lists\b", t) or re.fullmatch(r"(show|list|view)\s+lists", t):
        rows = await fetchall("SELECT name FROM user_collections WHERE sender_id=? ORDER BY updated_at DESC", (sender_id,))
        _ = [r[0] for r in rows]
        return True

    # Add items: "add milk, eggs to shopping list"
    m_add = re.match(r"^add\s+(.+?)\s+to\s+(.+?)\s+list$", t)
    if m_add:
        items_text = m_add.group(1)
        list_name = m_add.group(2).strip()
        items = []
        for chunk in re.split(r"[,;]", items_text):
            p = chunk.strip()
            if not p: continue
            items.append({"item_text": p, "qty": None, "unit": None})
        await add_collection_items(sender_id, list_name, items)
        return True

    # Mark done: "mark milk done in shopping"
    m_mark = re.match(r"^mark\s+(.+?)\s+done\s+in\s+(.+?)$", t)
    if m_mark:
        item = m_mark.group(1).strip(); lst = m_mark.group(2).strip()
        await mark_collection_item(sender_id, lst, item, status='done')
        return True

    # Clear done
    m_clear = re.match(r"^clear\s+done(\s+in\s+(.+))?$", t)
    if m_clear:
        lst = (m_clear.group(2) or LIST_DEFAULT).strip()
        await clear_done(sender_id, lst)
        return True

    # Simple fact: "city: Hyderabad"
    m_fact = re.match(r"^(city|language|timezone)\s*:\s*(.+)$", (text or '').strip(), re.IGNORECASE)
    if m_fact:
        k = m_fact.group(1).strip().lower(); v = m_fact.group(2).strip()
        await upsert_fact(sender_id, k, v, 'text', 0.95)
        return True

    return False

