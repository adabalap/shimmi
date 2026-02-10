# app/handlers.py
from __future__ import annotations

import re
from typing import Tuple

from .db import (
    add_collection_items,
    mark_collection_item,
    clear_done,
    fetchall,
    upsert_fact,
    execute,
)

LIST_DEFAULT = "shopping"


async def handle_commands(sender_id: str, chat_id: str, text: str) -> Tuple[bool, str]:
    raw = (text or "").strip()
    t = raw.lower()

    # ---------- Help ----------
    if t in {"help", "help me", "capabilities", "what can you do"}:
        return True, """I can help with:
• Q&A
• Remember facts you explicitly share (e.g., “my name is …”, “I live in …”)
• Lists: add / mark / clear / show
• Admin (groups): /observe on|off|status|retention 30d|redaction on|off
"""

    # ---------- Privacy ----------
    if t in {"/forget me", "forget me"}:
        await execute("DELETE FROM user_facts WHERE sender_id=?", (sender_id,))
        await execute("DELETE FROM user_records WHERE sender_id=?", (sender_id,))
        return True, "✅ Done. I’ve forgotten your saved facts and records."

    if t in {"/purge", "/purge chat"}:
        # Purge ambient chat RAG memory (if chroma tables exist)
        try:
            await execute("DELETE FROM rag_chunks WHERE chat_id=?", (chat_id,))
            return True, "✅ Done. I’ve purged ambient memory for this chat."
        except Exception:
            # If chroma tables aren’t present, do nothing
            return True, "✅ Done."

    # ---------- Show lists ----------
    if re.search(r"\bmy\s+lists\b", t) or re.fullmatch(r"(show|list|view)\s+lists", t):
        rows = await fetchall(
            "SELECT name FROM user_collections WHERE sender_id=? ORDER BY updated_at DESC",
            (sender_id,),
        )
        names = [r[0] for r in rows]
        if not names:
            return True, "You don’t have any lists yet. Try: add milk, eggs to shopping list"
        return True, "Your lists: " + ", ".join(names[:20])

    # ---------- Show items in a list ----------
    m_show = re.match(r"^(show|list|view)\s+(.+?)(?:\s+list)?$", raw, re.IGNORECASE)
    if m_show:
        lst = (m_show.group(2) or LIST_DEFAULT).strip()
        rows = await fetchall(
            """SELECT ci.item_text, ci.status
               FROM collection_items ci
               JOIN user_collections uc ON uc.id=ci.collection_id
              WHERE uc.sender_id=? AND uc.name=?
              ORDER BY ci.updated_at DESC LIMIT 50""",
            (sender_id, lst),
        )
        if not rows:
            return True, f"Your '{lst}' list is empty."
        open_items = [r[0] for r in rows if r[1] == "open"]
        done_items = [r[0] for r in rows if r[1] != "open"]
        msg = f"{lst} (open): " + (", ".join(open_items) if open_items else "none")
        if done_items:
            msg += "\nDone: " + ", ".join(done_items[:10])
        return True, msg

    # ---------- Add items ----------
    m_add = re.match(r"^add\s+(.+?)(?:\s+to\s+(.+?))?(?:\s+list)?$", raw, re.IGNORECASE)
    if m_add:
        items_text = (m_add.group(1) or "").strip()
        list_name = (m_add.group(2) or LIST_DEFAULT).strip()
        items = []
        for chunk in re.split(r"[,;]|\band\b", items_text, flags=re.IGNORECASE):
            p = chunk.strip()
            if not p:
                continue
            items.append({"item_text": p[:120], "qty": None, "unit": None})
        n = await add_collection_items(sender_id, list_name, items)
        return True, f"✅ Added {n} item(s) to {list_name}."

    # ---------- Mark done ----------
    m_mark = re.match(
        r"^mark\s+(.+?)\s+(?:as\s+)?done(?:\s+in\s+(.+?))?(?:\s+list)?$",
        raw,
        re.IGNORECASE,
    )
    if m_mark:
        item = m_mark.group(1).strip()
        lst = (m_mark.group(2) or LIST_DEFAULT).strip()
        ok = await mark_collection_item(sender_id, lst, item, status="done")
        return True, ("✅ Marked done." if ok else f"Couldn’t find '{item}' as open in {lst}.")

    # ---------- Clear done ----------
    m_clear = re.match(r"^clear\s+done(?:\s+in\s+(.+))?$", raw, re.IGNORECASE)
    if m_clear:
        lst = (m_clear.group(1) or LIST_DEFAULT).strip()
        n = await clear_done(sender_id, lst)
        return True, ("Nothing to clear." if n == 0 else f"✅ Cleared {n} done item(s) from {lst}.")

    # ---------- Simple facts ----------
    m_fact = re.match(r"^(city|language|timezone)\s*:\s*(.+)$", raw, re.IGNORECASE)
    if m_fact:
        k = m_fact.group(1).strip().lower()
        v = m_fact.group(2).strip()[:256]
        await upsert_fact(sender_id, k, v, "text", 0.95, namespace="default")
        return True, f"✅ Saved {k}: {v}"

    return False, ""
