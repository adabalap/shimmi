from __future__ import annotations

from typing import List, Tuple


def format_history(history: List[Tuple[str, str, str]], max_turns: int = 10) -> str:
    return "\n".join([f"{role.upper()}: {content}" for _ts, role, content in history[-max_turns:]])


def format_facts(facts: list[tuple[str, str, float]], max_items: int = 40) -> str:
    if not facts:
        return ''
    lines = ['FACTS']
    for k, v, c in facts[:max_items]:
        lines.append(f"{k}: {v} (conf={c:.2f})")
    return "\n".join(lines)


def build_system(persona: str) -> str:
    return "\n".join([
        f"You are {persona}, a privacy-safe stateful WhatsApp assistant.",
        "You may use FACTS, SNIPPETS, and HISTORY for grounding.",
        "CRITICAL: Never invent personal details (name, age, location, job, relationships).",
        "Only mention a personal detail if it appears verbatim in FACTS/HISTORY/SNIPPETS.",
        "Do NOT address the user by name unless FACTS contains key 'name' or the user explicitly said their name in HISTORY.",
        "If uncertain, say you don't know and ask a brief follow-up question.",
        "Keep responses concise: 1–4 short sentences.",
        "Avoid markdown formatting.",
    ])


def build_user_prompt(user_text: str, *, facts_block: str = '', snippets: str = '', history_block: str = '') -> str:
    parts = [f"USER: {user_text}"]
    if facts_block:
        parts.append(facts_block)
    if snippets:
        parts.append('SNIPPETS\n' + snippets)
    if history_block:
        parts.append('HISTORY\n' + history_block)
    return "\n\n".join(parts)
