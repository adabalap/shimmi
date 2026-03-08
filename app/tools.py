# tools.py

import asyncio
import logging
from typing import Dict, List, Any

from . import database
from .structured_actions import actions_store
from .reminder_manager import ReminderManager
from .reminder_commands import handle_reminder_command

logger = logging.getLogger("app.tools")

# --- Fact Management Tools ---

async def get_user_facts(whatsapp_id: str) -> Dict[str, Any]:
    """Tool to retrieve all known facts about a user."""
    if not database.sqlite_store:
        return {}
    logger.info("tool.get_user_facts user=%s", whatsapp_id)
    return await database.sqlite_store.get_all_facts(whatsapp_id)

async def upsert_user_fact(whatsapp_id: str, key: str, value: Any) -> str:
    """Tool to add or update a single fact about the user."""
    if not database.sqlite_store:
        return "Database not configured."
    logger.info("tool.upsert_user_fact user=%s key=%s", whatsapp_id, key)
    status = await database.sqlite_store.upsert_fact(whatsapp_id, key, str(value))
    return f"Fact '{key}' was {status}."

# --- Structured Action Tools (Lists) ---

async def manage_list(whatsapp_id: str, list_name: str, action: str, items: List[str] = None) -> str:
    """Tool to manage user lists (create, add, view)."""
    if not actions_store:
        return "List functionality is not enabled."

    if action == "create":
        await actions_store.create_list(whatsapp_id, list_name)
        if items:
            await actions_store.add_to_list(whatsapp_id, list_name, items)
            return f"Created list '{list_name}' with {len(items)} items."
        return f"Created empty list '{list_name}'."

    elif action == "add":
        if not items:
            return "You must provide items to add."
        result = await actions_store.add_to_list(whatsapp_id, list_name, items)
        return f"Added {result['count']} items to '{list_name}'."

    elif action == "view":
        list_items = await actions_store.get_list_items(whatsapp_id, list_name)
        if not list_items:
            return f"Your '{list_name}' list is empty."
        formatted_items = "\n".join(f"• {item.capitalize()}" for item in list_items)
        return f"📋 Here's your '{list_name}' list:\n{formatted_items}"

    else:
        return f"Unknown list action: {action}"

# --- Reminder Tool ---

async def manage_reminders(text: str, whatsapp_id: str, chat_id: str, manager: ReminderManager) -> str:
    """Tool to create, list, or manage reminders using natural language."""
    if not manager:
        return "Reminder functionality is not enabled."
    logger.info("tool.manage_reminders text='%s'", text)
    response = await handle_reminder_command(text, whatsapp_id, chat_id, manager)
    return response or "I couldn't understand that reminder command. Try 'remind me to call John tomorrow at 5pm'."

# --- Contextual Search Tool ---

async def search_ambient_memory(chat_id: str, query: str, k: int = 5) -> List[str]:
    """Tool to search the conversation history for relevant context."""
    if not database.chroma_store:
        return []
    logger.info("tool.search_ambient_memory query='%s'", query)
    snippets = await database.chroma_store.search(chat_id=chat_id, query=query, k=k)
    return [s.text for s in snippets]


