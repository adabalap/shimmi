# fact_mining.py

import logging
from typing import Callable, Coroutine

from . import database
from .prompts import FACT_MINING_PROMPT # This prompt is now defined in the corrected prompts.py

logger = logging.getLogger("app.fact_mining")

async def mine_and_store_facts(
    whatsapp_id: str,
    text: str,
    llm_func: Callable[[str, str], Coroutine[None, None, str]],
) -> None:
    """
    Uses an LLM to find storable facts in user text and saves them to the database.
    """
    logger.info("fact_mining.starting user_text='%s'", text)
    try:
        # Ask the LLM to extract facts
        facts_text = await llm_func(system=FACT_MINING_PROMPT, user=text)
        
        if not facts_text or facts_text.strip().lower() == "none":
            logger.info("fact_mining.no_facts_found")
            return

        # The LLM should return facts in a "key: value" format, one per line
        for line in facts_text.strip().split('\n'):
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip().replace(" ", "_").lower()
            value = value.strip()
            if key and value:
                await database.sqlite_store.upsert_fact(whatsapp_id, key, value)
                logger.info("fact_mining.fact_stored key=%s", key)

    except Exception:
        logger.exception("fact_mining.failed")

