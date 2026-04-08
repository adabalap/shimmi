"""
Agent Engine - Updated for Multi-Provider LLM
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .config import settings
from .retry import async_retry
from .prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT,
    REPAIR_PROMPT,
    FORMATTER_PROMPT,
    LIVE_SEARCH_PROMPT,
)
from .utils import sanitize_for_whatsapp

logger = logging.getLogger("app.agent")


class MemoryUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    importance: float = Field(default=0.5)
    category: str = Field(default="context")


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


# These are set by multi_provider_llm module
async def init_llm() -> None:
    from .multi_provider_llm import init_llm as llm_init
    await llm_init()


async def close_llm() -> None:
    from .multi_provider_llm import close_llm as llm_close
    await llm_close()


def _extract_json(text: str) -> dict:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty")
    if s.startswith("{"):
        return json.loads(s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])
    raise ValueError("no_json")


async def run_agent(
    *,
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
    llm_complete_fn
) -> AgentResult:
    """
    Main agent logic using multi-provider LLM backend
    """
    
    # Extract memory
    try:
        raw = await llm_complete_fn(
            system=MEMORY_EXTRACTOR_PROMPT,
            user=user_text,
            temperature=0.0,
            max_tokens=450
        )
        data = _extract_json(raw)
        proposed = [MemoryUpdate.model_validate(m) for m in data.get("memory_updates", [])]
    except Exception as e:
        if settings.debug_agent:
            logger.info("memory.extract_failed err=%s", str(e)[:120])
        proposed = []

    # Verify memory
    verified = []
    if proposed and settings.facts_verification:
        try:
            payload = {
                "user_message": user_text,
                "proposed_memory_updates": [m.model_dump() for m in proposed]
            }
            raw = await llm_complete_fn(
                system=VERIFIER_PROMPT,
                user=json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_tokens=450
            )
            data = _extract_json(raw)
            for item in data.get("approved", []):
                if item.get("confidence", 0) >= settings.facts_min_conf:
                    verified.append(MemoryUpdate(
                        key=str(item["key"]).strip(),
                        value=str(item["value"]).strip(),
                        importance=item.get("confidence", 0.5),
                        category="context"
                    ))
        except Exception as e:
            if settings.debug_agent:
                logger.info("memory.verify_failed err=%s", str(e)[:120])

    logger.info("🧠 memory.extracted count=%s verified=%s", len(proposed), len(verified))

    # Plan
    try:
        payload = {"user_message": user_text, "facts": facts, "context": context}
        raw = await llm_complete_fn(
            system=PLANNER_PROMPT,
            user=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            max_tokens=520
        )
        plan = _extract_json(raw)
    except Exception:
        plan = {"mode": "answer"}

    # Check if locale required
    if plan.get("requires_locale") and not _has_locale(facts):
        q = plan.get("question") or "What city and country should I use?"
        return AgentResult(
            reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(q)),
            memory_updates=verified
        )

    # Handle based on mode
    mode = plan.get("mode", "answer")

    if mode == "ask_facts" and plan.get("question"):
        return AgentResult(
            reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(plan["question"])),
            memory_updates=verified
        )

    if mode == "live_search" and settings.live_search_enabled:
        if plan.get("missing_facts"):
            q = plan.get("question") or "What information do you need?"
            return AgentResult(
                reply=ReplyPayload(type="text", text=sanitize_for_whatsapp(q)),
                memory_updates=verified
            )
        
        query = plan.get("search_query", user_text)
        search_payload = {"query": query, "facts": facts}
        
        try:
            ans = await llm_complete_fn(
                system=LIVE_SEARCH_PROMPT,
                user=json.dumps(search_payload, ensure_ascii=False),
                temperature=0.2,
                max_tokens=900
            )
            ans = await _format_whatsapp(ans, llm_complete_fn)
            return AgentResult(
                reply=ReplyPayload(type="text", text=ans),
                memory_updates=verified
            )
        except Exception as e:
            logger.error("live_search.failed err=%s", str(e)[:200])

    # Default: answer mode
    bundle = {"user": user_text, "facts": facts, "context": context}
    
    try:
        raw = await llm_complete_fn(
            system=SYSTEM_PROMPT,
            user=json.dumps(bundle, ensure_ascii=False),
            temperature=0.25,
            max_tokens=900
        )
        data = _extract_json(raw)
    except Exception:
        try:
            repaired = await llm_complete_fn(
                system=REPAIR_PROMPT,
                user=raw,
                temperature=0.0,
                max_tokens=550
            )
            data = _extract_json(repaired)
        except Exception as e:
            logger.error("agent.failed err=%s", str(e)[:200])
            return AgentResult(
                reply=ReplyPayload(type="text", text="I had trouble processing that. Could you rephrase?"),
                memory_updates=verified
            )

    result = AgentResult.model_validate(data)
    result.reply.text = await _format_whatsapp(result.reply.text, llm_complete_fn)

    # Merge memory updates
    all_updates = verified + result.memory_updates
    result.memory_updates = all_updates

    return result


async def _format_whatsapp(text: str, llm_complete_fn) -> str:
    """Format text for WhatsApp"""
    try:
        raw = await llm_complete_fn(
            system=FORMATTER_PROMPT,
            user=text,
            temperature=0.0,
            max_tokens=350
        )
        data = _extract_json(raw)
        return sanitize_for_whatsapp(data.get("text", text))
    except Exception:
        return sanitize_for_whatsapp(text)


def _has_locale(facts: Dict[str, str]) -> bool:
    """Check if user locale is known"""
    for k in ("city", "country", "postal_code", "locale"):
        if (facts.get(k) or "").strip():
            return True
    return False
