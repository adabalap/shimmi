"""
Agent Engine with Hallucination Detection
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .retry import async_retry
from .prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
    VERIFIER_PROMPT,
    REPAIR_PROMPT,
    FORMATTER_PROMPT,
)
from .utils import sanitize_for_whatsapp

logger = logging.getLogger("app.agent")


class MemoryUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    source: str = Field(default="user_stated")


class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)


class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


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


def _detect_hallucination(response_text: str, facts: Dict[str, str]) -> bool:
    """
    Detect if response contains hallucinated claims.
    Returns True if hallucination detected.
    """
    response_lower = response_text.lower()
    
    # Red flag phrases indicating claimed knowledge
    claim_phrases = [
        "your favorite", "you mentioned", "you told me",
        "you said", "you prefer", "you like",
        "your route", "your drink", "your place",
        "based on our conversations", "i remember you",
        "you love", "you enjoy"
    ]
    
    for phrase in claim_phrases:
        if phrase in response_lower:
            # Check if we actually have supporting facts
            has_supporting_fact = False
            
            # Extract key words from the claim
            words = response_lower.split()
            phrase_idx = response_lower.find(phrase)
            context = response_lower[max(0, phrase_idx):min(len(response_lower), phrase_idx+100)]
            
            # Check if any fact values appear in the claim
            for fact_key, fact_value in facts.items():
                if fact_value.lower() in context:
                    has_supporting_fact = True
                    break
            
            if not has_supporting_fact:
                logger.warning(
                    "hallucination.detected phrase='%s' context='%s' facts=%s",
                    phrase, context[:50], list(facts.keys())
                )
                return True
    
    return False


async def run_agent(
    *,
    chat_id: str,
    user_text: str,
    facts: Dict[str, str],
    context: List[Dict[str, Any]],
    llm_complete_fn,
    whatsapp_id: str = None  # NEW: Pass user ID for tracking
) -> AgentResult:
    """
    Main agent logic with hallucination detection.
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
        logger.info("memory.extract_failed err=%s", str(e)[:120])
        proposed = []

    # Verify memory
    from .config import settings
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
                        source="user_stated"
                    ))
        except Exception as e:
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
    
    # CRITICAL: Detect hallucinations
    if _detect_hallucination(result.reply.text, facts):
        logger.error(
            "hallucination.blocked original='%s' facts=%s",
            result.reply.text[:100], list(facts.keys())
        )
        result.reply.text = "I don't have that information about you yet. Can you tell me?"
        result.memory_updates = verified  # Reset to only verified

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
