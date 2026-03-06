"""
Agent Engine
- Production Grade v7.3
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .prompts import (
    SYSTEM_PROMPT,
    DIRECT_ANSWER_EXTRACTOR_PROMPT,
    MEMORY_EXTRACTOR_PROMPT,
)
from .utils import sanitize_for_whatsapp

logger = logging.getLogger("app.agent")

# --- Utilities ---
def estimate_tokens(text: str) -> int:
    return len(text or "") // 4

def extract_fact_key(query: str) -> Optional[str]:
    q = (query or "").lower()
    patterns = {
        "favorite drink": "favorite_drink", "favourite drink": "favorite_drink",
        "morning drink": "favorite_morning_drink", "bike": "bike_model",
        "vehicle": "vehicle_model", "car": "vehicle_model", "city": "city",
        "where.*live": "city", "location": "location", "name": "name",
    }
    for pattern, fact_key in patterns.items():
        if re.search(pattern, q):
            return fact_key
    return None

def _extract_json(text: str) -> Optional[Dict]:
    s = (text or "").strip()
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def _is_question(text: str) -> bool:
    text_lower = text.lower().strip()
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'do', 'is', 'are', 'can', 'tell me']
    if any(text_lower.startswith(word) for word in question_words) or text_lower.endswith('?'):
        return True
    return False

# --- Pydantic Models ---
class MemoryUpdate(BaseModel):
    key: str
    value: str

class ReplyPayload(BaseModel):
    type: str = "text"
    text: str

class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    question_asked: Optional[str] = None

# --- Agent Core Logic ---

async def run_agent(
    *,
    user_text: str,
    facts: Dict[str, str],
    context: List[str],
    llm_complete_fn,
    is_follow_up: bool = False,
    last_question: Optional[str] = None
) -> AgentResult:
    
    # 1. Handle direct follow-up answers to bot's questions
    if is_follow_up and last_question:
        if fact_key := extract_fact_key(last_question):
            try:
                prompt = DIRECT_ANSWER_EXTRACTOR_PROMPT.format(question=last_question, answer=user_text, key=fact_key)
                raw = await llm_complete_fn(system="You are a data extraction tool.", user=prompt, temperature=0.0, max_tokens=100)
                if (data := _extract_json(raw)) and (value := data.get('value')):
                    if str(value).lower() != 'none':
                        memory_update = MemoryUpdate(key=fact_key, value=value)
                        reply_text = f"Got it. I'll remember that."
                        return AgentResult(reply=ReplyPayload(text=reply_text), memory_updates=[memory_update])
            except Exception as e:
                logger.error("agent.follow_up_handler.failed err=%s", str(e))

    # 2. Handle questions about existing facts
    if _is_question(user_text):
        if (fact_key := extract_fact_key(user_text)):
            # If we have a valid fact, answer from it
            if (answer := facts.get(fact_key)) and str(answer).lower() != 'none':
                return AgentResult(reply=ReplyPayload(text=str(answer)))
            # Otherwise, ask for the missing fact
            else:
                question = f"I don't know your {fact_key.replace('_', ' ')}. Can you tell me?"
                return AgentResult(reply=ReplyPayload(text=question), question_asked=question)

    # 3. Handle statements: try to extract new facts
    proposed_updates = []
    if not _is_question(user_text):
        try:
            raw = await llm_complete_fn(system=MEMORY_EXTRACTOR_PROMPT, user=user_text, temperature=0.0, max_tokens=300)
            if (data := _extract_json(raw)) and (updates := data.get("memory_updates")):
                proposed_updates = [MemoryUpdate.model_validate(m) for m in updates]
        except Exception:
            logger.info("memory.extract_failed")
    
    # If we extracted a fact, confirm with the user
    if proposed_updates:
        # For simplicity, just confirm the first one for now
        fact = proposed_updates[0]
        reply_text = f"Did you say your {fact.key.replace('_', ' ')} is {fact.value}? I'll remember that."
        return AgentResult(reply=ReplyPayload(text=reply_text), memory_updates=proposed_updates)

    # 4. Fallback to a general conversational response
    bundle = {"user": user_text, "facts": facts, "context": "\n".join(context)}
    try:
        raw = await llm_complete_fn(system=SYSTEM_PROMPT, user=json.dumps(bundle, ensure_ascii=False), temperature=0.1, max_tokens=400)
        data = _extract_json(raw)
        return AgentResult.model_validate(data)
    except Exception as e:
        logger.error("agent.fallback.failed err=%s", str(e))
        return AgentResult(reply=ReplyPayload(text="I'm not sure how to respond to that. Can you rephrase?"))

