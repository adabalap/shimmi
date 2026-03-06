"""
Token-Optimized Agent Engine
- Production Grade v7.1
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

# --- Utilities moved from utils.py for code locality ---

def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in a string."""
    return len(text or "") // 4

def extract_fact_key(query: str) -> Optional[str]:
    """Extract a potential fact key from a user query."""
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

# --- Pydantic Models ---

class MemoryUpdate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)

class ReplyPayload(BaseModel):
    type: str = Field("text", pattern=r"^(text)$")
    text: str = Field(..., min_length=1)

class AgentResult(BaseModel):
    reply: ReplyPayload
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    question_asked: Optional[str] = None

# --- Core Agent Logic ---

def _extract_json(text: str) -> dict:
    """Finds and parses a JSON object within a string, even if surrounded by other text."""
    s = (text or "").strip()
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON found in string") from e
    raise ValueError("No JSON object found in string")

def _is_question(text: str) -> bool:
    """Check if the text is likely a question."""
    text_lower = text.lower().strip()
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should', 'tell me']
    if any(text_lower.startswith(word) for word in question_words):
        return True
    if text_lower.endswith('?'):
        return True
    return False

def _handle_simple_greeting(text: str) -> Optional[AgentResult]:
    """Handles simple greetings and pleasantries to avoid unnecessary LLM calls."""
    text_lower = (text or "").lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "spock"]
    thanks = ["thank you", "thanks", "thx", "ok", "okay", "got it"]

    if text_lower in greetings:
        reply = "Hello! How can I help you today?"
        return AgentResult(reply=ReplyPayload(type="text", text=reply))
    if text_lower in thanks:
        reply = "You're welcome! Is there anything else I can assist with?"
        return AgentResult(reply=ReplyPayload(type="text", text=reply))
    return None

async def run_agent(
    *,
    user_text: str,
    facts: Dict[str, str],
    context: List[str],
    llm_complete_fn,
    is_follow_up: bool = False,
    last_question: Optional[str] = None
) -> AgentResult:
    
    if greeting_response := _handle_simple_greeting(user_text):
        logger.info("🎯 agent.greeting_handler handled='true'")
        return greeting_response

    if is_follow_up and last_question:
        logger.info("🎯 agent.follow_up_handler triggered")
        if fact_key := extract_fact_key(last_question):
            try:
                prompt = DIRECT_ANSWER_EXTRACTOR_PROMPT.format(question=last_question, answer=user_text, key=fact_key)
                raw = await llm_complete_fn(system="You are a data extraction tool.", user=prompt, temperature=0.0, max_tokens=100)
                if data := _extract_json(raw):
                    if value := data.get('value'):
                        memory_update = MemoryUpdate(key=fact_key, value=value)
                        reply_text = f"Got it. I'll remember that your {fact_key.replace('_', ' ')} is {value}."
                        return AgentResult(reply=ReplyPayload(type="text", text=reply_text), memory_updates=[memory_update])
            except Exception as e:
                logger.error("agent.follow_up_handler.failed err=%s", str(e))
    
    if (fact_key := extract_fact_key(user_text)) and (answer := facts.get(fact_key)):
        response_text = sanitize_for_whatsapp(answer)
        return AgentResult(reply=ReplyPayload(type="text", text=response_text))

    proposed = []
    if not _is_question(user_text):
        try:
            raw = await llm_complete_fn(system=MEMORY_EXTRACTOR_PROMPT, user=user_text, temperature=0.0, max_tokens=300)
            if data := _extract_json(raw):
                proposed = [MemoryUpdate.model_validate(m) for m in data.get("memory_updates", [])]
        except Exception:
            logger.info("memory.extract_failed")

    if (fact_key := extract_fact_key(user_text)) and fact_key not in facts:
        question = f"I don't know your {fact_key.replace('_', ' ')}. Can you tell me?"
        return AgentResult(reply=ReplyPayload(type="text", text=question), memory_updates=proposed, question_asked=question)

    bundle = {"user": user_text, "facts": facts, "context": "\n".join(context)}
    
    try:
        raw = await llm_complete_fn(system=SYSTEM_PROMPT, user=json.dumps(bundle, ensure_ascii=False), temperature=0.1, max_tokens=400)
        data = _extract_json(raw)
    except Exception as e:
        logger.error("agent.failed err=%s", str(e))
        return AgentResult(reply=ReplyPayload(type="text", text="I had trouble processing that. Could you rephrase?"))

    result = AgentResult.model_validate(data)
    result.memory_updates.extend(proposed)
    return result

