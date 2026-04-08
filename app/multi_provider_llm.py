"""
Improved Multi-Provider LLM Engine with Smart Fallbacks
Addresses: Model rotation, rate limit handling, graceful degradation
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from groq import Groq
import google.generativeai as genai

from .config import settings
from .retry import async_retry

logger = logging.getLogger("app.agent")


# ============================================================================
# PROVIDER MANAGEMENT
# ============================================================================

@dataclass
class ProviderStatus:
    """Track health and availability of each provider"""
    name: str
    available: bool = True
    circuit_open_until: float = 0.0
    rate_limit_until: float = 0.0
    consecutive_failures: int = 0
    daily_tokens_used: int = 0
    daily_token_limit: int = 100000
    last_reset: float = field(default_factory=time.time)
    
    def is_available(self) -> bool:
        """Check if provider is currently available"""
        now = time.time()
        
        # Reset daily counters if needed (24h cycle)
        if now - self.last_reset > 86400:
            self.daily_tokens_used = 0
            self.last_reset = now
        
        # Check circuit breaker
        if now < self.circuit_open_until:
            return False
        
        # Check rate limit
        if now < self.rate_limit_until:
            return False
        
        # Check daily quota
        if self.daily_tokens_used >= self.daily_token_limit * 0.95:  # 95% threshold
            return False
        
        return self.available
    
    def record_success(self, tokens_used: int):
        """Record successful request"""
        self.consecutive_failures = 0
        self.daily_tokens_used += tokens_used
        logger.info(
            "provider.success provider=%s tokens_used=%d daily_total=%d",
            self.name, tokens_used, self.daily_tokens_used
        )
    
    def record_failure(self, is_rate_limit: bool = False):
        """Record failed request"""
        self.consecutive_failures += 1
        
        if is_rate_limit:
            # Rate limit hit - disable for 5-10 minutes
            cooldown = 300 + random.randint(0, 300)  # 5-10 minutes
            self.rate_limit_until = time.time() + cooldown
            logger.warning(
                "provider.rate_limited provider=%s cooldown=%ds",
                self.name, cooldown
            )
        elif self.consecutive_failures >= 3:
            # Multiple failures - open circuit for 30-60 seconds
            cooldown = 30 + random.randint(0, 30)
            self.circuit_open_until = time.time() + cooldown
            logger.error(
                "provider.circuit_opened provider=%s failures=%d cooldown=%ds",
                self.name, self.consecutive_failures, cooldown
            )


class MultiProviderLLM:
    """Manages multiple LLM providers with intelligent failover"""
    
    def __init__(self):
        self.providers: Dict[str, ProviderStatus] = {}
        self.groq_client: Optional[Groq] = None
        self.gemini_models: Dict[str, Any] = {}
        self._init_lock = asyncio.Lock()
        self._inflight = asyncio.Semaphore(5)
        
    async def initialize(self):
        """Initialize all available providers"""
        async with self._init_lock:
            # Initialize Groq
            if settings.groq_api_key:
                self.groq_client = Groq(
                    api_key=settings.groq_api_key,
                    timeout=settings.groq_timeout
                )
                self.providers["groq"] = ProviderStatus(
                    name="groq",
                    daily_token_limit=100000  # Free tier
                )
                logger.info("provider.init provider=groq models=%s", settings.groq_model_pool)
            
            # Initialize Gemini
            gemini_key = getattr(settings, 'gemini_api_key', None)
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.providers["gemini"] = ProviderStatus(
                    name="gemini",
                    daily_token_limit=1500000  # Free tier: 1.5M tokens/day
                )
                
                # Pre-load models
                model_pool = getattr(settings, 'gemini_model_pool', 'gemini-2.0-flash-exp').split(',')
                for model_name in model_pool:
                    model_name = model_name.strip()
                    try:
                        self.gemini_models[model_name] = genai.GenerativeModel(model_name)
                    except Exception as e:
                        logger.warning("gemini.model_load_failed model=%s error=%s", model_name, str(e))
                
                logger.info("provider.init provider=gemini models=%d", len(self.gemini_models))
    
    def _select_provider(self) -> tuple[str, ProviderStatus]:
        """Select best available provider"""
        # Priority order: Groq (fast) > Gemini (fallback)
        for provider_name in ["groq", "gemini"]:
            provider = self.providers.get(provider_name)
            if provider and provider.is_available():
                return provider_name, provider
        
        # No providers available - return least bad option
        if "gemini" in self.providers:
            logger.warning("provider.all_degraded using_gemini_anyway")
            return "gemini", self.providers["gemini"]
        
        if "groq" in self.providers:
            logger.warning("provider.all_degraded using_groq_anyway")
            return "groq", self.providers["groq"]
        
        raise RuntimeError("No LLM providers available")
    
    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> tuple[str, int]:
        """
        Complete a chat request using best available provider.
        Returns: (response_text, tokens_used)
        """
        provider_name, provider = self._select_provider()
        
        start_time = time.time()
        
        try:
            if provider_name == "groq":
                response, tokens = await self._call_groq(system, user, temperature, max_tokens)
            elif provider_name == "gemini":
                response, tokens = await self._call_gemini(system, user, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            duration = time.time() - start_time
            provider.record_success(tokens)
            
            logger.info(
                "llm.success provider=%s duration=%.2fs tokens=%d",
                provider_name, duration, tokens
            )
            
            return response, tokens
            
        except Exception as e:
            duration = time.time() - start_time
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str and "limit" in error_str
            
            provider.record_failure(is_rate_limit=is_rate_limit)
            
            logger.error(
                "llm.failed provider=%s duration=%.2fs error=%s",
                provider_name, duration, str(e)[:200]
            )
            
            # Try fallback provider if available
            if is_rate_limit and provider_name == "groq" and "gemini" in self.providers:
                logger.info("llm.fallback from=groq to=gemini")
                try:
                    return await self._call_gemini(system, user, temperature, max_tokens)
                except Exception as fallback_error:
                    logger.error("llm.fallback_failed error=%s", str(fallback_error)[:200])
            
            raise
    
    async def _call_groq(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int]:
        """Call Groq API"""
        if not self.groq_client:
            raise RuntimeError("Groq client not initialized")
        
        # Pick model from pool
        pool = settings.groq_model_pool or ["llama-3.3-70b-versatile"]
        model = random.choice(pool)
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async def _call():
            async with self._inflight:
                resp = await asyncio.to_thread(
                    lambda: self.groq_client.chat.completions.create(**payload)
                )
                text = (resp.choices[0].message.content or "").strip()
                # Estimate tokens (Groq doesn't always return usage)
                tokens = getattr(resp, 'usage', None)
                if tokens:
                    tokens_used = tokens.total_tokens
                else:
                    # Rough estimate: 1 token ≈ 0.75 words
                    tokens_used = int((len(system.split()) + len(user.split()) + len(text.split())) * 1.33)
                return text, tokens_used
        
        return await async_retry(_call, max_attempts=3, base_delay=0.5, max_delay=5.0)
    
    async def _call_gemini(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int]:
        """Call Gemini API"""
        if not self.gemini_models:
            raise RuntimeError("Gemini models not initialized")
        
        # Pick first available model
        model = list(self.gemini_models.values())[0]
        
        # Gemini doesn't have separate system/user roles in the same way
        # Combine system prompt into the user message
        combined_prompt = f"{system}\n\nUser message: {user}"
        
        async def _call():
            async with self._inflight:
                response = await asyncio.to_thread(
                    lambda: model.generate_content(
                        combined_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                )
                text = response.text.strip()
                
                # Get actual token usage
                try:
                    tokens_used = response.usage_metadata.total_token_count
                except AttributeError:
                    # Fallback estimate
                    tokens_used = int((len(combined_prompt.split()) + len(text.split())) * 1.33)
                
                return text, tokens_used
        
        return await async_retry(_call, max_attempts=3, base_delay=0.5, max_delay=5.0)


# Global instance
llm_manager = MultiProviderLLM()


async def init_llm() -> None:
    """Initialize LLM manager"""
    await llm_manager.initialize()
    logger.info("llm.init_complete providers=%d", len(llm_manager.providers))


async def close_llm() -> None:
    """Cleanup"""
    pass


# ============================================================================
# HIGH-LEVEL API (compatible with existing code)
# ============================================================================

async def smart_complete(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> str:
    """
    Smart completion with automatic provider selection and fallback.
    Returns response text.
    """
    try:
        response, tokens = await llm_manager.complete(system, user, temperature, max_tokens)
        return response
    except Exception as e:
        # All providers failed - return user-friendly error
        logger.exception("llm.all_failed")
        
        # Check if it's all rate limits
        all_rate_limited = all(
            p.rate_limit_until > time.time()
            for p in llm_manager.providers.values()
        )
        
        if all_rate_limited:
            return json.dumps({
                "reply": {
                    "type": "text",
                    "text": "I've hit my daily API limits across all providers. Please try again tomorrow or later today. 🙏"
                },
                "memory_updates": []
            })
        else:
            return json.dumps({
                "reply": {
                    "type": "text",
                    "text": "I'm having technical difficulties. Please try again in a few minutes. 🤖"
                },
                "memory_updates": []
            })


# Example usage in existing code:
async def example_usage():
    """Replace your existing _groq_raw calls with smart_complete"""
    
    # Old way:
    # raw = await _groq_raw(chat_id, PLANNER_PROMPT, user_msg, temperature=0.0, max_tokens=520)
    
    # New way (automatic fallback to Gemini if Groq fails):
    raw = await smart_complete(
        system=PLANNER_PROMPT,
        user=user_msg,
        temperature=0.0,
        max_tokens=520
    )
    
    return raw
