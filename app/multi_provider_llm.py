"""
Multi-Provider LLM Manager
Supports: Groq, Gemini (NEW google-genai SDK), Claude, OpenAI
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Optional, Dict, List

from groq import Groq

logger = logging.getLogger("app.llm")


class ProviderStatus:
    """Track provider availability and usage"""
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.available = True
        self.circuit_open_until = 0.0
        self.rate_limit_until = 0.0
        self.daily_tokens_used = 0
        self.daily_token_limit = capacity
        self.last_reset = time.time()
    
    def is_available(self) -> bool:
        now = time.time()
        # Reset daily counter
        if now - self.last_reset > 86400:
            self.daily_tokens_used = 0
            self.last_reset = now
        # Check availability
        if now < self.circuit_open_until or now < self.rate_limit_until:
            return False
        if self.daily_tokens_used >= self.daily_token_limit * 0.95:
            return False
        return self.available
    
    def record_success(self, tokens: int):
        self.daily_tokens_used += tokens
        logger.info("provider.success provider=%s tokens=%d remaining=%d", 
                   self.name, tokens, self.daily_token_limit - self.daily_tokens_used)
    
    def record_failure(self, is_rate_limit: bool = False):
        if is_rate_limit:
            self.rate_limit_until = time.time() + 300  # 5 min cooldown
            logger.warning("provider.rate_limited provider=%s cooldown=300s", self.name)
        else:
            self.circuit_open_until = time.time() + 30  # 30s circuit breaker
            logger.warning("provider.circuit_open provider=%s cooldown=30s", self.name)


class MultiProviderLLM:
    """Manages multiple LLM providers with intelligent failover"""
    
    def __init__(self):
        self.providers: Dict[str, ProviderStatus] = {}
        self.groq_client: Optional[Groq] = None
        self.gemini_client = None
        self.model_rotation: Dict[str, int] = {}  # Track model rotation
        self._inflight = asyncio.Semaphore(5)
        
    async def initialize(self):
        """Initialize all enabled providers"""
        from .config import settings
        
        # Groq
        if settings.groq_enabled and settings.groq_api_key:
            self.groq_client = Groq(
                api_key=settings.groq_api_key,
                timeout=getattr(settings, 'groq_timeout', 30.0)
            )
            self.providers["groq"] = ProviderStatus(
                "groq",
                getattr(settings, 'groq_daily_limit', 100000)
            )
            self.model_rotation["groq"] = 0
            logger.info("provider.init provider=groq models=%s", 
                       getattr(settings, 'groq_models', 'default'))
        
        # Gemini with NEW google-genai SDK
        gemini_key = getattr(settings, 'gemini_api_key', None)
        if settings.gemini_enabled and gemini_key:
            try:
                from google import genai
                # NEW SDK: Create client with API key
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.providers["gemini"] = ProviderStatus(
                    "gemini",
                    getattr(settings, 'gemini_daily_limit', 1500000)
                )
                self.model_rotation["gemini"] = 0
                logger.info("provider.init provider=gemini sdk=google-genai models=%s",
                           getattr(settings, 'gemini_models', 'default'))
            except Exception as e:
                logger.error("gemini.init_failed err=%s", str(e))
    
    def _select_provider(self) -> tuple[str, ProviderStatus]:
        """Select best available provider"""
        # Priority order: groq, gemini, claude, openai
        for provider_name in ["groq", "gemini", "claude", "openai"]:
            provider = self.providers.get(provider_name)
            if provider and provider.is_available():
                return provider_name, provider
        
        # No providers available - return least bad
        if "gemini" in self.providers:
            return "gemini", self.providers["gemini"]
        if "groq" in self.providers:
            return "groq", self.providers["groq"]
        raise RuntimeError("No LLM providers available")
    
    def _get_next_model(self, provider_name: str, models_str: str) -> str:
        """Get next model in rotation"""
        model_list = [m.strip() for m in models_str.split(',') if m.strip()]
        if not model_list:
            return ""
        
        # Rotate through models
        idx = self.model_rotation.get(provider_name, 0)
        model = model_list[idx % len(model_list)]
        self.model_rotation[provider_name] = idx + 1
        
        return model
    
    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> tuple[str, int]:
        """Complete with automatic provider selection and failover"""
        provider_name, provider = self._select_provider()
        
        try:
            if provider_name == "groq":
                response, tokens = await self._call_groq(system, user, temperature, max_tokens)
            elif provider_name == "gemini":
                response, tokens = await self._call_gemini(system, user, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            provider.record_success(tokens)
            return response, tokens
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = ("rate" in error_str and "limit" in error_str) or "429" in error_str
            provider.record_failure(is_rate_limit=is_rate_limit)
            
            # Try fallback to next provider
            if provider_name == "groq" and "gemini" in self.providers:
                logger.info("llm.fallback from=groq to=gemini reason=%s", str(e)[:100])
                try:
                    response, tokens = await self._call_gemini(system, user, temperature, max_tokens)
                    self.providers["gemini"].record_success(tokens)
                    return response, tokens
                except:
                    pass
            
            # All failed
            raise
    
    async def _call_groq(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int]:
        """Call Groq API with model rotation"""
        from .config import settings
        
        models_str = getattr(settings, 'groq_models', 'llama-3.3-70b-versatile')
        model = self._get_next_model("groq", models_str)
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        logger.info("groq.call model=%s", model)
        
        async def _call():
            async with self._inflight:
                resp = await asyncio.to_thread(
                    lambda: self.groq_client.chat.completions.create(**payload)
                )
                text = (resp.choices[0].message.content or "").strip()
                tokens = getattr(resp, 'usage', None)
                tokens_used = tokens.total_tokens if tokens else self._estimate_tokens(system, user, text)
                return text, tokens_used
        
        from .retry import async_retry
        return await async_retry(_call, max_attempts=2, base_delay=0.5)
    
    async def _call_gemini(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, int]:
        """Call Gemini API with NEW google-genai SDK"""
        from .config import settings
        
        models_str = getattr(settings, 'gemini_models', 'gemini-2.0-flash-exp')
        model = self._get_next_model("gemini", models_str)
        
        # Combine system and user prompts
        combined_prompt = f"{system}\n\nUser: {user}"
        
        logger.info("gemini.call model=%s", model)
        
        async def _call():
            async with self._inflight:
                # NEW SDK: Use client.models.generate_content()
                response = await asyncio.to_thread(
                    lambda: self.gemini_client.models.generate_content(
                        model=model,
                        contents=combined_prompt,
                        config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        }
                    )
                )
                
                # Extract text from response
                text = response.text.strip()
                
                # Get token usage
                try:
                    tokens_used = response.usage_metadata.total_token_count
                except:
                    tokens_used = self._estimate_tokens(system, user, text)
                
                return text, tokens_used
        
        from .retry import async_retry
        return await async_retry(_call, max_attempts=2, base_delay=0.5)
    
    def _estimate_tokens(self, system: str, user: str, response: str) -> int:
        """Rough token estimation"""
        return int((len(system) + len(user) + len(response)) / 4)


# Global instance
llm_manager = MultiProviderLLM()


async def init_llm() -> None:
    """Initialize LLM manager"""
    await llm_manager.initialize()
    logger.info("🧠 llm.init_complete providers=%d", len(llm_manager.providers))


async def close_llm() -> None:
    """Cleanup"""
    pass


async def smart_complete(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> str:
    """Smart completion with automatic provider selection and graceful fallback"""
    try:
        response, tokens = await llm_manager.complete(system, user, temperature, max_tokens)
        return response
    except Exception as e:
        logger.exception("llm.all_failed")
        
        # Check if all providers are rate limited
        all_rate_limited = all(
            p.rate_limit_until > time.time() 
            for p in llm_manager.providers.values()
        )
        
        if all_rate_limited:
            return json.dumps({
                "reply": {
                    "type": "text",
                    "text": "I've hit my daily API limits across all providers. Please try again in a few hours. 🙏"
                },
                "memory_updates": []
            })
        else:
            return json.dumps({
                "reply": {
                    "type": "text",
                    "text": "I'm experiencing technical difficulties. Please try again in a moment. 🤖"
                },
                "memory_updates": []
            })
