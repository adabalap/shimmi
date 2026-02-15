"""
ChromaDB Fact Mining System
Automatically extracts facts from conversation history and promotes to long-term memory
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("app.fact_miner")
UTC = timezone.utc


# ============================================================================
# FACT MINING FROM CHROMA
# ============================================================================

@dataclass
class MinedFact:
    """A fact discovered from conversation analysis"""
    key: str
    value: str
    confidence: float
    source_messages: List[str]  # IDs of messages that support this fact
    frequency: int  # How many times mentioned
    last_mentioned: str  # ISO timestamp


class ChromaFactMiner:
    """
    Mines structured facts from unstructured conversation history.
    Promotes high-confidence facts to long-term memory.
    """
    
    def __init__(self, chroma_store, sqlite_store, llm_complete_fn):
        self.chroma = chroma_store
        self.sqlite = sqlite_store
        self.llm_complete = llm_complete_fn
        
        # Track what we've already analyzed to avoid re-processing
        self.analyzed_messages: Set[str] = set()
    
    async def mine_facts_for_user(
        self,
        chat_id: str,
        whatsapp_id: str,
        lookback_days: int = 7
    ) -> List[MinedFact]:
        """
        Analyze recent conversations to extract facts.
        
        Process:
        1. Get recent user messages from ChromaDB
        2. Cluster messages by topic/theme
        3. Extract facts from clusters using LLM
        4. Validate and de-duplicate facts
        5. Return high-confidence facts
        """
        
        # STEP 1: Get recent user messages
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        
        messages = await self._get_recent_messages(chat_id, whatsapp_id, cutoff)
        
        if len(messages) < 5:
            # Not enough data to mine
            logger.info("fact_mining.insufficient_data chat=%s count=%d", chat_id, len(messages))
            return []
        
        logger.info("fact_mining.analyzing chat=%s messages=%d", chat_id, len(messages))
        
        # STEP 2: Group messages by semantic similarity
        clusters = await self._cluster_messages(messages)
        
        # STEP 3: Extract facts from each cluster
        all_mined_facts: Dict[str, MinedFact] = {}
        
        for cluster_id, cluster_messages in clusters.items():
            cluster_text = "\n".join([msg['text'] for msg in cluster_messages])
            
            # Use LLM to extract facts from this cluster
            facts = await self._extract_facts_from_cluster(cluster_text, cluster_messages)
            
            # Merge with existing facts
            for fact in facts:
                existing = all_mined_facts.get(fact.key)
                if existing:
                    # Merge: increase confidence and frequency
                    existing.confidence = min(1.0, existing.confidence + (fact.confidence * 0.5))
                    existing.frequency += fact.frequency
                    existing.source_messages.extend(fact.source_messages)
                    existing.last_mentioned = max(existing.last_mentioned, fact.last_mentioned)
                else:
                    all_mined_facts[fact.key] = fact
        
        # STEP 4: Filter to high-confidence facts
        high_confidence_facts = [
            fact for fact in all_mined_facts.values()
            if fact.confidence >= 0.75 and fact.frequency >= 2
        ]
        
        logger.info(
            "fact_mining.complete chat=%s total=%d high_conf=%d",
            chat_id, len(all_mined_facts), len(high_confidence_facts)
        )
        
        return high_confidence_facts
    
    async def _get_recent_messages(
        self,
        chat_id: str,
        whatsapp_id: str,
        since: str
    ) -> List[Dict]:
        """Get recent messages from ChromaDB"""
        
        # Get all messages for this chat
        result = await asyncio.to_thread(
            lambda: self.chroma.collection.get(
                where={"chat_id": chat_id, "whatsapp_id": whatsapp_id, "direction": "in"},
                include=["documents", "metadatas"]
            )
        )
        
        messages = []
        for msg_id, doc, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", [])
        ):
            # Filter by timestamp
            if meta.get("ts", "") >= since:
                messages.append({
                    "id": msg_id,
                    "text": doc,
                    "ts": meta.get("ts", ""),
                    "metadata": meta
                })
        
        # Sort by timestamp
        messages.sort(key=lambda x: x["ts"], reverse=True)
        
        return messages
    
    async def _cluster_messages(self, messages: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Cluster messages by semantic similarity.
        Simple clustering: just group by time windows for now.
        
        TODO: Could use embedding similarity for better clustering
        """
        
        clusters = defaultdict(list)
        
        # Simple time-based clustering (4-hour windows)
        for msg in messages:
            try:
                ts = datetime.fromisoformat(msg["ts"])
                # Cluster by 4-hour blocks
                cluster_id = int(ts.timestamp() / (4 * 3600))
                clusters[cluster_id].append(msg)
            except Exception:
                # If timestamp parsing fails, put in misc cluster
                clusters[0].append(msg)
        
        return dict(clusters)
    
    async def _extract_facts_from_cluster(
        self,
        cluster_text: str,
        messages: List[Dict]
    ) -> List[MinedFact]:
        """
        Use LLM to extract facts from a cluster of related messages.
        """
        
        prompt = f"""
Extract user facts from these messages. Only extract facts explicitly stated by the user.

MESSAGES:
{cluster_text[:2000]}  

Rules:
- Only extract if explicitly stated by user
- Use snake_case keys
- Return JSON: {{"facts": [{{"key": "...", "value": "...", "confidence": 0.0-1.0}}]}}
- High confidence (0.9+) for direct statements: "I live in Mumbai"
- Medium confidence (0.7-0.8) for indirect: "I'm from Mumbai" 
- Low confidence (0.5-0.6) for implied: "The weather in Mumbai is..."

Output JSON only:
"""
        
        try:
            response = await self.llm_complete(
                system="You are a fact extraction system. Output JSON only.",
                user=prompt,
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse response
            data = json.loads(response)
            facts_data = data.get("facts", [])
            
            mined_facts = []
            for f in facts_data:
                mined_facts.append(MinedFact(
                    key=f.get("key", "").strip(),
                    value=f.get("value", "").strip(),
                    confidence=float(f.get("confidence", 0.5)),
                    source_messages=[msg["id"] for msg in messages],
                    frequency=1,
                    last_mentioned=messages[0]["ts"]  # Most recent
                ))
            
            return mined_facts
            
        except Exception as e:
            logger.error("fact_extraction.failed error=%s", str(e)[:200])
            return []
    
    async def promote_to_long_term_memory(
        self,
        whatsapp_id: str,
        mined_facts: List[MinedFact],
        min_confidence: float = 0.75
    ) -> int:
        """
        Promote high-confidence mined facts to long-term SQLite memory.
        Returns: number of facts promoted
        """
        
        promoted = 0
        
        for fact in mined_facts:
            if fact.confidence < min_confidence:
                continue
            
            # Calculate importance based on confidence and frequency
            importance = min(1.0, fact.confidence * (1.0 + (fact.frequency * 0.1)))
            
            # Determine category
            category = self._categorize_fact(fact.key)
            
            try:
                status = await self.sqlite.upsert_fact(
                    whatsapp_id=whatsapp_id,
                    key=fact.key,
                    value=fact.value,
                    importance=importance,
                    category=category
                )
                
                if status in ("created", "updated"):
                    promoted += 1
                    logger.info(
                        "fact_promoted key=%s value=%s confidence=%.2f status=%s",
                        fact.key, fact.value[:50], fact.confidence, status
                    )
            
            except Exception as e:
                logger.error("fact_promotion.failed key=%s error=%s", fact.key, str(e))
        
        return promoted
    
    def _categorize_fact(self, key: str) -> str:
        """Categorize fact by key"""
        key_lower = key.lower()
        
        if any(k in key_lower for k in ['name', 'email', 'phone', 'city', 'country', 'location']):
            return 'profile'
        elif any(k in key_lower for k in ['favorite', 'prefer', 'like', 'dislike']):
            return 'preference'
        elif any(k in key_lower for k in ['current', 'last', 'recent', 'today']):
            return 'temporary'
        else:
            return 'context'


# ============================================================================
# BACKGROUND FACT MINING TASK
# ============================================================================

async def fact_mining_loop(
    chroma_store,
    sqlite_store,
    llm_complete_fn,
    interval_hours: int = 24
):
    """
    Background task that periodically mines facts from conversations.
    Run this as part of your app's background workers.
    """
    
    miner = ChromaFactMiner(chroma_store, sqlite_store, llm_complete_fn)
    
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            
            logger.info("fact_mining_loop.start")
            
            # Get all active chats
            result = await asyncio.to_thread(
                lambda: chroma_store.collection.get(include=["metadatas"])
            )
            
            # Group by (chat_id, whatsapp_id)
            user_chats = set()
            for meta in result.get("metadatas", []):
                chat_id = meta.get("chat_id")
                whatsapp_id = meta.get("whatsapp_id")
                direction = meta.get("direction")
                
                if chat_id and whatsapp_id and direction == "in":
                    user_chats.add((chat_id, whatsapp_id))
            
            total_promoted = 0
            
            for chat_id, whatsapp_id in user_chats:
                try:
                    # Mine facts from last 7 days
                    mined_facts = await miner.mine_facts_for_user(
                        chat_id=chat_id,
                        whatsapp_id=whatsapp_id,
                        lookback_days=7
                    )
                    
                    # Promote to long-term memory
                    if mined_facts:
                        promoted = await miner.promote_to_long_term_memory(
                            whatsapp_id=whatsapp_id,
                            mined_facts=mined_facts,
                            min_confidence=0.75
                        )
                        total_promoted += promoted
                
                except Exception as e:
                    logger.exception(
                        "fact_mining.user_failed chat=%s user=%s",
                        chat_id, whatsapp_id
                    )
            
            logger.info(
                "fact_mining_loop.complete chats=%d promoted=%d",
                len(user_chats), total_promoted
            )
        
        except Exception:
            logger.exception("fact_mining_loop.error")


# ============================================================================
# INTEGRATION WITH EXISTING APP
# ============================================================================

async def integrate_fact_mining(app_startup):
    """
    Add to your app's startup function:
    
    @app.on_event("startup")
    async def startup():
        # ... existing init code ...
        
        # Start fact mining background task
        from .fact_mining import fact_mining_loop, smart_complete
        asyncio.create_task(
            fact_mining_loop(
                chroma_store=database.chroma_store,
                sqlite_store=database.sqlite_store,
                llm_complete_fn=smart_complete,
                interval_hours=24
            )
        )
    """
    pass


# ============================================================================
# MANUAL FACT MINING (for testing or on-demand)
# ============================================================================

async def mine_facts_now(chat_id: str, whatsapp_id: str):
    """
    Manually trigger fact mining for a specific user.
    Useful for testing or admin commands.
    """
    from .improved_agent_engine import smart_complete
    from . import database
    
    miner = ChromaFactMiner(
        chroma_store=database.chroma_store,
        sqlite_store=database.sqlite_store,
        llm_complete_fn=smart_complete
    )
    
    # Mine facts from last 30 days
    mined_facts = await miner.mine_facts_for_user(
        chat_id=chat_id,
        whatsapp_id=whatsapp_id,
        lookback_days=30
    )
    
    if not mined_facts:
        return {"status": "no_facts_found", "count": 0}
    
    # Promote to memory
    promoted = await miner.promote_to_long_term_memory(
        whatsapp_id=whatsapp_id,
        mined_facts=mined_facts,
        min_confidence=0.7
    )
    
    return {
        "status": "success",
        "mined": len(mined_facts),
        "promoted": promoted,
        "facts": [
            {"key": f.key, "value": f.value, "confidence": f.confidence}
            for f in mined_facts[:10]  # Show top 10
        ]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example():
    """Example of manual fact mining"""
    
    result = await mine_facts_now(
        chat_id="919573717667@c.us",
        whatsapp_id="919573717667"
    )
    
    print(f"Mined {result['mined']} facts, promoted {result['promoted']} to long-term memory")
    print("Top facts:")
    for fact in result['facts']:
        print(f"  {fact['key']}: {fact['value']} (confidence: {fact['confidence']:.2f})")
