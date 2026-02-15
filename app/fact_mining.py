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
from typing import Dict, List, Set

logger = logging.getLogger("app.fact_miner")
UTC = timezone.utc


@dataclass
class MinedFact:
    """A fact discovered from conversation analysis"""

    key: str
    value: str
    confidence: float
    source_messages: List[str]
    frequency: int
    last_mentioned: str


class ChromaFactMiner:
    """
    Mines structured facts from unstructured conversation history.
    Promotes high-confidence facts to long-term memory.
    """

    def __init__(self, chroma_store, sqlite_store, llm_complete_fn):
        self.chroma = chroma_store
        self.sqlite = sqlite_store
        self.llm_complete = llm_complete_fn
        self.analyzed_messages: Set[str] = set()

    @staticmethod
    def _and_where(*clauses: Dict) -> Dict:
        clauses = [c for c in clauses if c]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    async def mine_facts_for_user(
        self,
        chat_id: str,
        whatsapp_id: str,
        lookback_days: int = 7,
    ) -> List[MinedFact]:
        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        messages = await self._get_recent_messages(chat_id, whatsapp_id, cutoff)

        if len(messages) < 5:
            logger.info("fact_mining.insufficient_data chat=%s count=%d", chat_id, len(messages))
            return []

        logger.info("fact_mining.analyzing chat=%s messages=%d", chat_id, len(messages))

        clusters = await self._cluster_messages(messages)

        all_mined_facts: Dict[str, MinedFact] = {}
        for _, cluster_messages in clusters.items():
            cluster_text = "\n".join([msg["text"] for msg in cluster_messages])
            facts = await self._extract_facts_from_cluster(cluster_text, cluster_messages)

            for fact in facts:
                existing = all_mined_facts.get(fact.key)
                if existing:
                    existing.confidence = min(1.0, existing.confidence + (fact.confidence * 0.5))
                    existing.frequency += fact.frequency
                    existing.source_messages.extend(fact.source_messages)
                    existing.last_mentioned = max(existing.last_mentioned, fact.last_mentioned)
                else:
                    all_mined_facts[fact.key] = fact

        high_confidence_facts = [
            fact
            for fact in all_mined_facts.values()
            if fact.confidence >= 0.75 and fact.frequency >= 2
        ]

        logger.info(
            "fact_mining.complete chat=%s total=%d high_conf=%d",
            chat_id,
            len(all_mined_facts),
            len(high_confidence_facts),
        )
        return high_confidence_facts

    async def _get_recent_messages(self, chat_id: str, whatsapp_id: str, since: str) -> List[Dict]:
        """Get recent inbound messages from ChromaDB"""
        where = self._and_where(
            {"chat_id": chat_id},
            {"whatsapp_id": whatsapp_id},
            {"direction": "in"},
        )

        result = await asyncio.to_thread(
            lambda: self.chroma.collection.get(where=where, include=["documents", "metadatas"])
        )

        messages: List[Dict] = []
        for msg_id, doc, meta in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            if (meta or {}).get("ts", "") >= since:
                messages.append(
                    {
                        "id": msg_id,
                        "text": doc,
                        "ts": (meta or {}).get("ts", ""),
                        "metadata": meta,
                    }
                )

        messages.sort(key=lambda x: x["ts"], reverse=True)
        return messages

    async def _cluster_messages(self, messages: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Simple clustering: group by 4-hour windows.
        """
        clusters = defaultdict(list)
        for msg in messages:
            try:
                ts = datetime.fromisoformat(msg["ts"])
                cluster_id = int(ts.timestamp() / (4 * 3600))
                clusters[cluster_id].append(msg)
            except Exception:
                clusters[0].append(msg)
        return dict(clusters)

    async def _extract_facts_from_cluster(self, cluster_text: str, messages: List[Dict]) -> List[MinedFact]:
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
                max_tokens=500,
            )
            data = json.loads(response)
            facts_data = data.get("facts", [])

            mined_facts: List[MinedFact] = []
            for f in facts_data:
                mined_facts.append(
                    MinedFact(
                        key=(f.get("key", "") or "").strip(),
                        value=(f.get("value", "") or "").strip(),
                        confidence=float(f.get("confidence", 0.5)),
                        source_messages=[msg["id"] for msg in messages],
                        frequency=1,
                        last_mentioned=messages[0]["ts"] if messages else "",
                    )
                )
            return mined_facts
        except Exception as e:
            logger.error("fact_extraction.failed error=%s", str(e)[:200])
            return []

    async def promote_to_long_term_memory(
        self,
        whatsapp_id: str,
        mined_facts: List[MinedFact],
        min_confidence: float = 0.75,
    ) -> int:
        """
        Promote high-confidence mined facts to long-term SQLite memory.
        NOTE: This uses sqlite_store.upsert_fact(whatsapp_id, key, value) to match your SQLiteMemory signature.
        """
        promoted = 0
        for fact in mined_facts:
            if fact.confidence < min_confidence:
                continue

            try:
                status = await self.sqlite.upsert_fact(
                    whatsapp_id=whatsapp_id,
                    key=fact.key,
                    value=fact.value,
                )
                if status in ("created", "updated"):
                    promoted += 1
                    logger.info(
                        "fact_promoted key=%s value=%s confidence=%.2f status=%s",
                        fact.key,
                        fact.value[:50],
                        fact.confidence,
                        status,
                    )
            except Exception as e:
                logger.error("fact_promotion.failed key=%s error=%s", fact.key, str(e))
        return promoted


async def fact_mining_loop(chroma_store, sqlite_store, llm_complete_fn, interval_hours: int = 24):
    """
    Background task that periodically mines facts from conversations.
    """
    miner = ChromaFactMiner(chroma_store, sqlite_store, llm_complete_fn)

    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)
            logger.info("fact_mining_loop.start")

            result = await asyncio.to_thread(lambda: chroma_store.collection.get(include=["metadatas"]))

            user_chats = set()
            for meta in result.get("metadatas", []):
                chat_id = (meta or {}).get("chat_id")
                whatsapp_id = (meta or {}).get("whatsapp_id")
                direction = (meta or {}).get("direction")
                if chat_id and whatsapp_id and direction == "in":
                    user_chats.add((chat_id, whatsapp_id))

            total_promoted = 0
            for chat_id, whatsapp_id in user_chats:
                try:
                    mined_facts = await miner.mine_facts_for_user(
                        chat_id=chat_id,
                        whatsapp_id=whatsapp_id,
                        lookback_days=7,
                    )

                    if mined_facts:
                        promoted = await miner.promote_to_long_term_memory(
                            whatsapp_id=whatsapp_id,
                            mined_facts=mined_facts,
                            min_confidence=0.75,
                        )
                        total_promoted += promoted
                except Exception:
                    logger.exception("fact_mining.user_failed chat=%s user=%s", chat_id, whatsapp_id)

            logger.info(
                "fact_mining_loop.complete chats=%d promoted=%d",
                len(user_chats),
                total_promoted,
            )
        except Exception:
            logger.exception("fact_mining_loop.error")
