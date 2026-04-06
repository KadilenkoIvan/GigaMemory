from typing import Dict, List, Optional, Sequence
import logging

from .embedder import TextEmbedder
from .vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    def __init__(self, store: InMemoryVectorStore, embedder: TextEmbedder):
        self.store = store
        self.embedder = embedder

    def search(
        self,
        dialogue_id: str,
        query: str,
        top_k: int,
        *,
        allowed_slots: Optional[Sequence[str]] = None,
        allowed_entities: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        logger.debug("Retriever search dialogue_id=%s top_k=%d", dialogue_id, top_k)
        q = self.embedder.encode([query])[0]
        all_hits = self.store.search(q, top_k=max(top_k * 5, top_k))
        filtered = [x for x in all_hits if x.get("dialogue_id") == dialogue_id]
        if allowed_slots:
            slot_set = set(str(s) for s in allowed_slots)
            filtered = [x for x in filtered if str(x.get("slot", "")) in slot_set]
        if allowed_entities:
            ent_set = set(str(e) for e in allowed_entities)
            filtered = [
                x
                for x in filtered
                if (str(x.get("subject", "")) in ent_set or str(x.get("object", "")) in ent_set)
            ]
        logger.info(
            "Retriever result dialogue_id=%s query_len=%d total_hits=%d filtered_hits=%d",
            dialogue_id,
            len(query),
            len(all_hits),
            len(filtered[:top_k]),
        )
        return filtered[:top_k]
