from dataclasses import dataclass
from typing import Any, Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorItem:
    embedding: List[float]
    payload: Dict[str, Any]


class InMemoryVectorStore:
    """
    Simple in-memory vector DB.
    TODO: replace with persistent backend if needed.
    """

    def __init__(self):
        self._items: List[VectorItem] = []
        self._matrix: np.ndarray | None = None

    def add(self, embedding: List[float], payload: Dict[str, Any]) -> None:
        self._items.append(VectorItem(embedding=embedding, payload=payload))
        vec = np.array([embedding], dtype=np.float32)
        if self._matrix is None:
            self._matrix = vec
        else:
            self._matrix = np.vstack([self._matrix, vec])
        logger.debug(
            "VectorStore add size=%d dialogue_id=%s",
            len(self._items),
            payload.get("dialogue_id"),
        )

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if self._matrix is None or len(self._items) == 0:
            logger.debug("VectorStore search skipped empty store")
            return []

        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        sims = np.matmul(q, self._matrix.T)[0]
        k = min(top_k, sims.shape[0])
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            item = self._items[int(idx)]
            row = dict(item.payload)
            row["score"] = float(sims[idx])
            results.append(row)
        logger.debug("VectorStore search top_k=%d returned=%d", top_k, len(results))
        return results

    def clear_dialogue(self, dialogue_id: str) -> None:
        filtered = [it for it in self._items if it.payload.get("dialogue_id") != dialogue_id]
        removed = len(self._items) - len(filtered)
        self._items = filtered
        if not filtered:
            self._matrix = None
            logger.info("VectorStore clear dialogue_id=%s removed=%d now_empty=true", dialogue_id, removed)
            return
        self._matrix = np.array([it.embedding for it in filtered], dtype=np.float32)
        logger.info("VectorStore clear dialogue_id=%s removed=%d remaining=%d", dialogue_id, removed, len(filtered))
