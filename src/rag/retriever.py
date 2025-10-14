from typing import List, Dict, Any
import numpy as np


class Retriever:
    def __init__(self, index: "MemoryIndex", embedder: "Embedder"):
        self.index = index
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedder.encode([query])
        D, I = self.index.index.search(np.array(query_emb, dtype=np.float32), k)
        results: List[Dict[str, Any]] = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self.index.data):
                entry = self.index.data[idx]
                results.append({
                    "dialogue_id": entry.get("dialogue_id"),
                    "chunk_id": entry.get("chunk_id"),
                    "role": entry.get("role"),
                    "text": entry.get("text"),
                    "score": float(score)
                })
        return results


