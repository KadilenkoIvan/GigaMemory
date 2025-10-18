from pathlib import Path
import json
from typing import List, Dict, Any, Tuple

import numpy as np


class MemoryIndex:
    def __init__(self, dim: int, index_path: str = None):
        # Default to module-relative path if not provided
        if index_path is None:
            module_dir = Path(__file__).resolve().parent
            index_path = str(module_dir / "index.npy")
        self.index_path = Path(index_path)
        self.dim = dim
        # Embedding matrix
        self._embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)
        # Stores structured entries: {dialogue_id, chunk_id, role, text}
        self.data: List[Dict[str, Any]] = []

    def add_entries(self, embeddings: List[List[float]], entries: List[Dict[str, Any]]):
        new_emb = np.array(embeddings, dtype=np.float32)
        if new_emb.ndim == 1:
            new_emb = new_emb.reshape(1, -1)
        if self._embeddings.size == 0:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        self.data.extend(entries)

    def search(self, query_embeddings: List[List[float]], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity search (dot-product on normalized embeddings).
        Returns (D, I) analogous to FAISS, shapes: (n_queries, k).
        """
        if self._embeddings.size == 0:
            n = len(query_embeddings) if query_embeddings else 1
            return np.zeros((n, 0), dtype=np.float32), np.zeros((n, 0), dtype=np.int64)

        Q = np.array(query_embeddings, dtype=np.float32)
        # Compute similarity
        sims = np.matmul(Q, self._embeddings.T)
        # Top-k per row
        k = min(k, sims.shape[1])
        if k <= 0:
            return np.zeros((sims.shape[0], 0), dtype=np.float32), np.zeros((sims.shape[0], 0), dtype=np.int64)
        idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        # sort top-k
        row_indices = np.arange(sims.shape[0])[:, None]
        topk_vals = sims[row_indices, idx]
        order = np.argsort(-topk_vals, axis=1)
        I = idx[row_indices, order]
        D = topk_vals[row_indices, order]
        return D.astype(np.float32), I.astype(np.int64)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.index_path, self._embeddings)
        with open(self.index_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def load(self):
        if self.index_path.exists():
            self._embeddings = np.load(self.index_path)
            with open(self.index_path.with_suffix(".json"), "r", encoding="utf-8") as f:
                self.data = json.load(f)


