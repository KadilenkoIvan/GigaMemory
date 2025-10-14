from pathlib import Path
import json
from typing import List, Dict, Any

import faiss
import numpy as np


class MemoryIndex:
    def __init__(self, dim: int, index_path: str = "rag/index.faiss"):
        self.index_path = Path(index_path)
        self.index = faiss.IndexFlatIP(dim)
        # Stores structured entries: {dialogue_id, chunk_id, role, text}
        self.data: List[Dict[str, Any]] = []

    def add_entries(self, embeddings: List[List[float]], entries: List[Dict[str, Any]]):
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.data.extend(entries)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.index_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.index_path.with_suffix(".json"), "r", encoding="utf-8") as f:
                self.data = json.load(f)


