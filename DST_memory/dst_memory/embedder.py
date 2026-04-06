from typing import List

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        # Keep embedder on CPU by default so it does not compete with slot LLM VRAM.
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
