import os
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Инициализирует эмбеддер в офлайн-режиме:
        - Если переменная окружения RAG_EMBEDDER_PATH указывает на локальную папку модели — используем её.
        - Иначе, если существует папка submit/rag/models/<model_name> — используем её.
        - Иначе используем строковое имя (надеемся, что модель также локально доступна через HF кэш).
        Интернет не используется, если передается локальный путь.
        """
        local_override = os.environ.get("RAG_EMBEDDER_PATH", "").strip()
        if local_override:
            model_ref = local_override
        else:
            base_dir = Path(__file__).resolve().parent
            candidate = base_dir / "../models" / model_name
            model_ref = str(candidate) if candidate.exists() else model_name

        self.model = SentenceTransformer(model_ref)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


