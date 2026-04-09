from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    importance_model_path: str = str(
        Path(__file__).resolve().parents[2]
        / "message_important_learning"
        / "best_model-full_tune"
    )
    importance_threshold: float = 0.5
    retrieval_top_k: int = 5
    # Если False — в финальную LLM попадают все активные слоты без отбора локальной LLM.
    use_memory_gate: bool = True
    # Если True — для отбора слотов под ответ используется эвристика (без вызова локальной модели).
    memory_gate_use_stub: bool = False
    # Источник текста памяти для финальной LLM: "slots" — записи из выбранных слотов; "vector" — top-k из векторного индекса.
    memory_context_source: str = "slots"

    # LLM mode:
    # - "stub": no external LLM call, returns template response
    # - "local": TODO implement local LLM backend
    # - "openrouter" / "api": OpenAI-compatible chat completions (OpenRouter default URL)
    llm_mode: str = "stub"
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b:free"
    llm_max_tokens: int = 1024
    openrouter_http_referer: str = ""
    openrouter_x_title: str = ""

    # Slot decision model config (Meno-Lite style)
    # If True -> use stub logic for slot decisions.
    # If False -> use slot decision model.
    slot_use_stub: bool = False
    slot_model_path: str = "models/Meno-Lite-0.1"
    slot_max_slots_per_message: int = 5
    # Генерация слотов: greedy + temperature=0 в slot_client (детерминированный выбор токена).

    # Финальная LLM (когда будут реализованы local/api): temperature=0 для воспроизводимости.
    llm_temperature: float = 0.0

    # RAGU knowledge-graph backend
    # If True, all triplets are mirrored to RAGU and "vector" retrieval uses
    # RAGU's LocalSearchEngine instead of the legacy InMemoryVectorStore.
    use_ragu: bool = False
    # SentenceTransformer model for RAGU's embedder.
    ragu_embedder_model: str = "deepvk/USER-bge-m3"
    # Folder where RAGU persists its storage files (graph, vectors, KV).
    # Empty string → <repo_root>/ragu_storage
    ragu_storage_path: str = ""
