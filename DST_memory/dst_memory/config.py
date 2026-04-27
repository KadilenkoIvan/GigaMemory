from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    importance_model_path: str = str(
        Path(__file__).resolve().parents[2]
        / "importants_classificator"
        / "training"
        / "results_e5"
        / "best_model"
    )
    importance_threshold: float = 0.5
    retrieval_top_k: int = 5
    graph_top_k_records: int = 20
    recent_history_pairs: int = 5
    # Если False — в финальную LLM попадают все активные слоты без отбора локальной LLM.
    use_memory_gate: bool = True
    # Если True — для отбора слотов под ответ используется эвристика (без вызова локальной модели).
    memory_gate_use_stub: bool = False
    # Strategy of memory payload for final answer:
    # - "full_graph_json": pass complete active memory graph as JSON.
    # - "relevant_slots_full": LLM gate selects slots, pass full selected slots.
    # - "topk_graph_records": RAGU semantic top-k over all graph.
    memory_strategy: str = "relevant_slots_full"

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

    # RAGU knowledge-graph backend (required in this project).
    use_ragu: bool = True
    # SentenceTransformer model for RAGU's embedder.
    ragu_embedder_model: str = "deepvk/USER-bge-m3"
    # Folder where RAGU persists its storage files (graph, vectors, KV).
    # Empty string → <repo_root>/ragu_storage
    ragu_storage_path: str = ""
