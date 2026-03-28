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
    use_memory_gate: bool = True

    # LLM mode:
    # - "stub": no external LLM call, returns template response
    # - "local": TODO implement local LLM backend
    # - "api": TODO implement remote API backend
    llm_mode: str = "stub"
    llm_api_url: str = ""
    llm_api_key: str = ""

    # Slot decision model config (Meno-Lite style)
    # If True -> use stub logic for slot decisions.
    # If False -> use slot decision model.
    slot_use_stub: bool = False
    slot_model_path: str = "models/Meno-Lite-0.1"
    slot_max_slots_per_message: int = 5
