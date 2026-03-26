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
