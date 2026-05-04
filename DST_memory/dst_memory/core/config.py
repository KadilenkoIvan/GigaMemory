from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Default TTL values per slot (used for Mode 1 and as fallback in Mode 2/3).
# Values must be from models.VALID_TTL_VALUES: 1d 3d 10d 2w 3w 1m 3m 6m 1y inf
SLOT_DEFAULT_TTL: Dict[str, str] = {
    "IDENTITY":     "inf",
    "FAMILY":       "inf",
    "FRIENDS":      "inf",
    "ROMANCE":      "1y",
    "WORK":         "1y",
    "EDUCATION":    "1y",
    "FINANCE":      "3m",
    "HEALTH":       "1y",
    "MENTAL_HEALTH":"6m",
    "HABITS":       "inf",
    "PREFERENCES":  "6m",
    "HOBBIES":      "6m",
    "SPORTS":       "6m",
    "FOOD":         "1m",
    "HOME":         "1y",
    "LOCATION":     "1y",
    "TRAVEL":       "3m",
    "PETS":         "inf",
    "TECH":         "6m",
    "VEHICLES":     "1y",
    "SCHEDULE":     "1m",
    "GOALS":        "3m",
    "EVENTS":       "2w",
}


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
    # - "local": HuggingFace causal LM via LocalHFServing (llm_model = path or HF id)
    # - "openrouter" / "api": OpenAI-compatible chat completions (OpenRouter default URL)
    llm_mode: str = "stub"
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b:free"
    # Weight dtype when llm_mode == "local" (torch_dtype in from_pretrained). Default FP16.
    llm_load_dtype: str = "float16"
    # When llm_mode == "local": none | 8bit | 4bit (BitsAndBytes; reduces VRAM).
    llm_load_quantization: str = "none"
    # Max prompt tokens (chat template) before generate; default 128k. Set 0 to disable clamp.
    llm_max_context_tokens: int = 128 * 1024
    # HF attention implementation when llm_mode == "local": eager | sdpa | flash_attention_2 | ...
    llm_attn_implementation: str = "eager"
    # Mistral-style sliding-window attention (model.config); default off.
    llm_use_sliding_window: bool = False
    llm_sliding_window: Optional[int] = None
    llm_max_tokens: int = 1024
    openrouter_http_referer: str = ""
    openrouter_x_title: str = ""

    # Slot decision model config (Meno-Lite style)
    slot_use_stub: bool = False
    slot_model_path: str = "models/Meno-Lite-0.1"
    slot_max_slots_per_message: int = 5
    # HF attention for local slot model (when slot_use_stub=False).
    slot_attn_implementation: str = "eager"
    slot_use_sliding_window: bool = False
    slot_sliding_window: Optional[int] = None

    # Финальная LLM temperature=0 для воспроизводимости.
    llm_temperature: float = 0.0

    # Hybrid thinking (Qwen3 / Qwen3.5) for local final LLM only; ignored for API modes.
    llm_enable_thinking: bool = True

    # RAGU knowledge-graph backend (required in this project).
    use_ragu: bool = True
    ragu_embedder_model: str = "deepvk/USER-bge-m3"
    ragu_storage_path: str = ""

    # -------------------------------------------------------------------
    # TTL (Time-To-Live) configuration
    # -------------------------------------------------------------------
    # TTL mode:
    # - "mode1": per-slot fixed TTL from slot_ttl_defaults (fast, coarse-grained)
    # - "mode2": model generates ttl field together with each triplet (default)
    # - "mode3": separate LLM call after triplet extraction (slower, not implemented yet)
    ttl_mode: str = "mode2"

    # Fallback TTL when mode2 model omits the ttl field, or for mode1.
    # Keys are canonical slot names; override any slot from SLOT_DEFAULT_TTL.
    ttl_slot_overrides: Dict[str, str] = field(default_factory=dict)

    # -------------------------------------------------------------------
    # Semantic deduplication
    # -------------------------------------------------------------------
    # When True: before inserting a new triplet, check cosine similarity with
    # existing active triplets in the SAME slot. If similarity >= threshold,
    # deactivate the old record and insert the new one (refreshed TTL).
    ttl_semantic_dedup_enabled: bool = True
    ttl_semantic_dedup_threshold: float = 0.9

    # -------------------------------------------------------------------
    # Slot context & deletion modes
    # -------------------------------------------------------------------
    # slot_context_enabled:
    #   When True — передавать текущие активные факты слота в промпт
    #   экстракции триплетов. Позволяет модели создавать исторические записи
    #   ("бывшее место жительства") и явно сигнализировать удаление.
    slot_context_enabled: bool = False

    # slot_context_max_facts:
    #   Максимальное кол-во фактов слота, передаваемых в контекст.
    #   Ограничивает раздувание промпта для маленьких моделей.
    slot_context_max_facts: int = 10

    # -------------------------------------------------------------------
    # Single-pass fallback modes
    # -------------------------------------------------------------------
    # slot_fallback_on_no_slots:
    #   When True (default) — если слот-селектор вернул пустой список слотов,
    #   запускается single-pass экстракция (без указания слота).
    #   When False — пустой список слотов означает конец обработки сообщения:
    #   никакие триплеты не извлекаются.
    slot_fallback_on_no_slots: bool = True

    # triplet_fallback_on_empty:
    #   When True (default) — если слоты были выделены, но все per-slot вызовы
    #   экстракции вернули пустые триплеты, запускается single-pass экстракция.
    #   When False — пустой результат по всем слотам означает конец обработки:
    #   никакие триплеты не сохраняются.
    triplet_fallback_on_empty: bool = True

    # triplet_deletion_mode:
    #   "none"         — удаление не выполняется (текущее поведение).
    #   "heuristic"    — Вариант C: rule-based паттерны отрицания без LLM.
    #   "llm_inline"   — Вариант A: сигналы удаления в том же вызове что и
    #                    экстракция. Требует slot_context_enabled=True.
    #   "llm_separate" — Вариант B: отдельный LLM-вызов для детекции удалений.
    #                    Сам вызов всегда получает контекст текущих фактов.
    triplet_deletion_mode: str = "none"

    # deletion_use_pymorphy:
    #   Использовать pymorphy2 для лемматизации при сравнении слов
    #   в режиме "heuristic". Улучшает попадание ("Москве" → "москва").
    deletion_use_pymorphy: bool = False

    # -------------------------------------------------------------------
    # Conflict resolver settings
    # -------------------------------------------------------------------
    # conflict_allow_multi_relation_same_object:
    #   When True (default): if an existing fact and a new fact have the SAME
    #   subject AND the same object but DIFFERENT relations, they are treated
    #   as complementary facts and the LLM conflict check is skipped.
    #   Example: "пользователь | есть партнёр | партнёр пользователя" and
    #            "пользователь | живёт вместе с | партнёр пользователя"
    #   → both survive without LLM intervention.
    #   Set to False to always run the LLM check for any same-subject pair.
    conflict_allow_multi_relation_same_object: bool = True

    # -------------------------------------------------------------------
    # Slot model thinking mode
    # -------------------------------------------------------------------
    # slot_model_enable_thinking:
    #   Enable or disable the thinking/reasoning phase for models that support it
    #   (Qwen3, Qwen3.5 and similar hybrid-thinking models).
    #   False (default) — thinking disabled; produces clean, compact JSON output.
    #   True — thinking enabled; model outputs a reasoning chain before the answer.
    #   When False: passes enable_thinking=False to apply_chat_template AND prepends
    #   '/no_think' to the system prompt as a fallback for older tokenizer versions.
    slot_model_enable_thinking: bool = False

    # Prompt UI language for slot/triplet/gate/deletion/conflict/final LLM ("ru" | "en").
    prompt_language: str = "ru"

    # When True (LongMemEval-style validation): parse row ``question_date`` once per dialogue,
    # stamp new facts with that instant, use it as "now" for TTL expiry, and show it to the
    # final LLM instead of the machine clock.
    use_dataset_datetime: bool = False

    # When True (default): every inserted fact gets ttl ``inf`` — model TTL output is ignored;
    # lazy TTL expiry never deactivates records. Set False to use normal ttl_mode / model TTL.
    force_infinite_ttl: bool = True

    # -------------------------------------------------------------------
    # Model unloading for local final LLM
    # -------------------------------------------------------------------
    # unload_models_before_final_llm:
    #   When True (default) and llm_mode="local" — before calling the final LLM,
    #   all other models (slot selector, triplet extractor, classifier, etc.)
    #   are unloaded from GPU to free memory for the final LLM.
    #   Ignored when llm_mode="openrouter" or "api" (no local models to unload).
    unload_models_before_final_llm: bool = True
