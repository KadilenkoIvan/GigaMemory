# LongMemEval Validation Config (v3)

Документ соответствует `validate_longmemeval.py`.

## Режимы валидации

```bash
--validation-mode full
--validation-mode memory_only
--validation-mode final_llm_only   # требует --input-state-dir
--validation-mode judge_only       # требует --input-answers-path
```

Дополнительно:
- `--input-state-dir` — директория с `memory_only_states.json` и `chunk_*/`
- `--input-answers-path` — путь к `intermediate_answers.json`

## Batch и метрики

- `--final-llm-batch-size`
- `--judge-batch-size`
- `--calculate-memory-hit-rate`

## Ключевые `--gm-*` override

- `--gm-importance-model-path`
- `--gm-slot-model-path`
- `--gm-importance-threshold`
- `--gm-memory-strategy` (`full_graph_json|relevant_slots_full|topk_graph_records`)
- `--gm-graph-top-k-records`
- `--gm-llm-mode` (`stub|local|openrouter|api|puter`)
- `--gm-llm-model`
- `--gm-llm-api-key`
- `--gm-llm-load-dtype`
- `--gm-llm-load-quantization` (`none|8bit|4bit`)
- `--gm-llm-max-context-tokens`
- `--gm-llm-tokenizer-model`
- `--gm-ragu-storage-path`
- `--gm-ragu-embedder-model`
- `--gm-slot-use-stub`
- `--gm-slot-context-enabled`
- `--gm-triplet-deletion-mode` (`none|heuristic|llm_inline|llm_separate`)
- `--gm-prompt-language` (`ru|en`)
- `--gm-unload-models-before-final-llm`
- `--gm-use-dataset-datetime`
- `--gm-force-infinite-ttl`

## Поведение local режимов

- Для `giga_memory.llm_mode=local` поддерживается выгрузка моделей памяти перед финальной LLM (`unload_models_before_final_llm=true`).
- Для `judge.mode=local` судья может работать локально, с отдельной моделью.

## Puter и лимиты контекста

- Для final LLM можно использовать `--gm-llm-mode puter` (OpenAI-compatible endpoint Puter).
- Для judge можно использовать `--judge-mode puter`.
- Для моделей с ограниченным окном задавайте:
  - `--gm-llm-max-context-tokens` для final LLM;
  - `--gm-llm-tokenizer-model` если `gm-llm-model` не HF-совместимый id;
  - `--judge-max-context-tokens` для judge;
  - `--judge-tokenizer-model` если `judge-model` не HF-совместимый id.

## Минимальные stage-команды

```bash
# Stage 1
python validate_longmemeval.py --validation-mode memory_only --config ./config_memory_only.json

# Stage 2
python validate_longmemeval.py --validation-mode final_llm_only --input-state-dir ./results_memory_only --config ./config_final_llm.json

# Stage 3
python validate_longmemeval.py --validation-mode judge_only --input-answers-path ./results_final_llm/intermediate_answers.json --input-state-dir ./results_memory_only --config ./config_judge.json
```

## Интерпретация метрик

- `average_score` — средний score judge по шкале 0..1.
- `memory_hit_rate` (если включён) — доля случаев, где нужный факт присутствовал в memory context.
- Разрыв `memory_hit_rate` vs `average_score` помогает разделять проблемы памяти и проблемы финальной LLM.
- `final_llm_prompt_chars_before_clamp_total` / `final_llm_prompt_chars_after_clamp_total` — сколько символов prompt было до обрезки и сколько реально ушло в final LLM.
