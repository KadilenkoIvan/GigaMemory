# Конфигурация `validate_longmemeval.py` (v3)

Этот файл описывает конфиг для `validation/GigaMemory_full/validate_longmemeval.py`.

## Быстрый старт

```bash
cd validation/GigaMemory_full
python validate_longmemeval.py --config ./run_config.json
```

## Структура JSON

```json
{
  "shared": {},
  "batch_processing": {},
  "judge": {},
  "giga_memory": {},
  "validation_mode": {}
}
```

- `shared` — путь к датасету, выходная директория, типы вопросов, логирование, сохранение промежуточных артефактов.
- `batch_processing` — `final_llm_batch_size`, `judge_batch_size`, `calculate_memory_hit_rate`.
- `judge` — параметры judge LLM (`openrouter|puter|local|none`), включая `max_context_tokens` и `tokenizer_model`.
- `giga_memory` — runtime-параметры пайплайна памяти (зеркалят `DST_memory/run_config.json`).
- `validation_mode` — режим запуска (`full|memory_only|final_llm_only|judge_only`) и входные пути для stage-режимов.

## Важные поля `giga_memory`

- `importance_model_path`
- `slot_model_path`
- `memory_strategy`
- `llm_mode`, `llm_model`, `llm_tokenizer_model`, `llm_load_dtype`, `llm_load_quantization`, `llm_max_context_tokens`
- `ragu_embedder_model`, `ragu_storage_path`
- `slot_context_enabled`, `triplet_deletion_mode`
- `prompt_language`
- `unload_models_before_final_llm`
- `use_dataset_datetime`
- `force_infinite_ttl`

## Режимы `validation_mode`

- `full` — память -> final LLM -> judge
- `memory_only` — только запись памяти + сохранение `memory_only_states.json`
- `final_llm_only` — только генерация ответов по сохранённым state (`input_state_dir` обязателен)
- `judge_only` — только оценка ответов (`input_answers_path` обязателен)

## CLI override

Поддерживаются:
- `--val-*` для блока валидации/батчей/judge;
- `--gm-*` для переопределения `giga_memory`;
- прямые флаги режима: `--validation-mode`, `--input-state-dir`, `--input-answers-path`.

Для моделей с ограниченным окном контекста настраивайте:
- `giga_memory.llm_max_context_tokens` для final LLM;
- `giga_memory.llm_tokenizer_model` как явный HF tokenizer id/path для final LLM clamp;
- `judge.max_context_tokens` для judge LLM;
- `judge.tokenizer_model` как явный HF tokenizer id/path для judge clamp.

Статистика `final_llm_only` теперь дополнительно сохраняет:
- `final_llm_prompt_chars_before_clamp_total`
- `final_llm_prompt_chars_after_clamp_total`
- число вызовов final LLM (`final_llm_calls`)

## Пример stage-прохода

```bash
# 1) Память
python validate_longmemeval.py \
  --validation-mode memory_only \
  --config ./config_memory_only.json

# 2) Финальная LLM по сохранённым state
python validate_longmemeval.py \
  --validation-mode final_llm_only \
  --input-state-dir ./results_memory_only \
  --config ./config_final_llm.json

# 3) Judge
python validate_longmemeval.py \
  --validation-mode judge_only \
  --input-answers-path ./results_final_llm/intermediate_answers.json \
  --input-state-dir ./results_memory_only \
  --calculate-memory-hit-rate \
  --config ./config_judge.json
```
