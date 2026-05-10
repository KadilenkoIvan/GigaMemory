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

- `shared` — путь к датасету, выходная директория, типы вопросов, логирование, сохранение промежуточных артефактов. Параметры пайплайна памяти (`conflict_rule_same_relation_updates`, `slot_model_path`, и т.д.) сюда **не подставляются** — их нужно задавать в `giga_memory` (или в `DST_memory/run_config.json` → `shared`, см. `build_pipeline_config` в `validate_longmemeval.py`).
- `batch_processing` — `final_llm_batch_size`, `judge_batch_size`, `calculate_memory_hit_rate`.
- `judge` — параметры judge LLM (`openrouter|puter|local|none`), включая `max_context_tokens` и `tokenizer_model`.
- `giga_memory` — runtime-параметры пайплайна памяти (зеркалят `DST_memory/run_config.json`).
- `validation_mode` — режим запуска (`full|memory_only|final_llm_only|judge_only`), входные пути и список стратегий для stage-режимов.

## Важные поля `giga_memory`

- `importance_model_path`
- `slot_model_path`
- `memory_strategy`
- `llm_mode`, `llm_model`, `llm_tokenizer_model`, `llm_load_dtype`, `llm_load_quantization`, `llm_max_context_tokens`
- `slot_llm_load_quantization` (`none`|`8bit`|`4bit`) — квантизация BitsAndBytes для **slot/triplet** локальной модели (отдельно от финальной LLM); нужны CUDA и `bitsandbytes`
- `ragu_embedder_model`, `ragu_storage_path`
- `slot_context_enabled`, `triplet_deletion_mode`
- `conflict_allow_multi_relation_same_object`, `conflict_rule_same_relation_updates` (`true` по умолчанию: автозамена при том же `subject`+`relation` и другом `object`; `false` — только LLM-резолвер)
- `prompt_language`
- `unload_models_before_final_llm`
- `use_dataset_datetime`
- `force_infinite_ttl`

## Режимы `validation_mode`

- `full` — память -> final LLM -> judge
- `memory_only` — только запись памяти + сохранение `memory_only_states.json`
- `final_llm_only` — только генерация ответов по сохранённым state (`input_state_dir` обязателен)
- `judge_only` — только оценка ответов (`input_answers_path` обязателен)

Дополнительно для `memory_only`:
- `memory_only_write_mode`:
  - `standard` — обычный путь (`slot_select` + per-slot extraction + fallback при необходимости),
  - `single_path_only` — форсировать только single-pass extraction (без отдельного этапа выбора слотов и per-slot extraction).

Дополнительно для `final_llm_only`:
- `final_llm_memory_strategies` — массив стратегий (`full_graph_json`, `relevant_slots_full`, `topk_graph_records`), которые будут запущены на одном и том же `memory_only` state.
- `final_llm_memory_payload_mode` — режим передачи памяти в final LLM:
  - `with_metadata` — полный JSON как сейчас;
  - `triplets_only` — только `subject/relation/object` (без служебных полей).

## CLI override

Поддерживаются:
- `--val-*` для блока валидации/батчей/judge;
- `--gm-*` для переопределения `giga_memory`;
- прямые флаги режима: `--validation-mode`, `--input-state-dir`, `--input-answers-path`, `--final-llm-memory-strategies`.
- прямые флаги режима: `--validation-mode`, `--input-state-dir`, `--input-answers-path`, `--final-llm-memory-strategies`, `--final-llm-memory-payload-mode`.

Для моделей с ограниченным окном контекста настраивайте:
- `giga_memory.llm_max_context_tokens` для final LLM;
- `giga_memory.llm_tokenizer_model` как явный HF tokenizer id/path для final LLM clamp;
- `judge.max_context_tokens` для judge LLM;
- `judge.tokenizer_model` как явный HF tokenizer id/path для judge clamp.

Статистика `final_llm_only` теперь дополнительно сохраняет:
- `final_llm_prompt_chars_before_clamp_total`
- `final_llm_prompt_chars_after_clamp_total`
- число вызовов final LLM (`final_llm_calls`)
- `dialogue_context_chars_total` / `dialogue_context_chars_avg` (размер исходного диалога в символах)

В `judge_only` эти же поля агрегируются в итоговом `validation_results.json`:
- `statistics.dialogue_context_chars.*`
- `results[].dialogue_context_chars`

## Пример stage-прохода

```bash
# 1) Память
python validate_longmemeval.py \
  --validation-mode memory_only \
  --memory-only-write-mode single_path_only \
  --config ./config_memory_only.json

# 2) Финальная LLM по сохранённым state
python validate_longmemeval.py \
  --validation-mode final_llm_only \
  --input-state-dir ./results_memory_only \
  --final-llm-memory-strategies full_graph_json,relevant_slots_full,topk_graph_records \
  --config ./config_final_llm.json

# 3) Judge
python validate_longmemeval.py \
  --validation-mode judge_only \
  --input-answers-path ./results_final_llm/full_graph_json/intermediate_answers.json \
  --input-state-dir ./results_memory_only \
  --calculate-memory-hit-rate \
  --config ./config_judge.json
```
