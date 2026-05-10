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

## memory_only: только часть диалогов

Чтобы после CUDA OOM не терять уже сохранённые `chunk_*` и строки в `memory_only_states.json`, можно перезапускать только нужные строки:

- `--memory-only-dialogue-row-indices` — индексы строк в **сбалансированном** списке (тот же номер, что в `chunk_XXXX`, например `7` → `chunk_0007`).
- `--memory-only-dialogue-ids` — значения поля `dialogue_id` в датасете (можно вместо или вместе с индексами; строка попадает в прогон, если совпал индекс **или** id).

В JSON (`validation_mode`): `memory_only_dialogue_row_indices`, `memory_only_dialogue_ids`.

При записи новые состояния **сливаются** с уже лежащим в `--output-dir` `memory_only_states.json` и `giga_memory_validation_logs.json`: строки для перезаписанных `dialogue_row_index` заменяются, остальные сохраняются. Папки `chunk_*` для необработанных строк не трогаются.

В конфиге пайплайна (`PipelineConfig` / `giga_memory`): для повторных попыток парсинга JSON после слота / триплетов / memory gate со второй попытки включается сэмплирование с повышением температуры (`llm_parse_retry_temperature`, `llm_parse_retry_temperature_increment`).

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
- `--gm-llm-load-quantization` (`none|8bit|4bit`) — финальная локальная LLM
- `--gm-slot-llm-load-quantization` (`none|8bit|4bit`) — локальная модель слотов/триплетов (`slot_model_path`), см. `shared.slot_llm_load_quantization` в `DST_memory/run_config.json`
- `--gm-llm-max-context-tokens`
- `--gm-llm-tokenizer-model`
- `--gm-ragu-storage-path`
- `--gm-ragu-embedder-model`
- `--gm-slot-use-stub`
- `--gm-slot-context-enabled`
- `--gm-triplet-deletion-mode` (`none|heuristic|llm_inline|llm_separate`)
- `--gm-conflict-rule-same-relation-updates` (`true|false`) — в `DST_memory/run_config.json` ключ `shared.conflict_rule_same_relation_updates`; при `false` нет автодеактивации при том же `subject`+`relation` и другом `object` (только LLM)
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
