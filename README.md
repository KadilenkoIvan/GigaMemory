# DST_memory

`DST_memory` — модуль долгосрочной памяти LLM на основе DST-графа фактов и RAGU retrieval.

## Что это за проект

- Память строится из сообщений пользователя.
- Из важных сообщений извлекаются триплеты `subject-relation-object`.
- Триплеты пишутся в состояние DST и синхронно зеркалятся в граф RAGU.
- **Поддерживается удаление фактов** тремя независимыми методами (см. ниже).
- При ответе формируется memory context одной из стратегий и передается в final LLM.
- Дополнительно передаются последние пары `user/assistant`.

## Что входит в каталог

- `run.py` — единая CLI-точка запуска.
- `dst_memory/` — вся логика пайплайна (разбита по подпакетам):
  - `core/` — pipeline, dst_manager, models, config, graph_backend
  - `prompts/` — сборщики промптов; тексты в `ru/` и `en/`; язык UI задаётся `prompt_language` в `run_config.json` / `--prompt-language`
  - `slots/` — онтология, нормализация, slot_select_client, slot_update_client
  - `triplets/` — extraction, conflict, deletion, negation_detector
  - `storage/` — RAGU backend (ragu_graph_processor)
  - `clients/` — serving, classifier, memory_gate_client, llm_client
  - `utils/` — io_utils, dotenv_loader, run_config_loader
- `run_config.json` — runtime-конфиг по умолчанию.
- `CONFIG.md` — описание параметров.
- `PIPELINE.md` — максимально подробная техническая документация по архитектуре и сценариям.

## Режимы запуска

### Test

Batch-прогон jsonl: сообщения проходят запись в память, затем вызывается ответ на финальный вопрос.

```bash
python DST_memory/run.py pipeline test --dataset-path data/format_example.jsonl --output-path DST_memory/output.json
```

### Inference Interactive

Пошаговый режим: новое сообщение -> запись в память -> ответ LLM.

```bash
python DST_memory/run.py pipeline inference interactive --dialogue-id demo
```

### Inference Single-turn

Один запрос на вход, один ответ на выход.

```bash
python DST_memory/run.py pipeline inference single-turn --dialogue-id d1 --message "..."
```

## Стратегии памяти

Переключаются `--memory-strategy`:

- `full_graph_json` — полный активный граф памяти в JSON.
- `relevant_slots_full` — LLM-gate выбирает слоты, передается полное содержимое выбранных слотов.
- `topk_graph_records` — top-k наиболее релевантных записей по всему графу (RAGU search).

## Важные флаги

- `--memory-strategy`
- `--graph-top-k-records`
- `--recent-history-pairs`
- `--slot-model-path`
- `--importance-model-path`
- `--ragu-embedder-model`
- `--ragu-storage-path`
- `--llm-mode` (`openrouter|api|stub|local`)
- `--no-final-llm`
- `--prompt-language` (`ru` \| `en`) — язык текстов промптов для slot/triplet/gate/deletion/conflict (в `run_config.json`: `prompt_language`)

## Режимы удаления фактов

Управляются двумя флагами `--slot-context-enabled` и `--triplet-deletion-mode`.

| Вариант | Флаги | Описание |
|---|---|---|
| A (inline) | `--slot-context-enabled --triplet-deletion-mode llm_inline` | Один LLM-вызов: модель видит текущие факты и выдаёт `delete`-сигналы вместе с новыми триплетами |
| B (separate) | `--triplet-deletion-mode llm_separate` | Отдельный LLM-вызов для детекции удалений, extraction без контекста |
| C (heuristic) | `--triplet-deletion-mode heuristic` | Rule-based паттерны отрицания без LLM, опционально `--deletion-use-pymorphy` |

Подробно — в `CONFIG.md`.

## Ключевые ограничения

- Проект зафиксирован как RAGU-only.
- `llm_mode=local` для final LLM пока не реализован.
- `llm_inline` режим автоматически включает контекст слота, даже если `slot_context_enabled=false`.
- Heuristic-детектор покрывает явные паттерны отрицания; косвенные семантические удаления — через LLM-режимы.

## Подробная техдокументация

См. `PIPELINE.md` — полный разбор всех этапов, связей между модулями, форматов данных и поведения в разных сценариях.
