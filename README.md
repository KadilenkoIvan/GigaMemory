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
  - `prompts/` — сборщики промптов; тексты в `ru/` и `en/`; язык UI задаётся `prompt_language` в `run_config.json` / `--prompt-language` (включая тексты **финальной** LLM: `prompts/<ru|en>/final_llm_messages.py`)
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
- `--prompt-language` (`ru` \| `en`) — язык текстов промптов для slot/triplet/gate/deletion/conflict и финальной LLM (в `run_config.json`: `prompt_language`)

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

## Валидация

### LongMemEval Benchmark

Для тестирования качества системы памяти используется датасет **LongMemEval** (`xiaowu0162/longmemeval-cleaned`).

### Структура валидации

```
validation/
├── GigaMemory_full/        # Полное тестирование GigaMemory (v3)
│   ├── validate_longmemeval.py      # Основной скрипт с 4 режимами
│   ├── run_config.json              # Базовый конфиг
│   ├── config_full.json             # Полный пайплайн
│   ├── config_memory_only.json      # Только память
│   ├── config_final_llm.json        # Только финальная LLM
│   ├── config_judge.json            # Только судья
│   ├── README.md
│   └── CONFIG.md
└── baseline/               # Baseline тестирование
    ├── validate_baseline.py
    ├── run_config.json
    └── README.md
```

### Режимы валидации (v3)

Скрипт `validate_longmemeval.py` поддерживает 4 режима работы:

| Режим | Описание | Команда |
|-------|----------|---------|
| `full` | Полный пайплайн: память → финальная LLM → судья | `python validate_longmemeval.py --validation-mode full` |
| `memory_only` | Только обработка памяти и сохранение состояний | `python validate_longmemeval.py --validation-mode memory_only` |
| `final_llm_only` | Загрузка сохранённой памяти → генерация ответов | `python validate_longmemeval.py --validation-mode final_llm_only --input-state-dir ./results_memory` |
| `judge_only` | Оценка сохранённых ответов с Memory Hit Rate | `python validate_longmemeval.py --validation-mode judge_only --input-answers-path ./results/answers.json` |

### 1. GigaMemory Full Validation

#### Полный пайплайн (режим `full`)

Полное тестирование с structured memory, batch processing и Memory Hit Rate.

```bash
cd validation/GigaMemory_full

# Используем конфиг файл (config_full.json)
python validate_longmemeval.py --config ./config_full.json

# Или с кастомным конфигом
python validate_longmemeval.py --config ./my_config.json

# Batch processing: накапливаем 5 диалогов перед вызовом финальной LLM
python validate_longmemeval.py \
    --validation-mode full \
    --val-shared-num-items-per-type 20 \
    --val-batch-final-llm-batch-size 5 \
    --val-batch-judge-batch-size 10

# С подсчётом Memory Hit Rate (дополнительная метрика)
python validate_longmemeval.py \
    --validation-mode full \
    --calculate-memory-hit-rate \
    --judge-mode openrouter
```

#### Пошаговая валидация (3 этапа)

Разделение процесса для экономии ресурсов и тестирования разных конфигураций:

```bash
# Шаг 1: Обработка памяти (один раз)
python validate_longmemeval.py \
    --validation-mode memory_only \
    --config ./config_memory_only.json

# Шаг 2: Генерация ответов (можно запускать с разными LLM)
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory_only \
    --gm-llm-model "openai/gpt-4o-mini" \
    --config ./config_final_llm.json

# Шаг 3: Оценка (можно менять judge модель)
python validate_longmemeval.py \
    --validation-mode judge_only \
    --input-answers-path ./results_final_llm/intermediate_answers.json \
    --calculate-memory-hit-rate \
    --config ./config_judge.json
```

**Преимущества разделения:**
- Обработка памяти выполняется один раз (самый быстрый этап)
- Можно тестировать разные финальные LLM на одних и тех же состояниях памяти
- GPU оптимизация: выгрузка моделей памяти перед загрузкой большой LLM
- Гибкость в выборе моделей для каждого этапа

**Метрики:**
- Accuracy (correct / total)
- Memory Hit Rate (дополнительно, через `--calculate-memory-hit-rate`)

### 2. Baseline Validation

Сравнение с простыми стратегиями передачи контекста.

```bash
cd validation/baseline

# Baseline: передаём ВЕСЬ контекст (все user + assistant)
python validate_baseline.py --strategy full_context

# Baseline: последние 10 пар + оставшиеся user сообщения
python validate_baseline.py --strategy recent_10_plus_user --output-dir ./results_recent10
```

**Стратегии:**
- `full_context` — все user и assistant сообщения из всех сессий
- `recent_10_plus_user` — последние 10 пар + все ранние user сообщения

**Метрики:**
- Accuracy (correct / total)

### Честное сравнение

Для честного сравнения используйте **одинаковые** настройки:

```bash
# 1. GigaMemory
cd validation/GigaMemory_full
python validate_longmemeval.py --config ./run_config.json

# 2. Baseline - Full Context
cd validation/baseline
python validate_baseline.py \
    --strategy full_context \
    --config ./run_config.json

# 3. Baseline - Recent 10 + User
python validate_baseline.py \
    --strategy recent_10_plus_user \
    --config ./run_config.json
```

### Структура вывода (GigaMemory)

```
output-dir/
├── validation_results.json     # Итоговые метрики
│   {
│     "statistics": {
│       "total": 50,
│       "correct": 42,
│       "incorrect": 8,
│       "accuracy": 0.84,
│       "memory_hit": 48,       # GigaMemory only
│       "memory_miss": 2,       # GigaMemory only
│       "memory_hit_rate": 0.96 # GigaMemory only
│     },
│     "results": [...]
│   }
├── validation.log              # Полный лог
└── chunk_*/                    # Состояние памяти (GigaMemory only)
```

Подробная документация:
- GigaMemory: `validation/GigaMemory_full/README.md`, `README_CONFIG.md`
- Baseline: `validation/baseline/README.md`

## Подробная техдокументация

См. `PIPELINE.md` — полный разбор всех этапов, связей между модулями, форматов данных и поведения в разных сценариях.
