# LongMemEval Validation Configuration v3

Документация для версии валидации (`validate_longmemeval.py`) с batch processing, Memory Hit Rate, полной CLI конфигурацией и **множественными режимами валидации**.

## Новые возможности v3

### Validation Modes - Режимы валидации

Новая система позволяет разделить процесс валидации на независимые этапы.

```bash
# Выбор режима валидации
--validation-mode full            # Полный пайплайн (default)
--validation-mode memory_only     # Только обработка памяти
--validation-mode final_llm_only  # Только финальная LLM (требует --input-state-dir)
--validation-mode judge_only      # Только судья (требует --input-answers-path)
```

**Параметры режимов:**

| Параметр | Режимы | Описание |
|----------|--------|----------|
| `--validation-mode` | Все | Основной режим работы |
| `--input-state-dir` | `final_llm_only`, `judge_only` | Путь к директории с `memory_only_states.json` и `chunk_*/` |
| `--input-answers-path` | `judge_only` | Путь к файлу `intermediate_answers.json` |
| `--memory-only-output-suffix` | `memory_only` | Суффикс для выходных директорий |

**Примечания:**
- `--input-state-dir` для `final_llm_only`: загружает состояния памяти для генерации ответов
- `--input-state-dir` для `judge_only`: альтернативный источник памяти для Memory Hit Rate (опционально, но рекомендуется если пути в `intermediate_answers.json` устарели)

### 1. Batch Processing Parameters

```bash
# Размер батча для финальной LLM (default: 1)
--final-llm-batch-size 5

# Размер батча для судьи (default: 1)
--judge-batch-size 10
```

**Как работает:**
1. **full/memory_only** Фаза 1: Последовательная обработка диалогов через memory pipeline
2. **full/final_llm_only** Фаза 2: Когда батч полон — последовательный вызов финальной LLM для всех накопленных
3. **full/judge_only** Фаза 3: Когда батч ответов полон — последовательная оценка судьей

**Рекомендации по режимам:**
- `memory_only`: batch параметры не используются (нет финальной LLM и судьи)
- `final_llm_only`: `final_llm_batch_size` определяет загрузку/выгрузку моделей
- `judge_only`: `judge_batch_size` должен быть >= количеству ответов для эффективности
- `full`: `final_llm_batch_size` <= `num_items`, `judge_batch_size` >= `final_llm_batch_size`

### 2. Memory Hit Rate Metric

```bash
# Включить подсчёт Memory Hit Rate (default: false)
--calculate-memory-hit-rate
```

**Что измеряется:**
- **Memory Hit Rate** = доля случаев, когда нужный факт попал в контекст финальной LLM
- Измеряется отдельным вызовом судьи, который анализирует `memory_context`

**Интерпретация результатов:**
```json
{
  "statistics": {
    "accuracy": 0.75,
    "memory_hit_rate": 0.90,
    "gap": -0.15
  }
}
```

- MHR=90%, Accuracy=75% → проблема в финальной LLM, не в памяти
- MHR=60%, Accuracy=55% → проблема в памяти (факты не сохраняются/не извлекаются)

### 3. Model Unloading для локальной финальной LLM

В `run_config.json`:
```json
{
  "shared": {
    "llm_mode": "local",
    "llm_model": "meta-llama/Llama-3.1-70B-Instruct",
    "unload_models_before_final_llm": true
  }
}
```

Или через CLI:
```bash
--gm-unload-models-before-final-llm true
```

**Когда срабатывает:**
- Только при `llm_mode=local`
- Перед обработкой батча финальной LLM выгружаются:
  - Slot serving model
  - Classifier
  - Другие локальные модели
- После батча модели перезагружаются автоматически

## Полная CLI конфигурация GigaMemory

Все параметры имеют префикс `--gm-` и переопределяют значения из `run_config.json`.

### Model Paths

| Parameter | Type | Example |
|-----------|------|---------|
| `--gm-importance-model-path` | string | `"./best_model"` |
| `--gm-slot-model-path` | string | `"Qwen/Qwen3-0.6B"` |

### Thresholds and Numbers

| Parameter | Type | Example |
|-----------|------|---------|
| `--gm-importance-threshold` | float | `0.25` |
| `--gm-graph-top-k-records` | int | `20` |
| `--gm-retrieval-top-k` | int | `5` |
| `--gm-recent-history-pairs` | int | `5` |
| `--gm-slot-max-slots-per-message` | int | `5` |
| `--gm-slot-context-max-facts` | int | `10` |

### Memory Strategy

| Parameter | Choices | Example |
|-----------|---------|---------|
| `--gm-memory-strategy` | `full_graph_json`, `relevant_slots_full`, `topk_graph_records` | `topk_graph_records` |

### LLM Settings

| Parameter | Type/Choices | Example |
|-----------|--------------|---------|
| `--gm-llm-mode` | `stub`, `local`, `openrouter`, `api` | `openrouter` |
| `--gm-llm-model` | string | `openai/gpt-oss-120b:free` |
| `--gm-llm-api-key` | string | `sk-or-v1-...` |
| `--gm-llm-api-url` | string | `https://openrouter.ai/api/v1` |
| `--gm-llm-temperature` | float | `0.0` |
| `--gm-llm-max-tokens` | int | `1024` |

### RAGU Settings

| Parameter | Type | Example |
|-----------|------|---------|
| `--gm-ragu-storage-path` | string | `./ragu_storage` |
| `--gm-ragu-embedder-model` | string | `deepvk/USER-bge-m3` |

### Slot and Deletion Settings

| Parameter | Type/Choices | Example |
|-----------|--------------|---------|
| `--gm-slot-use-stub` | `true`, `false` | `false` |
| `--gm-slot-context-enabled` | `true`, `false` | `true` |
| `--gm-triplet-deletion-mode` | `none`, `heuristic`, `llm_inline`, `llm_separate` | `llm_inline` |
| `--gm-deletion-use-pymorphy` | `true`, `false` | `false` |
| `--gm-slot-model-enable-thinking` | `true`, `false` | `false` |

### TTL Settings

| Parameter | Type | Example |
|-----------|------|---------|
| `--gm-ttl-mode` | `mode1`, `mode2`, `mode3` | `mode2` |
| `--gm-ttl-semantic-dedup-enabled` | `true`, `false` | `true` |
| `--gm-ttl-semantic-dedup-threshold` | float | `0.9` |

### Other Settings

| Parameter | Type/Choices | Example |
|-----------|--------------|---------|
| `--gm-prompt-language` | `ru`, `en` | `en` |
| `--gm-disable-memory-gate` | `true`, `false` | `false` |
| `--gm-memory-gate-use-stub` | `true`, `false` | `false` |
| `--gm-conflict-allow-multi-relation-same-object` | `true`, `false` | `true` |
| `--gm-slot-fallback-on-no-slots` | `true`, `false` | `true` |
| `--gm-triplet-fallback-on-empty` | `true`, `false` | `true` |
| `--gm-unload-models-before-final-llm` | `true`, `false` | `true` |

## Примеры конфигураций

### Пример 1: Тестирование разных стратегий памяти

```bash
# Strategy A: Full graph
python validate_longmemeval_v2.py ... \
    --gm-memory-strategy full_graph_json \
    --output-dir ./results_strategy_a

# Strategy B: Top-K records
python validate_longmemeval_v2.py ... \
    --gm-memory-strategy topk_graph_records \
    --gm-graph-top-k-records 30 \
    --output-dir ./results_strategy_b

# Strategy C: Relevant slots
python validate_longmemeval_v2.py ... \
    --gm-memory-strategy relevant_slots_full \
    --gm-disable-memory-gate false \
    --output-dir ./results_strategy_c
```

### Пример 2: Сравнение удаления фактов

```bash
# No deletion
python validate_longmemeval_v2.py ... \
    --gm-triplet-deletion-mode none \
    --output-dir ./results_del_none

# Heuristic deletion
python validate_longmemeval_v2.py ... \
    --gm-triplet-deletion-mode heuristic \
    --gm-deletion-use-pymorphy true \
    --output-dir ./results_del_heuristic

# LLM inline deletion
python validate_longmemeval_v2.py ... \
    --gm-triplet-deletion-mode llm_inline \
    --gm-slot-context-enabled true \
    --output-dir ./results_del_inline
```

### Пример 3: Оптимизация для большого датасета

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_optimized \
    --start-index 0 \
    --num-items 311 \
    --final-llm-batch-size 20 \
    --judge-batch-size 40 \
    --calculate-memory-hit-rate \
    --judge-mode openrouter \
    --config ../../DST_memory/run_config.json \
    \
    # GigaMemory optimizations
    --gm-memory-strategy topk_graph_records \
    --gm-graph-top-k-records 20 \
    --gm-retrieval-top-k 5 \
    --gm-slot-use-stub false \
    --gm-prompt-language en
```

### Пример 4: Локальная финальная LLM с выгрузкой

```bash
# run_config.json
# {
#   "shared": {
#     "llm_mode": "local",
#     "llm_model": "meta-llama/Llama-3.1-70B-Instruct",
#     "unload_models_before_final_llm": true,
#     "slot_model_path": "Qwen/Qwen3-0.6B",
#     "slot_use_stub": false
#   }
# }

python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_local_70b \
    --start-index 0 \
    --num-items 10 \
    --final-llm-batch-size 10 \
    --judge-mode local \
    --judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct" \
    --config ../../DST_memory/run_config.json
```

**Flow:**
1. Загружаются: classifier, Qwen3-0.6B (slot model)
2. Обрабатываются 10 диалогов через memory pipeline
3. **Выгружаются** classifier, Qwen3-0.6B
4. Загружается Llama-3.1-70B
5. Генерируются 10 ответов
6. Выгружается Llama-3.1-70B
7. Загружается judge model (Llama-3.2-1B)
8. Оцениваются 10 ответов

### Пример 5: Smoke test с stub режимом

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_stub \
    --start-index 0 \
    --num-items 5 \
    --final-llm-batch-size 5 \
    --judge-mode none \
    --config ../../DST_memory/run_config.json \
    --gm-slot-use-stub true \
    --gm-llm-mode stub \
    --gm-memory-gate-use-stub true
```

### Пример 6: Пошаговая валидация с разными режимами

```bash
# Шаг 1: Обработка памяти (memory_only)
python validate_longmemeval.py \
    --validation-mode memory_only \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_step1_memory \
    --num-items-per-type 20 \
    --config ./config_memory_only.json

# Шаг 2: Генерация ответов (final_llm_only)
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_step1_memory \
    --output-dir ./results_step2_llm \
    --final-llm-batch-size 10 \
    --gm-llm-model "openai/gpt-4o-mini" \
    --config ./config_final_llm.json

# Шаг 3: Оценка с Memory Hit Rate (judge_only)
# С указанием input-state-dir для гарантированного доступа к памяти
python validate_longmemeval.py \
    --validation-mode judge_only \
    --input-answers-path ./results_step2_llm/intermediate_answers.json \
    --input-state-dir ./results_step1_memory \
    --output-dir ./results_step3_judge \
    --judge-batch-size 20 \
    --calculate-memory-hit-rate \
    --judge-model "openai/gpt-4o" \
    --config ./config_judge.json
```

**Важно:** Для `judge_only` с `--calculate-memory-hit-rate` рекомендуется указывать `--input-state-dir` чтобы гарантировать доступ к состояниям памяти, особенно если файлы были перемещены.

## Конфигурационные файлы режимов

Предоставляются готовые конфигурационные файлы для каждого режима:

| Файл | Режим | Назначение |
|------|-------|-----------|
| `config_full.json` | `full` | Полный пайплайн (по умолчанию) |
| `config_memory_only.json` | `memory_only` | Только обработка памяти |
| `config_final_llm.json` | `final_llm_only` | Только финальная LLM |
| `config_judge.json` | `judge_only` | Только судья |

### Структура validation_mode в конфиге

```json
{
  "validation_mode": {
    "mode": "full",
    "input_state_dir": "",
    "input_answers_path": "",
    "memory_only_output_suffix": "_memory_only"
  }
}
```

**Описание полей:**

| Поле | Тип | Режимы | Описание |
|------|-----|--------|----------|
| `mode` | string | Все | `full`, `memory_only`, `final_llm_only`, `judge_only` |
| `input_state_dir` | string | `final_llm_only` | Путь к директории с результатами `memory_only` |
| `input_answers_path` | string | `judge_only` | Путь к файлу `intermediate_answers.json` |
| `memory_only_output_suffix` | string | `memory_only` | Суффикс для именования выходных директорий |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key для OpenRouter (final LLM и judge) |
| `DST_MEMORY_CONFIG` | Путь к конфигурации DST_memory |
| `HF_HOME` | HuggingFace cache directory |
| `TRANSFORMERS_CACHE` | Transformers cache directory |
| `CUDA_VISIBLE_DEVICES` | Выбор GPU для локальных моделей |

## Output Structure

```
output-dir/
├── validation.log              # Полный лог выполнения
├── validation_results.json     # Итоговые метрики
│   {
│     "metadata": {
│       "final_llm_batch_size": 5,
│       "judge_batch_size": 10,
│       "calculate_memory_hit_rate": true
│     },
│     "statistics": {
│       "total": 50,
│       "correct": 42,
│       "incorrect": 8,
│       "accuracy": 0.84,
│       "memory_hit": 48,
│       "memory_miss": 2,
│       "memory_hit_rate": 0.96
│     },
│     "results": [
│       {
│         "global_index": 0,
│         "question_id": "...",
│         "correct": true,
│         "memory_hit": true,
│         "judge_evaluation": {...},
│         "memory_hit_evaluation": {...}
│       }
│     ]
│   }
└── chunk_0000/                 # Состояние памяти после каждого диалога
    ├── dst_state.json
    └── ragu_storage/
```

## Метрики и их интерпретация

### Основная метрика: Accuracy

```
Accuracy = correct / total
```

### Дополнительная метрика: Memory Hit Rate

```
Memory Hit Rate = memory_hit / (memory_hit + memory_miss)
```

### Анализ разрыва (Gap)

```
Gap = Accuracy - Memory Hit Rate
```

| Gap | Интерпретация |
|-----|--------------|
| Gap ≈ 0 | Финальная LLM использует память эффективно |
| Gap > 0.1 | Проблема в финальной LLM (не использует контекст) |
| Gap < 0 | Редкий случай - LLM отвечает правильно даже без факта в памяти |

## Performance Tuning

### Баланс скорости и точности

| Config | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `slot_use_stub=true`, `llm_mode=stub` | Мгновенно | Низкая | Smoke test |
| `slot_use_stub=false`, `llm_mode=openrouter` | Средне | Высокая | Production eval |
| `slot_use_stub=false`, `llm_mode=local` | Медленно | Высокая | Offline eval |
| `final_llm_batch_size=20` | Быстрее | Та же | Local LLM оптимизация |

### API Cost Optimization (OpenRouter)

| Approach | Calls per item | Total for 311 items |
|----------|---------------|---------------------|
| Basic | 2 (LLM + Judge) | 622 |
| With MHR | 3 (LLM + Judge + MHR) | 933 |
| Batch judge (size=20) | ~2.05 | ~637 |

## Troubleshooting

### Out of Memory (OOM)

```bash
# Уменьшить батч
--final-llm-batch-size 1

# Использовать stub для слотов
--gm-slot-use-stub true

# Отключить сохранение состояния
--no-save-memory-state
```

### Slow processing

```bash
# Увеличить батчи
--final-llm-batch-size 20
--judge-batch-size 40

# Использовать более быструю judge model
--judge-model "openai/gpt-4o-mini"
```

### API rate limiting

```bash
# Увеличить батч judge (меньше вызовов)
--judge-batch-size 50

# Или использовать local judge
--judge-mode local
```
