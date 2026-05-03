# LongMemEval Validation - Тестирование GigaMemory

Этот документ описывает скрипт валидации (`validate_longmemeval.py`) для тестирования GigaMemory DST pipeline на LongMemEval датасете.

## Возможности

### 1. Сбалансированная выборка (Balanced Sampling)

Параметр `num_items_per_type` определяет количество примеров для каждого типа вопросов:

- Позволяет протестировать одинаковое количество примеров каждого типа
- Поддерживаемые типы: `single-session-user`, `single-session-preference`, `multi-session`, `knowledge-update`
- Обеспечивает честное сравнение между разными типами вопросов

### 2. Оценка по шкале 0-1 (Continuous Scoring)

Судья оценивает ответы по шкале от 0.0 до 1.0:

- **1.0**: Идеальное совпадение - все сущности и факты присутствуют
- **0.8**: Незначительные неточности (число, дата, имя)
- **0.6**: Частичный ответ - отсутствует одна из нескольких ключевых сущностей
- **0.4**: Слабое покрытие - только одна правильная сущность из нескольких
- **0.2**: Минимальное совпадение - тематически связано, но фактически нет
- **0.0**: Полное несовпадение, противоречие, или "не знаю"

### 3. Метрики по типам вопросов (Per-Question-Type Metrics)

Статистика собирается отдельно для каждого типа вопроса:

- Средний score по каждому типу
- Количество ошибок судьи по типу
- Позволяет выявить слабые места системы для конкретных типов

### 4. Расширенная статистика времени выполнения (Timing Stats)

Собираются детальные метрики производительности:

- Общее время выполнения
- Время на диалог (min, max, p50, p95, p99, mean)
- Время на сообщение (min, max, p50, p95, p99, mean)

### 5. Повторные попытки (Retry Logic)

Автоматические повторные попытки при HTTP ошибках (429, 500, 502, 503, 504):

- До 3 попыток с экспоненциальной задержкой
- Применяется к вызовам финальной LLM и судьи
- Улучшает надёжность при работе с API

### 6. Batch Processing для финальной LLM

Параметр `final_llm_batch_size` позволяет накапливать обработанные диалоги и вызывать финальную LLM пакетно:

- Значение `1` (по умолчанию): стандартное поведение
- Значение `N > 1`: накапливаем N диалогов, потом вызываем финальную LLM

### 7. Batch Processing для судьи

Параметр `judge_batch_size` позволяет накапливать ответы перед вызовом судьи:

- Значение `1` (по умолчанию): оценка после каждого ответа
- Значение `M > 1`: накапливаем M ответов, потом оцениваем пакетно

### 8. Memory Hit Rate Metric

Параметр `calculate_memory_hit_rate` включает дополнительную метрику:

- **Memory Hit Rate** = доля случаев, когда нужный факт был в контексте LLM
- Проверяется отдельным вызовом судьи, который анализирует `memory_context`
- Помогает различать проблемы: плохое сохранение в память vs плохой ответ LLM

### 9. Полная конфигурация через CLI

Все параметры можно переопределить через CLI:

```bash
--gm-memory-strategy topk_graph_records \
--gm-graph-top-k-records 50 \
--gm-prompt-language en \
--val-shared-num-items-per-type 20
```

## Архитектура обработки

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Memory Processing (Sequential)                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Dialog 1 │ -> │ Dialog 2 │ -> │ Dialog 3 │ -> ...     │
│  └────┬────┘    └────┬────┘    └────┬────┘             │
│       │              │              │                     │
│       v              v              v                     │
│  ┌─────────────────────────────────────────┐             │
│  │    Accumulator (final_llm_batch_size) │             │
│  │         Накапливаем диалоги            │             │
│  └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                            │
                            v (when batch full or end)
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Final LLM Generation (Sequential within batch)    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Answer 1 │ -> │ Answer 2 │ -> │ Answer 3 │ -> ...     │
│  └────┬────┘    └────┬────┘    └────┬────┘             │
│       │              │              │                     │
│       v              v              v                     │
│  ┌─────────────────────────────────────────┐             │
│  │      Accumulator (judge_batch_size)     │             │
│  │         Накапливаем ответы             │             │
│  └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                            │
                            v (when batch full or end)
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Judge Evaluation (Sequential within batch)       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Judge 1 │ -> │ Judge 2 │ -> │ Judge 3 │ -> ...     │
│  └─────────┘    └─────────┘    └─────────┘             │
│                                                           │
│  Optional: Memory Hit evaluation (extra call per item)    │
└─────────────────────────────────────────────────────────────┘
```

## Примеры использования

### Пример 1: Базовое тестирование (10 примеров на каждый тип)

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_basic \
    --num-items-per-type 10 \
    --config ../../DST_memory/run_config.json
```

**Что происходит:**
1. Загружается сбалансированная выборка: 10 single-session-user, 10 single-session-preference, 10 multi-session, 10 knowledge-update
2. Обрабатывается Dialog 1 -> сразу вызов финальной LLM -> сразу оценка судьей
3. ... и так далее для всех 40 диалогов

### Пример 2: Расширенное тестирование с batch processing

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_batch \
    --num-items-per-type 20 \
    --final-llm-batch-size 5 \
    --judge-batch-size 10 \
    --config ../../DST_memory/run_config.json
```

### Пример 3: Memory Hit Rate Metric

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_mhr \
    --num-items-per-type 10 \
    --calculate-memory-hit-rate \
    --judge-mode openrouter \
    --judge-model "openai/gpt-oss-120b:free" \
    --config ../../DST_memory/run_config.json
```

### Пример 4: Тестирование только одного типа вопросов

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_knowledge_update \
    --num-items-per-type 50 \
    --question-types "knowledge-update" \
    --config ../../DST_memory/run_config.json
```

**Вывод метрик:**
```
Statistics:
  Total processed: 50
  Correct: 35
  Incorrect: 15
  Accuracy: 70.00%
  Memory hits: 42
  Memory misses: 8
  Memory Hit Rate: 84.00%
```

**Интерпретация:**
- 84% случаев факт был в контексте (memory hit)
- Но accuracy только 70% - значит проблема в финальной LLM, не в памяти

### Пример 5: Полная конфигурация через CLI

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_custom \
    --num-items-per-type 20 \
    --config ../../DST_memory/run_config.json \
    \
    # Override GigaMemory settings
    --gm-memory-strategy topk_graph_records \
    --gm-graph-top-k-records 50 \
    --gm-prompt-language en \
    --gm-slot-use-stub false \
    --gm-slot-model-path "Qwen/Qwen3-0.6B" \
    --gm-importance-threshold 0.3 \
    --gm-triplet-deletion-mode llm_inline \
    --gm-slot-context-enabled true \
    \
    # Judge settings
    --judge-mode openrouter \
    --judge-model "anthropic/claude-3.5-sonnet"
```

### Пример 6: Локальная финальная LLM с выгрузкой моделей

```bash
# В run_config.json:
# {
#   "shared": {
#     "llm_mode": "local",
#     "llm_model": "meta-llama/Llama-3.1-70B-Instruct",
#     "unload_models_before_final_llm": true,
#     "slot_model_path": "Qwen/Qwen3-0.6B",
#     "slot_use_stub": false
#   }
# }

python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_local_llm \
    --num-items-per-type 5 \
    --final-llm-batch-size 5 \
    --config ../../DST_memory/run_config.json
```

**Что происходит:**
1. Загружаются модели GigaMemory (classifier, slot model)
2. Обрабатываются диалоги 1-5 через memory pipeline
3. **Выгружаются модели GigaMemory** (unload_local_models)
4. Загружается Llama-3.1-70B для финальных ответов
5. Генерируются ответы на диалоги 1-5
6. **Выгружается Llama-3.1-70B**
7. **Перезагружаются модели GigaMemory**

## Структура вывода

```
output-dir/
├── validation.log              # Полный лог выполнения
├── validation_results.json     # Итоговые метрики и результаты
│   {
│     "metadata": {
│       "num_items_per_type": 10,
│       "question_types": ["single-session-user", ...],
│       "final_llm_batch_size": 5,
│       "judge_batch_size": 10,
│       "calculate_memory_hit_rate": true,
│       ...
│     },
│     "statistics": {
│       "total": 40,
│       "total_score": 31.5,
│       "by_type": {
│         "knowledge-update": {"count": 10, "average_score": 0.85, "errors": 0},
│         ...
│       },
│       "memory_hit": 35,
│       "memory_miss": 5
│     },
│     "timing": {
│       "total_time": 245.3,
│       "total_items": 40,
│       "time_per_item": {"min": 3.1, "max": 8.5, "p50": 5.8, ...}
│     },
│     "results": [
│       {
│         "global_index": 0,
│         "question_id": "...",
│         "question": "...",
│         "question_type": "knowledge-update",
│         "reference_answer": "...",
│         "score": 1.0,
│         "reasoning": "Perfect match",
│         "predicted_answer": "...",
│         "correct": true,
│         "memory_hit": true,
│         "judge_evaluation": {...},
│         "memory_hit_evaluation": {...}
│       },
│       ...
│     ]
│   }
├── chunk_0000/                 # Состояние после каждого диалога
│   ├── dst_state.json         # Полное DST состояние
│   └── ragu_storage/          # RAGU хранилище
├── chunk_0001/
└── ...
```

## Полный список CLI параметров GigaMemory

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--gm-importance-model-path` | Путь к классификатору | `"./best_model"` |
| `--gm-slot-model-path` | Путь к slot model | `"Qwen/Qwen3-0.6B"` |
| `--gm-importance-threshold` | Порог важности | `0.25` |
| `--gm-memory-strategy` | Стратегия памяти | `full_graph_json`, `relevant_slots_full`, `topk_graph_records` |
| `--gm-graph-top-k-records` | Top-K для retrieval | `20` |
| `--gm-llm-mode` | Режим финальной LLM | `stub`, `local`, `openrouter`, `api` |
| `--gm-llm-model` | Имя модели | `openai/gpt-oss-120b:free` |
| `--gm-llm-api-key` | API ключ | `sk-or-v1-...` |
| `--gm-ragu-storage-path` | Путь к RAGU | `./ragu_storage` |
| `--gm-ragu-embedder-model` | Embedder модель | `deepvk/USER-bge-m3` |
| `--gm-slot-use-stub` | Stub режим | `true`, `false` |
| `--gm-slot-context-enabled` | Контекст слотов | `true`, `false` |
| `--gm-triplet-deletion-mode` | Режим удаления | `none`, `heuristic`, `llm_inline`, `llm_separate` |
| `--gm-prompt-language` | Язык промптов | `ru`, `en` |
| `--gm-unload-models-before-final-llm` | Выгрузка моделей | `true`, `false` |

## Таблица метрик и их интерпретация

| Метрика | Формула | Интерпретация |
|---------|---------|--------------|
| **Accuracy** | correct / total | Общая точность системы |
| **Memory Hit Rate** | hits / (hits + misses) | Насколько хорошо память сохраняет нужные факты |
| **Gap** | Accuracy - MHR | Разница показывает проблему в финальной LLM |

**Примеры интерпретации:**

```
Case 1: MHR=90%, Accuracy=85%
- Память работает хорошо (90% фактов сохранены)
- Проблема в финальной LLM (не использует контекст)

Case 2: MHR=60%, Accuracy=55%
- Проблема в памяти (40% фактов не сохранены)
- Финальная LLM работает нормально (использует то, что есть)

Case 3: MHR=95%, Accuracy=92%
- Идеальная работа системы
- Небольшие ошибки финальной LLM
```

## Примеры production-использования

### Полное тестирование датасета батчами

```bash
# Обрабатываем 311 примеров батчами по 20
for start in 0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300; do
    python validate_longmemeval_v2.py \
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
        --output-dir ./results_batch_${start} \
        --start-index ${start} \
        --num-items 20 \
        --final-llm-batch-size 10 \
        --judge-batch-size 20 \
        --calculate-memory-hit-rate \
        --judge-mode openrouter \
        --config ../../DST_memory/run_config.json
done

# Объединяем результаты
python merge_results.py ./results_batch_*/validation_results.json
```

### A/B тестирование разных стратегий памяти

```bash
# Strategy A: full_graph_json
python validate_longmemeval.py ... --gm-memory-strategy full_graph_json --output-dir ./results_strategy_a

# Strategy B: topk_graph_records
python validate_longmemeval.py ... --gm-memory-strategy topk_graph_records --gm-graph-top-k-records 30 --output-dir ./results_strategy_b

# Strategy C: relevant_slots_full
python validate_longmemeval.py ... --gm-memory-strategy relevant_slots_full --output-dir ./results_strategy_c

# Сравнить метрики
python compare_results.py ./results_strategy_*/validation_results.json
```

## Ограничения и рекомендации

### Batch sizes

- Рекомендуется: `judge_batch_size >= final_llm_batch_size`
- Для локальных моделей: используйте `final_llm_batch_size >= 5` чтобы амортизировать время загрузки/выгрузки

### Memory Hit Rate

- Увеличивает количество LLM вызовов в 2 раза (отдельная оценка для каждого примера)
- Использовать на подвыборке (~50 примеров) для анализа
- Не использовать на полном датасете (311 примеров) из-за стоимости API

### Local LLM режим

- Убедитесь, что GPU имеет достаточно памяти для финальной модели после выгрузки
- Рекомендуется `final_llm_batch_size >= 5` чтобы амортизировать время загрузки/выгрузки

## Сравнение v1 и v2

| Функция | v1 | v2 |
|---------|-----|-----|
| Sequential processing | ✓ | ✓ |
| Batch final LLM | ✗ | ✓ |
| Batch judge | ✗ | ✓ |
| Memory Hit Rate | ✗ | ✓ |
| Model unloading | ✗ | ✓ |
| CLI config override | Частично | Полная |
| Speed (local LLM) | Медленно | Быстро |
| API calls (OpenRouter) | 2N | 2N - 3N |
