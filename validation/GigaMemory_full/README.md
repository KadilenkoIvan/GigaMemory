# LongMemEval Validation v2 - Расширенное тестирование GigaMemory

Этот документ описывает расширенную версию скрипта валидации (`validate_longmemeval_v2.py`) с поддержкой batch processing, метрики Memory Hit Rate и полной конфигурации через CLI.

## Что нового в версии 2

### 1. Batch Processing для финальной LLM

Параметр `--final-llm-batch-size` позволяет накапливать обработанные диалоги и вызывать финальную LLM пакетно:

- Значение `1` (по умолчанию): стандартное поведение - ответ после каждого диалога
- Значение `N > 1`: накапливаем N диалогов, потом последовательно вызываем финальную LLM

**Зачем это нужно**: если финальная LLM локальная и большая, выгрузка/загрузка других моделей между вызовами оптимизирует использование GPU.

### 2. Batch Processing для судьи

Параметр `--judge-batch-size` позволяет накапливать ответы перед вызовом судьи:

- Значение `1` (по умолчанию): оценка после каждого ответа
- Значение `M > 1`: накапливаем M ответов, потом оцениваем пакетно

### 3. Memory Hit Rate Metric

Параметр `--calculate-memory-hit-rate` включает дополнительную метрику:

- **Memory Hit Rate** = доля случаев, когда нужный факт был в контексте LLM
- Проверяется отдельным вызовом судьи, который анализирует `memory_context`
- Помогает различать проблемы: плохое сохранение в память vs плохой ответ LLM

### 4. Model Unloading для локальной финальной LLM

Параметр в конфиге `unload_models_before_final_llm` (по умолчанию `true`):

- При `llm_mode=local` и значении `true`: перед вызовом финальной LLM выгружаются все другие модели
- Освобождает GPU память для большой финальной модели
- После обработки батча модели перезагружаются автоматически

### 5. Полная конфигурация GigaMemory через CLI

Все параметры GigaMemory можно переопределить через CLI без редактирования `run_config.json`:

```bash
--gm-memory-strategy topk_graph_records \
--gm-graph-top-k-records 50 \
--gm-prompt-language en \
--gm-slot-use-stub true \
--gm-llm-mode openrouter
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

### Пример 1: Базовое тестирование (batch_size=1)

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_basic \
    --start-index 0 \
    --num-items 10 \
    --config ../../DST_memory/run_config.json
```

**Что происходит:**
1. Обрабатывается Dialog 1 -> сразу вызов финальной LLM -> сразу оценка судьей
2. Обрабатывается Dialog 2 -> сразу вызов финальной LLM -> сразу оценка судьей
3. ... и так далее

### Пример 2: Batch Processing (final_llm_batch_size=5)

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_batch5 \
    --start-index 0 \
    --num-items 20 \
    --final-llm-batch-size 5 \
    --judge-batch-size 10 \
    --config ../../DST_memory/run_config.json
```

**Что происходит:**

1. **Memory Phase (последовательно):**
   - Обрабатываем Dialog 1-5 через memory pipeline (write_to_memory)
   - Сохраняем состояние памяти для каждого диалога
   - Накапливаем в буфере

2. **Final LLM Phase (когда буфер полон):**
   - Выгружаем модели GigaMemory (если local LLM)
   - Последовательно: Dialog 1 -> финальная LLM -> Answer 1
   - Последовательно: Dialog 2 -> финальная LLM -> Answer 2
   - ... Dialog 5 -> финальная LLM -> Answer 5
   - Перезагружаем модели GigaMemory

3. **Judge Phase (когда буфер ответов полон):**
   - Последовательно оцениваем Answer 1-10 судьей

### Пример 3: Memory Hit Rate Metric

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_mhr \
    --start-index 0 \
    --num-items 50 \
    --calculate-memory-hit-rate \
    --judge-mode openrouter \
    --judge-model "openai/gpt-oss-120b:free" \
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

### Пример 4: Полная конфигурация через CLI

```bash
python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_custom \
    --start-index 0 \
    --num-items 20 \
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

### Пример 5: Локальная финальная LLM с выгрузкой моделей

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

python validate_longmemeval_v2.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_local_llm \
    --start-index 0 \
    --num-items 5 \
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
│       "final_llm_batch_size": 5,
│       "judge_batch_size": 10,
│       "calculate_memory_hit_rate": true,
│       ...
│     },
│     "statistics": {
│       "total": 50,
│       "correct": 35,
│       "incorrect": 15,
│       "memory_hit": 42,
│       "memory_miss": 8
│     },
│     "results": [
│       {
│         "global_index": 0,
│         "question_id": "...",
│         "question": "...",
│         "reference_answer": "...",
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
python validate_longmemeval_v2.py ... --gm-memory-strategy full_graph_json --output-dir ./results_strategy_a

# Strategy B: topk_graph_records
python validate_longmemeval_v2.py ... --gm-memory-strategy topk_graph_records --gm-graph-top-k-records 30 --output-dir ./results_strategy_b

# Strategy C: relevant_slots_full
python validate_longmemeval_v2.py ... --gm-memory-strategy relevant_slots_full --output-dir ./results_strategy_c

# Сравнить метрики
python compare_results.py ./results_strategy_*/validation_results.json
```

## Ограничения и рекомендации

### Batch sizes

- `--final-llm-batch-size` должен быть <= `--num-items`
- `--judge-batch-size` должен быть кратен или превышать `--final-llm-batch-size`
- Рекомендуется: `judge_batch_size = 2 * final_llm_batch_size`

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
