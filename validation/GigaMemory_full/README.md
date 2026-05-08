# LongMemEval Validation - Тестирование GigaMemory v3

Этот документ описывает скрипт валидации (`validate_longmemeval.py`) для тестирования GigaMemory DST pipeline на LongMemEval датасете.

## Режимы валидации (Validation Modes) v3

Начиная с версии 3, скрипт поддерживает 4 режима работы, позволяющие разделить процесс валидации на этапы:

### Режимы

| Режим | Описание | Выходные файлы |
|-------|----------|----------------|
| `full` | Полный пайплайн: память → финальная LLM → судья | `validation_results.json` |
| `memory_only` | Только обработка памяти и сохранение состояний (без фиксированного final context) | `memory_only_states.json`, `chunk_*/` |
| `final_llm_only` | Загрузка сохранённой памяти → генерация ответов с выбранными стратегиями | `<strategy>/intermediate_answers.json` |
| `judge_only` | Оценка сохранённых ответов с Memory Hit Rate | `validation_results.json` |

### Когда использовать разные режимы

**`full`** - используйте по умолчанию для полной валидации в одном запуске.

**`memory_only`** - используйте когда:
- Хотите разделить процесс на этапы для экономии ресурсов
- Нужно сохранить промежуточные состояния памяти для анализа
- Планируете запускать финальную LLM позже с другими параметрами

**`final_llm_only`** - используйте когда:
- Уже обработали диалоги в режиме `memory_only`
- Хотите протестировать разные финальные LLM на одних и тех же состояниях памяти
- Нужно оптимизировать GPU (выгрузка моделей памяти перед загрузкой финальной LLM)

**`judge_only`** - используйте когда:
- Уже сгенерированы ответы в режиме `final_llm_only`
- Хотите переоценить ответы с другой моделью-судьёй
- Нужно включить Memory Hit Rate оценку после основного тестирования

### Пайплайн работы режимов

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  memory_only    │ -> │ final_llm_only │ -> │   judge_only    │
│                 │    │                 │    │                 │
│  Dataset        │    │  Saved Memory   │    │  Saved Answers  │
│  ↓              │    │  ↓              │    │  ↓              │
│  Memory States  │    │  Final LLM      │    │  Judge          │
│  (chunk_*/)     │    │  Answers        │    │  Evaluation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        v                       v                       v
memory_only_states.json  intermediate_answers.json  validation_results.json
```

`memory_only` теперь сохраняет именно состояние памяти, пригодное для повторного использования:
- полный DST snapshot + `chunk_*/ragu_storage` (состояние RAGU retrieval);
- компактные артефакты стратегий (`strategy_state_by_strategy`: релевантные слоты и retrieval-кандидаты), без хранения трёх полных memory-context;
- timing блок со средним временем `write_to_memory` на сообщение.

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

### 9. Статистика размера prompt final LLM

Для каждого вызова final LLM сохраняется размер prompt в символах:
- до обрезки (`before_clamp`);
- после обрезки (`after_clamp`).

Агрегаты по запуску попадают в `statistics`, а значения по каждому примеру — в `results`/`intermediate_answers`.

### 10. Размер исходного диалога vs переданный контекст

Дополнительно сохраняются:
- `dialogue_context_chars` — общий размер исходного диалога (все реплики) в символах;
- `final_llm_prompt_chars_after_clamp` — сколько символов реально ушло в final LLM после ограничений контекста.

Эти поля есть в `intermediate_answers.json` (`final_llm_only`) и дублируются в итоговых метриках `judge_only`.

### 10. Полная конфигурация через CLI

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

### Пример 1: Полный пайплайн (режим `full`)

Базовое тестирование - обработка памяти, генерация ответов и оценка судьёй в одном запуске.

```bash
python validate_longmemeval.py \
    --config ./config_full.json
```

Или с CLI параметрами:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_full \
    --num-items-per-type 10 \
    --validation-mode full \
    --config ../../DST_memory/run_config.json
```

**Что происходит:**
1. Загружается сбалансированная выборка: 10 примеров каждого типа (40 всего)
2. Каждый диалог обрабатывается через memory pipeline
3. Вызывается финальная LLM для генерации ответа
4. Судья оценивает ответ по шкале 0-1
5. Результаты сохраняются в `validation_results.json`

### Пример 2: Пошаговая валидация (3 этапа)

Разделение процесса на этапы для экономии ресурсов и гибкости.

#### Этап 1: Обработка памяти (`memory_only`)

```bash
python validate_longmemeval.py \
    --config ./config_memory_only.json
```

Или с CLI:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_memory_only \
    --num-items-per-type 10 \
    --validation-mode memory_only \
    --config ../../DST_memory/run_config.json
```

**Выходные файлы:**
```
results_memory_only/
├── memory_only_states.json     # Состояния всех диалогов
├── chunk_0000/
│   ├── dst_state.json         # DST состояние диалога 0
│   └── ragu_storage/          # Векторная БД диалога 0
├── chunk_0001/
│   ├── dst_state.json
│   └── ragu_storage/
└── ...
```

#### Этап 2: Генерация ответов (`final_llm_only`)

```bash
python validate_longmemeval.py \
    --config ./config_final_llm.json
```

Или с CLI:

```bash
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory_only \
    --output-dir ./results_final_llm \
    --final-llm-memory-strategies full_graph_json,relevant_slots_full,topk_graph_records \
    --final-llm-batch-size 5 \
    --config ../../DST_memory/run_config.json
```

**Параметры:**
- `--input-state-dir` - директория с результатами `memory_only` (обязательно)
- `--final-llm-batch-size` - размер батча для финальной LLM
- `--final-llm-memory-strategies` - список стратегий через запятую (если не указан, используется `giga_memory.memory_strategy`)

**Выходные файлы:**
```
results_final_llm/
├── full_graph_json/intermediate_answers.json
├── relevant_slots_full/intermediate_answers.json
└── topk_graph_records/intermediate_answers.json
```

#### Этап 3: Оценка судьёй (`judge_only`)

```bash
python validate_longmemeval.py \
    --config ./config_judge.json
```

Или с CLI:

```bash
python validate_longmemeval.py \
    --validation-mode judge_only \
    --input-answers-path ./results_final_llm/intermediate_answers.json \
    --input-state-dir ./results_memory_only \
    --output-dir ./results_judge \
    --judge-batch-size 10 \
    --calculate-memory-hit-rate \
    --config ../../DST_memory/run_config.json
```

**Параметры:**
- `--input-answers-path` - путь к `intermediate_answers.json` (обязательно)
- `--input-state-dir` - путь к сохранённым состояниям памяти (рекомендуется для Memory Hit Rate)
- `--calculate-memory-hit-rate` - включить оценку Memory Hit Rate

**Примечание:** `--input-state-dir` рекомендуется указывать для Memory Hit Rate, так как судья должен проверить наличие фактов в памяти. Если не указать, будут использованы пути из `intermediate_answers.json`, которые могут быть невалидны при перемещении файлов.

**Выходные файлы:**
```
results_judge/
└── validation_results.json     # Итоговые метрики
```

### Пример 3: Тестирование разных финальных LLM на одной памяти

Сначала обрабатываем диалоги один раз:

```bash
python validate_longmemeval.py \
    --validation-mode memory_only \
    --output-dir ./results_memory \
    --config ../../DST_memory/run_config.json
```

Затем тестируем разные финальные модели:

```bash
# GPT-4o-mini
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory \
    --output-dir ./results_llm_gpt4o_mini \
    --gm-llm-model "openai/gpt-4o-mini" \
    --config ../../DST_memory/run_config.json

# Claude 3.5 Sonnet
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory \
    --output-dir ./results_llm_claude \
    --gm-llm-model "anthropic/claude-3.5-sonnet" \
    --config ../../DST_memory/run_config.json

# Llama 3.1 70B (local)
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory \
    --output-dir ./results_llm_llama \
    --gm-llm-mode local \
    --gm-llm-model "meta-llama/Llama-3.1-70B-Instruct" \
    --config ../../DST_memory/run_config.json
```

И оцениваем всех одним судьёй:

```bash
for answers in ./results_llm_*/intermediate_answers.json; do
    output_dir=$(dirname "$answers")_judge
    python validate_longmemeval.py \
        --validation-mode judge_only \
        --input-answers-path "$answers" \
        --output-dir "$output_dir" \
        --judge-model "openai/gpt-4o"
done
```

### Пример 4: Memory Hit Rate Metric

```bash
# Полный пайплайн с MHR
python validate_longmemeval.py \
    --config ./config_full.json \
    --calculate-memory-hit-rate

# Или добавить MHR позже через judge_only
python validate_longmemeval.py \
    --validation-mode judge_only \
    --input-answers-path ./results/intermediate_answers.json \
    --calculate-memory-hit-rate \
    --output-dir ./results_with_mhr
```

**Вывод метрик:**
```
Statistics:
  Total processed: 40
  Correct: 35
  Incorrect: 5
  Average score: 0.875
  Memory hits: 38
  Memory misses: 2
  Memory Hit Rate: 0.95
```

**Интерпретация:**
- MHR = 95% - память сохраняет факты хорошо
- Accuracy = 87.5% - финальная LLM иногда ошибается
- Gap = 7.5% - небольшая проблема в использовании контекста

### Пример 5: Локальная финальная LLM с оптимизацией GPU

```bash
# Этап 1: Обработка памяти (использует GPU для classifier и slot model)
python validate_longmemeval.py \
    --validation-mode memory_only \
    --output-dir ./results_memory \
    --config ../../DST_memory/run_config.json

# Этап 2: Генерация ответов (выгружает модели памяти, загружает финальную LLM)
python validate_longmemeval.py \
    --validation-mode final_llm_only \
    --input-state-dir ./results_memory \
    --output-dir ./results_final_llm \
    --gm-llm-mode local \
    --gm-llm-model "meta-llama/Llama-3.1-70B-Instruct" \
    --gm-unload-models-before-final-llm true \
    --final-llm-batch-size 10 \
    --config ../../DST_memory/run_config.json

# Этап 3: Оценка (можно использовать local judge)
python validate_longmemeval.py \
    --validation-mode judge_only \
    --input-answers-path ./results_final_llm/intermediate_answers.json \
    --output-dir ./results_judge \
    --judge-mode local \
    --judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct" \
    --config ../../DST_memory/run_config.json
```

**Преимущества разделения:**
1. Модели памяти (небольшие) загружаются один раз
2. Финальная LLM (большая) загружается после выгрузки моделей памяти
3. Нет конфликтов за GPU память между компонентами

## Структура вывода

```
output-dir/
├── validation.log              # Полный лог выполнения
├── giga_memory_validation_logs.json  # Подробные логи GigaMemory (аналог DST_memory pipeline test *_logs.json): write_logs + answer_without_final_llm
├── validation_knowledge_graph.html   # HTML визуализация графа RAGU (если есть knowledge_graph.gml и save_intermediate=true)
├── validation_results.json     # Метрики и результаты (перезаписывается после каждой пары ответ+оценка; формат тот же)
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
│       "final_llm_prompt_chars": {
│         "calls": 40,
│         "before_clamp_total": 1223400,
│         "after_clamp_total": 988120
│       },
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
│         "final_llm_prompt_chars_before_clamp": 31240,
│         "final_llm_prompt_chars_after_clamp": 24577,
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
│   └── ragu_storage/          # Копия папки RAGU storage (тот же путь, что Settings.storage_folder / ragu_storage_path)
├── chunk_0001/
└── ...
```

## Полный список CLI параметров

### Параметры режима валидации

| Параметр | Описание | Возможные значения |
|----------|----------|-------------------|
| `--validation-mode` | Режим валидации | `full`, `memory_only`, `final_llm_only`, `judge_only` |
| `--input-state-dir` | Директория с состояниями (для `final_llm_only` и `judge_only`) | `./results_memory_only` |
| `--input-answers-path` | Путь к ответам (для `judge_only`) | `./results/full_graph_json/intermediate_answers.json` |
| `--memory-only-output-suffix` | Суффикс для выходных директорий | `_memory_only` |
| `--final-llm-memory-strategies` | Список стратегий памяти для `final_llm_only` | `full_graph_json,relevant_slots_full` |

### Параметры GigaMemory

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--gm-importance-model-path` | Путь к классификатору | `"./best_model"` |
| `--gm-slot-model-path` | Путь к slot model | `"Qwen/Qwen3-0.6B"` |
| `--gm-importance-threshold` | Порог важности | `0.25` |
| `--gm-memory-strategy` | Стратегия памяти | `full_graph_json`, `relevant_slots_full`, `topk_graph_records` |
| `--gm-graph-top-k-records` | Top-K для retrieval | `20` |
| `--gm-llm-mode` | Режим финальной LLM | `stub`, `local`, `openrouter`, `api`, `puter` |
| `--gm-llm-model` | Имя модели | `openai/gpt-oss-120b:free` |
| `--gm-llm-tokenizer-model` | HF tokenizer id/path для clamp | `Qwen/Qwen2.5-7B-Instruct` |
| `--gm-llm-api-key` | API ключ | `sk-or-v1-...` |
| `--gm-ragu-storage-path` | Путь к RAGU | `./ragu_storage` |
| `--gm-ragu-embedder-model` | Embedder модель | `deepvk/USER-bge-m3` |
| `--gm-slot-use-stub` | Stub режим | `true`, `false` |
| `--gm-slot-context-enabled` | Контекст слотов | `true`, `false` |
| `--gm-triplet-deletion-mode` | Режим удаления | `none`, `heuristic`, `llm_inline`, `llm_separate` |
| `--gm-prompt-language` | Язык промптов | `ru`, `en` |
| `--gm-unload-models-before-final-llm` | Выгрузка моделей | `true`, `false` |

### Параметры judge

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--judge-max-context-tokens` | Лимит prompt judge перед completion | `32768` |
| `--judge-tokenizer-model` | HF tokenizer id/path для clamp judge | `meta-llama/Llama-3.1-8B-Instruct` |

### Таблица сравнения режимов

| Характеристика | `full` | `memory_only` | `final_llm_only` | `judge_only` |
|----------------|--------|---------------|------------------|--------------|
| Загружает датасет | ✓ | ✓ | ✗ | ✗ |
| Обрабатывает память | ✓ | ✓ | ✗ | ✗ |
| Вызывает final LLM | ✓ | ✗ | ✓ | ✗ |
| Вызывает judge | ✓ | ✗ | ✗ | ✓ |
| Memory Hit Rate | ✓ | ✗ | ✗ | ✓ |
| Промежуточные файлы | chunk_*/ | memory_only_states.json | intermediate_answers.json | validation_results.json |
| Зависимости | Нет | Нет | memory_only | final_llm_only |
| API вызовы | 2N-3N | 0 | N | N-2N |
| GPU оптимизация | Ограничена | Да | Да | Не требуется |

**N** - количество тестовых примеров

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
    python validate_longmemeval.py \
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
        --output-dir ./results_batch_${start} \
        --num-items-per-type 5 \
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

## Сравнение версий

| Функция | v1 | v2 | v3 |
|---------|-----|-----|-----|
| Sequential processing | ✓ | ✓ | ✓ |
| Batch final LLM | ✗ | ✓ | ✓ |
| Batch judge | ✗ | ✓ | ✓ |
| Memory Hit Rate | ✗ | ✓ | ✓ |
| Model unloading | ✗ | ✓ | ✓ |
| CLI config override | Частично | Полная | Полная |
| Speed (local LLM) | Медленно | Быстро | Быстро |
| API calls (OpenRouter) | 2N | 2N - 3N | 2N - 3N |
| **Validation modes** | ✗ | ✗ | **✓** |
| **Separate stages** | ✗ | ✗ | **✓** |
| **A/B testing support** | ✗ | ✗ | **✓** |

### Режимы v3

```bash
# Полный пайплайн (как v2)
python validate_longmemeval.py --validation-mode full

# Только память - сохраняет состояния
python validate_longmemeval.py --validation-mode memory_only

# Только финальная LLM - загружает сохранённые состояния
python validate_longmemeval.py --validation-mode final_llm_only \
    --input-state-dir ./results_memory_only

# Только судья - оценивает сохранённые ответы
python validate_longmemeval.py --validation-mode judge_only \
    --input-answers-path ./results_final_llm/intermediate_answers.json
```
