# Конфигурация валидации через JSON файл

Скрипт `validate_longmemeval_v2.py` поддерживает конфигурацию через JSON файл, структура которого повторяет `DST_memory/run_config.json`.

## Структура конфига

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results",
    "start_index": 0,
    "num_items": 10,
    "log_level": "INFO",
    "log_file": true,
    "save_memory_state": true,
    "save_intermediate": true
  },
  "batch_processing": {
    "final_llm_batch_size": 1,
    "judge_batch_size": 1,
    "calculate_memory_hit_rate": false
  },
  "judge": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free",
    "api_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "temperature": 0.0,
    "max_tokens": 1024,
    "local_model_path": "",
    "unload_between_items": false
  },
  "giga_memory": {
    "importance_model_path": "",
    "importance_threshold": 0.25,
    "memory_strategy": "full_graph_json",
    "llm_mode": "openrouter",
    "llm_model": "openai/gpt-oss-120b:free",
    "slot_model_path": "DST_memory/models/Meno-Lite-0.1",
    "ragu_embedder_model": "deepvk/USER-bge-m3",
    "prompt_language": "ru",
    "unload_models_before_final_llm": true
  }
}
```

## Использование

### Базовый запуск (с дефолтным конфигом)

```bash
cd validation/full_pipeline
python validate_longmemeval_v2.py
```

Скрипт автоматически загрузит `run_config.json` из текущей директории.

### Кастомный конфиг

```bash
python validate_longmemeval_v2.py --config ./my_config.json
```

### Override через CLI

```bash
# Override параметров валидации
python validate_longmemeval_v2.py \
    --val-shared-start-index 20 \
    --val-shared-num-items 50 \
    --val-batch-final-llm-batch-size 10

# Override параметров судьи
python validate_longmemeval_v2.py \
    --val-judge-mode openrouter \
    --val-judge-model "anthropic/claude-3.5-sonnet"

# Override параметров GigaMemory
python validate_longmemeval_v2.py \
    --gm-memory-strategy topk_graph_records \
    --gm-graph-top-k-records 50
```

## Полный список --val-* параметров

### Validation (--val-shared-*)

| CLI Override | Соответствует в JSON | Тип |
|--------------|----------------------|-----|
| `--val-shared-dataset-path` | `shared.dataset_path` | string |
| `--val-shared-output-dir` | `shared.output_dir` | string |
| `--val-shared-start-index` | `shared.start_index` | int |
| `--val-shared-num-items` | `shared.num_items` | int |
| `--val-shared-log-level` | `shared.log_level` | DEBUG/INFO/WARNING/ERROR |
| `--val-shared-log-file` | `shared.log_file` | true/false |
| `--val-shared-save-memory-state` | `shared.save_memory_state` | true/false |
| `--val-shared-save-intermediate` | `shared.save_intermediate` | true/false |

### Batch Processing (--val-batch-*)

| CLI Override | Соответствует в JSON | Тип |
|--------------|----------------------|-----|
| `--val-batch-final-llm-batch-size` | `batch_processing.final_llm_batch_size` | int |
| `--val-batch-judge-batch-size` | `batch_processing.judge_batch_size` | int |
| `--val-batch-calculate-memory-hit-rate` | `batch_processing.calculate_memory_hit_rate` | true/false |

### Judge (--val-judge-*)

| CLI Override | Соответствует в JSON | Тип |
|--------------|----------------------|-----|
| `--val-judge-mode` | `judge.mode` | openrouter/local/none |
| `--val-judge-model` | `judge.model` | string |
| `--val-judge-api-key` | `judge.api_key` | string |
| `--val-judge-temperature` | `judge.temperature` | float |
| `--val-judge-max-tokens` | `judge.max_tokens` | int |
| `--val-judge-local-model-path` | `judge.local_model_path` | string |

## Примеры конфигов

### Конфиг для smoke test (быстрый тест)

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results_smoke",
    "start_index": 0,
    "num_items": 5,
    "log_level": "DEBUG"
  },
  "batch_processing": {
    "final_llm_batch_size": 5,
    "judge_batch_size": 5,
    "calculate_memory_hit_rate": false
  },
  "judge": {
    "mode": "none"
  },
  "giga_memory": {
    "llm_mode": "stub",
    "slot_use_stub": true
  }
}
```

### Конфиг для полного тестирования с Memory Hit Rate

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results_full",
    "start_index": 0,
    "num_items": 311,
    "log_level": "INFO"
  },
  "batch_processing": {
    "final_llm_batch_size": 20,
    "judge_batch_size": 40,
    "calculate_memory_hit_rate": true
  },
  "judge": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free",
    "api_key": "${OPENROUTER_API_KEY}"
  },
  "giga_memory": {
    "memory_strategy": "topk_graph_records",
    "graph_top_k_records": 30,
    "prompt_language": "en",
    "llm_mode": "openrouter",
    "llm_model": "openai/gpt-oss-120b:free"
  }
}
```

### Конфиг для локальной финальной LLM с выгрузкой

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results_local",
    "start_index": 0,
    "num_items": 50
  },
  "batch_processing": {
    "final_llm_batch_size": 10,
    "judge_batch_size": 10,
    "calculate_memory_hit_rate": false
  },
  "judge": {
    "mode": "local",
    "local_model_path": "meta-llama/Llama-3.2-1B-Instruct"
  },
  "giga_memory": {
    "llm_mode": "local",
    "llm_model": "meta-llama/Llama-3.1-70B-Instruct",
    "unload_models_before_final_llm": true,
    "slot_model_path": "Qwen/Qwen3-0.6B",
    "slot_use_stub": false,
    "memory_strategy": "full_graph_json"
  }
}
```

## Environment Variables в конфиге

Конфиг поддерживает подстановку переменных окружения:

```json
{
  "judge": {
    "api_key": "${OPENROUTER_API_KEY}"
  },
  "giga_memory": {
    "llm_api_key": "${OPENROUTER_API_KEY}",
    "importance_model_path": "${HOME}/models/best_model"
  }
}
```

Скрипт автоматически подставит значения из окружения.

## Приоритет параметров

1. **CLI аргументы** (`--val-*`, `--gm-*`) - высший приоритет
2. **Конфиг файл** (`--config` или `run_config.json`)
3. **Базовый конфиг** DST_memory (`DST_memory/run_config.json`)
4. **Default значения** - низший приоритет

## Миграция с v1 на v2

Если вы использовали старые CLI аргументы:

```bash
# Старый стиль (v1)
python validate_longmemeval.py \
    --dataset-path ... \
    --start-index 0 \
    --num-items 10

# Новый стиль (v2) - аналогично
python validate_longmemeval_v2.py \
    --val-shared-dataset-path ... \
    --val-shared-start-index 0 \
    --val-shared-num-items 10

# Или через конфиг файл
python validate_longmemeval_v2.py --config ./my_config.json
```
