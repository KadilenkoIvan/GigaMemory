# CONFIG reference (`run_config.json`)

`run.py` читает конфиг из:

1. `--config`, либо
2. `DST_MEMORY_CONFIG`, либо
3. `DST_memory/run_config.json`.

Перед этим загружается `DST_memory/.env` (`dotenv_loader.py`).

## `shared`

| key | type | description |
|---|---|---|
| `importance_model_path` | str | путь к модели бинарного классификатора важности |
| `importance_threshold` | float | порог класса important |
| `retrieval_top_k` | int | top-k для debug retrieval/сопутствующих запросов |
| `graph_top_k_records` | int | top-k для стратегии `topk_graph_records` |
| `recent_history_pairs` | int | размер окна последних пар user/assistant |
| `disable_memory_gate` | bool | отключить LLM gate для `relevant_slots_full` |
| `memory_gate_use_stub` | bool | использовать эвристику вместо локальной gate LLM |
| `memory_strategy` | str | `full_graph_json` \| `relevant_slots_full` \| `topk_graph_records` |
| `llm_mode` | str | `openrouter` \| `api` \| `stub` \| `local` |
| `llm_api_url` | str | OpenAI-compatible endpoint |
| `llm_api_key` | str | API key, можно через `${OPENROUTER_API_KEY}` |
| `llm_model` | str | model id провайдера |
| `llm_temperature` | float | температура final LLM |
| `llm_max_tokens` | int | max tokens final LLM |
| `openrouter_http_referer` | str | optional Referer header |
| `openrouter_x_title` | str | optional X-OpenRouter-Title |
| `no_final_llm` | bool | вернуть только структуру ответа без вызова final LLM |
| `log_level` | str | `INFO`, `DEBUG`, ... |
| `slot_use_stub` | bool | включить slot-triplet stub режим |
| `slot_model_path` | str | путь/id локальной slot LLM |
| `slot_max_slots_per_message` | int | лимит слотов на сообщение |
| `use_ragu` | bool | должен быть `true` (RAGU-only проект) |
| `ragu_embedder_model` | str | модель эмбеддингов для RAGU |
| `ragu_storage_path` | str | путь к RAGU storage |

## `pipeline_jsonl`

- `dataset_path`: path to jsonl
- `output_path`: output json

Используется в `pipeline test`.

## `pipeline_interactive`

- `dialogue_id`: default id для `pipeline inference interactive`.

## Пояснение стратегий

### `full_graph_json`

Final LLM получает полный JSON активной памяти:

```json
{
  "dialogue_id": "...",
  "slots": [
    {
      "slot": "FAMILY",
      "messages": [ ... ]
    }
  ]
}
```

### `relevant_slots_full`

1. Берутся активные слоты.
2. LLM-gate выбирает релевантные слоты.
3. В final LLM передается полное содержимое выбранных слотов.

Если `disable_memory_gate=true`, передаются все активные слоты.

### `topk_graph_records`

1. RAGU semantic search по всему графу.
2. Возвращаются top-k строк-графовых записей (`graph_top_k_records`).
3. Этот список идет в final LLM.

## История последних пар

`recent_history_pairs` задает размер окна из последних:

```json
[
  {"user": "...", "assistant": "..."},
  ...
]
```

Окно передается в final LLM вместе с memory context.
