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
| `ttl_mode` | str | режим TTL: `mode1` (per-slot defaults из `SLOT_DEFAULT_TTL`), `mode2` (модель генерирует поле `ttl` вместе с триплетом), `mode3` (отдельный вызов — резервировано) |
| `ttl_slot_overrides` | dict | JSON-словарь переопределений TTL по слоту, напр. `{"EVENTS":"1d","TRAVEL":"1m"}` |
| `ttl_semantic_dedup_enabled` | bool | включить семантическую дедупликацию триплетов внутри слота |
| `ttl_semantic_dedup_threshold` | float | порог косинусного сходства для семантической дедупликации (default `0.9`) |

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

---

## TTL (Time-To-Live)

### Режимы

| mode | описание |
|---|---|
| `mode1` | Каждому слоту назначается фиксированный TTL из словаря `SLOT_DEFAULT_TTL` в `config.py`. Применяется, если модель не сгенерировала TTL. |
| `mode2` | Модель генерирует поле `ttl` вместе с триплетом в одном вызове (быстро, рекомендуется). При невалидном значении — fallback на `mode1`. |
| `mode3` | Отдельный вызов модели только для TTL (резервировано; в текущей реализации ведёт себя как `mode2`). |

### Допустимые значения TTL

`1d`, `3d`, `10d`, `2w`, `3w`, `1m`, `3m`, `6m`, `1y`, `inf`

### Дефолты по слотам (`SLOT_DEFAULT_TTL`)

| slot | default TTL | обоснование |
|---|---|---|
| IDENTITY | inf | имя, пол — постоянно |
| FAMILY | inf | родственные связи |
| FRIENDS | inf | друзья |
| ROMANCE | 1y | отношения меняются |
| WORK | 1y | место работы |
| EDUCATION | 1y | учёба |
| FINANCE | 3m | финансы актуальны недолго |
| HEALTH | 1y | здоровье |
| MENTAL_HEALTH | 6m | психологическое состояние |
| HABITS | inf | привычки |
| PREFERENCES | 6m | предпочтения |
| HOBBIES | 6m | хобби |
| SPORTS | 6m | спорт |
| FOOD | 1m | пищевые привычки |
| HOME | 1y | место жительства |
| LOCATION | 1y | геолокация |
| TRAVEL | 3m | путешествия |
| PETS | inf | питомцы |
| TECH | 6m | техника |
| VEHICLES | 1y | транспорт |
| SCHEDULE | 1m | расписание |
| GOALS | 3m | цели/планы |
| EVENTS | 2w | текущие события |

### Поведение при истечении TTL

- Факт помечается `is_active=False` (мягкое удаление).
- Деактивированный факт не передаётся в final LLM.
- Синхронизируется в RAGU (удаляется из графа).
- В `*_logs.json` выводится в отдельном поле `expired_facts`.
- Проверка происходит лениво при каждом чтении/записи памяти.

### Семантическая дедупликация

- Включается через `ttl_semantic_dedup_enabled=true`.
- Сравнение только внутри одного слота.
- При косинусном сходстве ≥ `ttl_semantic_dedup_threshold` (default 0.9):
  - старый факт → `is_active=False`;
  - новый факт вставляется с актуальным `created_at_datetime` (таймер TTL обновляется).
