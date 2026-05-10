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
| `llm_mode` | str | `openrouter` \| `api` \| `puter` \| `stub` \| `local` |
| `llm_api_url` | str | OpenAI-compatible endpoint |
| `llm_api_key` | str | API key, можно через `${OPENROUTER_API_KEY}` |
| `llm_model` | str | model id провайдера |
| `llm_tokenizer_model` | str | HF tokenizer id/path для clamp, если `llm_model` не HF-совместимый |
| `llm_temperature` | float | температура final LLM |
| `llm_max_tokens` | int | max tokens final LLM |
| `llm_max_context_tokens` | int | максимальный размер prompt-контекста в токенах (0 = без ограничения) |
| `llm_load_dtype` | str | dtype загрузки финальной локальной LLM при `llm_mode=local` (`float16`, `bfloat16`, …) |
| `llm_load_quantization` | str | BitsAndBytes для **финальной** локальной LLM: `none`, `8bit`, `4bit` (не путать с `slot_llm_load_quantization`) |
| `openrouter_http_referer` | str | optional Referer header |
| `openrouter_x_title` | str | optional X-OpenRouter-Title |
| `no_final_llm` | bool | вернуть только структуру ответа без вызова final LLM |
| `log_level` | str | `INFO`, `DEBUG`, ... |
| `slot_use_stub` | bool | включить slot-triplet stub режим |
| `slot_model_path` | str | путь/id локальной slot LLM |
| `slot_max_slots_per_message` | int | лимит слотов на сообщение |
| `slot_model_enable_thinking` | bool | гибридный thinking у Qwen3/3.5 для slot-стека (default `false`) |
| `slot_llm_inject_no_think_prompt` | bool | при `true` (default) к system добавляется `/no_think`, если thinking выключен; при `false` только `enable_thinking` в chat template |
| `slot_llm_lm_format_enforcer` | bool | JSON schema через `lm-format-enforcer` для slot selector и triplet extraction (default `false`) |
| `slot_llm_load_quantization` | str | квантизация BitsAndBytes для **той же** локальной модели, что `slot_model_path` (слоты, триплеты, memory gate, конфликты): `none` (default), `8bit`, `4bit`. Нужны **CUDA** и пакет `bitsandbytes`. Отдельно от `llm_load_quantization` (финальная LLM). |
| `prompt_language` | str | язык UI промптов для slot/triplet/gate/deletion/conflict и **финальной LLM** (`dst_memory/prompts/<ru|en>/final_llm_messages.py`): `ru` или `en` (хранилище фактов и канонические ключи слотов не меняются) |
| `use_ragu` | bool | должен быть `true` (RAGU-only проект) |
| `ragu_embedder_model` | str | модель эмбеддингов для RAGU |
| `ragu_storage_path` | str | путь к RAGU storage |
| `ttl_mode` | str | режим TTL: `mode1` (per-slot defaults из `SLOT_DEFAULT_TTL`), `mode2` (модель генерирует поле `ttl` вместе с триплетом), `mode3` (отдельный вызов — резервировано) |
| `ttl_slot_overrides` | dict | JSON-словарь переопределений TTL по слоту, напр. `{"EVENTS":"1d","TRAVEL":"1m"}` |
| `ttl_semantic_dedup_enabled` | bool | включить семантическую дедупликацию триплетов внутри слота |
| `ttl_semantic_dedup_threshold` | float | порог косинусного сходства для семантической дедупликации (default `0.9`) |

## Контекст слота и удаление фактов

| key | type | description |
|---|---|---|
| `slot_context_enabled` | bool | передавать текущие активные факты слота в промпт экстракции (default `false`) |
| `slot_context_max_facts` | int | макс. кол-во фактов в контексте слота (default `10`, защита от раздувания промпта) |
| `triplet_deletion_mode` | str | режим удаления: `none` / `heuristic` / `llm_inline` / `llm_separate` |
| `deletion_use_pymorphy` | bool | использовать pymorphy2 для лемматизации в режиме `heuristic` (default `false`) |

Примечание по Puter:
- Для `llm_mode=puter` используется OpenAI-compatible backend Puter (`https://api.puter.com/puterai/openai/v1`).
- Ключ можно передать в `llm_api_key` или через `PUTER_API_KEY`.

### Режимы удаления (`triplet_deletion_mode`)

| mode | вариант | описание |
|---|---|---|
| `none` | — | удаление отключено (текущее поведение по умолчанию) |
| `heuristic` | C | rule-based детектор отрицания без LLM. Паттерны: «больше не», «перестал», «уволился» и т.д. Каскадное совпадение: точное (subj+rel+obj) → по (subj+rel). Опционально с pymorphy2. |
| `llm_inline` | A | сигналы удаления в том же LLM-вызове что и экстракция. Требует `slot_context_enabled=true`. Модель возвращает `{"triplets":[...], "delete":[...]}`. |
| `llm_separate` | B | отдельный LLM-вызов для детекции устаревших фактов. Вызов всегда получает контекст текущих фактов, независимо от `slot_context_enabled`. |

### Матрица совместимости

| `slot_context_enabled` | `triplet_deletion_mode` | поведение |
|---|---|---|
| `false` | `none` | текущее поведение, без удаления |
| `false` | `heuristic` | Вариант C: эвристика, EXTRACTION без контекста |
| `false` | `llm_separate` | Вариант B: EXTRACTION без контекста, deletion-вызов получает контекст |
| `false` | `llm_inline` | **автоматически** включает `slot_context_enabled=true` + Вариант A |
| `true` | `none` | контекст виден модели, удаление не обрабатывается |
| `true` | `heuristic` | контекст + эвристика |
| `true` | `llm_inline` | Вариант A: один вызов, модель видит контекст и выдаёт delete-сигналы |
| `true` | `llm_separate` | контекст в EXTRACTION + отдельный deletion-вызов |

### Что происходит при удалении

1. Факт помечается `is_active=False` (мягкое удаление).
2. Синхронизируется в RAGU (`delete_triplet_deltas`).
3. Удаление происходит **до** семантической дедупликации и конфликт-резолвера.
4. Каскадная стратегия совпадения для `heuristic` и `llm_separate`:
   - Сначала точное `subject+relation+object`.
   - Если нет совпадений — по `subject+relation` (любой object).

### Семантика обновлений в контекст-режиме (`slot_context_enabled=true`)

При передаче контекста модель генерирует более богатые обновления:
- **Переезд в новое место** (`переехал в Сызрань`):
  - `DELETE` старый факт места жительства
  - `ADD` новый факт места жительства (Сызрань)
  - `ADD` исторический факт «бывшее место жительства» (Москва)
- **Явный отказ без нового значения** (`больше не живу в Москве`):
  - `DELETE` старый факт
  - `ADD` исторический факт «бывшее место жительства»

## Конфликт-резолвер

| key | type | description |
|---|---|---|
| `conflict_allow_multi_relation_same_object` | bool | Если `true` (default): два факта с одинаковым `subject` + одинаковым `object` но разными `relation` считаются **дополняющими** и LLM-вызов конфликт-резолвера для них пропускается. Пример: `есть партнёр` и `живёт вместе с` одного и того же объекта — оба факта сохраняются. Установи `false` чтобы всегда вызывать LLM при любом `same-subject` совпадении. |
| `conflict_rule_same_relation_updates` | bool | Если `true` (default): при **том же** `subject` и **той же** `relation`, но **другом** `object`, старые рёбра деактивируются **детерминированно** (без LLM), как смена значения предиката (`works at`). Если `false` — это правило отключено; такой кейс уходит **только в LLM** конфликт-резолвер. Точный дубликат (совпадение S+R+O) по-прежнему отсекается правилом без LLM. CLI: `--no-conflict-rule-same-relation-updates`. При прогоне `validate_longmemeval.py` ключ задаётся в **`giga_memory`** JSON или через `--gm-conflict-rule-same-relation-updates`, не в блоке `shared` валидационного конфига. На уровне **INFO** при вызове модели в лог пишется строка `Conflict resolver: calling LLM slot=... subjects=...`; при отложенном кейсе (same S+R, different O) при **DEBUG** — `Conflict deferred to LLM (same S+R, different O): ...`. |

Формат ответа LLM-конфликт-резолвера:
- Поддерживаются оба варианта JSON: `{"deactivate":[...], "skip_new":[...]}` и сокращённый `{"deactivate":[...]}`.
- При отсутствии `skip_new` парсер трактует его как пустой список (без ошибки пайплайна).

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

### Важно для валидации `validation/GigaMemory_full`

В stage-режиме:
- `memory_only` сохраняет состояние памяти (DST + RAGU + компактные strategy artifacts) без хранения трёх полных memory-context;
- `memory_only_write_mode=single_path_only` форсирует single-pass extraction (без `slot_select` и per-slot extraction) для A/B-сравнения write-path;
- `final_llm_only` может запускаться по одной или нескольким стратегиям на одном и том же state (без повторного `write_to_memory`);
- в `final_llm_only` можно выбрать формат передачи памяти в final LLM:
  - `with_metadata` — с метаданными записей;
  - `triplets_only` — только триплеты (`subject/relation/object`);
- RAGU state для retrieval берётся из `chunk_*/ragu_storage`, поэтому `topk_graph_records` использует тот же сохранённый граф.

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
