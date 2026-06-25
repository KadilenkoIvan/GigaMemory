# PIPELINE: полная техническая документация

Этот документ описывает полную техническую картину `DST_memory`: архитектуру, жизненный цикл данных, режимы запуска, форматы, ограничения и практические сценарии.

---

## 1. Назначение

`DST_memory` реализует долговременную память для LLM в виде структурированного графа фактов.

Цели:
- записывать только значимую пользовательскую информацию;
- поддерживать обновление/деактивацию конфликтующих фактов;
- **удалять устаревшие факты** при явном отказе пользователя (новые режимы);
- управлять временем жизни (TTL) каждого факта и автоматически «протухать» устаревшие записи;
- дедуплицировать семантически близкие факты в рамках одного слота;
- извлекать релевантный memory context для ответа;
- передавать memory context в final LLM в контролируемом формате с полными промптами в логах.

---

## 2. Высокоуровневая схема

```mermaid
---
config:
  theme: base
  look: classic
  layout: dagre
  themeVariables:
    fontSize: 30px
    fontFamily: Arial
    primaryColor: '#f3f4f6'
    primaryTextColor: '#111827'
    primaryBorderColor: '#374151'
    lineColor: '#000000'
    secondaryColor: '#ffffff'
    tertiaryColor: '#ffffff'
  flowchart:
    nodeSpacing: 300
    rankSpacing: 50
    curve: basis
---
flowchart TB
    writePath["Механизм записи в память"] --> importance_block["Блок определения важности"]
    importance_block --> importance{"Сообщение важное?"}
    importance -- нет --> earlyExit["Память без изменений"]
    importance -- да --> slotSelect["Блок выбора слотов"]
    slotSelect --> slotFound{"Слоты найдены?"}
    slotFound -- нет --> singlePath["Single-pass fallback"]
    slotFound -- да --> tripletExtract["Блок извлечения триплетов"]
    tripletExtract --> tripletsFound{"Триплеты найдены?"}
    tripletsFound -- нет --> singlePath
    tripletsFound -- да --> semanticDedup["Блок нахождения семантических повторений"]
    singlePath --> singlePathFound{"Single-pass<br>вернул триплеты?"}
    singlePathFound -- нет --> earlyExit
    singlePathFound -- да --> semanticDedup
    semanticDedup --> semanticDedupPath{"Семантически схожие факты найдены?"}
    semanticDedupPath -- да --> replaceFact["Деактивация старого и запись нового факта"]
    semanticDedupPath -- нет --> keepFact["Запись нового факта"]
    replaceFact --> conflictResolve["Блок разрешения конфликтов"]
    keepFact --> conflictResolve
    conflictResolve --> dstState["Обновление состояния памяти"]
    dstState --> raguSync["Синхронизация с RAGU"]
    earlyExit --> writeDone["Завершение write-path"]
    raguSync --> writeDone

     writePath:::process
     importance_block:::process
     importance:::decision
     earlyExit:::terminal
     slotSelect:::process
     slotFound:::decision
     singlePath:::process
     tripletExtract:::process
     tripletsFound:::decision
     semanticDedup:::process
     singlePathFound:::decision
     semanticDedupPath:::decision
     replaceFact:::process
     keepFact:::process
     conflictResolve:::process
     dstState:::process
     raguSync:::process
     writeDone:::terminal
    classDef titleClass fill:none,stroke:none,color:#111827,font-size:120px,font-weight:bold
    classDef process fill:#ffca86,stroke:#374151,stroke-width:5px,color:#111827
    classDef decision fill:#f3f4f6,stroke:#374151,stroke-width:5px,color:#111827
    classDef terminal fill:#ffca86,stroke:#374151,stroke-width:5px,color:#111827
```

```mermaid
%%{init: {"theme": "default", "themeVariables": {"fontSize": "22px"}}}%%
flowchart TD

    writePath[Механизм записи в память] --> importance_block[Блок определения важности]
    importance_block[Блок определения важности] --> importance{Сообщение важное?}
    importance -- "нет" --> earlyExit[Память без изменений]
    importance -- "да" --> slotSelect[Блок выбора слотов]
    slotSelect --> slotFound{Слоты найдены?}
    slotFound -- "нет" --> singlePath[Single-pass fallback]
    slotFound -- "да" --> tripletExtract[Блок извлечение триплетов]
    tripletExtract --> tripletsFound{Триплеты найдены?}
    tripletsFound -- "нет" --> singlePath
    singlePath --> singlePathFound{Single-pass вернул триплеты?}
    singlePathFound -- "нет" --> earlyExit[Память без изменений]
    singlePathFound -- "да" --> semanticDedup
    tripletsFound -- "да" --> semanticDedup[Блок нахождение семантических повторений]
    semanticDedup[Блок нахождение семантических повторений] --> semanticDedupPath{Семантически схожие факты найдены?}
    semanticDedupPath -- "да" --> replaceFact[Деактивация старого и запись нового факта]
    semanticDedupPath -- "нет" --> keepFact[Запись нового факта]
    replaceFact --> conflictResolve
    keepFact --> conflictResolve
    conflictResolve --> dstState[Обновление состояния памяти]
    dstState --> raguSync[Синхронизация с RAGU]
    earlyExit --> writeDone[Завершение write-path]
    raguSync --> writeDone

    askMsg[Вопрос пользователя] --> answerPath[Подготовка ответа]
    answerPath --> strategy[Выбор стратегии памяти]
    strategy --> fullGraph[Полный граф памяти]
    strategy --> relSlots[Релевантные слоты]
    strategy --> topk[Топ-K записей графа]
    fullGraph --> promptBuild[Сборка промпта]
    relSlots --> promptBuild
    topk --> promptBuild
    history[Последние 10 пар диалога] --> promptBuild
    promptBuild --> finalLLM[Финальная LLM]
    finalLLM --> response[Ответ ассистента]
```

---

## 3. Модульная структура

```
GigaMemory/
├── run.py                      — CLI точка запуска (pipeline test / inference)
├── api.py                      — FastAPI REST сервер (Этап 2)
├── run_config.json             — default конфиг (валидация, pipeline test)
├── run_config_local.json       — конфиг для локального интерактивного режима
├── run_config_api.json         — конфиг для REST API сервера
└── dst_memory/
    ├── __init__.py             — экспорт PipelineConfig, Message, MemoryFact
    ├── core/                   — ядро пайплайна
    │   ├── pipeline.py         — DSTMemoryPipeline (write/answer/clear/save_session)
    │   ├── dst_manager.py      — DSTManager: слоты, TTL, конфликты, дедуп, RAGU-sync
    │   ├── models.py           — Message, FactRecord, MemoryFact, DialogueMemoryState, TTL_TO_TIMEDELTA
    │   ├── config.py           — PipelineConfig, SLOT_DEFAULT_TTL
    │   └── graph_backend.py    — GraphEdge (dataclass)
    ├── prompts/                — сборщики промптов и few-shot банки по языку UI
    │   ├── loader.py, parsers.py  — выбор `ru`/`en`, общие JSON-парсеры
    │   ├── ru/                 — русскоязычные тексты (system/user, few-shots,
    │   │                         realtime_mode_notice, parallel_write_notice)
    │   └── en/                 — English UI (аналогичная структура)
    ├── slots/                  — слоты и онтология
    │   ├── ontology.py         — SlotOntology, DEFAULT_USER_SLOTS, метки RU
    │   ├── slot_name_normalize.py
    │   ├── slot_model_path.py  — разрешение путей к модели слотов
    │   ├── slot_select_client.py  — SlotSelectClient
    │   └── slot_update_client.py  — SlotUpdateClient
    ├── triplets/               — извлечение и управление триплетами
    │   ├── triplet_client.py   — TripletExtractionClient
    │   ├── conflict_client.py  — TripletConflictClient (rule + LLM)
    │   ├── deletion_client.py  — TripletDeletionClient (llm_separate)
    │   └── negation_detector.py   — NegationDeletionDetector (heuristic)
    ├── storage/                — RAGU backend
    │   └── ragu_graph_processor.py — RaguGraphProcessor, build_ragu_processor
    ├── clients/                — LLM-клиенты и serving
    │   ├── serving.py          — LocalHFServing (HF CausalLM; Python 3.13 compat)
    │   ├── classifier.py       — ImportanceClassifier (use_fast=False для Py3.13)
    │   ├── memory_gate_client.py  — MemoryGateClient
    │   └── llm_client.py       — FinalLLMClient (realtime_mode, parallel_write_mode)
    └── utils/
        ├── io_utils.py         — read_jsonl, iter_user_messages, iter_dialogue_messages
        ├── dotenv_loader.py    — загрузка .env
        └── run_config_loader.py   — load_run_config, shared_section, subsection
```

### 3.1 Entry-points

- `run.py` — CLI-пайплайн:
  - парсинг CLI;
  - загрузка конфига (`run_config.json` + `.env`);
  - bootstrap RAGU-пути (`_ensure_local_ragu_import`);
  - автогенерация datetime-суффикса для `dialogue_id` в интерактивном режиме;
  - сборка `DSTMemoryPipeline`;
  - выполнение `module`/`pipeline` команд.

- `api.py` — FastAPI REST сервер (Этап 2):
  - singleton `DSTMemoryPipeline` инициализируется при старте (`lifespan`);
  - конфиг из `GIGAMEMORY_CONFIG` env или `run_config_api.json`;
  - per-dialogue threading.Lock для последовательного режима;
  - поддержка parallel_write через background threading.Thread;
  - 6 HTTP endpoints (message, graph, graph_short, graph/image, graph/html, DELETE).

### 3.2 Core pipeline

- `dst_memory/core/pipeline.py`
  - запись в память (`write_to_memory`);
  - формирование memory context (`_memory_context_for_question`);
  - генерация ответа (`answer`);
  - диагностический режим без final LLM (`answer_without_final_llm`);
  - окно последних пар (`add_recent_pair`, `recent_pairs`).

### 3.3 DST state manager

- `dst_memory/core/dst_manager.py`
  - хранит `DialogueMemoryState`;
  - записывает факты в слоты;
  - **семантическая дедупликация** (pre-pass, threshold 0.9): при обнаружении семантически близкого факта в том же слоте старый деактивируется, новый вставляется;
  - конфликт-резолвинг (rule-based + LLM);
  - **TTL expiry**: ленивая проверка `is_active=False` при каждом чтении/записи;
  - синхронизация вставок/удалений в RAGU;
  - выдача active slots, expired facts и slot payload.

### 3.4 Модели и форматы

- `dst_memory/core/models.py`
  - `Message`
  - `FactRecord` — включает `ttl: str` и `created_at_datetime: str` (ISO); метод `is_expired()` для ленивой проверки
  - `MemoryFact`
  - `DialogueMemoryState`:
    - `step`
    - `slots`
    - `next_record_id`
    - `recent_pairs` (последние `user/assistant` пары)

### 3.5 LLM-компоненты

- `dst_memory/clients/classifier.py` — importance classifier.
- `dst_memory/slots/slot_select_client.py` — выбор слотов.
- `dst_memory/triplets/triplet_client.py` — извлечение триплетов.
- `dst_memory/triplets/conflict_client.py` — LLM-конфликт-резолвер.
- `dst_memory/clients/memory_gate_client.py` — выбор релевантных слотов для ответа.
- `dst_memory/clients/llm_client.py` — финальная генерация ответа.

### 3.6 RAGU интеграция

- `dst_memory/storage/ragu_graph_processor.py`
  - адаптер `SentenceTransformerEmbedder`;
  - мост sync/async;
  - upsert/delete triplets;
  - semantic search через `LocalSearchEngine`.

---

## 4. Write-path (запись сообщения в память)

Функция: `DSTMemoryPipeline.write_to_memory(dialogue_id, Message(role="user", ...))`

Шаги:
1. Проверка роли: записываются только `user`-сообщения.
2. Importance classifier:
   - если сообщение неважное → `saved=False, reason=not_important`.
3. Slot selection (`SlotSelectClient`):
   - выбираются целевые слоты по онтологии.
4. TTL expiry pre-pass:
   - все факты в затронутых слотах проверяются на `is_expired()`; протухшие деактивируются (`is_active=False`) и синхронизируются в RAGU.
5. Triplet extraction (`TripletExtractionClient`):
   - извлекаются триплеты по слотам (lowercase/русский язык);
   - в режиме `ttl_mode=mode2` модель дополнительно генерирует поле `ttl` для каждого триплета.
6. Семантическая дедупликация (pre-pass, только в рамках одного слота):
   - для каждого нового триплета вычисляется косинусное сходство с активными записями слота;
   - если сходство ≥ `ttl_semantic_dedup_threshold` (default 0.9), старый факт деактивируется, новый вставляется (таймер TTL обновляется).
7. Conflict resolution (`TripletConflictClient`):
   - rule-layer + LLM-layer;
   - по умолчанию правило: тот же `subject`+`relation`, другой `object` → авто-деактивация старых (см. `conflict_rule_same_relation_updates` в `CONFIG.md`; при `false` только LLM);
   - возможна деактивация старых записей и/или skip новых;
   - `skip_new` в ответе LLM является опциональным: парсер корректно обрабатывает ответы только вида `{"deactivate":[...]}`.
8. Обновление DST state:
   - добавляются `FactRecord` с полями `ttl` и `created_at_datetime`;
   - обновляется `step`, `record_id`.
9. RAGU sync:
   - новые записи → `upsert_triplet_deltas` (TTL хранится в поле `description` ребра);
   - деактивации → `delete_triplet_deltas`.

---

## 5. Answer-path (формирование context и ответ)

Функция: `DSTMemoryPipeline.answer(dialogue_id, question)`

1. Выбор стратегии memory context.
2. Сбор context payload.
3. Добавление последних `recent_pairs`.
4. Передача в `FinalLLMClient.generate(...)`.

### 5.1 Стратегия `full_graph_json`

В final LLM идет полный JSON активной памяти:

```json
{
  "dialogue_id": "d1",
  "slots": [
    {
      "slot": "FAMILY",
      "messages": [
        {
          "record_id": 1,
          "message_text": "...",
          "source_text": "...",
          "subject": "...",
          "relation": "...",
          "object": "...",
          "created_at_step": 1,
          "updated_at_step": 1,
          "is_active": true,
          "ttl": "inf",
          "created_at_datetime": "2025-01-01T12:00:00"
        }
      ]
    }
  ]
}
```

### 5.2 Стратегия `relevant_slots_full`

1. Берутся active slots.
2. `MemoryGateClient` решает `use_memory` и выбирает слоты.
3. В context идет полное содержимое выбранных слотов.
4. Если `disable_memory_gate=true`, берутся все active slots.

### 5.3 Стратегия `topk_graph_records`

1. Вопрос -> RAGU semantic search по всему графу.
2. Берется top-k (`graph_top_k_records`, default 20).
3. В context идет список наиболее релевантных графовых строк.

### 5.4 Stage-валидация (`validation/GigaMemory_full`)

- В `memory_only` сохраняется не "фиксированный final prompt-context", а переиспользуемое состояние памяти:
  - DST snapshot;
  - `chunk_*/ragu_storage` (для retrieval);
  - компактные strategy artifacts (выбор релевантных слотов и retrieval candidates), без дублирования трёх полных memory-context.
- Для `memory_only` можно включить `single_path_only` режим записи: slot selection bypassed, экстракция идет только single-pass путём.
- В `final_llm_only` стратегия выбирается при генерации ответа (`--final-llm-memory-strategies`), можно запускать сразу несколько стратегий на одном `memory_only` результате.
- Дополнительно есть режим payload-а для final LLM:
  - `with_metadata` (полные записи),
  - `triplets_only` (только `subject/relation/object`).

---

## 6. Последние 5 пар user/assistant

Хранятся в `DialogueMemoryState.recent_pairs`.

Поток:
- В `test` режиме пары восстанавливаются из входного диалога (`iter_dialogue_messages`).
- В `inference` режиме пара добавляется после получения финального ответа.

Формат:

```json
[
  {"user": "....", "assistant": "...."},
  {"user": "....", "assistant": "...."}
]
```

---

## 7. Режимы запуска

## 7.1 `pipeline test`

Назначение: offline-прогон датасета.

Логика:
- все user-реплики пишутся в память;
- пары user/assistant попадают в `recent_pairs`;
- финальный вопрос задается один раз;
- сохраняются:
  - compact output,
  - подробные логи.

Команда:

```bash
python DST_memory/run.py pipeline test --dataset-path data/format_example.jsonl --output-path DST_memory/output.json
```

## 7.2 `pipeline inference interactive`

Назначение: онлайн real-time диалог с долговременной памятью.

**Изоляция сессий:** при каждом старте к `dialogue_id` автоматически добавляется суффикс `_YYYY-MM-DD_HH-MM-SS`. Каждая сессия хранится в `<session_dir>/<dialogue_id>/` — DST-state (`state.json`) и RAGU-граф (`ragu/`). Явный `--dialogue-id full_id_with_timestamp` восстанавливает конкретную сессию.

**Логика на шаг (sequential mode):**
1. `write_to_memory` — важность → слоты → триплеты → dedup → конфликты → DST → RAGU
2. `answer` — memory context + recent_pairs → финальная LLM → ответ
3. `add_recent_pair` — добавить пару в окно истории

**Логика на шаг (parallel_write mode, `--parallel-write`):**
1. Запускается фоновый поток: `write_to_memory` (весь write-path)
2. Одновременно: `answer` из текущего графа (до нового сообщения)
3. `add_recent_pair` — сразу после получения ответа
4. Финальная LLM получает `parallel_write_notice()` в system prompt: явно указывает, что `recent_pairs` актуальнее графа

**Real-time prompt:** при интерактивном режиме финальная LLM получает `realtime_mode_notice()` в system prompt — переключает фрейм с «ассистент на основе памяти» на «общий AI-ассистент, память — контекст персонализации». Не влияет на оценочный пайплайн (`pipeline test`).

**Команды в диалоге:**
- `/clear` — сбросить память текущего диалога
- `/exit` — завершить сессию
- `/memory` — вывести активные слоты и факты в JSON
- `/expired` — вывести протухшие факты (is_active=False) в JSON

## 7.3 `pipeline inference single-turn`

Назначение: API-friendly single request.

Логика:
- одно сообщение записывается;
- сразу один ответ;
- возвращается json с write_log + answer.

## 7.4 REST API (`DST_memory/api.py`)

Назначение: продуктовый HTTP-слой поверх пайплайна. FastAPI приложение с одним singleton-пайплайном на процесс.

### Инициализация

При запуске сервера:
1. Читается конфиг (`GIGAMEMORY_CONFIG` env → `DST_memory/run_config_api.json`).
2. Загружается `DSTMemoryPipeline` (все модели + RAGU backend).
3. На `pipeline.final_llm` устанавливается `realtime_mode=True` (real-time промпт для финальной LLM).

### Endpoints

| Метод | Путь | Логика |
|---|---|---|
| `POST /dialogue/{id}/message` | Принять `content`, запустить `write_to_memory` → `answer` → `add_recent_pair`. При `parallel_write=true` — запись в фоновом потоке, ответ строится сразу из текущего графа. |
| `GET /dialogue/{id}/graph` | Вызов `dst.slots_with_messages(id)` — все активные факты со всеми метаданными. |
| `GET /dialogue/{id}/graph_short` | Только активные триплеты (subject/relation/object/ttl) + вычисленное поле `expires_at` (ISO datetime или null для `inf`). |
| `GET /dialogue/{id}/graph/image` | PNG-визуализация через networkx + matplotlib. Узлы — сущности, рёбра — отношения, цвет — по слоту. |
| `GET /dialogue/{id}/graph/html` | Интерактивный HTML через pyvis (тёмный фон, drag & drop узлов, hover tooltips, легенда слотов). |
| `DELETE /dialogue/{id}` | `pipeline.clear_memory(id)` — сброс DST-состояния и RAGU-графа. |

### Параллельная запись

`POST /message` с `parallel_write: true`:
- Запись в граф стартует в background-треде.
- Ответ строится немедленно на основе текущего состояния памяти (до текущего сообщения).
- Поведение аналогично `--parallel-write` в интерактивном CLI-режиме.

### Изоляция диалогов

Каждый `dialogue_id` — независимое состояние. Один сервер может обслуживать произвольное количество диалогов одновременно. Per-dialogue lock гарантирует последовательность операций внутри одного диалога в синхронном режиме.

### Персистентность

Если `api.session_dir` задан, после каждого `write_to_memory` состояние сохраняется в `<session_dir>/<dialogue_id>/state.json`.  
RAGU-граф хранится в `api.ragu_storage_path` (разделяется между всеми диалогами).

---

## 8. RAGU: хранение и релевантность

### 8.1 Как сохраняются данные в RAGU

Для каждого триплета:
- `Entity(subject, entity_type=slot)`
- `Entity(object, entity_type=slot)`
- `Relation(subject → object, relation_type, slot)`

TTL хранится в поле `description` ребра в виде аннотации `[ttl:Xm]`, например:
```
пользователь есть сестра сестра пользователя [ttl:inf]
```

`record_id` хранится в mapping для последующего delete.

### 8.2 Где хранится

- Персистентные файлы RAGU в `ragu_storage` (или `ragu_storage_path`).
- Внешняя серверная БД (Postgres/Qdrant/Milvus) в текущем проекте не используется.

### 8.3 Как считается релевантность

- Semantic ranking `LocalSearchEngine.a_search(query, top_k)`.
- Для `relevant_slots_full` дополнительно действует slot-filter через gate.

---

## 9. Final LLM prompt

`FinalLLMClient` получает:
- `question`
- `memory_context` (структурный JSON/records)
- `recent_pairs`

Язык system/user текстов финальной LLM совпадает с `prompt_language` (`ru` \| `en`): шаблоны в `dst_memory/prompts/ru/final_llm_messages.py` и `dst_memory/prompts/en/final_llm_messages.py` (префикс политики API без инструментов — одноязычный внутри выбранной локали). Константа `CHAT_API_OUTPUT_POLICY` в `llm_client.py` остаётся двуязычной только для **judge**-вызовов в скриптах валидации.

В `openrouter/api` режиме отправляется OpenAI-compatible `chat/completions`.

### Поддерживаемые режимы final LLM

- `stub` — локальный шаблонный ответ.
- `openrouter` — основной runtime режим.
- `api` — OpenAI-compatible endpoint.
- `puter` — OpenAI-compatible endpoint Puter (`https://api.puter.com/puterai/openai/v1`).
- `local` — локальная HF-модель через `LocalHFServing` (поддерживаются `llm_load_dtype`, `llm_load_quantization`, `llm_max_context_tokens`).

Локальная модель **слотов/триплетов/gate** (`slot_model_path`) настраивается отдельно: `slot_llm_load_quantization` (`none` \| `8bit` \| `4bit`, BitsAndBytes + CUDA), см. `CONFIG.md` и `DST_memory/run.py --slot-llm-load-quantization`.

---

## 10. Форматы выходов

### 10.1 `answer_without_final_llm`

Возвращает:
- `dialogue_id`
- `question`
- `use_memory`
- `memory_gate` (метаданные режима)
- `memory_context_for_final_llm`
- `recent_pairs`
- `retrieved`
- `memory_slots`
- `final_llm_prompt` — полные system + user промпты для final LLM
- `expired_facts` — список деактивированных фактов (TTL-expiry)

### 10.2 `pipeline test` файлы

- `<output>.json` — compact результат по диалогам.
- `<output>_logs.json` — подробные write/answer-логи, включая `final_llm_prompt` (полные промпты) и `expired_facts`.

---

## 11. Конфиг-параметры (критичные)

**Стратегия и память:**
- `memory_strategy` — `full_graph_json` | `relevant_slots_full` | `topk_graph_records`
- `graph_top_k_records` — top-k для `topk_graph_records`
- `recent_history_pairs` — размер окна последних пар user/assistant
- `disable_memory_gate`, `memory_gate_use_stub`

**Модели:**
- `slot_use_stub`, `importance_model_path`
- `ragu_embedder_model`, `ragu_storage_path`
- `llm_mode`, `llm_api_url`, `llm_api_key`, `llm_model`

**Сессии и параллельный режим:**
- `session_dir` — директория сессий; автоматически создаёт изолированные подпапки с datetime
- `parallel_write_mode` — параллельная запись (или `--parallel-write` в CLI)
- `force_infinite_ttl` — если `true`, все факты получают TTL `inf`; установить `false` для реального TTL

**TTL и дедупликация:**
- `ttl_mode` — `mode1` (per-slot defaults) | `mode2` (модель генерирует поле `ttl`)
- `ttl_slot_overrides` — JSON-словарь переопределений, напр. `{"EVENTS": "1d"}`
- `ttl_semantic_dedup_enabled`, `ttl_semantic_dedup_threshold`

Полная таблица: `CONFIG.md`.

---

## 12. Набор smoke-команд без внешних LLM

### full_graph_json

```bash
python DST_memory/run.py --llm-mode stub --slot-use-stub --memory-gate-use-stub --no-final-llm --memory-strategy full_graph_json pipeline test --dataset-path data/format_example.jsonl --output-path DST_memory/smoke_full_graph.json
```

### relevant_slots_full

```bash
python DST_memory/run.py --llm-mode stub --slot-use-stub --memory-gate-use-stub --no-final-llm --memory-strategy relevant_slots_full pipeline test --dataset-path data/format_example.jsonl --output-path DST_memory/smoke_relevant_slots.json
```

### topk_graph_records

```bash
python DST_memory/run.py --llm-mode stub --slot-use-stub --memory-gate-use-stub --no-final-llm --memory-strategy topk_graph_records --graph-top-k-records 20 pipeline test --dataset-path data/format_example.jsonl --output-path DST_memory/smoke_topk.json
```

---

## 13. Текущие технические ограничения

- **Python 3.13 + tokenizers Rust**: `encode_batch` падает с TypeError на Python 3.13. Обход: `use_fast=False` для BERT-классификатора, `backend_tokenizer.encode()` fallback в LocalHFServing. Долгосрочный фикс: `pip install tokenizers --upgrade` до версии ≥ 0.21.
- `ttl_mode=mode3` (отдельный вызов модели для TTL) не реализован — используется `mode2`.
- Качество retrieval зависит от embedder-модели и качества триплетов.
- Семантическая дедупликация требует инициализации embedder в `RaguGraphProcessor`; если embedder не загружен — пропускается с предупреждением.
- REST API: RAGU-граф разделяется между всеми диалогами (один `ragu_storage_path`); для полной изоляции — запускать отдельные процессы с разными конфигами.
- Параллельный режим (`--parallel-write`) не защищён от конкурентных POST-запросов к одному `dialogue_id` через API — в sequential режиме это гарантирует per-dialogue lock.

---

## 14. Что смотреть при отладке

- `run.py` — CLI и routing команд.
- `dst_memory/core/pipeline.py` — memory strategy и answer flow.
- `dst_memory/core/dst_manager.py` — запись фактов и конфликт-логика.
- `dst_memory/storage/ragu_graph_processor.py` — синхронизация и поиск.
- `dst_memory/clients/llm_client.py` — финальный prompt и вызов API.
- `run_config.json` — фактический runtime профиль.
