# GigaMemory — план доработки и развития

Этот файл описывает приоритеты улучшений проекта для участия в Junior ML Contest 2026 (ai.itmo.ru/junior_ml_contest) и общего развития продукта.

---

## Текущее состояние

### Сильные стороны
- Нетривиальная архитектура: DST-граф триплетов + RAGU semantic retrieval
- Три независимых режима удаления фактов (inline, separate, heuristic)
- Три стратегии памяти (`full_graph_json`, `relevant_slots_full`, `topk_graph_records`)
- Валидация на LongMemEval: измерены accuracy и memory hit rate для 3 моделей и 3 стратегий
- Инженерная инфраструктура: CI/CD, Docker, mypy, pre-commit, 89 тестов
- Real-time интерактивный режим с изоляцией сессий по datetime
- Параллельный путь записи: ответ немедленно, граф обновляется в фоне
- REST API сервер (FastAPI) с 6 endpoints и автодокументацией Swagger

### Что пока отсутствует
- Telegram-бот (строится поверх REST API)
- Графики экспериментов не добавлены в репозиторий (есть отдельно, будут добавлены)
- DS артефакты (Jupyter notebook с анализом результатов валидации)

---

## Этапы доработки

### Этап 0 — добавление результатов экспериментов (ожидает файлы)

Когда появятся графики из LLM-as-a-judge экспериментов:
- Добавить в `docs/results/` или `notebooks/results/`
- Добавить таблицу сравнения стратегий и моделей в README
- Добавить секцию Benchmark Results в README с числами accuracy и memory hit rate

**Ожидаемые графики:**
- Accuracy по стратегиям (`full_graph_json` vs `relevant_slots_full` vs `topk_graph_records`)
- Accuracy по моделям (3 модели × 3 стратегии)
- Memory hit rate как отдельная метрика
- Сравнение с baseline

---

### Этап 1 — улучшение и тестирование core-функциональности ✅ DONE

#### 1.1 Real-time режим (inference interactive) ✅

**Реализовано:**
- Автоматическая изоляция сессий: к `dialogue_id` добавляется суффикс `_YYYY-MM-DD_HH-MM-SS` при каждом запуске, каждая сессия хранится в отдельной папке
- Сохранение сессий на диск через `session_dir` (DST state + RAGU граф)
- Корректная обработка прерываний (Ctrl+C, пустой ввод, `/exit`)
- `realtime_mode_notice()` — аддитивный промпт в системном сообщении, переключает финальную LLM с «ассистент на основе памяти» на «общий AI-ассистент, память — контекст для персонализации»
- Команды в интерактивном режиме: `/clear`, `/exit`, `/memory`, `/expired`

**Конфиг:** `DST_memory/run_config_local.json`

#### 1.2 Параллельный путь записи и ответа ✅

**Реализовано:**
- Флаг `--parallel-write` в `pipeline inference interactive`
- Запись в граф (extraction → dedup → conflict → DST → RAGU) запускается в фоновом потоке (`threading.Thread`)
- Ответ строится немедленно из текущего графа (без фактов из текущего сообщения)
- Промпт `parallel_write_notice()` — явно указывает финальной LLM, что последние сообщения в `recent_pairs` актуальнее графа
- Параметр `parallel_write_mode: bool` в `PipelineConfig` и `run_config.json`

**Дополнительные исправления этапа:**
- Python 3.13 + tokenizers Rust: `use_fast=False` для BERT-классификатора, `backend_tokenizer.encode()` fallback в serving.py
- `uv sync` conflicting indexes: переход с `extra = "cuda"` на `sys_platform != 'darwin'` маркеры в `[tool.uv.sources]`

#### 1.3 Покрытие тестами
- Тесты для параллельного режима (mock async) — TODO (в следующих этапах)

---

### Этап 2 — продуктовый слой: REST API ✅ DONE

#### 2.1 FastAPI сервер (`DST_memory/api.py`)

**Реализованные endpoints:**
- `POST /dialogue/{dialogue_id}/message` — принять сообщение, вернуть ответ LLM; поддерживает `parallel_write: true`
- `GET /dialogue/{dialogue_id}/graph` — полный граф памяти (JSON со всеми метаданными)
- `GET /dialogue/{dialogue_id}/graph_short` — только активные триплеты + `expires_at` (ISO или null)
- `GET /dialogue/{dialogue_id}/graph/image` — PNG-визуализация (networkx + matplotlib; узлы по слотам, рёбра по TTL)
- `GET /dialogue/{dialogue_id}/graph/html` — интерактивный HTML (pyvis, тёмная тема, forceAtlas2, drag & drop)
- `DELETE /dialogue/{dialogue_id}` — сбросить память диалога

**Конфигурация:**
- Отдельный конфиг `DST_memory/run_config_api.json`
- Ключ OpenRouter через `OPENROUTER_API_KEY` env
- Путь к конфигу через `GIGAMEMORY_CONFIG` env
- Swagger UI автоматически на `/docs`

**Инфраструктура:**
- `pyproject.toml`: extras `[api]` — fastapi, uvicorn, matplotlib, networkx; `[vllm]` — vllm>=0.8.0
- `Makefile`: `make install` = всё, `make install-local` = только пайплайн, `make install-api` = CUDA + API; `make serve` запускает FastAPI; `make vllm` запускает vLLM; `make start` запускает оба вместе
- `docker-compose.yml`: сервис `api` на порту 8000, volume `api_sessions`

#### 2.2 vLLM inference backend для слот-модели ✅ DONE

**Реализовано:**
- Новый класс `VLLMSlotServing` (`DST_memory/dst_memory/clients/vllm_serving.py`) — drop-in замена `LocalHFServing`
- Вызывает vLLM через OpenAI-compatible API (`/v1/chat/completions`)
- Структурированный вывод через `guided_json` (server-side constrained decoding, аналог lm-format-enforcer но быстрее)
- Поддержка retry с temperature при ошибках парсинга
- Новые поля в `PipelineConfig`: `slot_llm_mode` ("local" | "vllm"), `slot_llm_api_url`, `slot_llm_api_key`
- `pipeline.py`: фабричный метод `_build_slot_serving()` — автоматически выбирает backend
- `run_config_api.json`: `slot_llm_mode: "vllm"` по умолчанию

**Преимущества vLLM перед transformers:**
- PagedAttention — революционный KV-cache менеджмент, 4B-модель в AWQ умещается в ~4GB VRAM
- Flash Attention 2 — встроено, автоматически
- Throughput выше в 5-10× за счёт continuous batching
- Отдельный процесс — FastAPI не ждёт inference, может масштабироваться независимо

**Отключение thinking:** `VLLMSlotServing` всегда передаёт `chat_template_kwargs.enable_thinking=false` (для Qwen3/3.5), иначе reasoning съедает бюджет токенов и JSON-ответ приходит пустым/обрезанным. Бэкенд constrained decoding — `xgrammar` (быстрее outlines); из триплет-схем убран `additionalProperties:false`, провоцировавший фолбэк в медленный outlines.

**Запуск (Linux/WSL — vLLM не работает на Windows нативно):**
```bash
# Установить vLLM (в отдельном venv WSL)
pip install vllm

# Скачать 4-bit AWQ модель слотов, напр.:
huggingface-cli download cyankiwi/Qwen3.5-9B-AWQ-4bit --local-dir models/Qwen3.5-AWQ

# Запустить vLLM сервер (WSL, Terminal 1) — модель видна по /mnt/...
make vllm

# Запустить FastAPI (Terminal 2) — ходит к vLLM по localhost:8001
make serve

# Или оба вместе (Linux/WSL)
make start
```

#### 2.3 Доработки пайплайна и визуализации ✅ DONE

**Конфигурируемые бюджеты токенов:**
- Убран хардкод `max_new_tokens` из всех slot-клиентов; вынесено в `PipelineConfig`:
  `slot_select_max_tokens`, `triplet_extract_max_tokens`, `conflict_max_tokens`,
  `deletion_max_tokens`, `memory_gate_max_tokens`.

**Флаг `relevant_slots_always_include_identity`:**
- В режиме `relevant_slots_full` принудительно добавляет слот `IDENTITY` в контекст финальной LLM, даже если memory gate его не выбрал (обход недобора у маленьких slot-моделей).

**Переписана визуализация графа (PNG + HTML):**
- Автономная реализация логики отрисовки RAGU (без зависимости от пакета RAGU): узлы по слотам, размер по степени, рёбра по TTL, легенды слотов + TTL.
- Сущности scoped по слоту (`slot\0entity`) — каждый слот образует раздельный непересекающийся кластер со своим узлом «пользователь».
- PNG: `_layout_disjoint` раскладывает каждый слот в свою ячейку сетки; HTML: pyvis forceAtlas2.

**Расширенное логирование:**
- `llm_client`: вопрос, memory_context, превью system/user промптов, ответ.
- `vllm_serving`: `finish_reason` + длина reasoning, предупреждения о пустом/обрезанном ответе.

---

### Этап 3 — продуктовый слой: Telegram-бот

Строится поверх REST API (этап 2), чтобы бот был тонким клиентом без бизнес-логики.

#### 3.1 Базовый функционал
- Каждый чат = отдельный `dialogue_id`
- Пользователь пишет сообщение → бот отвечает с учётом памяти
- Используется режим `relevant_slots_full` (лучший по метрикам)

#### 3.2 Команды
- `/start` — приветствие, объяснение что это такое
- `/graph` — показать текущий граф знаний как изображение
- `/memory` — показать список активных слотов и фактов в текстовом виде
- `/forget` — сбросить память текущего диалога
- `/stats` — количество фактов, слотов, записей в графе

#### 3.3 Визуализация графа знаний
- `pyvis` уже есть в зависимостях — генерация PNG через headless
- Альтернатива: `networkx` + `matplotlib` для более простого рендеринга
- Граф отправляется как фото в Telegram

#### 3.4 Параллельный режим в боте
- Пользователь пишет → бот немедленно начинает отвечать
- Запись в память — фоновая задача
- Индикатор "typing..." пока идёт генерация ответа

---

### Этап 4 — Data Science артефакты

#### 4.1 Jupyter notebook с анализом
- Загрузка результатов валидации
- Графики: accuracy по стратегиям, по моделям, memory hit rate
- Ablation: вклад каждого компонента (TTL, semantic dedup, conflict resolution)
- Сравнение с baseline (`full_context`, `recent_10_plus_user`)

#### 4.2 Experiment tracking
- Добавить логирование метрик в W&B или MLflow в `validation/` скриптах
- Артефакты: конфиги, метрики, примеры правильных/неправильных ответов

---

### Этап 5 — документация для конкурса

#### 5.1 README — добавить секции
- Benchmark Results: таблица с числами (ожидает графики из этапа 0)
- Конкурентный анализ: чем GigaMemory отличается от MemGPT / Zep / Mem0
- Use cases: конкретные сценарии применения с объяснением ценности

#### 5.2 Описание проекта (PDF, до 3 страниц)
Структура:
1. Проблема: LLM теряют контекст за пределами окна — особенно критично для малых моделей
2. Решение: DST-граф триплетов как долговременная персонализированная память
3. Архитектура: mermaid-диаграмма, описание ключевых компонентов
4. Результаты: таблица метрик по LongMemEval
5. Отличие от конкурентов

---

## Сценарии применения

Режим `relevant_slots_full` — лучший по метрикам, используется во всех продуктовых сценариях.

### Приоритетные кейсы

**1. Замена памяти у моделей с малым контекстом**
Малые модели (7–9B, 32K контекст) на edge/мобильных устройствах получают персонализированную долговременную память. Граф хранит факты компактно — не нужен полный raw-диалог.

**2. Параллельный real-time ассистент**
Пользователь говорит → ответ строится немедленно на recent_history_pairs → граф обновляется в фоне. Latency ≈ 0 добавочного времени. Факты из нового сообщения попадут в граф к следующему обмену.

**3. Диалоговые системы / чаты поддержки**
Агент помнит историю клиента по слотам (имя, продукт, проблема, предыдущие обращения). Конфиденциальность: хранятся структурированные факты, а не raw-диалог. Компактность: тысячи сообщений → несколько десятков триплетов.

**4. Middleware-слой для любой LLM**
GigaMemory как прослойка между пользователем и любой LLM (GPT, Claude, Llama). Персонализация без fine-tuning и без передачи полного контекста.

### Перспективные кейсы

**5. Голосовые ассистенты**
Окно контекста ещё меньше чем у текстовых моделей, персонализация критична. Граф хранит долгосрочные предпочтения пользователя.

**6. EdTech / тьютор**
Хранит прогресс студента, пробелы в знаниях, предпочтения по стилю объяснения. Адаптирует ответы к истории взаимодействия.

**7. Корпоративный knowledge bot**
Команда общается с ботом о проекте → граф накапливает факты о решениях, архитектуре, договорённостях. Новый участник задаёт вопросы — бот отвечает с памятью о всей истории проекта.

---

## Конкурентный анализ (заготовка)

| Система | Подход к памяти | Отличие от GigaMemory |
|---|---|---|
| MemGPT / OpenMemory | Иерархический контекст + vector store | Нет структурированного графа, нет детерминированного удаления |
| Zep | Knowledge graph + NER | Проприетарный, нет кастомных режимов удаления |
| Mem0 | Vector + graph hybrid | Нет триплетной структуры, нет LongMemEval валидации |
| Langchain Memory | Buffer / Summary / Vector | Нет граф-структуры, нет conflict resolution |
| **GigaMemory** | DST-граф триплетов + RAGU | Структурированные факты, три режима удаления, проверено на бенчмарке |

---

## Порядок реализации

```
[Сейчас]
  ↓
Этап 0 — добавить графики экспериментов (ожидаем файлы) IN PROGRESS 
  ↓
Этап 1 — real-time режим + параллельный путь DONE
  ↓
Этап 2 — FastAPI DONE
  ↓
Этап 3 — Telegram-бот DONE
  ↓
Этап 4 — DS артефакты (notebook + tracking) TODO
  ↓
Этап 5 — документация для конкурса (README + PDF) TODO
```