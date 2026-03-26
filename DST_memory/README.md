# DST_memory

Независимый модуль долгосрочной памяти в стиле DST для LLM, выделенный из текущего состояния репозитория и приведенный к рабочему, чистому виду.

## Что реализовано сейчас

- Отдельный Python-модуль `dst_memory` (без Docker/окружений/соревновательного адаптера).
- Подключен классификатор значимости сообщений из:
  - `message_important_learning/best_model-full_tune`
- Реализован in-memory векторный стор как отдельный модуль:
  - `dst_memory/vector_store.py`
- Реализован оркестратор пайплайна:
  - `dst_memory/pipeline.py`
- Реализован запуск:
  - отдельных модулей (`module ...`)
  - полного пайплайна (`pipeline ...`)
- Финальная LLM сейчас в режиме `stub` (шаблонный ответ).
- Есть режим работы без финальной LLM:
  - `--no-final-llm` возвращает структуру памяти и retrieved факты.

## Структура

- `dst_memory/config.py` — конфиг пайплайна и порог классификатора.
- `dst_memory/classifier.py` — бинарный классификатор значимости.
- `dst_memory/dst_manager.py` — менеджер слотов (upsert + заглушка delete policy).
- `dst_memory/embedder.py` — эмбеддинги.
- `dst_memory/vector_store.py` — in-memory vector DB.
- `dst_memory/retriever.py` — retrieval по векторному стору.
- `dst_memory/llm_client.py` — клиент финальной LLM (`stub`/`TODO local`/`TODO api`).
- `dst_memory/pipeline.py` — полный пайплайн.
- `dst_memory/io_utils.py` — чтение jsonl и извлечение user-сообщений.
- `run.py` — единая CLI-точка запуска.

## Быстрый старт

Установка:

```bash
pip install -r DST_memory/requirements.txt
```

### 1) Запуск отдельных модулей

Классификатор:

```bash
python DST_memory/run.py module classifier --text "Я живу в Москве и люблю футбол"
```

DST (заглушка выделения слота):

```bash
python DST_memory/run.py module dst --dialogue-id d1 --text "У меня есть кот Барсик"
```

Векторный стор:

```bash
python DST_memory/run.py module vector --dialogue-id d1 --query "Как зовут кота?" --memory-lines "питомцы: кот Барсик" "город: Москва"
```

### 2) Запуск полного пайплайна

По `jsonl`:

```bash
python DST_memory/run.py --no-final-llm pipeline jsonl --dataset-path data/format_example.jsonl --output-path DST_memory/output.json
```

Интерактивно:

```bash
python DST_memory/run.py --no-final-llm pipeline interactive --dialogue-id demo
```

Команды в интерактивном режиме:
- обычный текст: записать user-сообщение в память;
- `/ask <вопрос>`: получить ответ;
- `/clear`: очистить память;
- `/exit`: выйти.

## Важные ограничения текущей версии

- DST extraction сейчас заглушка (`facts: <текст>`), без настоящего slot-matching.
- Логика удаления/обновления противоречивых фактов — заглушка (`TODO`).
- Финальная генерация ответа:
  - `stub` работает,
  - `local` и `api` пока `NotImplementedError` с явным `TODO`.
- Векторный стор в оперативной памяти, без персистентности.

## TODO (следующие шаги)

- Реализовать LLM-based выделение слотов и update/delete политику.
- Добавить темпоральные правила актуальности фактов по типам слотов.
- Реализовать backend финальной LLM:
  - `local` (например vLLM/transformers),
  - `api` (HTTP client + retries + timeouts).
- Добавить персистентное векторное хранилище и миграцию с in-memory.
- Добавить unit-тесты на DST state transitions и retrieval.
