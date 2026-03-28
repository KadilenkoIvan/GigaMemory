# DST_memory

Независимый модуль долгосрочной памяти в стиле DST для LLM, выделенный из текущего состояния репозитория и приведенный к рабочему, чистому виду.

## Что реализовано сейчас

- Отдельный Python-модуль `dst_memory` (без Docker/окружений/соревновательного адаптера).
- Подключен классификатор значимости сообщений из:
  - `message_important_learning/best_model-full_tune`
- Добавлен отдельный клиент для модели слотов (`Meno-Lite`-style):
  - получает список существующих слотов + системный промпт + user-сообщение
  - модель возвращает JSON с массивом строк `slot_assignments` (только имена широких категорий)
  - имена приводятся к lower case, нормализуются (pymorphy2; при необходимости pyspellchecker ru)
  - новый слот создаётся, если такого имени ещё нет в состоянии; иначе запись добавляется в существующий слот
  - до 5 имён на сообщение (с требованием минимизировать число слотов)
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
- Компактный output хранит память в формате:
  - `memory_slots: [ { "slot": "<имя>", "messages": [ ... ] }, ... ]`
  - в каждом слоте — массив сохранённых пользовательских сообщений (полный текст реплики).
- Векторная индексация выполняется по названию слота, а текст сообщения хранится в метаданных.

## Структура

- `dst_memory/config.py` — конфиг пайплайна и порог классификатора.
- `dst_memory/classifier.py` — бинарный классификатор значимости.
- `dst_memory/dst_manager.py` — менеджер слотов (upsert + заглушка delete policy).
- `dst_memory/slot_client.py` — клиент модели принятия решения по слотам.
- `dst_memory/slot_name_normalize.py` — нормализация имён слотов (pymorphy2 / pyspellchecker).
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

DST (решение по слотам). Для запуска без модели слотов включи заглушку:

```bash
python DST_memory/run.py --slot-use-stub module dst --dialogue-id d1 --text "У меня есть кот Барсик"
```

Векторный стор:

```bash
python DST_memory/run.py module vector --dialogue-id d1 --query "Как зовут кота?" --memory-lines "питомцы: кот Барсик" "город: Москва"
```

### 2) Запуск полного пайплайна

По `jsonl`:

```bash
python DST_memory/run.py --slot-use-stub --no-final-llm pipeline jsonl --dataset-path data/format_example.jsonl --output-path DST_memory/output.json
```

Интерактивно:

```bash
python DST_memory/run.py --slot-use-stub --no-final-llm pipeline interactive --dialogue-id demo
```

Команды в интерактивном режиме:
- обычный текст: записать user-сообщение в память;
- `/ask <вопрос>`: получить ответ;
- `/clear`: очистить память;
- `/exit`: выйти.

### 3) Флаги для модели слотов

- `--slot-use-stub` — использовать заглушку вместо модели слотов.
- `--slot-model-path` — путь до модели слотов (по умолчанию `models/Meno-Lite-0.1`).
- `--slot-max-slots-per-message` — максимум имён слотов на сообщение (по умолчанию `5`).

### 4) Подробный запуск с реальной slot-моделью

Ниже пример для режима **без заглушки**, когда решения по слотам делает сама модель.

#### 4.1 Где должны лежать веса

Рекомендуемый вариант:

- положить модель в `DST_memory/models/Meno-Lite-0.1`
- запускать с `--slot-model-path DST_memory/models/Meno-Lite-0.1`

Пример структуры (минимум):

```text
GigaMemory/
  DST_memory/
    models/
      Meno-Lite-0.1/
        config.json
        tokenizer.json (или tokenizer.model + tokenizer_config.json)
        model.safetensors (или sharded safetensors)
        special_tokens_map.json
```

Важно:
- путь `--slot-model-path` должен указывать **на директорию модели**, а не на отдельный файл;
- формат должен быть HF-совместимым для `AutoTokenizer` + `AutoModelForCausalLM`;
- если путь неверный, пайплайн упадет с явной ошибкой.

#### 4.2 Запуск jsonl-прогона со slot-моделью

```bash
python DST_memory/run.py \
  --slot-model-path DST_memory/models/Meno-Lite-0.1 \
  --slot-max-slots-per-message 5 \
  --no-final-llm \
  pipeline jsonl \
  --dataset-path dataset_generation/GigaMemory_data/format_example_short.jsonl \
  --output-path DST_memory/test_output.json
```

Что получишь:
- `DST_memory/test_output.json` — компактная память по слотам;
- `DST_memory/test_output_logs.json` — подробные логи решений.

#### 4.3 Запуск интерактивного режима со slot-моделью

```bash
python DST_memory/run.py \
  --slot-model-path DST_memory/models/Meno-Lite-0.1 \
  --no-final-llm \
  pipeline interactive \
  --dialogue-id demo
```

Команды:
- обычный ввод: новое user-сообщение (пройдет классификатор + slot-модель);
- `/ask <вопрос>`: retrieval + вывод памяти;
- `/clear`: очистка состояния диалога;
- `/exit`: выход.

## Важные ограничения текущей версии

- Модель слотов ожидается как локальная HF-совместимая CausalLM; API-режим для нее пока не добавлен.
- Если модель слотов вернула невалидный JSON после 1 ретрая — сообщение пропускается (fallback отсутствует).
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
