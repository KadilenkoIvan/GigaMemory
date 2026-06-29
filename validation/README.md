# Валидация GigaMemory на LongMemEval

Этот раздел описывает, **как** тестировалась система GigaMemory, **на чём** и **какие результаты** получены. Здесь же — точка входа в подробную документацию по скриптам валидации.

- [`baseline/`](baseline/README.md) — baseline без памяти (full-context): диалог целиком кладётся в промпт финальной LLM.
- [`GigaMemory_full/`](GigaMemory_full/README.md) — полный пайплайн памяти GigaMemory (извлечение фактов → DST-граф → ответ финальной LLM по контексту памяти).
- [`GigaMemory_full/CONFIG.md`](GigaMemory_full/CONFIG.md), [`GigaMemory_full/README_CONFIG.md`](GigaMemory_full/README_CONFIG.md) — формат конфигов и CLI-параметры.
- [`metrics/`](metrics) — сводные выжимки метрик (`metrics-baseline.json`, `metrics-gigamemory.json`).

---

## 1. Что и зачем сравниваем

Цель валидации — проверить гипотезу: **структурированная долговременная память даёт более точные ответы, чем подача «сырого» диалога в контекст LLM**, и при этом на порядок сокращает объём, который реально читает финальная модель.

Сравниваются два подхода с **одной и той же** финальной LLM:

| Подход | Что видит финальная LLM |
|--------|--------------------------|
| **Baseline (full-context)** | Весь диалог (все реплики) — с обрезкой по окну токенов модели |
| **GigaMemory** | Компактный контекст памяти: релевантные слоты / триплеты DST-графа |

## 2. Датасет

[**LongMemEval**](https://github.com/xiaowu0162/LongMemEval) (`longmemeval_s_cleaned.json`) — бенчмарк долговременной памяти диалоговых ассистентов. Каждый пример — длинная история сессий (haystack) + вопрос, ответ на который требует вспомнить факт из прошлого.

Сбалансированная выборка по 4 типам вопросов (по 50 на тип):

| Тип | Что проверяет |
|-----|---------------|
| `single-session-user` | факт о пользователе из одной сессии |
| `single-session-preference` | предпочтение пользователя из одной сессии |
| `multi-session` | агрегация фактов из нескольких сессий |
| `knowledge-update` | обновление факта (важно: новое значение, а не старое) |

Средняя длина исходного диалога в выборке — **≈ 489 000 символов**.

## 3. Модели

- **Извлечение фактов (GigaMemory):** `Qwen3.5-4B` (slot/triplet-экстрактор) — строит DST-граф.
- **Финальная LLM (ответ на вопрос):** одни и те же три модели для baseline и для GigaMemory —
  - `LLaMA-3-8B-Instruct`
  - `Qwen2.5-7B-Instruct`
  - `Mistral-Nemo`
- **Судья (оценка ответа):** LLM-judge по непрерывной шкале 0–1 (см. ниже).

## 4. Методика тестирования

### Стратегии памяти GigaMemory

Финальной LLM можно отдавать память тремя способами (`memory_strategy`):

| Стратегия | Что подаётся в финальную LLM |
|-----------|------------------------------|
| `full_graph_json` | весь DST-граф фактов в JSON |
| `relevant_slots_full` | только слоты, релевантные вопросу (полные) |
| `topk_graph_records` | top-K записей графа по retrieval |

Каждая стратегия проверяется в двух вариантах графа:
- `active_only` — только активные (актуальные) факты;
- `with_inactive` — включая деактивированные (история изменений).

### Поэтапный прогон

Чтобы экономить ресурсы и переиспользовать состояние памяти, пайплайн разбит на стадии (подробно — в [GigaMemory_full/README.md](GigaMemory_full/README.md)):

```
memory_only  →  final_llm_only  →  judge_only
(строим память) (ответы по стратегиям) (оценка + Memory Hit Rate)
```

Память диалогов считается **один раз**, затем на одних и тех же состояниях прогоняются разные финальные LLM и стратегии — это даёт честное сравнение.

### Оценка (LLM-judge, шкала 0–1)

| Балл | Значение |
|------|----------|
| 1.0 | идеальное совпадение |
| 0.8 | незначительная неточность (число/дата/имя) |
| 0.6 | частичный ответ |
| 0.4 | слабое покрытие |
| 0.2 | минимальное совпадение |
| 0.0 | несовпадение / противоречие / «не знаю» |

Особые правила: для `knowledge-update` старый факт вместо нового → 0.0.

### Memory Hit Rate (MHR)

Дополнительная метрика: доля случаев, когда нужный факт **присутствовал** в контексте памяти (отдельный вызов судьи). Помогает отделить проблему памяти (факт не сохранён) от проблемы финальной LLM (факт был, но не использован).

---

## 5. Результаты

> Метрика — средний балл судьи (0–1). Baseline — 200 вопросов (50×4), GigaMemory — 195 вопросов.

### 5.1. Baseline (full-context, без памяти)

| Финальная LLM | Средний балл | single-user | single-pref | multi | knowledge-upd |
|---------------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-3-8B    | 0.035 | 0.040 | 0.040 | 0.020 | 0.040 |
| Qwen2.5-7B    | 0.164 | 0.184 | 0.288 | 0.060 | 0.124 |
| Mistral-Nemo  | 0.311 | 0.360 | 0.204 | 0.276 | 0.404 |

![Метрики baseline](<../val_images/baseline метрики.png>)

### 5.2. GigaMemory по стратегиям

Средний балл по стратегиям (экстрактор — Qwen3.5-4B):

| Финальная LLM | `full_graph_json` | `relevant_slots_full` | `topk_graph_records` |
|---------------|:---:|:---:|:---:|
| LLaMA-3-8B   | 0.134 | **0.203** | 0.108 |
| Qwen2.5-7B   | 0.255 | **0.548** | 0.268 |
| Mistral-Nemo | 0.526 | **0.620** | 0.299 |

> Для каждой ячейки взят лучший из вариантов `active_only` / `with_inactive`. Победитель по строке выделен — во всех случаях это `relevant_slots_full`.

![GigaMemory по стратегиям](<../val_images/GigaMemory по стратегиям.png>)

### 5.3. Главное: GigaMemory vs Baseline

Лучшая стратегия (`relevant_slots_full`) против baseline с той же финальной LLM:

| Финальная LLM | Baseline | GigaMemory | Прирост |
|---------------|:---:|:---:|:---:|
| LLaMA-3-8B   | 0.035 | **0.203** | ×5.8 |
| Qwen2.5-7B   | 0.164 | **0.548** | ×3.3 |
| Mistral-Nemo | 0.311 | **0.620** | ×2.0 |

![Сравнение метрик с baseline](<../val_images/сравнение метрик с baseline.png>)

### 5.4. Объём контекста


![Сравнение объёма контекста](<../val_images/сравнение конеткста.png>)

### 5.5. Выводы

- Структурированная память **превосходит** подачу полного диалога для всех трёх финальных LLM.
- Чем хуже модель работает с длинным контекстом, тем больше выигрыш памяти (LLaMA — ×5.8).
- Лучшая стратегия — **`relevant_slots_full`**: подавать только релевантные вопросу слоты, а не весь граф и не top-K retrieval.
- Memory Hit Rate для `relevant_slots_full` — **0.53–0.62**: основной запас роста — в полноте извлечения фактов, а не в финальной LLM.

---

## 6. Воспроизведение

```bash
# Baseline (full-context)
cd validation/baseline
python validate_baseline.py --config ./run_config_full_context_Qwen25.json

# GigaMemory: поэтапно (память → ответы → судья)
cd validation/GigaMemory_full
python validate_longmemeval.py --validation-mode memory_only     --config ./config_full.json
python validate_longmemeval.py --validation-mode final_llm_only  --config ./config_final_llm_only_extactor-model-Qwen_final-LLM-Qwen25.json
python validate_longmemeval.py --validation-mode judge_only      --config ./config_judge_only_bundle_Qwen_final-LLM-Qwen25.json
```

Полный список режимов, CLI-параметров и форматов вывода — в [GigaMemory_full/README.md](GigaMemory_full/README.md) и [baseline/README.md](baseline/README.md).

## 7. Что лежит в репозитории, а что — нет

Сырые прогоны весят гигабайты (векторные БД RAGU, DST-снапшоты, промежуточные ответы) и поэтому их нет в репозитоии, в нём только лёгкие и значимые артефакты:

| Коммитится | Не коммитится (генерируется скриптами) |
|------------|----------------------------------------|
| скрипты `validate_*.py`, `aggregate_*` | `chunk_*/`, `ragu_storage/`, `dst_state.json` |
| документация (`README.md`, `CONFIG.md`) | `memory_only_states.json`, `intermediate_answers.json` |
| конфиги `config_*.json`, `run_config_*.json` | `*.log`, `*.html`, `*.gml`, ноутбуки `*.ipynb` |
| выжимки `metrics/metrics-*.json` | большие деревья `result_*/`, `results_memory_only_*/` |
| итоговые `validation_results.json` (baseline + judge-бандлы) | `test_data/` |
