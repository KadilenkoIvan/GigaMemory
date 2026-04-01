# slot_eval

Отдельный каталог для проверки **slot-update** модели на JSON-датасетах (формат как в `dataset_generation/slot_eval_dataset*.json`).

## Логика

Инференс slot-update реализован **отдельно** в `slot_eval/pipeline_slot_update.py` (класс `PipelineSlotUpdate`) — та же последовательность, что и `SlotUpdateClient` в DST_memory, без изменений в основном репозитории.

Из DST_memory подтягиваются только **загрузка модели** (`LocalHFServing`) и **промпт** (`build_update_messages`).

## Зависимости

Код подключает **DST_memory** из соседней папки: ожидается структура:

```
GigaMemory/
  DST_memory/
    dst_memory/
  slot_eval/
    run_eval.py
    ...
```

Установите зависимости (см. `DST_memory/requirements.txt` для полного набора):

```bash
pip install -r requirements.txt
pip install -r ../DST_memory/requirements.txt
```

## Запуск

Из каталога `slot_eval`:

```bash
python run_eval.py --dataset ../dataset_generation/slot_eval_dataset-GPT_DLC.json --model Qwen/Qwen3.5-0.8B --output report.json
```

Опции:

- `--limit N` — только первые N примеров
- `--judge-output judge.json` — дополнительно записать результат **заглушки** LLM-as-judge (см. `slot_eval/judge.py`)

## Формат выходного JSON

Массив объектов:

```json
{
  "id": "s071a",
  "slot_name": "семья",
  "dataset": {
    "existing_records": [{"id": 1, "value": "..."}],
    "user_message": "...",
    "expected_operations": [{"op": "add", "value": "..."}]
  },
  "model": {
    "operations": [{"op": "add", "value": "...", "id": null}],
    "eval_meta": {
      "primary_raw": "<сырой текст ответа модели>",
      "effective_raw": "...",
      "used_json_fix": false,
      "used_fallback": false
    },
    "error": null
  }
}
```

При ошибке инференса `model.error` и `operations` могут быть пустыми.

## LLM-as-a-judge

Шаблон в `slot_eval/judge.py`: функция `judge_example` и `judge_report` сейчас возвращают заглушку без вызова LLM. Подключите свой API или локальную модель в `judge_example` по пометкам `TODO`.

## Отдельный git-репозиторий

Чтобы сделать из этого отдельный репозиторий:

```bash
cd slot_eval
git init
git add .
git commit -m "Initial slot eval harness"
```

При переносе в другое место укажите путь к **DST_memory** или установите пакет `DST_memory` в окружение и поправьте `slot_eval/paths.py`.
