# Baseline Validation for LongMemEval

Baseline testing with simple context-passing strategies. Includes timing metrics, retry logic, and 0-1 scoring scale.

## Features

- **Two baseline strategies:**
  - `full_context` — all user + assistant messages
  - `recent_10_plus_user` — last 10 pairs + remaining user messages

- **Timing metrics:**
  - Total processing time
  - Time per item (min, max, p50, p95, p99)
  - Time per message (min, max, p50, p95, p99)

- **Retry logic:** 3 attempts with exponential backoff for HTTP 429/500 errors

- **Judge scoring 0-1 scale:**
  - 1.0 = Perfect match
  - 0.8 = Minor inaccuracy
  - 0.6 = Partial answer
  - 0.4 = Weak coverage
  - 0.2 = Minimal match
  - 0.0 = No match

- **Per-question-type metrics:** aggregated scores by type

- **Final LLM prompt size stats:** total/average symbols before and after **tokenizer** clamp (`max_context_tokens`; символы полного chat-пrompt).

- **Размер диалога и контекста (сопоставимо с GigaMemory_full, без MHR):**
  - `dialogue_context_chars` — сумма длин текстов **всех** реплик в `haystack_sessions` строки датасета (как в `validate_longmemeval`), одинакова для всех вопросов одной строки.
  - `conversation_context_chars_*` — сумма длин полей `content` у **переданных в промпт** реплик (после стратегии `full_context` / `recent_10_plus_user`), до и после обрезки по токенам.

- **Balanced sampling:** N items per question type

## Configuration

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results",
    "num_items_per_type": 10,
    "question_types": [
      "single-session-user",
      "single-session-preference",
      "multi-session",
      "knowledge-update"
    ]
  },
  "baseline": {
    "strategy": "full_context",
    "final_llm_batch_size": 1,
    "judge_batch_size": 1
  },
  "final_llm": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free",
    "tokenizer_model": ""
  },
  "judge": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free"
  }
}
```

`final_llm.tokenizer_model` — optional HF model id/path for tokenizer used in `max_context_tokens` truncation.
Use it when API model id is provider-specific (e.g., `qwen/qwen-2.5-7b-instruct`) and not resolvable by `AutoTokenizer`.

### OpenRouter: поле `reasoning`

См. [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

На части маршрутов OpenRouter ответ **400**: *«Reasoning is mandatory for this endpoint and cannot be disabled»*, если в теле запроса передано отключение reasoning (`effort: none`, `max_tokens: 0` и т.п.). Поэтому **`validate_baseline.py` по умолчанию не добавляет `reasoning`** — ключ нужно задавать в JSON только если вы точно знаете, что модель и провайдер это поддерживают.

- Ключа **`reasoning` нет** или **`"reasoning": null`** → поле **`reasoning` не отправляется** в `/chat/completions`.
- Непустой объект `"reasoning": { ... }` → передаётся как есть (на свой риск).
- Интерleaved-блоки в `content` после ответа подчищает `_strip_provider_thinking_blocks`.

Если нужен именно endpoint без обязательного reasoning — меняют модель/роут на OpenRouter или конфигурацию у провайдера; это не решается только промптом.

## Usage

```bash
# Full context baseline
python validate_baseline.py --config ./run_config.json

# Recent 10 + user strategy
python validate_baseline.py --strategy recent_10_plus_user --output-dir ./results_recent10
```

## Output Format

```json
{
  "metadata": {...},
  "statistics": {
    "total": 40,
    "errors_final_llm": 0,
    "errors_judge": 0,
    "average_score": 0.75,
    "final_llm_calls": 40,
    "final_llm_prompt_chars_before_clamp_total": 1200000,
    "final_llm_prompt_chars_after_clamp_total": 980000,
    "final_llm_prompt_chars_before_clamp_avg": 30000.0,
    "final_llm_prompt_chars_after_clamp_avg": 24500.0,
    "dialogue_context_chars": {
      "count": 40,
      "total": 4800000,
      "avg": 120000.0
    },
    "conversation_context_chars": {
      "calls": 40,
      "before_clamp_total": 4500000,
      "after_clamp_total": 3600000,
      "before_clamp_avg": 112500.0,
      "after_clamp_avg": 90000.0
    },
    "final_llm_prompt_chars": {
      "calls": 40,
      "before_clamp_total": 1200000,
      "after_clamp_total": 980000,
      "before_clamp_avg": 30000.0,
      "after_clamp_avg": 24500.0
    },
    "by_type": {
      "single-session-user": {"count": 10, "average_score": 0.82, "errors": 0},
      "single-session-preference": {"count": 10, "average_score": 0.78, "errors": 0},
      "multi-session": {"count": 10, "average_score": 0.65, "errors": 0},
      "knowledge-update": {"count": 10, "average_score": 0.75, "errors": 0}
    }
  },
  "timing": {...},
  "results": [
    {
      "global_index": 0,
      "dialogue_context_chars": 118000,
      "conversation_context_chars_before_clamp": 115000,
      "conversation_context_chars_after_clamp": 92000,
      "final_llm_prompt_chars_before_clamp": 31240,
      "final_llm_prompt_chars_after_clamp": 24577,
      "...": "..."
    }
  ]
}
```

Файл `validation_results.json` в каталоге результатов **атомарно перезаписывается** после каждого завершённого примера (ответ финальной LLM + оценка судьи). Пока накопленный батч судьи не сброшен, новые строки в `results` не появляются (логика батчей сохранена).

## Scoring Criteria

| Score | Description | Criteria |
|-------|-------------|----------|
| 1.0 | Perfect | All key entities match, meaning identical |
| 0.8 | Minor error | All entities present, one slightly distorted |
| 0.6 | Partial | Most covered, one important entity missing |
| 0.4 | Weak | One correct entity from several needed |
| 0.2 | Minimal | Related domain, but content doesn't match |
| 0.0 | None | Incorrect, contradicts, or "don't know" |

### Special Rules

- **knowledge-update:** Old fact instead of new = 0.0
- **single-session-preference:** Correct fact used = 1.0 (regardless of phrasing)
- **multi-session:** Partial aggregation scored proportionally

## Comparison with GigaMemory_full

| Feature | GigaMemory_full | Baseline |
|---------|-----------|----------|
| Memory | Structured slots + RAGU | Raw context |
| Scoring | 0-1 scale | 0-1 scale |
| Retry | Yes (3 attempts) | Yes (3 attempts) |
| Timing | Full metrics | Full metrics |
| Per-type metrics | Yes | Yes |
| Prompt/context chars in JSON | `statistics.final_llm_prompt_chars`, `statistics.dialogue_context_chars`; nested judge export | Same-style nested blocks + `conversation_context_chars` (strategy turns before/after clamp) |
| Memory hit evaluation | Optional | No |
