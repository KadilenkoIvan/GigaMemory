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
    "model": "openai/gpt-oss-120b:free"
  },
  "judge": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free"
  }
}
```

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
    "by_type": {
      "single-session-user": {"count": 10, "average_score": 0.82, "errors": 0},
      "single-session-preference": {"count": 10, "average_score": 0.78, "errors": 0},
      "multi-session": {"count": 10, "average_score": 0.65, "errors": 0},
      "knowledge-update": {"count": 10, "average_score": 0.75, "errors": 0}
    }
  },
  "timing": {
    "total_time": 120.5,
    "total_items": 40,
    "total_messages": 2400,
    "time_per_item": {"min": 1.2, "max": 5.8, "p50": 2.8, "p95": 4.5, "p99": 5.2},
    "time_per_message": {"min": 0.02, "max": 0.15, "p50": 0.05, "p95": 0.08, "p99": 0.10}
  },
  "results": [
    {
      "global_index": 0,
      "question_id": "...",
      "question": "...",
      "reference_answer": "...",
      "predicted_answer": "...",
      "question_type": "single-session-user",
      "score": 1.0,
      "reasoning": "Perfect match",
      "final_llm_error": null,
      "judge_error": null
    }
  ]
}
```

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

## Comparison with GigaMemory

| Feature | GigaMemory | Baseline |
|---------|-----------|----------|
| Memory | Structured slots | Raw context |
| Scoring | 0-1 scale | 0-1 scale |
| Retry | Yes (3 attempts) | Yes (3 attempts) |
| Timing | Full metrics | Full metrics |
| Per-type metrics | Yes | Yes |
