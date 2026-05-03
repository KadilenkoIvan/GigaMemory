# Baseline Validation for LongMemEval

This directory contains baseline validation scripts for comparing against GigaMemory DST pipeline.

## Baseline Strategies

### 1. `full_context`
Pass **ALL** user and assistant messages from all sessions to the final LLM.

**Context format:**
```
User: Message 1
Assistant: Response 1
User: Message 2
Assistant: Response 2
...
User: Message N
Assistant: Response N

Question: {question}
```

### 2. `recent_10_plus_user`
Pass:
- Last **10 complete user/assistant pairs** (most recent)
- PLUS **all remaining user messages** from earlier sessions

**Context format:**
```
User: [Early message 1]
User: [Early message 2]
...
User: [Early message M]
User: [Recent message M+1]
Assistant: [Recent response M+1]
...
User: [Recent message M+10]
Assistant: [Recent response M+10]

Question: {question}
```

This strategy tests whether the "lost in the middle" effect affects retrieval of older information when recent context is present.

## Structure

```
baseline/
├── validate_baseline.py      # Main validation script
├── run_config.json          # Configuration file
└── README.md                # This file
```

## Usage

### Basic Usage (Full Context)

```bash
cd validation/baseline
python validate_baseline.py
```

### Recent 10 + User Strategy

```bash
python validate_baseline.py --strategy recent_10_plus_user --output-dir ./results_recent10
```

### With Custom Config

```bash
python validate_baseline.py --config ./my_config.json --output-dir ./results
```

## Configuration

Edit `run_config.json`:

```json
{
  "shared": {
    "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
    "output_dir": "./results",
    "start_index": 0,
    "num_items": 311,
    "log_level": "INFO"
  },
  "baseline": {
    "strategy": "full_context",
    "final_llm_batch_size": 10,
    "judge_batch_size": 20
  },
  "final_llm": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free",
    "api_key": "${OPENROUTER_API_KEY}"
  },
  "judge": {
    "mode": "openrouter",
    "model": "openai/gpt-oss-120b:free"
  }
}
```

## Comparison with GigaMemory

For fair comparison:

1. **Use same final LLM** (copy settings from GigaMemory config)
2. **Use same judge** (copy settings from GigaMemory config)
3. **Use same dataset range** (same `start_index` and `num_items`)
4. **Use same batch sizes** (for performance consistency)

### Example Comparison Run

```bash
# 1. Run GigaMemory validation
cd validation/GigaMemory_full
python validate_longmemeval.py --config ./run_config.json

# 2. Run Baseline - Full Context
cd validation/baseline
python validate_baseline.py \
    --strategy full_context \
    --output-dir ./results_full_context

# 3. Run Baseline - Recent 10 + User
python validate_baseline.py \
    --strategy recent_10_plus_user \
    --output-dir ./results_recent10

# 4. Compare results
python compare_results.py \
    ../GigaMemory_full/results/validation_results.json \
    ./results_full_context/validation_results.json \
    ./results_recent10/validation_results.json
```

## Output Format

Output matches GigaMemory format:

```json
{
  "metadata": {
    "strategy": "full_context",
    "dataset_path": "...",
    "start_index": 0,
    "num_items": 50,
    "final_llm_mode": "openrouter",
    "final_llm_model": "openai/gpt-oss-120b:free",
    "judge_mode": "openrouter",
    "judge_model": "openai/gpt-oss-120b:free",
    "timestamp": "2025-01-01 12:00:00"
  },
  "statistics": {
    "total": 50,
    "correct": 42,
    "incorrect": 8
  },
  "results": [
    {
      "global_index": 0,
      "question_id": "abc123",
      "question": "What is my dog's name?",
      "reference_answer": "Max",
      "predicted_answer": "Your dog's name is Max",
      "question_type": "single-session-user",
      "correct": true,
      "judge_evaluation": {
        "correct": true,
        "reasoning": "Correctly identifies the dog's name"
      }
    }
  ]
}
```

## Metrics

- **Accuracy** = correct / total (primary metric for comparison)

Unlike GigaMemory, baseline does **not** calculate:
- Memory Hit Rate (no "memory" in baseline - just raw context)
- Deleted facts tracking
- Slot-based metrics

## Expected Results

Based on "Lost in the Middle" research:

| Strategy | Expected Accuracy | Notes |
|----------|-------------------|-------|
| `full_context` | ~40-60% | Degrades with longer context |
| `recent_10_plus_user` | ~50-70% | Better if answer in recent context |
| GigaMemory | ~70-85% | Structured memory should outperform |

## Troubleshooting

### Out of Memory

The `full_context` strategy with 50+ sessions may exceed context window:

```bash
# Reduce batch size
--val-batch-final-llm-batch-size 1

# Or use a model with larger context window
--gm-llm-model "anthropic/claude-3-opus-200k"
```

### Slow Processing

```bash
# Increase batch sizes
--val-batch-final-llm-batch-size 20
--val-batch-judge-batch-size 40
```

## Differences from GigaMemory

| Feature | GigaMemory | Baseline |
|---------|-----------|----------|
| Memory mechanism | Structured slots + RAGU | Raw context |
| Context length | Controlled (slots) | Full (can be very long) |
| TTL / Deletion | Yes | No |
| Memory Hit Rate | Yes | No |
| State saving | Per-dialogue | None |
| Processing time | Slower (extraction) | Faster (no extraction) |
| Cost | Higher (more LLM calls) | Lower (just final LLM + judge) |
