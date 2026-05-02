# LongMemEval Validation for GigaMemory

This directory contains tools for validating the GigaMemory DST (Dialogue State Tracking) pipeline using the LongMemEval benchmark dataset.

## Overview

The validation script (`validate_longmemeval.py`) tests the full memory pipeline by:

1. Loading the LongMemEval dataset
2. Filtering relevant question types (facts about the user)
3. Processing each example through the full DST pipeline:
   - Extracting and classifying user messages
   - Writing important facts to memory slots
   - Generating answers using retrieved memory context
4. Evaluating answer correctness using LLM-as-judge
5. Saving detailed results and memory states

## Dataset

**LongMemEval** (`xiaowu0162/longmemeval-cleaned`) is a benchmark for evaluating long-term memory in conversational AI systems.

### Relevant Question Types

The validation focuses on question types that test user fact memory:

| Type | Description | Count |
|------|-------------|-------|
| `single-session-user` | Facts mentioned by user in one session | 70 |
| `single-session-preference` | Preferences requiring personalization | 30 |
| `multi-session` | Facts scattered across multiple sessions | 133 |
| `knowledge-update` | Updated facts (user changed information) | 78 |

**Total relevant examples:** 311 out of 500

## Quick Start

### Prerequisites

1. Ensure you have the LongMemEval dataset:
   ```bash
   # Download from HuggingFace
   wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
   ```

2. Configure your DST_memory pipeline (see CONFIG.md)

3. Set up API keys (for OpenRouter mode):
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

### Basic Usage

#### Process first 10 items with OpenRouter judge:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results \
    --start-index 0 \
    --num-items 10 \
    --judge-mode openrouter \
    --config ../../DST_memory/run_config.json
```

#### Process items 20-30 with local judge:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_local \
    --start-index 20 \
    --num-items 10 \
    --judge-mode local \
    --judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct" \
    --config ../../DST_memory/run_config.json
```

#### Test memory without final LLM generation:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_no_llm \
    --start-index 0 \
    --num-items 5 \
    --no-final-llm \
    --judge-mode none \
    --config ../../DST_memory/run_config.json
```

## Command Line Arguments

### Dataset and Output

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to LongMemEval JSON | Required |
| `--output-dir` | Output directory for results | Required |
| `--start-index` | Start index in filtered dataset | 0 |
| `--num-items` | Number of items to process | 10 |

### Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to run_config.json | `../../DST_memory/run_config.json` |
| `--ragu-storage-path` | Custom RAGU storage path | "" |

### Memory State Saving

| Argument | Description |
|----------|-------------|
| `--save-memory-state` | Save memory state after each chunk (default) |
| `--no-save-memory-state` | Disable memory state saving |
| `--save-intermediate` | Save intermediate results (default) |

### Judge Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--judge-mode` | Judge type: `openrouter`, `local`, `none` | `openrouter` |
| `--judge-model` | Model for OpenRouter judge | `openai/gpt-oss-120b:free` |
| `--judge-api-url` | API URL for judge | OpenRouter endpoint |
| `--judge-api-key` | API key (or use env var) | "" |
| `--judge-temperature` | Judge LLM temperature | 0.0 |
| `--judge-max-tokens` | Judge max tokens | 1024 |
| `--judge-local-model-path` | Path to local judge model | "" |
| `--unload-judge-between-items` | Unload judge between items | False |

### Pipeline Options

| Argument | Description |
|----------|-------------|
| `--no-final-llm` | Skip final LLM generation |

### Logging

| Argument | Description | Default |
|----------|-------------|---------|
| `--log-level` | Logging level | `INFO` |
| `--log-file` | Save log to file (default) |
| `--no-log-file` | Disable file logging |

## Output Structure

```
output-dir/
├── validation.log              # Full execution log
├── validation_results.json     # Aggregated results with statistics
├── result_0000.json            # Individual result per item (if --save-intermediate)
├── result_0001.json
├── ...
├── chunk_0000/                 # Saved memory state per item
│   ├── dst_state.json         # DST state (slots, deleted facts)
│   └── ragu_storage/          # RAGU knowledge graph storage
├── chunk_0001/
│   ├── dst_state.json
│   └── ragu_storage/
└── ...
```

## Result Format

Each result file contains:

```json
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "reference_answer": "Business Administration",
  "predicted_answer": "Business Administration",
  "num_sessions": 53,
  "num_user_messages": 127,
  "write_logs_summary": {
    "total_messages": 127,
    "saved_messages": 23
  },
  "memory_state": {
    "slots": [...],
    "expired_facts": [...],
    "deleted_facts_with_reasons": [
      {
        "slot": "EDUCATION",
        "record_id": 5,
        "subject": "пользователь",
        "relation": "изучал",
        "object": "инженерия",
        "deletion_reason": "conflict_resolution",
        "deletion_source": "conflict_resolver",
        ...
      }
    ]
  },
  "answer_details": {
    "use_memory": true,
    "memory_strategy": "full_graph_json",
    "retrieved": [...]
  },
  "judge_evaluation": {
    "correct": true,
    "reasoning": "The predicted answer matches the reference exactly."
  },
  "correct": true,
  "global_index": 0,
  "slice_index": 0
}
```

## Deleted Facts Tracking

The validation script captures all deletions with reasons:

| Deletion Reason | Description | Source |
|----------------|-------------|--------|
| `ttl_expired` | Fact exceeded its time-to-live | TTL checker |
| `deletion_signal` | User explicitly removed fact | LLM (inline/separate) or heuristic |
| `conflict_resolution` | Fact replaced by newer conflicting fact | Conflict resolver |
| `semantic_dedup` | Near-duplicate fact removed | Semantic dedup engine |
| `manual` | Manual deactivation | API call |

## Evaluation Metrics

The script reports:

- **Total processed**: Number of items attempted
- **Correct**: Judge marked answer as correct
- **Incorrect**: Judge marked answer as incorrect
- **Accuracy**: Percentage of correct answers

## Troubleshooting

### Memory Issues

If running out of memory:

1. Use `--unload-judge-between-items` for local judge mode
2. Process in smaller batches with `--num-items`
3. Use `--no-final-llm` to skip generation during memory testing

### API Rate Limits

For OpenRouter judge mode:

1. Use smaller `--num-items` batches
2. Add delays between requests (modify script if needed)
3. Consider switching to local judge mode

### Import Errors

Ensure paths are correct:

```python
# The script adds these to sys.path automatically
repo_root = Path(__file__).resolve().parents[2]
dst_memory_path = repo_root / "DST_memory"
ragu_path = repo_root / "RAGU"
```

## Smoke Test

Run a quick smoke test without LLM calls:

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./smoke_test \
    --start-index 0 \
    --num-items 2 \
    --no-final-llm \
    --judge-mode none \
    --config ../../DST_memory/run_config.json \
    --log-level DEBUG
```

## Integration with DST_memory

The validation script reuses DST_memory components:

- `PipelineConfig` - Configuration dataclass
- `DSTMemoryPipeline` - Main pipeline orchestrator
- `Message` - User message model
- `build_ragu_processor` - RAGU storage initialization

See `../../DST_memory/run.py` for the reference CLI implementation.
