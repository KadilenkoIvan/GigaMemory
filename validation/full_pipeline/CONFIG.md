# LongMemEval Validation Configuration

This document describes the configuration options for the LongMemEval validation script.

## Configuration Files

### DST_memory Config

The validation script uses the same `run_config.json` as DST_memory. Key parameters:

```json
{
  "shared": {
    "importance_model_path": "path/to/importance/classifier",
    "importance_threshold": 0.25,
    "memory_strategy": "full_graph_json",
    "llm_mode": "openrouter",
    "llm_model": "openai/gpt-oss-120b:free",
    "slot_use_stub": false,
    "slot_model_path": "DST_memory/models/Meno-Lite-0.1",
    "ragu_embedder_model": "deepvk/USER-bge-m3",
    "triplet_deletion_mode": "none",
    "prompt_language": "ru"
  }
}
```

**Important**: The LongMemEval dataset is in English, so you may want to set `"prompt_language": "en"` for better extraction quality.

## Validation-Specific Configuration

### Command Line Overrides

These settings are specific to the validation script and passed via CLI:

#### Dataset Selection

```bash
# Process a specific range of examples
--start-index 0      # Start from first relevant example
--num-items 50       # Process 50 examples
```

The script automatically filters to relevant question types:
- `single-session-user` (70 examples)
- `single-session-preference` (30 examples)
- `multi-session` (133 examples)
- `knowledge-update` (78 examples)

#### Judge Configuration

**OpenRouter Mode** (recommended for accuracy):
```bash
--judge-mode openrouter
--judge-model "openai/gpt-oss-120b:free"
--judge-api-key "${OPENROUTER_API_KEY}"
```

**Local Mode** (for offline operation):
```bash
--judge-mode local
--judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct"
--unload-judge-between-items  # Save memory between items
```

**No Judge** (memory testing only):
```bash
--judge-mode none
--no-final-llm
```

#### Memory State Persistence

```bash
--save-memory-state        # Save DST + RAGU state after each item (default)
--no-save-memory-state     # Disable state saving
--save-intermediate        # Save individual result files (default)
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter (final LLM and judge) |
| `DST_MEMORY_CONFIG` | Path to DST_memory config file |

## Complete Configuration Examples

### Full Test with OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_full \
    --start-index 0 \
    --num-items 311 \
    --judge-mode openrouter \
    --judge-model "openai/gpt-oss-120b:free" \
    --config ../../DST_memory/run_config.json \
    --log-level INFO
```

### Memory-Only Test (No LLM)

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_memory \
    --start-index 0 \
    --num-items 50 \
    --no-final-llm \
    --judge-mode none \
    --config ../../DST_memory/run_config.json
```

### Local Testing (All Local Models)

```bash
python validate_longmemeval.py \
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
    --output-dir ./results_local \
    --start-index 0 \
    --num-items 10 \
    --judge-mode local \
    --judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct" \
    --config ../../DST_memory/run_config.json \
    --unload-judge-between-items
```

With `run_config.json`:
```json
{
  "shared": {
    "llm_mode": "local",
    "slot_use_stub": false,
    "slot_model_path": "local/path/to/slot-model",
    "prompt_language": "en"
  }
}
```

## Batch Processing Configuration

For processing the full dataset in batches:

```bash
# Batch 1: Items 0-50
python validate_longmemeval.py ... --start-index 0 --num-items 50 --output-dir ./batch_1

# Batch 2: Items 50-100
python validate_longmemeval.py ... --start-index 50 --num-items 50 --output-dir ./batch_2

# Batch 3: Items 100-150
python validate_longmemeval.py ... --start-index 100 --num-items 50 --output-dir ./batch_3

# ... and so on
```

## Performance Tuning

### Memory Optimization

For systems with limited RAM:

```bash
# Process in very small batches
--num-items 5

# Unload models between items
--unload-judge-between-items

# Disable intermediate saves to reduce I/O
--no-save-intermediate
```

### Speed Optimization

For faster validation (trading accuracy):

```bash
# Use stub mode for slot operations (if implemented)
--config stub_config.json

# Skip judge evaluation
--judge-mode none

# Reduce retrieval top-k
# (in run_config.json: "retrieval_top_k": 3)
```

## Debugging Configuration

Enable detailed logging:

```bash
--log-level DEBUG
```

This will log:
- Full prompts sent to LLMs
- Detailed extraction results
- RAGU operations
- Memory state changes

## Integration Testing

Test specific components:

### Importance Classifier Only
```bash
# Check DST_memory classifier directly
python ../../DST_memory/run.py module classifier --text "I have a dog named Max"
```

### Slot Selection Only
```bash
# Check slot selection
python ../../DST_memory/run.py module dst --dialogue-id test --text "My wife is a doctor"
```

### Full Pipeline on Single Example
```bash
# Create a single-example JSONL file
python ../../DST_memory/run.py pipeline test \
    --dataset-path single_example.jsonl \
    --output-path test_output.json
```

## Troubleshooting Configuration

### Path Issues

If imports fail, verify directory structure:
```
GigaMemory/
├── DST_memory/          # Pipeline code
├── RAGU/                # RAGU storage
├── LongMemEval/         # Dataset
└── validation/
    └── full_pipeline/   # This directory
```

### Model Loading Issues

For local models, ensure HuggingFace cache is accessible:
```bash
export HF_HOME="/path/to/huggingface/cache"
export TRANSFORMERS_CACHE="/path/to/transformers/cache"
```

### API Timeout Issues

Increase timeout in code if needed (in `validate_longmemeval.py`):
```python
# Increase from default 120 seconds
with urllib.request.urlopen(req, timeout=300) as resp:
```

## Output Configuration

The script produces:

1. **Console output** - Real-time progress
2. **validation.log** - Detailed execution log
3. **validation_results.json** - Aggregated results
4. **result_*.json** - Individual item results
5. **chunk_*/dst_state.json** - DST memory state
6. **chunk_*/ragu_storage/** - RAGU graph storage

Configure retention:
```bash
# Keep only final results (smaller output)
--no-save-intermediate
--no-save-memory-state

# Keep everything (full audit trail, larger output)
--save-intermediate
--save-memory-state
```
