@echo off
REM LongMemEval Validation Example Script for Windows
REM This is an example batch script for running validation

REM Set your OpenRouter API key (or set as environment variable)
REM set OPENROUTER_API_KEY=your-key-here

REM Create output directory
if not exist "results" mkdir results

echo Running LongMemEval Validation...
echo =======================================

REM Example 1: Quick smoke test (no LLM, no judge)
echo [1] Running smoke test (2 items, no LLM)...
python validate_longmemeval.py ^
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json ^
    --output-dir ./results/smoke_test ^
    --start-index 0 ^
    --num-items 2 ^
    --no-final-llm ^
    --judge-mode none ^
    --config ../../DST_memory/run_config.json ^
    --log-level DEBUG

if %ERRORLEVEL% neq 0 (
    echo Smoke test failed!
    exit /b 1
)

echo Smoke test completed successfully!
echo.

REM Example 2: Small batch with OpenRouter judge
echo [2] Running small batch with OpenRouter judge (5 items)...
python validate_longmemeval.py ^
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json ^
    --output-dir ./results/batch_5 ^
    --start-index 0 ^
    --num-items 5 ^
    --judge-mode openrouter ^
    --judge-model "openai/gpt-oss-120b:free" ^
    --config ../../DST_memory/run_config.json ^
    --log-level INFO

if %ERRORLEVEL% neq 0 (
    echo Batch validation failed!
    exit /b 1
)

echo Batch validation completed!
echo.

REM Example 3: Process items 10-20
echo [3] Processing items 10-20...
python validate_longmemeval.py ^
    --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json ^
    --output-dir ./results/items_10_20 ^
    --start-index 10 ^
    --num-items 10 ^
    --judge-mode openrouter ^
    --config ../../DST_memory/run_config.json

echo.
echo =======================================
echo All validation runs completed!
echo Check results/ directory for outputs.
pause
