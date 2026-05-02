"""
Smoke test for LongMemEval validation script.
This test runs without loading actual models to verify the script structure.
"""

import sys
import os
from pathlib import Path

# Add paths
repo_root = Path(__file__).resolve().parents[2]
dst_memory_path = repo_root / "DST_memory"
ragu_path = repo_root / "RAGU"
if str(dst_memory_path) not in sys.path:
    sys.path.insert(0, str(dst_memory_path))
if str(ragu_path) not in sys.path:
    sys.path.insert(0, str(ragu_path))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from dst_memory.utils.dotenv_loader import load_dst_memory_dotenv
        from dst_memory.utils.run_config_loader import load_run_config, shared_section
        from dst_memory import PipelineConfig
        from dst_memory.core.pipeline import DSTMemoryPipeline
        print("  [OK] Core imports successful")
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False
    return True


def test_dataset_loading():
    """Test loading minimal test dataset."""
    print("Testing dataset loading...")
    try:
        import json
        test_data_path = Path(__file__).parent / "test_data" / "minimal_test.json"
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [OK] Loaded {len(data)} test items")
        print(f"    - Item 0: {data[0].get('question_id')} (type: {data[0].get('question_type')})")
        print(f"    - Item 1: {data[1].get('question_id')} (type: {data[1].get('question_type')})")
        return True
    except Exception as e:
        print(f"  [FAIL] Dataset loading failed: {e}")
        return False


def test_judge_client():
    """Test JudgeClient creation."""
    print("Testing JudgeClient...")
    try:
        # Import from validation script
        from validate_longmemeval import JudgeClient

        # Create client in stub-like mode (none)
        client = JudgeClient(mode="none")
        print("  [OK] JudgeClient created (mode=none)")
        return True
    except Exception as e:
        print(f"  [FAIL] JudgeClient failed: {e}")
        return False


def test_config_loading():
    """Test loading smoke test config."""
    print("Testing config loading...")
    try:
        from dst_memory.utils.run_config_loader import load_run_config, shared_section

        config_path = Path(__file__).parent / "test_data" / "smoke_test_config.json"
        file_cfg = load_run_config(str(config_path))
        shared = shared_section(file_cfg)

        print(f"  [OK] Config loaded")
        print(f"    - llm_mode: {shared.get('llm_mode')}")
        print(f"    - memory_strategy: {shared.get('memory_strategy')}")
        print(f"    - slot_use_stub: {shared.get('slot_use_stub')}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config loading failed: {e}")
        return False


def test_pipeline_build():
    """Test building pipeline with stub config."""
    print("Testing pipeline build (stub mode)...")
    try:
        from validate_longmemeval import build_pipeline_from_config

        config_path = Path(__file__).parent / "test_data" / "smoke_test_config.json"
        pipeline = build_pipeline_from_config(str(config_path))

        print("  [OK] Pipeline built successfully")
        print(f"    - Memory strategy: {pipeline.config.memory_strategy}")
        print(f"    - LLM mode: {pipeline.config.llm_mode}")
        print(f"    - Slot stub: {pipeline.config.slot_use_stub}")
        return True
    except Exception as e:
        print(f"  [FAIL] Pipeline build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("LongMemEval Validation Smoke Test")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Dataset Loading", test_dataset_loading),
        ("Judge Client", test_judge_client),
        ("Config Loading", test_config_loading),
        ("Pipeline Build", test_pipeline_build),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        print("-" * 40)
        success = test_func()
        results.append((name, success))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll smoke tests passed!")
        return 0
    else:
        print("\nSome tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
