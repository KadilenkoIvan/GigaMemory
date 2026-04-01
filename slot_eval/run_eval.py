#!/usr/bin/env python3
"""
CLI: run slot-update model on a slot eval JSON dataset, write comparison JSON.

Example:
  python run_eval.py --dataset ../dataset_generation/slot_eval_dataset-GPT_DLC.json \\
    --model Qwen/Qwen3.5-0.8B --output report.json

  python run_eval.py --dataset ... --output report.json --judge-output judge_stub.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python run_eval.py` from slot_eval/ without installing the package
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from slot_eval.judge import judge_report
from slot_eval.runner import run_dataset, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Slot-update eval: dataset vs model output")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to slot eval JSON (array of examples with expected_operations)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path (same as DST_memory slot model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="slot_eval_report.json",
        help="Output JSON path",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N examples")
    parser.add_argument(
        "--judge-output",
        type=str,
        default=None,
        help="Optional path to write stub judge summary JSON",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    rows = run_dataset(
        args.dataset,
        args.model,
        max_retries=args.max_retries,
        limit=args.limit,
    )
    write_report(rows, args.output)
    logging.info("Wrote %s (%d rows)", args.output, len(rows))

    if args.judge_output:
        judged = judge_report(rows)
        Path(args.judge_output).write_text(
            json.dumps(judged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logging.info("Wrote judge stub %s", args.judge_output)


if __name__ == "__main__":
    main()
