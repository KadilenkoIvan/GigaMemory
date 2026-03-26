"""
Extract user messages from a JSONL dialogues dataset and save to CSV.

Each line in the input JSONL is expected to follow the structure shown in
`dataset_generation/GigaMemory_data/format_example.jsonl`, containing a top-level
`sessions` array with `messages`, where each message has `role` and `content`.

The output CSV contains two columns:
- message: text of the user message
- label: constant value -1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def extract_user_messages(record: dict) -> Iterable[str]:
    """Yield user message contents from a single JSON record."""
    for session in record.get("sessions", []):
        for message in session.get("messages", []):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    yield content


def parse_jsonl(path: Path) -> Iterable[str]:
    """Iterate through all user messages in a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}"
                ) from exc
            yield from extract_user_messages(record)


def write_csv(messages: Iterable[str], output_path: Path) -> None:
    """Write messages with label -1 to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["message", "label"])
        for message in messages:
            writer.writerow([message, -1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert user messages from JSONL dialogues to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("GigaMemory_data/format_example.jsonl"),
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GigaMemory_data/user_messages.csv"),
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    messages = parse_jsonl(args.input)
    write_csv(messages, args.output)


if __name__ == "__main__":
    main()
