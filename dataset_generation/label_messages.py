"""
Interactive labeling with incremental saves and restart support.

Поведение:
- Сообщения берутся по порядку из input CSV.
- Метка вводится 0/1, Enter = 0.
- После каждого сообщения файл output пересохраняется: можно прерывать работу
  и продолжать позже.
- Флаг --start-index позволяет начать с произвольного сообщения (0-based):
  - Если старт после последней сохранённой строки, новые метки будут добавлены.
  - Если старт внутри уже размеченных, их метки перезапишутся.
"""

import argparse
import csv
from pathlib import Path


def prompt_label(message: str) -> str:
    """Prompt until the user enters 0 or 1."""
    while True:
        print("\nСообщение:")
        print(message)
        value = input("Введите метку (1 или 0): ").strip()
        if value == "":
            return "0"
        if value in {"0", "1"}:
            return value
        print("Нужно ввести 1 или 0.")


def read_messages(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_existing_labels(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_labels(output_path: Path, rows: list[dict]) -> None:
    """Persist labeled rows (only those что уже размечены)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["message", "label"])
        writer.writeheader()
        writer.writerows(rows)


def label_messages(
    input_path: Path, output_path: Path, max_messages: int, start_index: int
) -> None:
    """
    Label messages with incremental saving.

    - start_index (0-based): позиция, с которой начинать.
    - При старте внутри уже размеченных — метки перезапишутся.
    - При старте после последних размеченных — новые метки добавятся.
    """
    messages = read_messages(input_path)
    if not messages:
        print("Нет сообщений для разметки.")
        return
    if start_index < 0 or start_index >= len(messages):
        raise ValueError(
            f"start_index {start_index} вне диапазона [0, {len(messages)-1}]"
        )

    labeled_rows = read_existing_labels(output_path)

    # Синхронизируем длину уже размеченных с доступными сообщениями
    if len(labeled_rows) > len(messages):
        labeled_rows = labeled_rows[: len(messages)]

    processed = 0
    for idx in range(start_index, len(messages)):
        if processed >= max_messages:
            break
        message_row = messages[idx]
        label = prompt_label(message_row.get("message", ""))

        if idx < len(labeled_rows):
            labeled_rows[idx]["label"] = label
        else:
            labeled_rows.append(
                {"message": message_row.get("message", ""), "label": label}
            )

        write_labels(output_path, labeled_rows)
        processed += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label messages from CSV by entering 1 or 0."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("GigaMemory_data/user_messages.csv"),
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GigaMemory_data/user_messages_labeled.csv"),
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--max_messages",
        type=int,
        default=5,
        help="How many messages to label interactively.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index to start labeling from.",
    )
    args = parser.parse_args()

    label_messages(args.input, args.output, args.max_messages, args.start_index)


if __name__ == "__main__":
    main()
