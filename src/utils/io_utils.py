import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from models import Dialog


def read(file_path: str) -> List[Dialog]:
    with open(file_path, "r") as file:
        dialogs_data = [json.loads(line) for line in file]
    return [Dialog.from_dict(_) for _ in dialogs_data]


def save_result(result: dict, output_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = Path(output_dir)
    output_file_path = output_dir_path / "output.json"
    evaluate_output_file_path = output_dir_path / "submit.csv"
    timestamped_evaluate_output_file_path = (
        output_dir_path / "timestamped" / f"submit-{timestamp}.csv"
    )

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Результат сохранен в {output_file_path}")

    evals = [
        {"id": r["id"], "answer": r["pred_answer"], "answer_time": r["answer_time"]}
        for r in result["responses"]
    ]
    with open(evaluate_output_file_path, "w", encoding="utf-8") as eval:
        writer = csv.DictWriter(eval, fieldnames=["id", "answer", "answer_time"])
        writer.writeheader()
        writer.writerows(evals)

    os.makedirs(timestamped_evaluate_output_file_path.parent, exist_ok=True)
    shutil.copy(evaluate_output_file_path, timestamped_evaluate_output_file_path)

    print(
        f"Answers saved to {evaluate_output_file_path} и {timestamped_evaluate_output_file_path}"
    )
