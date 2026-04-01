"""
LLM-as-a-judge template: compare expected_operations vs model.operations.

Сейчас — заглушка без вызова LLM. Позже: сформировать промпт из пары (expected, actual)
и отправить в судью-модель, вернуть score + rationale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def judge_example(
    example_id: str,
    slot_name: str,
    user_message: str,
    expected_operations: List[Dict[str, Any]],
    model_operations: List[Dict[str, Any]],
    *,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Заглушка: не вызывает LLM.

    TODO:
      - Построить текст промпта: контекст слота, сообщение, эталон, ответ модели.
      - Вызвать API или локальную LLM (temperature 0 для стабильности).
      - Вернуть, например: {"match": 0.85, "verdict": "...", "issues": [...]}
    """
    _ = (judge_model, user_message, slot_name)
    return {
        "example_id": example_id,
        "status": "stub",
        "message": "LLM judge not implemented; plug in your API or LocalHFServing here.",
        "quick_checks": {
            "same_len": len(expected_operations) == len(model_operations),
            "expected_count": len(expected_operations),
            "model_count": len(model_operations),
        },
        "expected_operations": expected_operations,
        "model_operations": model_operations,
    }


def judge_report(report_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Прогон заглушки по всем строкам отчёта run_dataset."""
    summaries: List[Dict[str, Any]] = []
    for row in report_rows:
        if row.get("model", {}).get("error"):
            summaries.append(
                {
                    "id": row["id"],
                    "judge": {"status": "error", "skipped": True},
                }
            )
            continue
        j = judge_example(
            example_id=row["id"],
            slot_name=row.get("slot_name", ""),
            user_message=row["dataset"]["user_message"],
            expected_operations=row["dataset"]["expected_operations"],
            model_operations=row["model"]["operations"],
        )
        summaries.append({"id": row["id"], "judge": j})
    return {
        "status": "stub",
        "total": len(summaries),
        "per_example": summaries,
    }
