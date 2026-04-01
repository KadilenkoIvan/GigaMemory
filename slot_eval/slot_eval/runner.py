"""Run slot-update model on each dataset example; build JSON report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import load_examples
from .paths import ensure_dst_memory_on_path
from .pipeline_slot_update import PipelineSlotUpdate, trace_to_dict

logger = logging.getLogger(__name__)


def _ops_to_jsonable(ops: List[SlotOperation]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in ops:
        d: Dict[str, Any] = {"op": o.op}
        if o.value is not None:
            d["value"] = o.value
        if o.record_id is not None:
            d["id"] = o.record_id
        out.append(d)
    return out


def run_dataset(
    dataset_path: Path | str,
    model_path: str,
    max_retries: int = 1,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ensure_dst_memory_on_path()
    from dst_memory.serving import LocalHFServing

    examples = load_examples(dataset_path)
    if limit is not None:
        examples = examples[:limit]

    serving = LocalHFServing(model_path)
    pipeline = PipelineSlotUpdate(serving=serving, max_retries=max_retries)

    results: List[Dict[str, Any]] = []
    for ex in examples:
        try:
            ops = pipeline.plan_operations(
                ex.slot_name,
                ex.existing_records,
                ex.user_message,
            )
            model_ops = _ops_to_jsonable(ops)
            err: Optional[str] = None
        except Exception as e:
            logger.exception("Example %s failed", ex.id)
            model_ops = []
            err = f"{type(e).__name__}: {e}"

        row: Dict[str, Any] = {
            "id": ex.id,
            "slot_name": ex.slot_name,
            "dataset": {
                "existing_records": ex.existing_records,
                "user_message": ex.user_message,
                "expected_operations": ex.expected_operations,
            },
            "model": {
                "operations": model_ops,
                "eval_meta": trace_to_dict(pipeline.last_trace),
                "error": err,
            },
        }
        results.append(row)

    return results


def write_report(rows: List[Dict[str, Any]], output_path: Path | str) -> None:
    Path(output_path).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
