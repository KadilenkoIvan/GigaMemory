"""Load slot eval JSON datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SlotEvalExample:
    id: str
    slot_name: str
    existing_records: List[Dict[str, Any]]
    user_message: str
    expected_operations: List[Dict[str, Any]]
    raw: Dict[str, Any]


def load_examples(path: Path | str) -> List[SlotEvalExample]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of objects")

    out: List[SlotEvalExample] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} is not an object")
        eid = str(row.get("id", f"idx_{i}"))
        slot_name = str(row.get("slot_name", ""))
        existing = row.get("existing_records") or []
        if not isinstance(existing, list):
            raise ValueError(f"{eid}: existing_records must be a list")
        msg = str(row.get("user_message", ""))
        exp = row.get("expected_operations")
        if exp is None:
            raise ValueError(f"{eid}: missing expected_operations")
        if not isinstance(exp, list):
            raise ValueError(f"{eid}: expected_operations must be a list")
        out.append(
            SlotEvalExample(
                id=eid,
                slot_name=slot_name,
                existing_records=existing,
                user_message=msg,
                expected_operations=exp,
                raw=row,
            )
        )
    return out
