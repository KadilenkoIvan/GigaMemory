"""Language-agnostic JSON parsers for LLM outputs (shared by ru/en prompt packs)."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def parse_conflict_response(text: str) -> Dict[str, List[int]]:
    """Parse model response → {deactivate: [...], skip_new: [...]}."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)

    def _to_int(x: Any) -> int:
        if isinstance(x, dict):
            return int(x.get("idx") or x.get("record_id") or x.get("id") or 0)
        return int(x)

    return {
        "deactivate": [_to_int(x) for x in data.get("deactivate", [])],
        "skip_new": [_to_int(x) for x in data.get("skip_new", [])],
    }


def parse_deletion_response(text: str) -> List[Dict[str, str]]:
    """
    Parse LLM response into objects for deletion.

    Returns
    -------
    List of {"subject": ..., "relation": ..., "object": ...} dicts.
    Raises ValueError on parse failure.
    """
    blob = (text or "").strip()
    if blob.startswith("```"):
        lines = blob.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        blob = "\n".join(lines).strip()

    try:
        obj = json.loads(blob)
    except Exception:
        start = blob.find("{")
        if start < 0:
            raise ValueError(f"No JSON object found in: {blob[:200]!r}")
        depth = 0
        for i in range(start, len(blob)):
            c = blob[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(blob[start : i + 1])
                    break
        else:
            raise ValueError(f"Unbalanced JSON in: {blob[:200]!r}")

    if not isinstance(obj, dict) or "delete" not in obj:
        raise ValueError(f'Expected {{"delete": [...]}} but got: {blob[:200]!r}')

    items = obj["delete"]
    if not isinstance(items, list):
        raise ValueError(f'"delete" field is not a list: {blob[:200]!r}')

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip().lower()
        r = str(item.get("relation", "")).strip().lower()
        o = str(item.get("object", "")).strip().lower()
        if s and r and o:
            result.append({"subject": s, "relation": r, "object": o})
    return result


def parse_update_response(response_text: str) -> list[dict[str, Any]]:
    """Parse slot-update model reply → list of operations."""
    data = json.loads(response_text.strip())
    return data.get("operations", [])
