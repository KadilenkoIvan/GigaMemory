import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .serving import GenerationConfig, LocalHFServing
from .slot_update_messages import build_update_messages, build_triplets_messages

logger = logging.getLogger(__name__)


@dataclass
class SlotOperation:
    op: str  # add|update|delete|nothing
    record_id: Optional[int] = None
    value: Optional[str] = None
    triplets: List[Dict[str, str]] | None = None
    graph_artifacts: Dict[str, Any] | None = None


class SlotUpdateClient:
    def __init__(self, serving: Optional[LocalHFServing] = None, max_retries: int = 1):
        self.serving = serving
        self.max_retries = max_retries

    def plan_operations(
        self,
        slot_name: str,
        existing_records: List[Dict[str, Any]],
        user_message: str,
    ) -> List[SlotOperation]:
        if self.serving is None:
            logger.info("SlotUpdate STUB: add(full message) slot=%s", slot_name)
            return (
                [SlotOperation(op="add", value=user_message.strip())]
                if user_message.strip()
                else []
            )

        messages = build_update_messages(slot_name, existing_records, user_message)

        raw = self._generate_with_retries(messages)
        ops = self._parse_operations(raw)
        if ops is not None:
            ops = self._sanitize_ops(existing_records, ops)
            return [self._attach_triplets(slot_name, user_message, o) for o in ops]

        # Attempt to "fix" JSON once, then retry parse.
        fixed = self._attempt_fix_json(raw)
        if fixed:
            ops = self._parse_operations(fixed)
            if ops is not None:
                ops = self._sanitize_ops(existing_records, ops)
                return [self._attach_triplets(slot_name, user_message, o) for o in ops]

        # Final fallback: add a single record with the full message (as requested).
        logger.info("SlotUpdate fallback to add(full message) slot=%s", slot_name)
        return [SlotOperation(op="add", value=user_message.strip())] if user_message.strip() else []

    def _sanitize_ops(
        self,
        existing_records: List[Dict[str, Any]],
        ops: List[SlotOperation],
    ) -> List[SlotOperation]:
        """
        Make operations safe and applicable:
        - In empty slot: drop update/delete; convert update(with value) to add(value).
        - If update/delete references unknown id: convert update(with value) to add(value), drop delete.
        - If nothing is present with other ops: remove nothing.
        """
        existing_ids = {
            int(r["id"]) for r in existing_records if isinstance(r, dict) and isinstance(r.get("id"), int)
        }
        out: List[SlotOperation] = []
        for o in ops:
            if o.op == "nothing":
                out.append(o)
                continue

            if o.op in {"update", "delete"}:
                if not existing_ids:
                    if o.op == "update" and o.value:
                        out.append(SlotOperation(op="add", value=o.value))
                    continue
                if o.record_id is None or o.record_id not in existing_ids:
                    if o.op == "update" and o.value:
                        out.append(SlotOperation(op="add", value=o.value))
                    continue

            out.append(o)

        has_non_nothing = any(o.op != "nothing" for o in out)
        if has_non_nothing:
            out = [o for o in out if o.op != "nothing"]
        if not out:
            return [SlotOperation(op="nothing")]
        return out

    def _attach_triplets(
        self,
        slot_name: str,
        user_message: str,
        op: SlotOperation,
    ) -> SlotOperation:
        if op.op not in {"add", "update"} or not op.value:
            return op
        messages = build_triplets_messages(slot_name, user_message, op.value)
        raw = self.serving.generate_chat(
            messages,
            generation_config=GenerationConfig(max_new_tokens=220, do_sample=False),
        )
        try:
            obj = json.loads(raw.strip())
        except Exception:
            return op
        if isinstance(obj, dict):
            op.graph_artifacts = obj
        arr = obj.get("triplets") if isinstance(obj, dict) else None
        if isinstance(arr, list):
            out: List[Dict[str, str]] = []
            for x in arr:
                if not isinstance(x, dict):
                    continue
                s = str(x.get("subject", "")).strip()
                r = str(x.get("relation", "")).strip()
                o = str(x.get("object", "")).strip()
                if not s or not r or not o:
                    continue
                out.append({"subject": s, "relation": r, "object": o})
            op.triplets = out[:3]
        return op

    def _generate_with_retries(self, messages: List[Dict[str, str]]) -> str:
        tries = self.max_retries + 1
        last = ""
        for attempt in range(1, tries + 1):
            last = self.serving.generate_chat(
                messages,
                generation_config=GenerationConfig(max_new_tokens=400, do_sample=False),
            )
            logger.info("SlotUpdate raw response attempt=%d: %s", attempt, last)
            if self._parse_operations(last) is not None:
                return last
        return last

    def _attempt_fix_json(self, bad_text: str) -> str:
        prompt = (
            "Исправь следующий ответ так, чтобы он был ВАЛИДНЫМ JSON строго формата:\n"
            '{ "operations": [ {"op":"add","value":"..."}, {"op":"update","id":1,"value":"..."}, {"op":"delete","id":2}, {"op":"nothing"} ] }\n'
            "Никакого текста кроме JSON.\n\n"
            f"Ответ для исправления:\n```text\n{bad_text}\n```"
        )
        messages = [{"role": "system", "content": "Ты исправляешь JSON."}, {"role": "user", "content": prompt}]
        fixed = self.serving.generate_chat(
            messages,
            generation_config=GenerationConfig(max_new_tokens=250, do_sample=False),
        )
        logger.info("SlotUpdate fixed JSON candidate: %s", fixed)
        return fixed

    def _parse_operations(self, text: str) -> Optional[List[SlotOperation]]:
        try:
            obj = json.loads(text.strip())
        except Exception:
            return None
        if not isinstance(obj, dict) or "operations" not in obj:
            return None
        ops = obj["operations"]
        if not isinstance(ops, list):
            return None

        out: List[SlotOperation] = []
        for item in ops:
            if not isinstance(item, dict):
                continue
            op = str(item.get("op", "")).strip().lower()
            if op not in {"add", "update", "delete", "nothing"}:
                continue
            if op == "add":
                v = item.get("value")
                if isinstance(v, str) and v.strip():
                    out.append(SlotOperation(op="add", value=v.strip()))
            elif op == "update":
                rid = item.get("id")
                v = item.get("value")
                if isinstance(rid, int) and isinstance(v, str) and v.strip():
                    out.append(SlotOperation(op="update", record_id=rid, value=v.strip()))
            elif op == "delete":
                rid = item.get("id")
                if isinstance(rid, int):
                    out.append(SlotOperation(op="delete", record_id=rid))
            elif op == "nothing":
                out.append(SlotOperation(op="nothing"))
        return out

