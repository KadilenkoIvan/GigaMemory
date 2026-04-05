import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .serving import GenerationConfig, LocalHFServing
from .slot_update_messages import build_update_messages

logger = logging.getLogger(__name__)


@dataclass
class SlotOperation:
    op: str  # add|update|delete|nothing
    record_id: Optional[int] = None
    value: Optional[str] = None


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
            return ops

        # Attempt to "fix" JSON once, then retry parse.
        fixed = self._attempt_fix_json(raw)
        if fixed:
            ops = self._parse_operations(fixed)
            if ops is not None:
                return ops

        # Final fallback: add a single record with the full message (as requested).
        logger.info("SlotUpdate fallback to add(full message) slot=%s", slot_name)
        return [SlotOperation(op="add", value=user_message.strip())] if user_message.strip() else []

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

