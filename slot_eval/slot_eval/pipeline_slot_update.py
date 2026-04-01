"""
Автономная копия логики slot-update из DST_memory (SlotUpdateClient).

Подключение к основному коду только:
  - dst_memory.serving (LocalHFServing, GenerationConfig)
  - dst_memory.slot_update_messages.build_update_messages

При изменении SlotUpdateClient в DST_memory — при необходимости синхронизируйте этот файл вручную.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from dst_memory.serving import LocalHFServing

logger = logging.getLogger(__name__)


@dataclass
class SlotOperation:
    op: str
    record_id: Optional[int] = None
    value: Optional[str] = None


@dataclass
class SlotUpdateTrace:
    """Снимок последнего вызова (для отчёта eval)."""

    primary_raw: str = ""
    json_fix_raw: Optional[str] = None
    effective_raw: Optional[str] = None
    used_json_fix: bool = False
    used_fallback: bool = False


class PipelineSlotUpdate:
    """Та же последовательность, что SlotUpdateClient.plan_operations."""

    def __init__(self, serving: Optional["LocalHFServing"], max_retries: int = 1):
        self.serving = serving
        self.max_retries = max_retries
        self.last_trace = SlotUpdateTrace()

    def plan_operations(
        self,
        slot_name: str,
        existing_records: List[Dict[str, Any]],
        user_message: str,
    ) -> List[SlotOperation]:
        self.last_trace = SlotUpdateTrace()

        if self.serving is None:
            logger.info("SlotUpdate eval STUB: add(full message) slot=%s", slot_name)
            self.last_trace.used_fallback = True
            return (
                [SlotOperation(op="add", value=user_message.strip())]
                if user_message.strip()
                else []
            )

        from dst_memory.slot_update_messages import build_update_messages

        messages = build_update_messages(slot_name, existing_records, user_message)

        model_response = self._generate_with_retries(messages)
        self.last_trace.primary_raw = model_response

        ops = self._parse_operations(model_response)
        if ops is not None:
            self.last_trace.effective_raw = model_response
            return ops

        json_fix_candidate = self._attempt_fix_json(model_response)
        if json_fix_candidate:
            self.last_trace.json_fix_raw = json_fix_candidate
            ops = self._parse_operations(json_fix_candidate)
            if ops is not None:
                self.last_trace.effective_raw = json_fix_candidate
                self.last_trace.used_json_fix = True
                return ops

        logger.info("SlotUpdate eval fallback to add(full message) slot=%s", slot_name)
        self.last_trace.used_json_fix = bool(json_fix_candidate)
        self.last_trace.used_fallback = True
        self.last_trace.effective_raw = None
        return [SlotOperation(op="add", value=user_message.strip())] if user_message.strip() else []

    def _generate_with_retries(self, messages: List[Dict[str, str]]) -> str:
        from dst_memory.serving import GenerationConfig

        tries = self.max_retries + 1
        last_attempt_text = ""
        for attempt in range(1, tries + 1):
            last_attempt_text = self.serving.generate_chat(
                messages,
                generation_config=GenerationConfig(max_new_tokens=400, do_sample=False),
            )
            logger.info("SlotUpdate eval raw attempt=%d: %s", attempt, last_attempt_text)
            if self._parse_operations(last_attempt_text) is not None:
                return last_attempt_text
        return last_attempt_text

    def _attempt_fix_json(self, bad_text: str) -> str:
        from dst_memory.serving import GenerationConfig

        prompt = (
            "Исправь следующий ответ так, чтобы он был ВАЛИДНЫМ JSON строго формата:\n"
            '{ "operations": [ {"op":"add","value":"..."}, {"op":"update","id":1,"value":"..."}, {"op":"delete","id":2}, {"op":"nothing"} ] }\n'
            "Никакого текста кроме JSON.\n\n"
            f"Ответ для исправления:\n```text\n{bad_text}\n```"
        )
        messages = [{"role": "system", "content": "Ты исправляешь JSON."}, {"role": "user", "content": prompt}]
        fixed_text = self.serving.generate_chat(
            messages,
            generation_config=GenerationConfig(max_new_tokens=250, do_sample=False),
        )
        logger.info("SlotUpdate eval fixed JSON candidate: %s", fixed_text)
        return fixed_text

    def _parse_operations(self, text: str) -> Optional[List[SlotOperation]]:
        try:
            parsed = json.loads(text.strip())
        except Exception:
            return None
        if not isinstance(parsed, dict) or "operations" not in parsed:
            return None
        ops = parsed["operations"]
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


def trace_to_dict(t: SlotUpdateTrace) -> Dict[str, Any]:
    return {
        "primary_raw": t.primary_raw,
        "json_fix_raw": t.json_fix_raw,
        "effective_raw": t.effective_raw,
        "used_json_fix": t.used_json_fix,
        "used_fallback": t.used_fallback,
    }
