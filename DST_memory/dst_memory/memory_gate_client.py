"""Локальная LLM решает, какие слоты памяти релевантны сообщению (для финального ответа)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory_gate_messages import build_memory_gate_messages
from .serving import GenerationConfig, LocalHFServing
from .slot_name_normalize import normalize_slot_label, resolve_slot_key_to_existing

logger = logging.getLogger(__name__)

_MARKERS = frozenset(
    (
        "я",
        "мой",
        "моя",
        "моё",
        "мое",
        "меня",
        "у меня",
        "помнишь",
        "помни",
        "запомни",
        "мы говорили",
    )
)


@dataclass
class MemoryGateSelection:
    """Результат шлюза: подставлять ли память в финальную LLM и какие слоты."""

    use_memory: bool
    slot_names: List[str]


class MemoryGateClient:
    def __init__(
        self,
        use_stub: bool,
        serving: Optional[LocalHFServing] = None,
        max_retries: int = 1,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_retries = max_retries
        if self.use_stub:
            logger.info("Memory gate client in STUB mode (эвристика по маркерам)")
        elif self.serving is None:
            raise ValueError("MemoryGateClient requires serving when use_stub is False")
        else:
            logger.info(
                "Memory gate client using LocalHFServing device=%s",
                self.serving.device,
            )

    def select_slots(
        self,
        user_message: str,
        existing_slot_names: List[str],
        *,
        for_vector_context: bool = False,
    ) -> MemoryGateSelection:
        if not existing_slot_names:
            return MemoryGateSelection(use_memory=False, slot_names=[])

        if self.use_stub or self.serving is None:
            return self._stub_select(
                user_message, existing_slot_names, for_vector_context=for_vector_context
            )

        messages = build_memory_gate_messages(
            user_message,
            existing_slot_names,
            for_vector_context=for_vector_context,
        )
        tries = self.max_retries + 1
        for attempt in range(1, tries + 1):
            raw = self.serving.generate_chat(
                messages,
                generation_config=GenerationConfig(max_new_tokens=200, do_sample=False),
            )
            logger.info("Memory gate raw attempt=%d: %s", attempt, raw[:500])
            parsed = self._parse_response(raw)
            if parsed is None:
                logger.warning("Memory gate parse failed attempt=%d/%d", attempt, tries)
                continue
            finalized = self._finalize(
                existing_slot_names,
                parsed,
                for_vector_context=for_vector_context,
            )
            logger.info(
                "Memory gate attempt=%d use_memory=%s slots=%s",
                attempt,
                finalized.use_memory,
                finalized.slot_names,
            )
            return finalized

        logger.warning("Memory gate falling back to no memory after failed parses")
        return MemoryGateSelection(use_memory=False, slot_names=[])

    def _stub_select(
        self,
        user_message: str,
        existing_slot_names: List[str],
        *,
        for_vector_context: bool,
    ) -> MemoryGateSelection:
        lower = user_message.lower()
        if any(m in lower for m in _MARKERS):
            if for_vector_context:
                return MemoryGateSelection(use_memory=True, slot_names=[])
            return MemoryGateSelection(use_memory=True, slot_names=list(existing_slot_names))
        return MemoryGateSelection(use_memory=False, slot_names=[])

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        s = text.strip()
        if not s.startswith("```"):
            return s
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        cleaned = self._strip_markdown_fence(text)
        blob = cleaned
        if not blob.strip().startswith("{"):
            ext = self._extract_first_json_object(cleaned)
            if ext:
                blob = ext
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _finalize(
        self,
        existing_slot_names: List[str],
        obj: Dict[str, Any],
        *,
        for_vector_context: bool,
    ) -> MemoryGateSelection:
        use_mem = obj.get("use_memory")
        if isinstance(use_mem, str):
            use_mem = use_mem.strip().lower() in ("true", "1", "yes", "да")
        use_memory = bool(use_mem)

        raw_slots = obj.get("slots")
        if raw_slots is None:
            for key in ("relevant_slots", "selected_slots", "memory_slots"):
                if key in obj:
                    raw_slots = obj[key]
                    break
        if not isinstance(raw_slots, list):
            raw_slots = []

        names_out: List[str] = []
        seen: set[str] = set()
        canonical = list(existing_slot_names)

        for item in raw_slots:
            if isinstance(item, str):
                s = item.strip()
            elif isinstance(item, dict) and "slot" in item:
                s = str(item["slot"]).strip()
            elif isinstance(item, dict) and "slot_name" in item:
                s = str(item["slot_name"]).strip()
            else:
                continue
            matched = MemoryGateClient._match_slot_name(s, canonical)
            if matched is None or matched in seen:
                continue
            seen.add(matched)
            names_out.append(matched)

        if not use_memory:
            return MemoryGateSelection(use_memory=False, slot_names=[])
        if not names_out:
            if for_vector_context:
                return MemoryGateSelection(use_memory=True, slot_names=[])
            return MemoryGateSelection(use_memory=False, slot_names=[])
        return MemoryGateSelection(use_memory=True, slot_names=names_out)

    @staticmethod
    def _match_slot_name(s: str, canonical: List[str]) -> Optional[str]:
        if not s:
            return None
        for ex in canonical:
            if ex == s or ex.lower() == s.lower():
                return ex
        normalized = normalize_slot_label(s)
        if normalized:
            resolved = resolve_slot_key_to_existing(canonical, normalized)
            if resolved in canonical:
                return resolved
        return None
