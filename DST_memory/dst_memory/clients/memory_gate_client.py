"""Локальная LLM решает, какие слоты памяти релевантны сообщению (для финального ответа)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..prompts.loader import (
    PromptModules,
    load_prompt_modules,
    normalize_prompt_language,
)
from ..slots.ontology import DEFAULT_USER_SLOTS
from ..slots.slot_name_normalize import (
    normalize_slot_label,
    resolve_slot_key_to_existing,
)
from .serving import GenerationConfig, SlotServing

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
    slot_names: list[str]


class MemoryGateClient:
    def __init__(
        self,
        use_stub: bool,
        serving: SlotServing | None = None,
        max_retries: int = 1,
        prompt_language: str = "ru",
        max_new_tokens: int = 200,
        parse_retry_temperature: float = 0.65,
        parse_retry_temperature_increment: float = 0.08,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_retries = max_retries
        self.prompt_language = prompt_language
        self.max_new_tokens = int(max_new_tokens)
        self.parse_retry_temperature = float(parse_retry_temperature)
        self.parse_retry_temperature_increment = float(
            parse_retry_temperature_increment
        )
        self._prompt_modules_cache: dict[str, PromptModules] = {}
        if self.use_stub:
            logger.info("Memory gate client in STUB mode (эвристика по маркерам)")
        elif self.serving is None:
            raise ValueError("MemoryGateClient requires serving when use_stub is False")
        else:
            logger.info(
                "Memory gate client using LocalHFServing device=%s",
                self.serving.device,
            )

    def _modules(self, prompt_language: str | None = None) -> PromptModules:
        lang = normalize_prompt_language(prompt_language or self.prompt_language)
        cached = self._prompt_modules_cache.get(lang)
        if cached is None:
            cached = load_prompt_modules(lang)
            self._prompt_modules_cache[lang] = cached
        return cached

    def select_slots(
        self,
        user_message: str,
        existing_slot_names: list[str],
        *,
        for_vector_context: bool = False,
        prompt_language: str | None = None,
    ) -> MemoryGateSelection:
        if not existing_slot_names:
            return MemoryGateSelection(use_memory=False, slot_names=[])

        if self.use_stub or self.serving is None:
            return self._stub_select(
                user_message, existing_slot_names, for_vector_context=for_vector_context
            )

        pm = self._modules(prompt_language)
        messages = pm.memory_gate_messages.build_memory_gate_messages(
            user_message,
            existing_slot_names,
            for_vector_context=for_vector_context,
        )
        tries = self.max_retries + 1
        for attempt in range(1, tries + 1):
            if attempt == 1:
                gen_cfg = GenerationConfig(
                    max_new_tokens=self.max_new_tokens, do_sample=False
                )
            else:
                t = min(
                    1.0,
                    self.parse_retry_temperature
                    + self.parse_retry_temperature_increment * float(attempt - 2),
                )
                gen_cfg = GenerationConfig(
                    max_new_tokens=self.max_new_tokens, do_sample=True, temperature=t
                )
            raw = self.serving.generate_chat(messages, generation_config=gen_cfg)
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
        existing_slot_names: list[str],
        *,
        for_vector_context: bool,
    ) -> MemoryGateSelection:
        lower = user_message.lower()
        if any(m in lower for m in _MARKERS):
            if for_vector_context:
                return MemoryGateSelection(use_memory=True, slot_names=[])
            return MemoryGateSelection(
                use_memory=True, slot_names=list(existing_slot_names)
            )
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
    def _extract_first_json_object(text: str) -> str | None:
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

    def _parse_response(self, text: str) -> dict[str, Any] | None:
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
        existing_slot_names: list[str],
        obj: dict[str, Any],
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

        names_out: list[str] = []
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
    def _match_slot_name(s: str, canonical: list[str]) -> str | None:
        if not s:
            return None
        allowed = set(canonical)
        for ex in canonical:
            if ex == s or ex.lower() == s.lower():
                return ex
        # RU few-shots / model output: «СЕМЬЯ», «РАБОТА» → canonical FAMILY, WORK
        resolved = DEFAULT_USER_SLOTS.resolve(s)
        if resolved and resolved in allowed:
            return resolved
        normalized = normalize_slot_label(s)
        if normalized:
            r2 = resolve_slot_key_to_existing(canonical, normalized)
            if r2 in allowed:
                return r2
        return None
