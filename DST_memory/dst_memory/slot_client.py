import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .serving import GenerationConfig, LocalHFServing
from .slot_messages import build_messages
from .slot_model_path import resolve_slot_model_path
from .slot_ontology import SLOT_ID_BY_LABEL, SLOT_IDS
from .slot_name_normalize import (
    normalize_slot_label,
    resolve_slot_key_to_existing,
)

logger = logging.getLogger(__name__)

# Backwards compatibility: other code may import from slot_client.
__all__ = ["SlotDecision", "SlotDecisionClient", "resolve_slot_model_path"]


@dataclass
class SlotDecision:
    """Имя слота после нормализации; создание/добавление решает DSTManager по state.slots."""
    slot_name: str


class SlotDecisionClient:
    """
    Slot name decisions via the same LocalHFServing instance as SlotUpdateClient
    (single model load in GPU memory).
    """

    def __init__(
        self,
        use_stub: bool,
        serving: Optional[LocalHFServing] = None,
        max_slots: int = 5,
        max_retries: int = 1,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_slots = max_slots
        self.max_retries = max_retries

        if self.use_stub:
            logger.info("Slot decision client in STUB mode")
        elif self.serving is None:
            raise ValueError(
                "SlotDecisionClient requires serving=LocalHFServing(...) when use_stub is False"
            )
        else:
            logger.info(
                "Slot decision client using shared LocalHFServing device=%s",
                self.serving.device,
            )

    def decide_slots(
        self,
        existing_slots: List[str],
        user_message: str,
        force_at_least_one: bool = False,
    ) -> List[SlotDecision]:
        if self.use_stub:
            return [SlotDecision(slot_name="home_daily_life")]
        messages = build_messages(
            user_message,
            self.max_slots,
            existing_slots,
            force_at_least_one=force_at_least_one,
        )

        tries = self.max_retries + 1
        for attempt in range(1, tries + 1):
            raw = self._generate(messages)
            logger.info("Slot model raw response attempt=%d: %s", attempt, raw)
            raw_names = self._parse_slot_names_from_response(raw)
            if raw_names is None:
                logger.warning("Failed to parse slot JSON attempt=%d/%d", attempt, tries)
                continue

            decisions = self._finalize_decisions(existing_slots, raw_names)
            decisions = [d for d in decisions if d.slot_name in SLOT_IDS]
            # If model produced only unknown labels, treat as parse failure to retry.
            if raw_names and not decisions:
                logger.warning(
                    "Slot model attempt=%d returned unknown slots raw=%s",
                    attempt,
                    raw_names,
                )
                continue
            logger.info(
                "Slot model attempt=%d parsed=%s decisions=%s",
                attempt,
                raw_names,
                [d.slot_name for d in decisions],
            )
            return decisions

        return []

    def _finalize_decisions(
        self, existing_slots: List[str], raw_names: List[str]
    ) -> List[SlotDecision]:
        """Нормализация имён, сопоставление с уже существующими ключами, дедуп."""
        canonical_existing: List[str] = list(existing_slots)
        seen: set[str] = set()
        out: List[SlotDecision] = []

        for name in raw_names:
            raw = str(name).strip()
            # Fixed ontology: allow exact slot ids without ru-normalization.
            if raw in SLOT_IDS:
                resolved = raw
            elif raw.upper() in SLOT_ID_BY_LABEL:
                resolved = SLOT_ID_BY_LABEL[raw.upper()]
            else:
                normalized = normalize_slot_label(raw)
                if not normalized:
                    continue
                resolved = resolve_slot_key_to_existing(canonical_existing, normalized)
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(SlotDecision(slot_name=resolved))
            if resolved not in canonical_existing:
                canonical_existing.append(resolved)
            if len(out) >= self.max_slots:
                break

        return out

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        if self.use_stub or self.serving is None:
            raise RuntimeError(
                "Slot model is not initialized. "
                "Disable --slot-use-stub and pass a shared LocalHFServing instance."
            )
        logger.info("Slot decision generation via shared serving device=%s", self.serving.device)
        return self.serving.generate_chat(
            messages,
            generation_config=GenerationConfig(max_new_tokens=300, do_sample=False),
        )

    def _coerce_to_slot_list(self, obj: Any) -> Optional[List[str]]:
        """Возвращает список имён слотов или None, если структура не распознана."""
        if isinstance(obj, dict):
            for key in ("slot_assignments", "slots", "decisions"):
                if key in obj:
                    v = obj[key]
                    if isinstance(v, list):
                        return self._extract_slot_names_from_list(v)
                    return None
            return None
        if isinstance(obj, list):
            return self._extract_slot_names_from_list(obj)
        return None

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
    def _extract_first_balanced_json_object(text: str) -> Optional[str]:
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

    @staticmethod
    def _extract_first_balanced_json_array(text: str) -> Optional[str]:
        start = text.find("[")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _extract_slot_names_from_list(items: List[Any]) -> List[str]:
        out: List[str] = []
        for item in items:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict) and "slot_name" in item:
                s = str(item["slot_name"]).strip()
                if s:
                    out.append(s)
        return out

    def _parse_legacy_line_objects(self, text: str) -> List[str]:
        chunks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        out: List[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "slot_name" not in obj:
                continue
            s = str(obj["slot_name"]).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _parse_slot_names_from_response(self, text: str) -> Optional[List[str]]:
        """Список слотов, [] если модель явно вернула пустой список; None при ошибке разбора."""
        cleaned = self._strip_markdown_fence(text)

        for candidate in (cleaned, self._extract_first_balanced_json_object(cleaned) or ""):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            names = self._coerce_to_slot_list(obj)
            if names is not None:
                return names[: self.max_slots]

        extracted = self._extract_first_balanced_json_object(cleaned)
        if extracted:
            try:
                obj = json.loads(extracted)
                names = self._coerce_to_slot_list(obj)
                if names is not None:
                    return names[: self.max_slots]
            except json.JSONDecodeError:
                pass

        arr = self._extract_first_balanced_json_array(cleaned)
        if arr:
            try:
                obj = json.loads(arr)
                if isinstance(obj, list):
                    return self._extract_slot_names_from_list(obj)[: self.max_slots]
            except json.JSONDecodeError:
                pass

        legacy = self._parse_legacy_line_objects(text)
        if legacy:
            return legacy[: self.max_slots]
        return None
