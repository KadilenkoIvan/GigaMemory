import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .slot_name_normalize import (
    normalize_slot_label,
    resolve_slot_key_to_existing,
)

logger = logging.getLogger(__name__)


def _is_hf_repo_id(s: str) -> bool:
    """
    True for HuggingFace hub ids like 'Qwen/Qwen2.5-0.5B' (namespace/model).
    False for local paths (absolute, or multi-segment relative paths).
    """
    if "/" not in s:
        return False
    if Path(s).is_absolute():
        return False
    parts = s.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    if Path(s).exists():
        return False
    return True


def resolve_slot_model_path(model_path: str) -> str:
    """
    Return a path/id suitable for AutoTokenizer/AutoModel.from_pretrained.

    - Existing directory (relative or absolute) -> resolved absolute path.
    - HuggingFace repo id -> unchanged string.
    - Otherwise -> FileNotFoundError with a clear message.
    """
    raw = str(model_path).strip()
    if not raw:
        raise ValueError("slot model path is empty")

    local = Path(raw).expanduser()
    if local.is_dir():
        return str(local.resolve())

    if _is_hf_repo_id(raw):
        return raw

    raise FileNotFoundError(
        f"Slot model directory not found: {model_path!r}. "
        "Use an existing folder with tokenizer + weights, a HuggingFace id (e.g. org/model), "
        "or --slot-use-stub."
    )


@dataclass
class SlotDecision:
    """Имя слота после нормализации; создание/добавление решает DSTManager по state.slots."""
    slot_name: str


class SlotDecisionClient:
    def __init__(
        self,
        use_stub: bool,
        model_path: str,
        max_slots: int = 5,
        max_retries: int = 1,
    ):
        self.use_stub = use_stub
        self.model_path = model_path
        self.max_slots = max_slots
        self.max_retries = max_retries

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_model_ready = False

        if not self.use_stub:
            resolved = resolve_slot_model_path(model_path)
            logger.info(
                "Loading slot decision model path=%s (resolved=%s) device=%s",
                model_path,
                resolved,
                self.device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            self._is_model_ready = True
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = "unknown"
            logger.info("Slot model loaded on device=%s", model_device)
        else:
            logger.info("Slot decision client in STUB mode")
            self.tokenizer = None
            self.model = None

    def decide_slots(self, existing_slots: List[str], user_message: str) -> List[SlotDecision]:
        if self.use_stub:
            return [SlotDecision(slot_name="facts")]

        prompt_system = self._build_system_prompt(existing_slots)
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": user_message},
        ]

        tries = self.max_retries + 1
        for attempt in range(1, tries + 1):
            raw = self._generate(messages)
            logger.info("Slot model raw response attempt=%d: %s", attempt, raw)
            raw_names = self._parse_slot_names_from_response(raw)
            if not raw_names:
                logger.warning("Failed to parse slot names attempt=%d/%d", attempt, tries)
                continue

            decisions = self._finalize_decisions(existing_slots, raw_names)
            if decisions:
                logger.info(
                    "Slot model decisions attempt=%d: %s",
                    attempt,
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
            normalized = normalize_slot_label(name)
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
        if not self._is_model_ready or self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Slot model is not initialized. "
                "Disable --slot-use-stub and provide valid --slot-model-path."
            )
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = "unknown"
        logger.info("Slot generation using device=%s", model_device)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return result.strip()

    def _build_system_prompt(self, existing_slots: List[str]) -> str:
        slots_text = ", ".join(existing_slots) if existing_slots else "<пусто>"
        max_s = self.max_slots
        return (
            "Ты классификатор тем. Твоя единственная задача — определить, к каким слотам относится сообщение.\n\n"
            "СЛОТ = широкая категория жизни человека.\n"
            "Правильно: еда, спорт, семья, работа, здоровье, транспорт, хобби, питомцы, финансы, покупки.\n"
            "Неправильно: эстрагон, маршрут 12, iPhone 15 — слишком узко.\n\n"
            "ПРАВИЛА (выполняй все):\n"
            "1. Имя слота: 1–3 слова, только русский язык.\n"
            f"2. Максимум слотов в ответе: {max_s}.\n"
            "3. Если подходит существующий слот — используй его имя точно как написано в списке.\n"
            "4. Новый слот создавай только если ни один существующий не подходит.\n"
            "5. Не объясняй выбор. Не добавляй текст вне JSON.\n\n"
            "СУЩЕСТВУЮЩИЕ СЛОТЫ:\n"
            f"{slots_text}\n\n"
            "ФОРМАТ ОТВЕТА — строго этот JSON и ничего больше:\n"
            '{"slot_assignments":["слот1","слот2"]}\n\n'
            "ПРИМЕРЫ:\n"
            "Слоты: [питомцы, семья] | Сообщение: У кота Барсика линька\n"
            '→ {"slot_assignments":["питомцы"]}\n\n'
            "Слоты: [работа] | Сообщение: Начал готовиться к марафону\n"
            '→ {"slot_assignments":["спорт"]}\n\n'
            "Слоты: [еда] | Сообщение: Добавь в салат эстрагон и лимон\n"
            '→ {"slot_assignments":["еда"]}\n\n'
            "Слоты: [семья, хобби] | Сообщение: В субботу едем с женой в горы, вечером почитаю книгу\n"
            '→ {"slot_assignments":["семья","хобби"]}\n\n'
            "Слоты: [] | Сообщение: Купил новый велосипед\n"
            '→ {"slot_assignments":["хобби"]}\n'
        )

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

    def _names_from_parsed_json(self, obj: Any) -> List[str]:
        if isinstance(obj, list):
            return self._extract_slot_names_from_list(obj)
        if not isinstance(obj, dict):
            return []
        for key in ("slot_assignments", "slots", "decisions"):
            raw = obj.get(key)
            if isinstance(raw, list):
                return self._extract_slot_names_from_list(raw)
        return []

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

    def _parse_slot_names_from_response(self, text: str) -> List[str]:
        cleaned = self._strip_markdown_fence(text)

        for candidate in (
            cleaned,
            self._extract_first_balanced_json_object(cleaned) or "",
            self._extract_first_balanced_json_array(cleaned) or "",
        ):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            names = self._names_from_parsed_json(obj)
            if names:
                return names[: self.max_slots]

        extracted = self._extract_first_balanced_json_object(cleaned)
        if extracted:
            try:
                obj = json.loads(extracted)
                names = self._names_from_parsed_json(obj)
                if names:
                    return names[: self.max_slots]
            except json.JSONDecodeError:
                pass

        arr = self._extract_first_balanced_json_array(cleaned)
        if arr:
            try:
                obj = json.loads(arr)
                if isinstance(obj, list):
                    names = self._extract_slot_names_from_list(obj)
                    if names:
                        return names[: self.max_slots]
            except json.JSONDecodeError:
                pass

        legacy = self._parse_legacy_line_objects(text)
        return legacy[: self.max_slots]
