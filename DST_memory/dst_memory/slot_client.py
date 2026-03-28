import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    create_new: bool
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
            # In stub mode model/tokenizer are intentionally not initialized.
            self.tokenizer = None
            self.model = None

    def decide_slots(self, existing_slots: List[str], user_message: str) -> List[SlotDecision]:
        if self.use_stub:
            # Stub behavior mirrors old logic: one generic slot.
            return [SlotDecision(create_new=True, slot_name="facts")]

        prompt_system = self._build_system_prompt(existing_slots)
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": user_message},
        ]

        tries = self.max_retries + 1
        for attempt in range(1, tries + 1):
            raw = self._generate(messages)
            logger.info("Slot model raw response attempt=%d: %s", attempt, raw)
            decisions = self._parse_json_decisions(raw)
            if decisions:
                logger.info(
                    "Slot model parsed decisions attempt=%d: %s",
                    attempt,
                    [
                        {"create_new": d.create_new, "slot_name": d.slot_name}
                        for d in decisions[: self.max_slots]
                    ],
                )
                logger.info("Slot decisions parsed count=%d attempt=%d", len(decisions), attempt)
                return decisions[: self.max_slots]
            logger.warning("Failed to parse slot decisions attempt=%d/%d", attempt, tries)

        # No fallback by request: skip message on parse failure.
        return []

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
            "Ты модуль управления слотами долгосрочной памяти пользователя.\n"
            "Тебе передан список существующих слотов и новое сообщение пользователя.\n\n"
            "ВАЖНО: в одном сообщении пользователя может быть несколько разных тем. "
            f"Ты можешь (и должен при необходимости) вернуть НЕСКОЛЬКО слотов сразу — "
            f"до {max_s} элементов в массиве slot_assignments.\n"
            "Если смысл касается только одной темы — верни ровно один элемент в массиве.\n\n"
            "Существующие слоты:\n"
            f"{slots_text}\n\n"
            "Верни РОВНО один JSON-объект (без текста до или после). Формат:\n"
            "{\n"
            '  "slot_assignments": [\n'
            '    {"create_new": false, "slot_name": "краткое имя слота"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Поле create_new: boolean true/false (не строка).\n"
            "Правила:\n"
            f"1) В slot_assignments от 1 до {max_s} объектов.\n"
            "2) МИНИМИЗИРУЙ число слотов: новый слот только если фраза явно не укладывается в существующие.\n"
            "3) slot_name: lower case, русский язык, кратко (1–3 слова).\n"
            "4) Для уже существующего слота: create_new=false и точное имя из списка (в lower case).\n"
            "5) Несколько разных тем в одной реплике — несколько элементов в slot_assignments.\n"
            "6) Не используй markdown и не оборачивай ответ в ```.\n\n"
            "Few-shot примеры:\n"
            "Пример 1 (один слот):\n"
            "Существующие: [\"питомцы\", \"семья\"]\n"
            "Сообщение: \"У моего кота Барсика опять линька\"\n"
            "Ответ:\n"
            '{"slot_assignments":[{"create_new":false,"slot_name":"питомцы"}]}\n\n'
            "Пример 2 (новый слот):\n"
            "Существующие: [\"работа\"]\n"
            "Сообщение: \"Я начал готовиться к марафону\"\n"
            "Ответ:\n"
            '{"slot_assignments":[{"create_new":true,"slot_name":"спорт"}]}\n\n'
            "Пример 3 (несколько слотов в одном сообщении):\n"
            "Существующие: [\"семья\", \"хобби\"]\n"
            "Сообщение: \"В субботу поедем с женой в горы, а вечером почитаю книгу\"\n"
            "Ответ:\n"
            '{"slot_assignments":['
            '{"create_new":false,"slot_name":"семья"},'
            '{"create_new":false,"slot_name":"хобби"}]}'
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
    def _coerce_create_new(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "да"):
                return True
            if v in ("false", "0", "no", "нет"):
                return False
        return None

    def _parse_assignment_items(self, items: List[Any]) -> List[SlotDecision]:
        out: List[SlotDecision] = []
        seen_slots: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            if "slot_name" not in item:
                continue
            slot_name = str(item["slot_name"]).strip().lower()
            if not slot_name or slot_name in seen_slots:
                continue
            if "create_new" not in item:
                continue
            create_new = self._coerce_create_new(item["create_new"])
            if create_new is None:
                continue
            seen_slots.add(slot_name)
            out.append(SlotDecision(create_new=create_new, slot_name=slot_name))
        return out

    def _parse_envelope_object(self, obj: Any) -> List[SlotDecision]:
        if isinstance(obj, list):
            return self._parse_assignment_items(obj)
        if not isinstance(obj, dict):
            return []
        for key in ("slot_assignments", "slots", "decisions"):
            raw = obj.get(key)
            if isinstance(raw, list):
                return self._parse_assignment_items(raw)
        return []

    def _parse_legacy_line_objects(self, text: str) -> List[SlotDecision]:
        chunks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        out: List[SlotDecision] = []
        seen_slots: set[str] = set()
        for chunk in chunks:
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "create_new" not in obj or "slot_name" not in obj:
                continue
            slot_name = str(obj["slot_name"]).strip().lower()
            if not slot_name or slot_name in seen_slots:
                continue
            create_new = self._coerce_create_new(obj["create_new"])
            if create_new is None:
                create_new_raw = str(obj["create_new"]).strip().lower()
                create_new = create_new_raw == "true"
            seen_slots.add(slot_name)
            out.append(SlotDecision(create_new=create_new, slot_name=slot_name))
        return out

    def _parse_json_decisions(self, text: str) -> List[SlotDecision]:
        cleaned = self._strip_markdown_fence(text)

        for candidate in (cleaned, self._extract_first_balanced_json_object(cleaned) or ""):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            decisions = self._parse_envelope_object(obj)
            if decisions:
                return decisions[: self.max_slots]

        extracted = self._extract_first_balanced_json_object(cleaned)
        if extracted:
            try:
                obj = json.loads(extracted)
                decisions = self._parse_envelope_object(obj)
                if decisions:
                    return decisions[: self.max_slots]
            except json.JSONDecodeError:
                pass

        legacy = self._parse_legacy_line_objects(text)
        return legacy[: self.max_slots]
