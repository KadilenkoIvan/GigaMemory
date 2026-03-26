import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


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
            model_dir = Path(model_path)
            # Allow two modes:
            # 1) local directory path
            # 2) Hugging Face model id, e.g. "Qwen/Qwen3.5-0.8B"
            if not model_dir.exists() and "/" not in model_path:
                raise FileNotFoundError(
                    f"Slot model directory not found: {model_path}. "
                    "Provide valid local --slot-model-path, HF model id, or use --slot-use-stub."
                )
            logger.info("Loading slot decision model path=%s device=%s", model_path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
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
        return (
            "Ты модуль управления слотами долгосрочной памяти пользователя.\n"
            "Тебе передан список существующих слотов и новое сообщение пользователя.\n"
            "Твоя задача: вернуть JSON-объекты решения по слотам.\n\n"
            "Существующие слоты:\n"
            f"{slots_text}\n\n"
            "Формат КАЖДОГО JSON ОБЪЕКТА строго:\n"
            '{"create_new": "false | true", "slot_name": "string"}\n\n'
            "Правила:\n"
            "1) Объектов может быть несколько, но не больше 5.\n"
            "2) МИНИМИЗИРУЙ количество слотов: создавай новый слот только если сообщение явно не подходит ни к одному существующему.\n"
            "3) slot_name: lower case, русский язык, краткий и понятный (1-3 слова).\n"
            "4) Если подходит существующий слот, возвращай create_new=false и его имя.\n"
            "5) Не добавляй никакой текст вне JSON.\n\n"
            "Few-shot примеры:\n"
            "Пример 1:\n"
            "Существующие: [\"питомцы\", \"семья\"]\n"
            "Сообщение: \"У моего кота Барсика опять линька\"\n"
            "Ответ:\n"
            '{"create_new":"false","slot_name":"питомцы"}\n\n'
            "Пример 2:\n"
            "Существующие: [\"работа\"]\n"
            "Сообщение: \"Я начал готовиться к марафону\"\n"
            "Ответ:\n"
            '{"create_new":"true","slot_name":"спорт"}\n\n'
            "Пример 3:\n"
            "Существующие: [\"семья\", \"хобби\"]\n"
            "Сообщение: \"В субботу поедем с женой в горы, а вечером почитаю книгу\"\n"
            "Ответ:\n"
            '{"create_new":"false","slot_name":"семья"}\n'
            '{"create_new":"false","slot_name":"хобби"}'
        )

    def _parse_json_decisions(self, text: str) -> List[SlotDecision]:
        chunks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        out: List[SlotDecision] = []
        for chunk in chunks:
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "create_new" not in obj or "slot_name" not in obj:
                continue
            slot_name = str(obj["slot_name"]).strip().lower()
            if not slot_name:
                continue
            create_new_raw = str(obj["create_new"]).strip().lower()
            create_new = create_new_raw == "true"
            out.append(SlotDecision(create_new=create_new, slot_name=slot_name))
        return out
