"""Промпты извлечения триплетов (русские few-shot, system без markdown)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .ontology import CANONICAL_TO_RU_LABEL, DEFAULT_USER_SLOTS, RU_SLOT_LABELS_ORDERED
from .prompt_fewshots_ru import (
    triplet_per_slot_few_shot_messages,
    triplet_single_pass_few_shot_messages,
)


def build_triplet_messages(
    user_message: str,
    *,
    slot_name: str | None = None,
    include_slot: bool = False,
    ontology_slots: List[str] | None = None,
    max_triplets: int = 12,
) -> List[Dict[str, Any]]:
    _ = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_ru_json = json.dumps(RU_SLOT_LABELS_ORDERED, ensure_ascii=False)

    ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name) if slot_name else None

    # Slot header goes FIRST in system prompt so the model sees it immediately.
    slot_header = ""
    if slot_name and ru_slot:
        slot_header = (
            f"ТЕКУЩИЙ СЛОТ: {ru_slot} ({slot_name}).\n"
            f"ИЗВЛЕКАЙ ТОЛЬКО ФАКТЫ, ОТНОСЯЩИЕСЯ К СЛОТУ «{ru_slot}».\n"
            f"ЕСЛИ В СООБЩЕНИИ НЕТ ФАКТОВ ДЛЯ СЛОТА «{ru_slot}» — ВЕРНИ {{\"triplets\":[]}}.\n"
            "В JSON НЕ УКАЗЫВАЙ ПОЛЕ slot — СЛОТ ЗАДАЁТСЯ СИСТЕМОЙ.\n\n"
        )

    # Schema examples use lowercase (model must output lowercase, postprocessing uppercases).
    output_schema_no_slot = (
        '{"triplets":[{"subject":"пользователь","relation":"работает как","object":"водитель такси"}]}'
    )
    output_schema_with_slot = (
        '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер"}]}'
    )

    output_schema = output_schema_with_slot if include_slot else output_schema_no_slot

    system = (
        slot_header
        + "ТЫ СИСТЕМА ИЗВЛЕЧЕНИЯ ФАКТОВ ИЗ РЕПЛИКИ ПОЛЬЗОВАТЕЛЯ.\n"
        "ПРЕДСТАВЬ ФАКТЫ КАК ТРИПЛЕТЫ: СУБЪЕКТ, СВЯЗЬ, ОБЪЕКТ.\n"
        "СУБЪЕКТ, СВЯЗЬ И ОБЪЕКТ ПИШИ СТРОЧНЫМИ БУКВАМИ (lowercase).\n"
        "НЕ ИСПОЛЬЗУЙ СИМВОЛ ПОДЧЁРКИВАНИЯ «_» — РАЗДЕЛЯЙ СЛОВА ТОЛЬКО ПРОБЕЛАМИ.\n"
        "ДЛЯ ФАКТОВ О САМОМ ПОЛЬЗОВАТЕЛЕ ИСПОЛЬЗУЙ СУБЪЕКТ: пользователь.\n"
        "РАЗРЕШИ КОРЕФЕРЕНЦИЮ ВНУТРИ СООБЩЕНИЯ.\n"
        "НЕ ВЫДУМЫВАЙ ФАКТЫ.\n"
        "ИГНОРИРУЙ ЧИСТЫЕ ЭМОЦИИ БЕЗ ПРОВЕРЯЕМЫХ ФАКТОВ.\n"
        f"ОНТОЛОГИЯ СЛОТОВ (СПРАВОЧНО): {slots_ru_json}\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON. БЕЗ MARKDOWN. БЕЗ ТЕКСТА ВНЕ JSON.\n"
        "СХЕМА ОТВЕТА:\n"
        f"{output_schema}\n"
        f"МАКСИМУМ ТРИПЛЕТОВ: {max_triplets}.\n"
        'ЕСЛИ НЕТ УСТОЙЧИВЫХ ФАКТОВ: {"triplets":[]}.'
    )

    def user_turn_no_slot(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}\n\nИзвлеки триплеты."

    def user_turn_with_slot(msg: str) -> str:
        return (
            f"Слот: {ru_slot}\n"
            f"Сообщение пользователя:\n{msg}\n\n"
            f"Извлеки триплеты только для слота «{ru_slot}»."
        )

    if include_slot:
        # Single-pass mode: extract all slots at once, slot field appears in JSON output.
        few_shot = triplet_single_pass_few_shot_messages(user_turn_no_slot)
        user_turn = user_turn_no_slot
    elif slot_name and ru_slot:
        # Per-slot mode: shared examples without slot hint, per-slot examples with slot hint.
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_with_slot, slot_name
        )
        user_turn = user_turn_with_slot
    else:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_no_slot, None
        )
        user_turn = user_turn_no_slot

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
