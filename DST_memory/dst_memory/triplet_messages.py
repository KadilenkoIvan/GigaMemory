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

    slot_constraint = ""
    if slot_name:
        ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name)
        slot_constraint = (
            f"\nТЕКУЩИЙ СЛОТ (СТРОГО): {ru_slot} ({slot_name}).\n"
            "ИЗВЛЕКАЙ ТОЛЬКО ФАКТЫ, ОТНОСЯЩИЕСЯ К ЭТОМУ СЛОТУ.\n"
            "В JSON НЕ УКАЗЫВАЙ ПОЛЕ slot — СЛОТ ЗАДАЁТСЯ СИСТЕМОЙ.\n"
            "ЕСЛИ ДЛЯ ЭТОГО СЛОТА НЕТ ФАКТОВ — ПУСТОЙ СПИСОК triplets.\n"
        )

    output_schema_no_slot = '{"triplets":[{"subject":"ПОЛЬЗОВАТЕЛЬ","relation":"РАБОТАЕТ_КАК","object":"водитель такси"}]}'
    output_schema_with_slot = (
        '{"triplets":[{"slot":"РАБОТА","subject":"ПОЛЬЗОВАТЕЛЬ","relation":"РАБОТАЕТ_КАК","object":"инженер"}]}'
    )

    output_schema = output_schema_with_slot if include_slot else output_schema_no_slot

    system = (
        "ТЫ СИСТЕМА ИЗВЛЕЧЕНИЯ ФАКТОВ ИЗ РЕПЛИКИ ПОЛЬЗОВАТЕЛЯ.\n"
        "ПРЕДСТАВЬ ФАКТЫ КАК ТРИПЛЕТЫ: СУБЪЕКТ, СВЯЗЬ, ОБЪЕКТ.\n"
        "СВЯЗЬ — КРАТКАЯ РУССКАЯ МЕТКА В ВЕРХНЕМ РЕГИСТРЕ С ПОДЧЁРКИВАНИЯМИ.\n"
        "ДЛЯ ФАКТОВ О САМОМ ПОЛЬЗОВАТЕЛЕ ИСПОЛЬЗУЙ СУБЪЕКТ ПОЛЬЗОВАТЕЛЬ.\n"
        "РАЗРЕШИ КОРЕФЕРЕНЦИЮ ВНУТРИ СООБЩЕНИЯ.\n"
        "НЕ ВЫДУМЫВАЙ ФАКТЫ.\n"
        "ИГНОРИРУЙ ЧИСТЫЕ ЭМОЦИИ БЕЗ ПРОВЕРЯЕМЫХ ФАКТОВ.\n"
        f"ОНТОЛОГИЯ СЛОТОВ (СПРАВОЧНО, ДЛЯ ОБЩЕГО РЕЖИМА): {slots_ru_json}\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON. БЕЗ MARKDOWN. БЕЗ ТЕКСТА ВНЕ JSON.\n"
        "СХЕМА ОТВЕТА:\n"
        f"{output_schema}\n"
        f"МАКСИМУМ ТРИПЛЕТОВ: {max_triplets}.\n"
        'ЕСЛИ НЕТ УСТОЙЧИВЫХ ФАКТОВ: {"triplets":[]}.'
        f"{slot_constraint}"
    )

    def user_turn(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}\n\nИзвлеки триплеты."

    if include_slot:
        few_shot = triplet_single_pass_few_shot_messages(user_turn)
    else:
        few_shot = triplet_per_slot_few_shot_messages(user_turn, slot_name)

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
