from __future__ import annotations

import json
from typing import Any, Dict, List

from .ontology import RU_SLOT_LABELS_ORDERED
from .prompt_fewshots_ru import slot_select_few_shot_messages


def build_slot_select_messages(
    user_message: str,
    *,
    ontology_slots: List[str] | None = None,
    max_slots: int = 5,
) -> List[Dict[str, Any]]:
    _ = ontology_slots
    slots_json = json.dumps(RU_SLOT_LABELS_ORDERED, ensure_ascii=False)

    system = (
        "ТЫ КЛАССИФИКАТОР СЛОТОВ ДОЛГОВРЕМЕННОЙ ПАМЯТИ ПОЛЬЗОВАТЕЛЯ.\n"
        "ВЫБЕРИ ИЗ ФИКСИРОВАННОЙ ОНТОЛОГИИ СЛОТЫ, К КОТОРЫМ ОТНОСИТСЯ СООБЩЕНИЕ.\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON, БЕЗ MARKDOWN, БЕЗ ПОЯСНЕНИЙ.\n\n"
        "ПРАВИЛА:\n"
        "ВЫБИРАЙ ТОЛЬКО СЛОТЫ С УСТОЙЧИВОЙ ПОЛЕЗНОЙ ИНФОРМАЦИЕЙ О ПОЛЬЗОВАТЕЛЕ.\n"
        "НЕСКОЛЬКО СЛОТОВ РАЗРЕШЕНО.\n"
        "ЕСЛИ НЕТ СТАБИЛЬНЫХ ФАКТОВ О ПОЛЬЗОВАТЕЛЕ — ПУСТОЙ СПИСОК.\n"
        "ИМЕНА СЛОТОВ В ОТВЕТЕ — РУССКИЕ МЕТКИ ИЗ СПИСКА НИЖЕ (КАК В СПИСКЕ).\n\n"
        f"ДОПУСТИМЫЕ СЛОТЫ (РУССКИЕ МЕТКИ): {slots_json}\n"
        f"МАКСИМУМ СЛОТОВ В ОТВЕТЕ: {max_slots}\n\n"
        'СХЕМА ОТВЕТА: {"slot_assignments":["СЕМЬЯ","РАБОТА"]}'
    )

    def user_turn(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}"

    few_shot = slot_select_few_shot_messages(user_turn)

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
