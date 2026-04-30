from __future__ import annotations

import json
from typing import Any, Dict, List

from ..slots.ontology import RU_SLOT_LABELS_ORDERED
from .prompt_fewshots_ru import slot_select_few_shot_messages


def build_slot_select_messages(
    user_message: str,
    *,
    ontology_slots: List[str] | None = None,
    max_slots: int = 5,
) -> List[Dict[str, Any]]:
    _ = ontology_slots
    slots_json = json.dumps([s.lower() for s in RU_SLOT_LABELS_ORDERED], ensure_ascii=False)

    system = (
        "Ты классификатор слотов долговременной памяти пользователя.\n"
        "Выбери из фиксированной онтологии слоты, к которым относится сообщение.\n"
        "Ответ только валидный json, без markdown, без пояснений.\n\n"
        "Правила:\n"
        "1. Выбирай только слоты с устойчивой полезной информацией о пользователе.\n"
        "2. Несколько слотов разрешено.\n"
        "3. Если нет фактов о пользователе, его окружении, имуществе или относящихся к нему вещах — пустой список.\n"
        "4. При косвенном, не прямом упоминании факта о пользователе, его окружении, имуществе или относящихся к нему вещах, выбирай слоты, которые могут быть связаны с этим фактом.\n"
        "5. Имена слотов в ответе — русские метки из списка ниже, строго в верхнем регистре (как в списке).\n\n"
        f"Допустимые слоты (русские метки): {slots_json}\n"
        f"Максимум слотов в ответе: {max_slots}\n\n"
        'Схема ответа: {"slot_assignments":["slot1","slot2", ...]}'
    )

    def user_turn(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}"

    few_shot = slot_select_few_shot_messages(user_turn, lowercase_slots=False)

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
