from __future__ import annotations

import json
from typing import Any, Dict, List

from .ontology import DEFAULT_USER_SLOTS


def build_slot_select_messages(
    user_message: str,
    *,
    ontology_slots: List[str] | None = None,
    max_slots: int = 5,
) -> List[Dict[str, Any]]:
    slots = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_json = json.dumps(slots, ensure_ascii=False)

    system = (
        "You are a slot classifier for user long-term memory.\n"
        "Pick which slots from a fixed ontology are relevant to the user message.\n"
        "Return ONLY JSON.\n\n"
        "Rules:\n"
        "- Select only slots that are truly about the user and useful later.\n"
        "- Multiple slots are allowed.\n"
        "- If there is no stable user-related information, return empty list.\n"
        "- Use EXACT slot names from ontology, uppercase.\n\n"
        f"Allowed slots: {slots_json}\n"
        f"Max slots in response: {max_slots}\n\n"
        'Output schema: {"slot_assignments": ["FAMILY", "WORK"]}'
    )

    def user_turn(msg: str) -> str:
        return f"Message:\n```text\n{msg}\n```"

    few_shot = [
        {"role": "user", "content": user_turn("мы с женой женаты 10 лет, у нас есть сын")},
        {"role": "assistant", "content": '{"slot_assignments":["FAMILY"]}'},
        {"role": "user", "content": user_turn("работаю водителем такси и по выходным играю в футбол")},
        {"role": "assistant", "content": '{"slot_assignments":["WORK","SPORTS"]}'},
        {"role": "user", "content": user_turn("окей, понял, спасибо")},
        {"role": "assistant", "content": '{"slot_assignments":[]}'},
    ]

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )

