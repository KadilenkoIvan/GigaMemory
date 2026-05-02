from __future__ import annotations

import json
from typing import Any, Dict, List

from ...slots.ontology import DEFAULT_USER_SLOTS
from .prompt_fewshots import slot_select_few_shot_messages


def build_slot_select_messages(
    user_message: str,
    *,
    ontology_slots: List[str] | None = None,
    max_slots: int = 5,
) -> List[Dict[str, Any]]:
    names = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_json = json.dumps([s.upper() for s in names], ensure_ascii=False)

    system = (
        "You are a classifier for long-term user-memory slots.\n"
        "Pick slots from the fixed ontology that the message belongs to.\n"
        "Answer with valid JSON only, no markdown, no commentary.\n\n"
        "Rules:\n"
        "1. Choose only slots with stable, useful information about the user.\n"
        "2. Multiple slots are allowed.\n"
        "3. If there are no facts about the user, their environment, property, or related things — empty list.\n"
        "4. On indirect mentions of such facts, pick slots that could plausibly relate.\n"
        "5. Slot names in the answer must be canonical English keys from the list below, UPPERCASE as stored in memory (e.g. FAMILY, WORK).\n\n"
        f"Allowed slots: {slots_json}\n"
        f"Maximum slots in the answer: {max_slots}\n\n"
        'Response schema: {"slot_assignments":["SLOT1","SLOT2", ...]}'
    )

    def user_turn(msg: str) -> str:
        return f"User message:\n{msg}"

    few_shot = slot_select_few_shot_messages(user_turn, lowercase_slots=False)

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
