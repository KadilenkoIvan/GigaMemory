"""Triplet extraction prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .ontology import DEFAULT_USER_SLOTS


def build_triplet_messages(
    user_message: str,
    *,
    slot_name: str | None = None,
    include_slot: bool = False,
    ontology_slots: List[str] | None = None,
    max_triplets: int = 12,
) -> List[Dict[str, Any]]:
    slots = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_json = json.dumps(slots, ensure_ascii=False)

    slot_constraint = ""
    if slot_name:
        slot_constraint = (
            f"\nCURRENT SLOT IS {slot_name}.\n"
            "EXTRACT ONLY FACTS FOR THIS SLOT.\n"
            "IF MESSAGE HAS NO FACTS FOR THIS SLOT, RETURN EMPTY LIST.\n"
        )

    output_schema = '{"triplets":[{"subject":"USER","relation":"WORKS_AS","object":"Taxi driver"}]}'
    if include_slot:
        output_schema = (
            '{"triplets":[{"slot":"WORK","subject":"USER","relation":"WORKS_AS","object":"Taxi driver"}]}'
        )

    system = (
        "YOU ARE AN EXPERT INFORMATION EXTRACTION SYSTEM.\n"
        "EXTRACT STABLE USER-RELEVANT FACTS FROM ONE MESSAGE AS TRIPLETS.\n"
        "TRIPLET FORMAT IS SUBJECT, RELATION, OBJECT.\n"
        "A FACT MUST BE USEFUL IN FUTURE DIALOGUES.\n"
        "IGNORE PURE EMOTIONS WITHOUT CONCRETE FACTS.\n"
        "USE USER AS CANONICAL SUBJECT FOR USER FACTS.\n"
        "RESOLVE COREFERENCE INSIDE THE MESSAGE.\n"
        "DO NOT HALLUCINATE.\n"
        f"ALLOWED SLOT ONTOLOGY: {slots_json}\n"
        "RETURN ONLY VALID JSON. NO MARKDOWN. NO COMMENTS.\n"
        "OUTPUT SCHEMA:\n"
        f"{output_schema}\n"
        "RELATION LABELS MUST BE UPPER_SNAKE_CASE.\n"
        f"MAX TRIPLETS: {max_triplets}.\n"
        'IF NO STABLE FACTS, RETURN {"triplets":[]}.'
        f"{slot_constraint}"
    )

    def user_turn(msg: str) -> str:
        return (
            "User message (Russian):\n"
            f"```text\n{msg}\n```\n"
            "Extract triplets now."
        )

    few_shot_triplets_1 = [
        {"subject": "USER", "relation": "MARRIED_TO", "object": "WIFE"},
        {"subject": "USER", "relation": "HAS_CHILD", "object": "SON"},
        {"subject": "USER", "relation": "WORKS_AS", "object": "Taxi driver"},
    ]
    few_shot_triplets_2 = [
        {"subject": "USER", "relation": "HAS_PET", "object": "CAT_Barsik"},
        {"subject": "CAT_Barsik", "relation": "AFRAID_OF", "object": "Dogs"},
        {"subject": "USER", "relation": "AVOIDS", "object": "Meat"},
    ]
    if include_slot:
        few_shot_triplets_1 = [
            {"slot": "FAMILY", "subject": "USER", "relation": "MARRIED_TO", "object": "WIFE"},
            {"slot": "FAMILY", "subject": "USER", "relation": "HAS_CHILD", "object": "SON"},
            {"slot": "WORK", "subject": "USER", "relation": "WORKS_AS", "object": "Taxi driver"},
        ]
        few_shot_triplets_2 = [
            {"slot": "PETS", "subject": "USER", "relation": "HAS_PET", "object": "CAT_Barsik"},
            {"slot": "PETS", "subject": "CAT_Barsik", "relation": "AFRAID_OF", "object": "Dogs"},
            {"slot": "FOOD", "subject": "USER", "relation": "AVOIDS", "object": "Meat"},
        ]

    few_shot = [
        {
            "role": "user",
            "content": user_turn("мы с женой уже 10 лет женаты, у нас есть сын. я работаю водителем такси"),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "triplets": few_shot_triplets_1
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": user_turn("сегодня на работе был полный завал, ненавижу эти смены"),
        },
        {
            "role": "assistant",
            "content": '{"triplets":[]}',
        },
        {
            "role": "user",
            "content": user_turn("у меня кот Барсик, он боится собак. я не ем мясо"),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "triplets": few_shot_triplets_2
                },
                ensure_ascii=False,
            ),
        },
    ]

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )

