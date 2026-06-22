"""
Variant B: separate LLM call for deletion detection (English UI).

Used when triplet_deletion_mode=\"llm_separate\".
Input: current slot facts + new user message.
Output: {\"delete\": [{subject, relation, object}, ...]}

Always receives current facts context.
Slot ids in prompts are canonical English (LOCATION, WORK, …).
Fact lines and JSON use English lemmas when prompt_language is en.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

_DELETION_FEWSHOTS: list[tuple[str, str, str, str]] = [
    (
        "LOCATION",
        "user | residence | moscow",
        "I don't live in Moscow anymore, I moved out",
        '{"delete":[{"subject":"user","relation":"residence","object":"moscow"}]}',
    ),
    (
        "WORK",
        "user | works as | engineer\nuser | workplace | yandex",
        "I left Yandex a month ago",
        '{"delete":[{"subject":"user","relation":"workplace","object":"yandex"}]}',
    ),
    (
        "ROMANCE",
        "user | dating | user's girlfriend\nuser's girlfriend | name | katya",
        "Katya and I broke up",
        '{"delete":[{"subject":"user","relation":"dating","object":"user\'s girlfriend"},{"subject":"user\'s girlfriend","relation":"name","object":"katya"}]}',
    ),
    (
        "PETS",
        "user | has cat | user's cat\nuser's cat | name | ryzhik",
        "Thanks, got it",
        '{"delete":[]}',
    ),
    (
        "HABITS",
        "user | smokes | yes\nuser | amount | pack a day",
        "I quit smoking three weeks ago",
        '{"delete":[{"subject":"user","relation":"smokes","object":"yes"},{"subject":"user","relation":"amount","object":"pack a day"}]}',
    ),
]


def build_deletion_messages(
    user_message: str,
    slot_name: str,
    existing_triplets: list[str],
) -> list[dict[str, Any]]:
    facts_block = "\n".join(existing_triplets) if existing_triplets else "(no facts)"

    system = (
        f"SLOT (canonical id): {slot_name}.\n"
        "YOU DETECT OBSOLETE FACTS.\n"
        "YOU RECEIVE CURRENT SLOT FACTS AND THE NEW USER MESSAGE.\n"
        "DECIDE WHICH FACTS SHOULD BE REMOVED BASED ON THE MESSAGE.\n\n"
        "RULES:\n"
        "  - Remove only facts clearly contradicted or cancelled by the new message.\n"
        "  - If the user updates a fact — remove the OLD fact.\n"
        "  - If the user says a fact is no longer valid — remove it.\n"
        "  - If the message cancels nothing — return empty delete.\n"
        "  - Do NOT remove facts based on guesses.\n"
        "  - Do NOT invent facts — only those present in the list.\n\n"
        "OUTPUT ONLY VALID JSON. NO MARKDOWN. NO TEXT OUTSIDE JSON.\n"
        'SCHEMA: {"delete":[{"subject":"...","relation":"...","object":"..."}]}\n'
        'IF NOTHING TO REMOVE: {"delete":[]}'
    )

    def _user_turn(slot_id: str, facts: str, msg: str) -> str:
        return (
            f"Slot: {slot_id}\n"
            f"Current facts:\n{facts}\n\n"
            f"New message: {msg}\n\n"
            "Which facts should be removed?"
        )

    few_shots: list[dict[str, Any]] = []
    for fs_slot, fs_facts, fs_msg, fs_ans in _DELETION_FEWSHOTS:
        few_shots.append(
            {"role": "user", "content": _user_turn(fs_slot, fs_facts, fs_msg)}
        )
        few_shots.append({"role": "assistant", "content": fs_ans})

    return (
        [{"role": "system", "content": system}]
        + few_shots
        + [
            {
                "role": "user",
                "content": _user_turn(slot_name, facts_block, user_message),
            }
        ]
    )
