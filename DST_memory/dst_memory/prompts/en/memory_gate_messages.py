"""Memory gate prompts — English UI; alternating user + assistant few-shots."""

from __future__ import annotations

from typing import Dict, List

from .prompt_fewshots import (
    MEMORY_GATE_FEWSHOT,
    MEMORY_GATE_FEWSHOT_VECTOR,
    memory_gate_user_block,
)


def build_memory_gate_messages(
    user_message: str,
    slot_names: list[str],
    *,
    for_vector_context: bool = False,
) -> list[dict[str, str]]:
    extra_block = ""
    if for_vector_context:
        extra_block = (
            " IF MEMORY IS NEEDED FOR THE ANSWER BUT NO SPECIFIC SLOT FROM THE LIST FITS — "
            "set use_memory: true and slots: []."
        )

    system = (
        "YOU HELP SELECT RELEVANT LONG-TERM MEMORY.\n"
        "YOU ARE GIVEN THE USER MESSAGE AND A LIST OF SLOT NAMES (NAMES ONLY, NO CONTENT).\n"
        "DECIDE WHETHER SAVED MEMORY IS NEEDED TO ANSWER.\n"
        "IF THE QUESTION IS ABSTRACT, GENERAL, OR NOT PERSONAL — MEMORY IS NOT NEEDED.\n"
        "OTHERWISE — SELECT ALL SLOTS THAT MIGHT HELP GIVE A BETTER ANSWER.\n"
        "RULE: BETTER TO INCLUDE AN EXTRA SLOT THAN MISS A USEFUL ONE.\n"
        "LIST SELECTED SLOT NAMES EXACTLY AS IN THE LIST.\n"
        "RETURN EXACTLY ONE JSON OBJECT, NO MARKDOWN, NO PROSE:\n"
        '{"use_memory": true or false, "slots": ["SLOT_NAME", ...]}\n'
        'WHEN use_memory=false THE "slots" ARRAY MUST BE [].'
    )

    few_shot: list[dict[str, str]] = []
    for question, slots_block, assistant_json in MEMORY_GATE_FEWSHOT:
        slot_list = [s.strip() for s in slots_block.split("\n") if s.strip()]
        few_shot.append(
            {
                "role": "user",
                "content": memory_gate_user_block(question, slot_list, ""),
            }
        )
        few_shot.append({"role": "assistant", "content": assistant_json})

    if for_vector_context:
        for question, slots_block, assistant_json in MEMORY_GATE_FEWSHOT_VECTOR:
            slot_list = [s.strip() for s in slots_block.split("\n") if s.strip()]
            few_shot.append(
                {
                    "role": "user",
                    "content": memory_gate_user_block(
                        question, slot_list, extra_block.strip()
                    ),
                }
            )
            few_shot.append({"role": "assistant", "content": assistant_json})

    final_user = memory_gate_user_block(user_message, slot_names, extra_block)

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": final_user}]
    )
