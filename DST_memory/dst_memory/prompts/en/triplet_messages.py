"""Triplet extraction prompts — English UI; lemmas are lowercase English (storage format)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...slots.ontology import (
    DEFAULT_USER_SLOTS,
    TRIPLET_CLARIFY_EVENTS_TRAVEL_CHAIN,
    TRIPLET_CLARIFY_FAMILY_ROMANCE_BOUNDARY,
    TRIPLET_CLARIFY_KINSHIP_FOREIGN_FAMILY,
    triplet_prompt_show_clarification,
)
from .prompt_fewshots import (
    triplet_context_few_shot_messages,
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
    ttl_mode: str = "mode2",
    existing_triplets: Optional[List[str]] = None,
    enable_deletion: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build chat messages for triplet extraction (English prompt pack).

    Parameters
    ----------
    existing_triplets :
        Current active slot facts as "subject | relation | object" lines.
        If passed (including empty list), a context block is added.
        None means no context (legacy behaviour).
    enable_deletion :
        Extend the response schema with "delete" for explicit removals.
    """
    _ = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_catalog_json = json.dumps(DEFAULT_USER_SLOTS.slot_names, ensure_ascii=False)

    use_context = existing_triplets is not None
    use_deletion = enable_deletion

    slot_header = ""
    if slot_name:
        slot_header = (
            f"Current slot: {slot_name}.\n"
            f"Extract only facts that belong to slot \"{slot_name}\".\n"
            f"Do not include facts that belong to other slots, even if they appear in the message.\n"
            f"If there are no facts for this slot in the message — return {{\"triplets\":[]}}.\n"
            "Do not output a \"slot\" field in JSON — the slot is fixed by the system.\n\n"
        )

    slot_boundary_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_FAMILY_ROMANCE_BOUNDARY):
        slot_boundary_block = (
            "Boundary between slots FAMILY and ROMANCE:\n"
            "  FAMILY — blood relatives and legal family OF THE USER:\n"
            "    husband, wife, son, daughter, mother, father, brother, sister, grandparents, etc.\n"
            "  ROMANCE — romantic relationships:\n"
            "    boyfriend, girlfriend, partner, love interest, ex, etc.\n"
            "  Rule: wife/husband → FAMILY; girlfriend/boyfriend → ROMANCE. Never swap.\n"
            "  SOMEONE ELSE'S FAMILY (friend's family, colleague's, another person's)\n"
            "    → MUST NOT go into the user's FAMILY slot.\n"
            "    Example: \"my friend has a sister\" — do NOT add to the user's FAMILY; "
            "add to FRIENDS if the current slot is FRIENDS.\n\n"
        )

    events_travel_chain_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_EVENTS_TRAVEL_CHAIN):
        events_travel_chain_block = (
            "For EVENTS and TRAVEL — chain (place/event becomes subject of its attributes):\n"
            "  user → action → place or event\n"
            "  place or event → attribute → value\n"
            "  Wrong: {\"subject\":\"user\",\"relation\":\"trip\",\"object\":\"tokyo september\"}\n"
            "  Right: {\"subject\":\"user\",\"relation\":\"trip\",\"object\":\"tokyo\"}\n"
            "         {\"subject\":\"tokyo\",\"relation\":\"date\",\"object\":\"september\"}\n"
            "         {\"subject\":\"tokyo\",\"relation\":\"travels with\",\"object\":\"family\"}\n\n"
        )

    kinship_foreign_family_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_KINSHIP_FOREIGN_FAMILY):
        kinship_foreign_family_block = (
            "Kinship accuracy:\n"
            "  Be clear who is whose. Subject is the one who \"has\" the relation.\n"
            "  If the relative belongs to someone else, not the user, do not describe them as the user's relative.\n"
            "  Someone else's family — do NOT add those facts to the user's FAMILY slot; "
            "use FRIENDS when the current slot is FRIENDS.\n\n"
        )

    context_block = ""
    if use_context:
        if existing_triplets:
            facts_lines = "\n".join(f"  {line}" for line in existing_triplets)
            if use_deletion:
                context_instructions = (
                    "Instructions for working with current facts:\n"
                    "  1. If a new fact replaces an old one — add it to \"triplets\" AND add\n"
                    "     the old fact to \"delete\". For history, also add to \"triplets\"\n"
                    "     a fact with a former/previous relation (e.g. \"former residence\").\n"
                    "  2. If the user explicitly revokes a fact without replacement — add the old fact\n"
                    "     to \"delete\". For history, add to \"triplets\" a former/previous fact.\n"
                    "  3. If the fact is only refined — update via \"delete\" + new \"triplets\".\n"
                    "  4. If the new message does not change known facts — \"delete\":[].\n"
                    "  5. Do not duplicate facts already listed in \"triplets\".\n\n"
                )
            else:
                context_instructions = (
                    "Instructions for working with current facts:\n"
                    "  1. Do not duplicate existing facts in \"triplets\" — only add new ones.\n"
                    "  2. If a fact changed — add the new fact and if needed a historical\n"
                    "     (former/previous) one. The system removes the old one automatically.\n"
                    "  3. If the message adds no new facts — return {\"triplets\":[]}.\n\n"
                )
            context_block = (
                f"Current facts in slot"
                + (f" \"{slot_name}\"" if slot_name else "")
                + " (do not duplicate them unless there is new information):\n"
                + facts_lines + "\n\n"
                + context_instructions
            )
        else:
            context_block = (
                "Current facts in slot"
                + (f" \"{slot_name}\"" if slot_name else "")
            )

    use_ttl = (ttl_mode == "mode2")

    if use_ttl:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"WORK","subject":"user","relation":"works as","object":"engineer","ttl":"1y"}]}'
                if not use_deletion else
                '{"triplets":[{"slot":"WORK","subject":"user","relation":"works as","object":"engineer","ttl":"1y"}],"delete":[{"subject":"user","relation":"works as","object":"driver"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"user","relation":"works as","object":"taxi driver","ttl":"1y"}]}'
                if not use_deletion else
                '{"triplets":[{"subject":"user","relation":"residence","object":"syzran","ttl":"1y"},{"subject":"user","relation":"former residence","object":"moscow","ttl":"1y"}],"delete":[{"subject":"user","relation":"residence","object":"moscow"}]}'
            )
        ttl_block = (
            "\nAdd a TTL field to every triplet (fact lifetime).\n"
            "Allowed TTL values: 6h, 12h, 1d, 3d, 10d, 2w, 3w, 1m, 3m, 6m, 1y, inf\n"
            "TTL guidelines:\n"
            "  inf  — name, gender, nationality, family, pets, stable habits (morning coffee)\n"
            "  1y   — work, study, housing, health (diagnoses), car, location\n"
            "  6m   — hobbies, sports, preferences, mental state, dating\n"
            "  3m   — goals, romance, financial plans\n"
            "  1m   — schedule, near-future plans, food\n"
            "  2w   — specific events (went to a wedding, passed an exam)\n"
            "  1d   — day-level states (stressful day, calm day, feel bad)\n"
            "  12h  — within-day mood (cheerful, down, productive)\n"
            "  6h   — short states (tired, energized, burned out, going to bed)\n"
        )
    else:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"WORK","subject":"user","relation":"works as","object":"engineer"}]}'
                if not use_deletion else
                '{"triplets":[{"slot":"WORK","subject":"user","relation":"works as","object":"engineer"}],"delete":[{"subject":"user","relation":"works as","object":"driver"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"user","relation":"works as","object":"taxi driver"}]}'
                if not use_deletion else
                '{"triplets":[{"subject":"user","relation":"residence","object":"syzran"}],"delete":[{"subject":"user","relation":"residence","object":"moscow"}]}'
            )
        ttl_block = ""

    delete_block = ""
    if use_deletion:
        delete_block = (
            "\nThe \"delete\" field — facts to remove from memory explicitly.\n"
            "Only include facts from the current slot's fact list.\n"
            "If nothing to remove — \"delete\":[].\n"
        )

    system = (
        slot_header
        + slot_boundary_block
        + "You extract facts from the user's message.\n"
        "Represent facts as triplets: subject, relation, object.\n"
        "Write subject, relation, and object in lowercase.\n"
        "Do not use underscores \"_\" — separate words with spaces only.\n"
        "For facts about the user themselves, use subject: user.\n"

        "When a related entity is mentioned (pet, family member, colleague, etc.):\n"
        "  1. Link triplet — always add the role first, then the name if present:\n"
        "     Wrong: {\"subject\":\"user\",\"relation\":\"has cat\",\"object\":\"ryzhik\"}\n"
        "     Right: {\"subject\":\"user\",\"relation\":\"has cat\",\"object\":\"user's cat\"}\n"
        "            {\"subject\":\"user's cat\",\"relation\":\"name\",\"object\":\"ryzhik\"}\n"
        "  2. Entity attributes — always use the role first, then the name if present:\n"
        "     Wrong: {\"subject\":\"ryzhik\",\"relation\":\"sick\",\"object\":\"yes\"}\n"
        "     Right: {\"subject\":\"user's cat\",\"relation\":\"name\",\"object\":\"ryzhik\"}\n"
        "            {\"subject\":\"ryzhik\",\"relation\":\"condition\",\"object\":\"sick\"}\n\n"
        + events_travel_chain_block
        + "Each triplet stands alone — understandable when read alone.\n"
        "  Subject and object must uniquely name the entity without context:\n"
        "  Wrong: {\"subject\":\"elder\",\"relation\":\"name\",\"object\":\"alesha\"}\n"
        "  Right: {\"subject\":\"user's elder son\",\"relation\":\"name\",\"object\":\"alesha\"}\n"
        "Do not pack several facts into one object.\n"
        "The relation must be unambiguous — use a chain:\n"
        "  Wrong: {\"subject\":\"user\",\"relation\":\"frequency\",\"object\":\"once a week\"}\n"
        "  Right: {\"subject\":\"user\",\"relation\":\"goes to\",\"object\":\"fishing\"}\n"
        "         {\"subject\":\"fishing\",\"relation\":\"frequency\",\"object\":\"once a week\"}\n\n"
        + kinship_foreign_family_block
        + "Do not invent facts — only what is explicitly stated.\n"
        "It is forbidden to create entries that mention the user but do not belong to the current slot.\n"
        "Do not record missing information — if a fact is not mentioned or unknown,\n"
        "  do not create triplets like \"user | hobby | unknown\",\n"
        "  Forbidden: \"user | age | not specified\", etc. Only concrete facts.\n\n"
        "VERY IMPORTANT: A fact can be indicated without an object, if it is obvious from the context, such facts need to be added.\n"
        "In the message, the fact can be indicated indirectly, you need to understand it from the context and add it.\n"
        "The message may contain not only a fact, but also a question, reasoning, emotions, assessments, etc. In this case, it is necessary to add facts, even if it is not mentioned directly.\n\n"
        + ttl_block
        + delete_block
        + context_block
        + f"Slot ontology (reference): {slots_catalog_json}\n"
        "Answer with valid JSON only, no markdown, no commentary.\n"
        "Response schema:\n"
        f"{output_schema}\n"
        f"Maximum triplets: {max_triplets}.\n"
        "If there are no facts: {"triplets":[]}.\n\n"
        "VERY IMPORTANT: A fact can be indicated without an object, if it is obvious from the context, such facts need to be added.\n"
        "In the message, the fact can be indicated indirectly, you need to understand it from the context and add it.\n"
        "The message may contain not only a fact, but also a question, reasoning, emotions, assessments, etc. In this case, it is necessary to add facts, even if it is not mentioned directly.\n\n"
    )

    def user_turn_no_slot(msg: str) -> str:
        return f"User message:\n{msg}\n\nExtract triplets."

    def user_turn_with_slot(msg: str) -> str:
        return (
            f"Slot: {slot_name}\n"
            f"User message:\n{msg}\n\n"
            f"Extract triplets only for slot \"{slot_name}\"."
        )

    def user_turn_with_context(msg: str) -> str:
        slot_part = f"Slot: {slot_name}\n" if slot_name else ""
        action = (
            "Extract new/changed facts and list facts to delete."
            if use_deletion
            else "Extract only new facts; do not duplicate existing ones."
        )
        return (
            f"{slot_part}"
            f"User message:\n{msg}\n\n"
            f"{action}"
        )

    if use_context:
        few_shot = triplet_context_few_shot_messages(
            user_turn_with_context, slot_name=slot_name, use_ttl=use_ttl,
            enable_deletion=use_deletion,
        )
        user_turn = user_turn_with_context
    elif include_slot:
        few_shot = triplet_single_pass_few_shot_messages(user_turn_no_slot, use_ttl=use_ttl)
        user_turn = user_turn_no_slot
    elif slot_name:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_with_slot, slot_name, use_ttl=use_ttl
        )
        user_turn = user_turn_with_slot
    else:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_no_slot, None, use_ttl=use_ttl
        )
        user_turn = user_turn_no_slot

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
