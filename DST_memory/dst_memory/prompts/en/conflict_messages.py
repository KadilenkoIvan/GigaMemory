"""
English UI for triplet-level conflict resolution.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .prompt_fewshots import CONFLICT_RESOLUTION_FEWSHOT


def build_conflict_messages(
    slot_name: str,
    existing_triplets: List[Dict[str, Any]],
    new_triplets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    system = (
        "YOU ARE A MEMORY CONFLICT RESOLVER FOR A USER KNOWLEDGE GRAPH.\n"
        "You receive EXISTING active triplets (with record_id) and NEW incoming triplets (with idx).\n"
        "Your task: detect contradictions, superseded facts, and exact duplicates.\n\n"
        "OUTPUT ONLY VALID JSON. NO MARKDOWN. NO EXPLANATION.\n\n"
        "SCHEMA: {\"deactivate\": [<record_id>, ...], \"skip_new\": [<idx>, ...]}\n\n"
        "RULES:\n"
        "DEACTIVATE an existing record when the new triplet supersedes it "
        "(same subject, semantically equivalent or contradicting relation — old value is now wrong or outdated).\n"
        "SKIP a new triplet (add its idx to skip_new) when it is an exact duplicate of an existing record "
        "and adds no new information.\n"
        "IF there is no conflict and no duplicate — return {\"deactivate\":[], \"skip_new\":[]}.\n"
        "DO NOT deactivate facts that remain valid alongside the new ones.\n"
        "DO NOT skip new triplets that add genuinely new information.\n"
        "IMPORTANT: If the existing and new triplet have the SAME subject and SAME object but "
        "DIFFERENT relations, they express COMPLEMENTARY information about the same entity "
        "(e.g. 'есть партнёр' and 'живёт вместе с' for the same partner) — "
        "keep BOTH, return {\"deactivate\":[], \"skip_new\":[]}.\n"
        "CRITICAL: if you deactivate an existing record BECAUSE a new triplet replaces it,"
        "this new triplet SHOULD NOT be included in skip_new — it carries true information and should be inserted.\n"
        "YOU CANNOT simultaneously deactivate a record and skip a new triplet that caused the deactivation.\n"
        "If you deactivated an existing entry because a new triplet replaces it, then you need to insert all the information provided that relates to the new triplet.\n\n"
        f"SLOT: {slot_name}\n"
    )

    def user_turn(existing: str, new: str) -> str:
        return (
            f"Existing triplets: {existing}\n"
            f"New triplets: {new}"
        )

    few_shot: List[Dict[str, Any]] = []
    for existing_json, new_json, answer in CONFLICT_RESOLUTION_FEWSHOT:
        few_shot.append({"role": "user", "content": user_turn(existing_json, new_json)})
        few_shot.append({"role": "assistant", "content": answer})

    existing_json = json.dumps(existing_triplets, ensure_ascii=False)
    new_json = json.dumps(new_triplets, ensure_ascii=False)
    final = {"role": "user", "content": user_turn(existing_json, new_json)}

    return [{"role": "system", "content": system}] + few_shot + [final]
