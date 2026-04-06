"""
Prompt for triplet-level conflict resolution.

Logic:
  Rule layer (no LLM): exact (subject, relation) match → auto-deactivate old record.
  LLM layer: same subject, different relation, potential semantic conflict → ask LLM.

LLM input:  slot name, existing active triplets (with record_id), new incoming triplets (indexed).
LLM output: {"deactivate": [record_ids_to_deactivate], "skip_new": [new_indices_to_drop]}

Default: add all new triplets, keep all existing. LLM specifies only exceptions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .prompt_fewshots_ru import CONFLICT_RESOLUTION_FEWSHOT


def build_conflict_messages(
    slot_name: str,
    existing_triplets: List[Dict[str, Any]],
    new_triplets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build LLM messages for conflict resolution.

    existing_triplets: list of dicts with keys record_id, subject, relation, object.
    new_triplets: list of dicts with keys idx, subject, relation, object.
    """
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
        "DO NOT skip new triplets that add genuinely new information.\n\n"
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


def parse_conflict_response(text: str) -> Dict[str, List[int]]:
    """Parse model response → {deactivate: [...], skip_new: [...]}."""
    # Strip markdown fences if model adds them
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    return {
        "deactivate": [int(x) for x in data.get("deactivate", [])],
        "skip_new": [int(x) for x in data.get("skip_new", [])],
    }
