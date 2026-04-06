from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SlotOntology:
    """
    Fixed slot ontology.

    Slot names are short, UPPERCASE, and represent everyday areas of life.
    Each slot corresponds to a "subgraph" of a single user knowledge graph.
    """

    slot_names: List[str]

    def normalize(self, name: str) -> str:
        s = (name or "").strip().upper()
        if not s:
            return ""
        # allow minor formatting variants (spaces/dashes/underscores)
        s = s.replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s

    def resolve(self, name: str) -> str:
        """
        Resolve a candidate slot name to a canonical ontology slot.
        Returns "" if the slot is not in ontology.
        """
        cand = self.normalize(name)
        if not cand:
            return ""
        allowed = set(self.slot_names)
        if cand in allowed:
            return cand
        return ""

    def as_prompt_list(self) -> str:
        return ", ".join(self.slot_names)


DEFAULT_USER_SLOTS = SlotOntology(
    slot_names=[
        "IDENTITY",
        "FAMILY",
        "FRIENDS",
        "ROMANCE",
        "WORK",
        "EDUCATION",
        "FINANCE",
        "HEALTH",
        "MENTAL_HEALTH",
        "HABITS",
        "PREFERENCES",
        "HOBBIES",
        "SPORTS",
        "FOOD",
        "HOME",
        "LOCATION",
        "TRAVEL",
        "PETS",
        "TECH",
        "VEHICLES",
        "SCHEDULE",
        "GOALS",
        "EVENTS",
    ]
)


def resolve_slot_or_none(name: str, *, ontology: SlotOntology = DEFAULT_USER_SLOTS) -> Optional[str]:
    resolved = ontology.resolve(name)
    return resolved or None


def filter_resolve_slots(names: Iterable[str], *, ontology: SlotOntology = DEFAULT_USER_SLOTS) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for n in names:
        r = ontology.resolve(n)
        if not r or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out

