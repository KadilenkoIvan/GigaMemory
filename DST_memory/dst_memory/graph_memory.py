from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GraphEdge:
    subject: str
    relation: str
    object: str
    slot: str
    dialogue_id: str
    source_text: str


@dataclass
class DialogueGraph:
    dialogue_id: str
    edges: List[GraphEdge] = field(default_factory=list)


class GraphMemory:
    """
    Lightweight in-process graph memory.
    Designed as a bridge before deeper RAGU index integration.
    """

    def __init__(self) -> None:
        self._by_dialogue: Dict[str, DialogueGraph] = {}

    def _get(self, dialogue_id: str) -> DialogueGraph:
        if dialogue_id not in self._by_dialogue:
            self._by_dialogue[dialogue_id] = DialogueGraph(dialogue_id=dialogue_id)
        return self._by_dialogue[dialogue_id]

    def upsert_triplets(
        self,
        dialogue_id: str,
        slot: str,
        source_text: str,
        triplets: List[Dict[str, str]],
    ) -> None:
        g = self._get(dialogue_id)
        for t in triplets:
            s = str(t.get("subject", "")).strip()
            r = str(t.get("relation", "")).strip()
            o = str(t.get("object", "")).strip()
            if not s or not r or not o:
                continue
            g.edges.append(
                GraphEdge(
                    subject=s,
                    relation=r,
                    object=o,
                    slot=slot,
                    dialogue_id=dialogue_id,
                    source_text=source_text,
                )
            )

    def slot_subgraph(self, dialogue_id: str, slots: List[str]) -> List[Dict[str, str]]:
        g = self._get(dialogue_id)
        allowed = set(slots)
        return [
            {
                "subject": e.subject,
                "relation": e.relation,
                "object": e.object,
                "slot": e.slot,
                "source_text": e.source_text,
            }
            for e in g.edges
            if e.slot in allowed
        ]

    def clear_dialogue(self, dialogue_id: str) -> None:
        self._by_dialogue.pop(dialogue_id, None)

    def as_ragu_artifacts(
        self,
        dialogue_id: str,
        slots: List[str] | None = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Exports in a shape close to RAGU Entity/Relation artifacts.
        """
        g = self._get(dialogue_id)
        edges = g.edges
        if slots:
            allowed = set(slots)
            edges = [e for e in edges if e.slot in allowed]
        entities_map: Dict[str, Dict] = {}
        relations: List[Dict] = []
        for e in edges:
            for name in (e.subject, e.object):
                key = name.lower()
                if key not in entities_map:
                    entities_map[key] = {
                        "entity_name": name,
                        "entity_type": "USER_MEMORY_ENTITY",
                        "description": f"Извлечено из памяти пользователя: {name}",
                    }
            relations.append(
                {
                    "subject_name": e.subject,
                    "object_name": e.object,
                    "relation_type": e.relation.upper().replace(" ", "_"),
                    "description": f"{e.subject} {e.relation} {e.object}",
                    "relation_strength": 1.0,
                    "slot": e.slot,
                }
            )
        return list(entities_map.values()), relations

